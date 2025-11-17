from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pennylane as qml
import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    """DCGAN style weight init."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class QuantumStyleEncoder(nn.Module):
    """Parameterized quantum circuit executed on CPU, independent of torch device."""

    def __init__(
        self,
        wires: int = 6,
        layers: int = 2,
        backend: str = "default.qubit",
        torch_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.wires = wires
        self.backend = backend

        dev = self._make_device(backend, wires, torch_device)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(wires), rotation="Y")
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

        weight_shapes = {"weights": (layers, wires, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_device = inputs.device
        cpu_inputs = inputs.to(torch.device("cpu"))
        outputs = self.quantum_layer(cpu_inputs)
        return outputs.to(original_device)

    def to(self, *args, **kwargs):  # noqa: D401
        """Override .to() so the quantum layer always stays on CPU."""
        super().to(torch.device("cpu"))
        return self

    @staticmethod
    def _make_device(backend: str, wires: int, torch_device: str):
        try:
            if backend.endswith(".torch"):
                return qml.device(backend, wires=wires, torch_device=torch_device)
            return qml.device(backend, wires=wires)
        except Exception as exc:  # pylint: disable=broad-except
            fallback = "default.qubit"
            print(
                f"[QuantumStyleEncoder] Backend '{backend}' unavailable ({exc}). Falling back to '{fallback}'."
            )
            return qml.device(fallback, wires=wires)


class HybridGenerator(nn.Module):
    """Quantum-assisted conditional generator."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        image_channels: int = 3,
        base_channels: int = 64,
        quantum_wires: int = 6,
        quantum_layers: int = 2,
        class_emb_dim: int = 64,
        image_size: int = 128,
        quantum_backend: str = "default.qubit.torch",
        quantum_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_channels = image_channels
        self.image_size = image_size

        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        quantum_input_dim = latent_dim + class_emb_dim
        self.quantum_proj = nn.Sequential(
            nn.Linear(quantum_input_dim, quantum_wires),
            nn.LayerNorm(quantum_wires),
            nn.Tanh(),
        )
        self.quantum_encoder = QuantumStyleEncoder(
            wires=quantum_wires,
            layers=quantum_layers,
            backend=quantum_backend,
            torch_device=quantum_device,
        )

        fused_dim = latent_dim + class_emb_dim + quantum_wires
        feature_map_size = image_size // 32  # Start from 4x4 for 128px
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, base_channels * 16 * feature_map_size * feature_map_size),
            nn.BatchNorm1d(base_channels * 16 * feature_map_size * feature_map_size),
            nn.GELU(),
        )
        self.feature_map_size = feature_map_size

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        class_emb = self.class_embed(class_ids)
        quantum_inputs = torch.cat([noise, class_emb], dim=1)
        quantum_angles = self.quantum_proj(quantum_inputs)
        quantum_style = self.quantum_encoder(quantum_angles)
        fused = torch.cat([noise, class_emb, quantum_style], dim=1)
        x = self.fc(fused)
        x = x.view(x.shape[0], -1, self.feature_map_size, self.feature_map_size)
        return self.deconv(x)


class ConditionalDiscriminator(nn.Module):
    """Patch-style discriminator with class conditioning."""

    def __init__(self, num_classes: int, image_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, image_channels)

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(image_channels * 2, base_channels, normalize=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
        )

    def forward(self, images: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        class_map = self.class_embed(class_ids)
        class_map = class_map.unsqueeze(-1).unsqueeze(-1).expand_as(images)
        x = torch.cat([images, class_map], dim=1)
        logits = self.model(x)
        return logits.view(images.size(0), -1).mean(dim=1)


@dataclass
class ModelConfig:
    latent_dim: int = 96
    image_size: int = 128
    image_channels: int = 3
    base_channels: int = 64
    quantum_wires: int = 6
    quantum_layers: int = 2
    class_emb_dim: int = 64
    quantum_backend: str = "default.qubit"
    quantum_device: str = "cpu"


def build_models(num_classes: int, cfg: ModelConfig) -> Tuple[HybridGenerator, ConditionalDiscriminator]:
    generator = HybridGenerator(
        latent_dim=cfg.latent_dim,
        num_classes=num_classes,
        image_channels=cfg.image_channels,
        base_channels=cfg.base_channels,
        quantum_wires=cfg.quantum_wires,
        quantum_layers=cfg.quantum_layers,
        class_emb_dim=cfg.class_emb_dim,
        image_size=cfg.image_size,
        quantum_backend=cfg.quantum_backend,
        quantum_device=cfg.quantum_device,
    )
    discriminator = ConditionalDiscriminator(
        num_classes=num_classes,
        image_channels=cfg.image_channels,
        base_channels=cfg.base_channels,
    )
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator
