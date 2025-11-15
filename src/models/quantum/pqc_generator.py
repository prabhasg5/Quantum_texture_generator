"""Quantum latent generator implemented with PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pennylane as qml
import torch
import torch.nn.functional as F


def _default_device(n_qubits: int) -> qml.Device:
    return qml.device("lightning.qubit", wires=n_qubits, shots=None)


@dataclass
class PQCConfig:
    n_qubits: int = 8
    depth: int = 4
    latent_dim: int = 8
    shots: Optional[int] = None
    device_name: str = "lightning.qubit"


class QuantumLatentGenerator(torch.nn.Module):
    """Maps conditioning vectors into a quantum latent representation."""

    def __init__(self, config: PQCConfig) -> None:
        super().__init__()
        self.config = config
        self.dev = qml.device(config.device_name, wires=config.n_qubits, shots=config.shots)
        self.encoding_layers = torch.nn.Linear(config.latent_dim, config.n_qubits)
        self.reupload_layers = torch.nn.Linear(config.latent_dim, config.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            self._encode(inputs)
            qml.layer(self._variational_block, config.depth, weights)
            return [qml.expval(qml.PauliZ(wire)) for wire in range(config.latent_dim)]

        self.circuit = circuit
        weight_shapes = {"weights": (config.depth, config.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embedding = F.silu(self.encoding_layers(conditioning))
        latent = self.qlayer(embedding)
        latent = F.layer_norm(latent, latent.shape[-1:])
        return latent

    def _encode(self, data: torch.Tensor) -> None:
        for wire, value in enumerate(data):
            qml.RX(value, wires=wire)
            qml.RY(value, wires=wire)

    def _variational_block(self, weights: torch.Tensor) -> None:
        for wire, params in enumerate(weights):
            qml.RZ(params[0], wires=wire)
            qml.RX(params[1], wires=wire)
            qml.RY(params[2], wires=wire)
        for wire in range(self.config.n_qubits - 1):
            qml.CZ(wires=[wire, wire + 1])
