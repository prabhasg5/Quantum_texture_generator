"""Classical latent generator baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    latent_dim: int = 8
    hidden_dim: int = 256
    depth: int = 6


class SineLayer(nn.Module):
    """Applies a linear transform followed by sine activation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sin(self.linear(x))


class ClassicalLatentGenerator(nn.Module):
    """Parameter-matched MLP replacing the PQC module."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        layers = [SineLayer(config.latent_dim, config.hidden_dim)]
        for _ in range(config.depth - 2):
            layers.append(SineLayer(config.hidden_dim, config.hidden_dim))
        layers.append(nn.Linear(config.hidden_dim, config.latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.network(conditioning)
        return torch.nn.functional.layer_norm(latent, latent.shape[-1:])
