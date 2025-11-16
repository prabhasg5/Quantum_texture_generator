"""Conditioning layers for class and palette embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConditioningConfig:
    class_count: int
    class_embed_dim: int = 64
    palette_embed_dim: int = 21
    noise_dim: int = 16
    latent_dim: int = 8

    @property
    def input_dim(self) -> int:
        return self.class_embed_dim + self.palette_embed_dim + self.noise_dim


class ConditioningEncoder(nn.Module):
    def __init__(self, config: ConditioningConfig) -> None:
        super().__init__()
        self.class_embedding = nn.Embedding(config.class_count, config.class_embed_dim)
        self.palette_projector = nn.Linear(config.palette_embed_dim, config.palette_embed_dim)
        self.noise_dim = config.noise_dim
        self.projector = nn.Linear(config.input_dim, config.latent_dim)
        self.config = config

    def forward(self, class_id: torch.Tensor, palette_embedding: torch.Tensor) -> torch.Tensor:
        class_vec = self.class_embedding(class_id)
        palette_vec = torch.tanh(self.palette_projector(palette_embedding))
        noise = torch.randn(class_vec.size(0), self.noise_dim, device=class_vec.device)
        conditioning = torch.cat([class_vec, palette_vec, noise], dim=1)
        return F.silu(self.projector(conditioning))
