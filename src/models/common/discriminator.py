"""Conditional discriminator shared by quantum and classical experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionDiscriminator(nn.Module):
    def __init__(self, class_count: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Conv2d(256, 1, kernel_size=4)
        self.class_embed = nn.Embedding(class_count, embed_dim)
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, image: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(image)
        patch_scores = self.output(features)
        pooled = features.mean(dim=[2, 3])
        class_vec = self.class_embed(class_id)
        projection = (self.proj(pooled) * class_vec).sum(dim=1, keepdim=True)
        return patch_scores.view(image.shape[0], -1).mean(dim=1, keepdim=True) + projection
