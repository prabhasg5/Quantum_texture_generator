"""Shared decoder that maps latent vectors to 32×32 texture tiles."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.silu(out + residual)


class TextureDecoder(nn.Module):
    """Converts latent codes into 32×32×3 RGB tiles."""

    def __init__(self, latent_dim: int, base_channels: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, base_channels * 4 * 4)
        self.blocks = nn.ModuleList([
            ResidualBlock(base_channels),
            ResidualBlock(base_channels // 2),
            ResidualBlock(base_channels // 4),
            ResidualBlock(base_channels // 8),
        ])
        self.upsample = nn.ModuleList([
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
        ])
        self.output = nn.Conv2d(base_channels // 8, 3, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = latent.shape[0]
        x = self.linear(latent).view(batch, -1, 4, 4)
        x = F.silu(x)
        x = self.blocks[0](x)
        x = self.upsample[0](x)
        x = self.blocks[1](x)
        x = self.upsample[1](x)
        x = self.blocks[2](x)
        x = self.upsample[2](x)
        x = self.blocks[3](x)
        x = torch.tanh(self.output(x))
        return x
