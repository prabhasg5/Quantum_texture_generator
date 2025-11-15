"""Palette encoding utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class PaletteEncoding(ABC):
    """Interface for palette encoders that output fixed-length embeddings."""

    @abstractmethod
    def encode(self, colors: torch.Tensor) -> torch.Tensor:
        """Encode an (N×3) palette tensor in RGB or Lab space into a 1D embedding."""


class SimplePaletteStats(PaletteEncoding):
    """Baseline palette encoder using simple channel statistics."""

    def __init__(self, n_colors: int = 7, color_space: str = "lab") -> None:
        self.n_colors = n_colors
        self.color_space = color_space

    def encode(self, colors: torch.Tensor) -> torch.Tensor:
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("Palette tensor must have shape (num_colors, 3).")

        padded = self._pad_palette(colors)
        mean = padded.mean(dim=0)
        std = padded.std(dim=0)
        return torch.cat([padded.flatten(), mean, std], dim=0)

    def _pad_palette(self, colors: torch.Tensor) -> torch.Tensor:
        if colors.shape[0] == self.n_colors:
            return colors
        if colors.shape[0] > self.n_colors:
            return colors[: self.n_colors]
        pad = torch.zeros((self.n_colors - colors.shape[0], 3), dtype=colors.dtype, device=colors.device)
        pad[:, 0] = colors[:, 0].mean() if colors.numel() else 0.0
        pad[:, 1] = colors[:, 1].mean() if colors.numel() else 0.0
        pad[:, 2] = colors[:, 2].mean() if colors.numel() else 0.0
        return torch.cat([colors, pad], dim=0)
