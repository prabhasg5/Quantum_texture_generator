"""Color and palette helper utilities."""

from __future__ import annotations

import torch


def extract_palette(image: torch.Tensor, n_colors: int = 5, max_iter: int = 10) -> torch.Tensor:
    """Derive a palette by running a lightweight k-means over image pixels."""

    if image.ndim != 3:
        raise ValueError("Image tensor must be C×H×W.")
    pixels = image.permute(1, 2, 0).reshape(-1, image.shape[0])
    if pixels.shape[0] < n_colors:
        padded = torch.zeros((n_colors, 3), dtype=image.dtype, device=image.device)
        padded[: pixels.shape[0]] = pixels
        return padded

    indices = torch.randperm(pixels.shape[0], device=image.device)[:n_colors]
    centers = pixels[indices]

    for _ in range(max_iter):
        distances = torch.cdist(pixels, centers)
        labels = distances.argmin(dim=1)
        new_centers = []
        for i in range(n_colors):
            cluster = pixels[labels == i]
            if cluster.numel() == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(cluster.mean(dim=0))
        new_centers_tensor = torch.stack(new_centers)
        if torch.allclose(new_centers_tensor, centers, atol=1e-4):
            centers = new_centers_tensor
            break
        centers = new_centers_tensor

    return centers.clamp(0.0, 1.0)
