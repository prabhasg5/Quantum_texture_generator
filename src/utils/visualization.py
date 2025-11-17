from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision.utils import make_grid, save_image


def denormalize(t: torch.Tensor) -> torch.Tensor:
    return t.mul(0.5).add(0.5).clamp(0, 1)


def plot_losses(history: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for label, values in history.items():
        plt.plot(values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_class_grid(
    generator,
    class_names: Sequence[str],
    device: torch.device,
    samples_per_class: int,
    latent_dim: int,
    out_path: Path,
) -> None:
    was_training = generator.training
    generator.eval()
    rows = []
    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            noise = torch.randn(samples_per_class, latent_dim, device=device)
            class_ids = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            imgs = generator(noise, class_ids)
            grid = make_grid(denormalize(imgs), nrow=samples_per_class)
            rows.append(grid)
    stacked = torch.cat(rows, dim=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(stacked, out_path)
    generator.train(was_training)


def save_lpips_hist(values: Sequence[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.histplot(values, bins=20, kde=True, color="#1f77b4")
    plt.xlabel("Nearest-train LPIPS distance")
    plt.ylabel("Frequency")
    plt.title("Novelty distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_coverage(real_scores, fake_scores, coverage: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.kdeplot(real_scores, label="Real radius", linestyle="--")
    sns.kdeplot(fake_scores, label="Real→Fake distance")
    plt.axvline(sum(real_scores) / len(real_scores), color="tab:green", linestyle=":", label="Real avg")
    plt.axvline(sum(fake_scores) / len(fake_scores), color="tab:red", linestyle=":", label="Fake avg")
    plt.title(f"Feature-space coverage: {coverage*100:.1f}%")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
