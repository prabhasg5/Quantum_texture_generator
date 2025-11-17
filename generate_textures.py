from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torchvision.utils import make_grid, save_image

from src.data.texture_dataset import TextureDataset
from src.models.qgan import ModelConfig, build_models
from src.utils.visualization import denormalize


def load_generator(checkpoint: Path, num_classes: int, device: torch.device):
    data = torch.load(checkpoint, map_location=device)
    model_cfg = ModelConfig(**data["config"]["model"])
    generator, _ = build_models(num_classes, model_cfg)
    generator.load_state_dict(data["generator"])
    generator.to(device)
    generator.eval()
    return generator, model_cfg


def main():
    parser = argparse.ArgumentParser(description="Generate textures from trained QGAN")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Dataset root to read class names")
    parser.add_argument("--class-name", type=str, default=None, help="Class to generate (defaults to all classes grid)")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of textures to create")
    parser.add_argument("--out", type=Path, default=Path("generated_samples.png"), help="Output image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dataset = TextureDataset(args.dataset_root, image_size=128, augment=False)
    class_names: List[str] = [dataset.idx_to_class[i] for i in range(len(dataset.idx_to_class))]

    generator, model_cfg = load_generator(args.checkpoint, len(class_names), device)

    if args.class_name and args.class_name not in dataset.class_to_idx:
        print("Available classes:")
        for name in class_names:
            print(f" - {name}")
        raise SystemExit(f"Unknown class: {args.class_name}")

    with torch.no_grad():
        if args.class_name:
            idx = dataset.class_to_idx[args.class_name]
            noise = torch.randn(args.num_samples, model_cfg.latent_dim, device=device)
            class_ids = torch.full((args.num_samples,), idx, dtype=torch.long, device=device)
            imgs = generator(noise, class_ids)
            grid = make_grid(denormalize(imgs), nrow=min(args.num_samples, 8))
            save_image(grid, args.out)
            print(f"Saved samples for class '{args.class_name}' to {args.out}")
        else:
            rows = []
            for idx, name in enumerate(class_names):
                noise = torch.randn(args.num_samples, model_cfg.latent_dim, device=device)
                class_ids = torch.full((args.num_samples,), idx, dtype=torch.long, device=device)
                imgs = generator(noise, class_ids)
                rows.append(make_grid(denormalize(imgs), nrow=args.num_samples))
            stacked = torch.cat(rows, dim=1)
            save_image(stacked, args.out)
            print(f"Saved per-class grid to {args.out}")


if __name__ == "__main__":
    main()
