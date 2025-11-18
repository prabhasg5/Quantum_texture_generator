from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import models
import lpips

from src.utils.visualization import denormalize


def _collect_subset(loader: DataLoader, max_samples: int, device: torch.device) -> torch.Tensor:
    images = []
    for batch in loader:
        img = batch["image"].to(device)
        images.append(img)
        if torch.cat(images).shape[0] >= max_samples:
            break
    return torch.cat(images)[:max_samples]


def compute_fid(
    generator,
    dataset,
    device: torch.device,
    latent_dim: int,
    batch_size: int,
    sample_count: int,
) -> float:
    fid_device = torch.device("cpu") if device.type == "mps" else device
    fid = FrechetInceptionDistance(feature=2048).to(fid_device)
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    processed = 0
    for batch in real_loader:
        imgs = denormalize(batch["image"].to(fid_device))
        imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        fid.update(imgs, real=True)
        processed += imgs.shape[0]
        if processed >= sample_count:
            break

    generator.eval()
    with torch.no_grad():
        remaining = sample_count
        class_ids = torch.arange(len(dataset.class_to_idx), device=device)
        class_ids = class_ids.repeat((sample_count + len(class_ids) - 1) // len(class_ids))[:sample_count]
        idx = 0
        while remaining > 0:
            cur = min(batch_size, remaining)
            classes = class_ids[idx : idx + cur]
            noise = torch.randn(cur, latent_dim, device=device)
            fake = denormalize(generator(noise, classes)).to(fid_device)
            fake = (fake * 255).clamp(0, 255).to(torch.uint8)
            fid.update(fake, real=False)
            remaining -= cur
            idx += cur
    return float(fid.compute().cpu())


def compute_lpips_novelty(
    generator,
    dataset,
    device: torch.device,
    latent_dim: int,
    sample_count: int = 128,
    reference_count: int = 512,
) -> Tuple[float, List[float]]:
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    ref_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    ref_images = []
    with torch.no_grad():
        for batch in ref_loader:
            ref_images.append(batch["image"].to(device))
            if torch.cat(ref_images).shape[0] >= reference_count:
                break
    ref_images = denormalize(torch.cat(ref_images)[:reference_count])

    generator.eval()
    novelty_scores: List[float] = []
    with torch.no_grad():
        for _ in range(sample_count):
            class_id = random.randint(0, len(dataset.class_to_idx) - 1)
            noise = torch.randn(1, latent_dim, device=device)
            fake = denormalize(generator(noise, torch.tensor([class_id], device=device)))
            dists = []
            for chunk in ref_images.split(8):
                d = lpips_model(fake, chunk).view(-1)
                dists.append(d)
            min_dist = torch.cat(dists).min().item()
            novelty_scores.append(min_dist)
    lpips_avg = float(sum(novelty_scores) / len(novelty_scores))
    return lpips_avg, novelty_scores


def _build_feature_extractor(device: torch.device):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    body = model.features
    body.eval()
    body.to(device)
    for p in body.parameters():
        p.requires_grad = False
    return body, weights.transforms()


def _extract_features(model, preprocess, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = model(preprocess(images))
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
    return feats


def compute_feature_coverage(
    generator,
    dataset,
    device: torch.device,
    latent_dim: int,
    sample_count: int = 256,
    k: int = 5,
) -> Tuple[float, List[float], List[float]]:
    model, preprocess = _build_feature_extractor(device)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    real_feats = []
    real_processed = 0
    for batch in loader:
        imgs = denormalize(batch["image"].to(device))
        feats = _extract_features(model, preprocess, imgs)
        real_feats.append(feats.cpu())
        real_processed += imgs.shape[0]
        if real_processed >= sample_count:
            break
    real_feats = torch.cat(real_feats)[:sample_count]

    generator.eval()
    fake_feats = []
    with torch.no_grad():
        remaining = sample_count
        while remaining > 0:
            cur = min(32, remaining)
            class_ids = torch.randint(0, len(dataset.class_to_idx), (cur,), device=device)
            noise = torch.randn(cur, latent_dim, device=device)
            imgs = denormalize(generator(noise, class_ids))
            feats = _extract_features(model, preprocess, imgs)
            fake_feats.append(feats.cpu())
            remaining -= cur
    fake_feats = torch.cat(fake_feats)[:sample_count]

    real_np = real_feats.numpy()
    fake_np = fake_feats.numpy()
    real_real = pairwise_distances(real_np, real_np)
    kth = np.partition(real_real, k, axis=1)[:, k]
    real_fake = pairwise_distances(real_np, fake_np)
    nearest = real_fake.min(axis=1)
    coverage = float(np.mean(nearest <= kth))
    return coverage, kth.tolist(), nearest.tolist()


def compute_colorfulness(
    generator,
    num_classes: int,
    device: torch.device,
    latent_dim: int,
    sample_count: int = 128,
) -> Dict[str, float]:
    generator.eval()
    stats = []
    with torch.no_grad():
        remaining = sample_count
        while remaining > 0:
            cur = min(32, remaining)
            class_ids = torch.randint(0, num_classes, (cur,), device=device)
            noise = torch.randn(cur, latent_dim, device=device)
            imgs = denormalize(generator(noise, class_ids))
            color_std = imgs.view(cur, imgs.shape[1], -1).std(dim=2).mean(dim=1)
            stats.append(color_std.cpu())
            remaining -= cur
    values = torch.cat(stats)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_metrics(metrics: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
