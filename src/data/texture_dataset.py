from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def discover_class_folders(root: Path) -> List[Path]:
    """Return all texture class folders sorted alphabetically."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir()])


class TextureDataset(Dataset):
    """Texture dataset where each subfolder acts as a class."""

    def __init__(
        self,
        root: Path,
        image_size: int = 128,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        class_dirs = discover_class_folders(self.root)
        if not class_dirs:
            raise RuntimeError(f"No class folders found inside {self.root}")

        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            self.idx_to_class[idx] = class_dir.name
            for path in sorted(class_dir.glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
                    self.samples.append((path, idx))

        if not self.samples:
            raise RuntimeError(f"No images found in dataset at {self.root}")

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if augment:
            transform_list = [
                transforms.Resize(image_size + 16),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            transform_list = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return {
            "image": tensor,
            "label": label,
            "path": str(path),
        }


@dataclass
class DataConfig:
    root: Path
    batch_size: int = 16
    image_size: int = 128
    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True


def build_dataloader(cfg: DataConfig, augment: bool = True) -> Tuple[DataLoader, TextureDataset]:
    dataset = TextureDataset(cfg.root, image_size=cfg.image_size, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=augment,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        pin_memory=cfg.pin_memory,
        drop_last=augment,
    )
    return loader, dataset
