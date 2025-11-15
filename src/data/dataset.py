"""Dataset loading utilities for the PTD texture corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ..utils.colors import extract_palette
from ..utils.palette import PaletteEncoding, SimplePaletteStats


@dataclass
class TextureSample:
    """Container describing a single texture example and its metadata."""

    image: torch.Tensor
    class_id: int
    palette_embedding: torch.Tensor
    palette_colors: torch.Tensor
    path: Path


class PTDTextureDataset(Dataset):
    """Loads PTD texture tiles, deriving palettes on the fly."""

    def __init__(
        self,
        root: Path,
        split_manifest: Optional[Path] = None,
        class_map_path: Optional[Path] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        palette_encoder: Optional[PaletteEncoding] = None,
        palette_size: int = 5,
    ) -> None:
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        if split_manifest is not None and not split_manifest.exists():
            raise FileNotFoundError(f"Split manifest missing: {split_manifest}")
        if class_map_path is not None and not class_map_path.exists():
            raise FileNotFoundError(f"Class map missing: {class_map_path}")

        self.root = root
        self.transform = transform
        self.palette_encoder = palette_encoder or SimplePaletteStats(n_colors=palette_size)
        self.palette_size = palette_size
        self.class_to_id = self._load_class_map(class_map_path)
        self.records = self._build_records(split_manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TextureSample:
        record = self.records[index]
        image = self._load_image(record["path"])
        if self.transform:
            image = self.transform(image)

        palette_colors = self._derive_palette(image)
        palette_embedding = self.palette_encoder.encode(palette_colors)

        return TextureSample(
            image=image,
            class_id=record["class_id"],
            palette_embedding=palette_embedding,
            palette_colors=palette_colors,
            path=record["path"],
        )

    def _load_class_map(self, mapping_path: Optional[Path]) -> Dict[str, int]:
        if mapping_path:
            classes = [line.strip() for line in mapping_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        return {label: idx for idx, label in enumerate(classes)}

    def _build_records(self, manifest_path: Optional[Path]) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        if manifest_path:
            lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            candidates = lines
        else:
            candidates = []
            for class_dir in sorted(self.root.iterdir()):
                if not class_dir.is_dir():
                    continue
                for image_path in sorted(class_dir.glob("*.png")):
                    rel = image_path.relative_to(self.root)
                    candidates.append(str(rel))

        for relative in candidates:
            class_name, _, file_name = relative.partition("/")
            if not file_name:
                continue
            full_path = self.root / class_name / file_name
            if not full_path.exists():
                continue
            class_id = self.class_to_id.get(class_name)
            if class_id is None:
                raise KeyError(f"Class '{class_name}' not found in class map.")
            records.append({"path": full_path, "class_id": class_id, "class_name": class_name})

        if not records:
            raise RuntimeError(f"No texture images discovered under {self.root}.")
        return records

    def _load_image(self, path: Path) -> torch.Tensor:
        tensor = read_image(str(path)).float() / 255.0
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def _derive_palette(self, image: torch.Tensor) -> torch.Tensor:
        palette = extract_palette(image, n_colors=self.palette_size)
        return palette
