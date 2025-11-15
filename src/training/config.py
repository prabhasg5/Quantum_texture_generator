"""Utilities for loading experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str
    variant: str
    dataset_root: Path
    manifest_path: Optional[Path]
    class_map_path: Optional[Path]
    class_count: int

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle)
        manifest_raw = raw.get("manifest_path")
        manifest_path = (
            Path(manifest_raw).expanduser().resolve() if manifest_raw else None
        )
        class_map_raw = raw.get("class_map_path")
        class_map_path = (
            Path(class_map_raw).expanduser().resolve() if class_map_raw else None
        )
        return cls(
            name=raw["name"],
            variant=raw["variant"],
            dataset_root=Path(raw["dataset_root"]).expanduser().resolve(),
            manifest_path=manifest_path,
            class_map_path=class_map_path,
            class_count=int(raw["class_count"]),
        )
