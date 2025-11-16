"""Utilities for loading experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainerSettings:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 3.0e-4
    beta1: float = 0.5
    beta2: float = 0.99
    weight_decay: float = 1.0e-4
    device: str = "auto"
    log_interval: int = 50
    checkpoint_interval: int = 1000
    checkpoint_epochs: int = 5
    max_steps: Optional[int] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    palette_size: int = 5
    lambda_palette: float = 5.0
    grad_clip: Optional[float] = 1.0
    num_workers: int = 8
    latent_dim: int = 8
    n_qubits: int = 8
    class_embed_dim: int = 64
    noise_dim: int = 16
    seed: Optional[int] = 42
    resume: bool = True
    sample_interval_epochs: int = 5
    history_interval_epochs: int = 5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainerSettings":
        if data is None:
            return cls()
        field_names = {field.name for field in fields(cls)}
        init_kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in field_names:
                continue
            if key == "output_dir" and value is not None:
                init_kwargs[key] = Path(value).expanduser().resolve()
            else:
                init_kwargs[key] = value
        return cls(**init_kwargs)


@dataclass
class ExperimentConfig:
    name: str
    variant: str
    dataset_root: Path
    manifest_path: Optional[Path]
    class_map_path: Optional[Path]
    class_count: int
    device_name: Optional[str]
    shots: Optional[int]
    training: TrainerSettings

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
        training_raw = raw.get("training")
        return cls(
            name=raw["name"],
            variant=raw["variant"],
            dataset_root=Path(raw["dataset_root"]).expanduser().resolve(),
            manifest_path=manifest_path,
            class_map_path=class_map_path,
            class_count=int(raw["class_count"]),
            device_name=raw.get("device_name"),
            shots=raw.get("shots"),
            training=TrainerSettings.from_dict(training_raw),
        )
