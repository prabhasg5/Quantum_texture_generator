"""Smoke test for configuration and model assembly."""

from __future__ import annotations

from pathlib import Path

import click

from ..data.dataset import PTDTextureDataset
from .config import ExperimentConfig
from .loop import build_models


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def dry_run(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    dataset = PTDTextureDataset(
        root=cfg.dataset_root,
        split_manifest=cfg.manifest_path,
        class_map_path=cfg.class_map_path,
        palette_size=cfg.training.palette_size,
    )
    palette_dim = dataset[0]["palette_embedding"].shape[-1]
    artifacts = build_models(cfg, cfg.training, palette_dim)
    click.echo("Dry run successful: models instantiated.")
    click.echo(f"Variant: {cfg.variant}")
    click.echo(f"Decoder parameters: {sum(p.numel() for p in artifacts.decoder.parameters())}")
    click.echo(f"Generator parameters: {sum(p.numel() for p in artifacts.generator.parameters())}")


if __name__ == "__main__":
    dry_run()
