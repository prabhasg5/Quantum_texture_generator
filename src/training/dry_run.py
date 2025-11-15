"""Smoke test for configuration and model assembly."""

from __future__ import annotations

from pathlib import Path

import click

from .config import ExperimentConfig
from .loop import build_models


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def dry_run(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    artifacts = build_models(cfg)
    click.echo("Dry run successful: models instantiated.")
    click.echo(f"Variant: {cfg.variant}")
    click.echo(f"Decoder parameters: {sum(p.numel() for p in artifacts.decoder.parameters())}")
    click.echo(f"Generator parameters: {sum(p.numel() for p in artifacts.generator.parameters())}")


if __name__ == "__main__":
    dry_run()
