"""CLI entry-point for training experiments."""

from __future__ import annotations

from pathlib import Path

import click

from .config import ExperimentConfig
from .loop import build_models


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def main(config_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config_path)
    artifacts = build_models(cfg)
    click.echo(f"Loaded experiment '{cfg.name}' with variant '{cfg.variant}'.")
    click.echo(f"Decoder params: {sum(p.numel() for p in artifacts.decoder.parameters())}")
    click.echo(f"Generator params: {sum(p.numel() for p in artifacts.generator.parameters())}")
    click.echo("Training loop not yet implemented.")


if __name__ == "__main__":
    main()
