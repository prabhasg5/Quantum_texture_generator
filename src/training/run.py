"""CLI entry-point for training experiments."""

from __future__ import annotations

from pathlib import Path

import click

from .config import ExperimentConfig
from .loop import train_experiment
from ..utils.logging import configure_logging


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def main(config_path: Path) -> None:
    configure_logging()
    cfg = ExperimentConfig.from_yaml(config_path)
    click.echo(f"Starting experiment '{cfg.name}' ({cfg.variant})")
    completed = train_experiment(cfg)
    if completed:
        click.echo("Training completed.")
    else:
        click.echo("Training interrupted; latest checkpoint saved.")


if __name__ == "__main__":
    main()
