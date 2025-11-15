"""Project entry point placeholder."""

from __future__ import annotations

from pathlib import Path

import click

from .training.run import main as train_command


@click.group()
def cli() -> None:
    """Command-line interface for the quantum texture generator."""


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
def train(config_path: Path) -> None:
    """Train either the quantum or classical variant based on config."""

    train_command.main(standalone_mode=False, config_path=config_path)


if __name__ == "__main__":
    cli()
