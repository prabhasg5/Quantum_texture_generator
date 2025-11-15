"""Run the full evaluation suite comparing quantum and classical outputs."""

from __future__ import annotations

from pathlib import Path

import click

from .metrics import summarize_metrics


@click.command()
@click.option("--quantum", type=click.Path(exists=True, path_type=Path), required=True, help="Path to quantum samples.")
@click.option("--classical", type=click.Path(exists=True, path_type=Path), required=True, help="Path to classical samples.")
def main(quantum: Path, classical: Path) -> None:
    # Placeholder until evaluation metrics are implemented.
    click.echo(f"Comparing quantum run at {quantum} vs classical run at {classical}.")
    click.echo("Metrics pending implementation.")
    click.echo(summarize_metrics({"novelty": 0.0, "coverage": 0.0}))


if __name__ == "__main__":
    main()
