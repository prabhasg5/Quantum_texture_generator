"""Report generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_report(metrics: Dict[str, float], output_path: Path) -> None:
    raise NotImplementedError("Generate HTML/PDF report summarizing experiment comparisons.")
