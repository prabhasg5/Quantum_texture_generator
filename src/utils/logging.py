"""Logging helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import logging


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@contextmanager
def log_section(name: str) -> Iterator[None]:
    logging.info("=== %s ===", name)
    try:
        yield
    finally:
        logging.info("=== end %s ===", name)
