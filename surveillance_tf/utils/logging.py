"""Logging helpers using Rich."""
from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console()


def _configure_root(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True)],
    )


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name: Optional[str]
        Logger namespace; defaults to project root.
    level: int
        Logging level; INFO by default.
    """
    root = logging.getLogger("surveillance_tf")
    if not root.handlers:
        _configure_root(level)
        root.setLevel(level)
    return root.getChild(name) if name else root


__all__ = ["get_logger"]
