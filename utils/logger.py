"""Centralized logging setup for the football_ai project.

Every module obtains its logger through :func:`get_logger`, which creates
children of the single ``football_ai`` root logger.  :func:`setup_logging`
is called exactly once (by the entry point) and attaches the console and
optional file handlers.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT_LOGGER_NAME = "football_ai"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure the project root logger.

    Args:
        level: Logging level name ("DEBUG", "INFO", ...).
        log_dir: Directory where log files are written.
        log_to_file: If True and ``log_dir`` is given, also log to a
            timestamped file inside ``log_dir``.

    Returns:
        The configured ``football_ai`` root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Re-running setup (e.g. in tests) must not duplicate handlers.
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"run_{timestamp}.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger("detection.detector")``."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
