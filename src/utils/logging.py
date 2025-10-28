"""Logging helpers shared across components."""
from __future__ import annotations

import logging
import os
from typing import Optional


_LOGGER_CACHE: dict[str, logging.Logger] = {}


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format."""
    if logging.getLogger().handlers:
        return
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler])


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a configured logger, caching it for reuse."""
    if name is None:
        name = os.getenv("APP_LOGGER_NAME", "translinguistic_nlp")
    if name not in _LOGGER_CACHE:
        configure_root_logger()
        _LOGGER_CACHE[name] = logging.getLogger(name)
    return _LOGGER_CACHE[name]
