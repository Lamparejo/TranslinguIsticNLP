"""Configuration loading helpers."""
from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = pathlib.Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
