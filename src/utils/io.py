"""File IO helpers."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Iterator

from .logging import get_logger

LOGGER = get_logger(__name__)


def read_jsonl(path: str | pathlib.Path) -> Iterator[dict]:
    """Yield JSON objects from a JSON Lines file."""
    data_path = pathlib.Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | pathlib.Path, records: Iterable[dict]) -> None:
    """Write an iterable of dictionaries to a JSON Lines file."""
    record_list = list(records)
    data_path = pathlib.Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as handle:
        for record in record_list:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %s records to %s", len(record_list), data_path)


def write_json(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    """Write a JSON document to disk."""
    data_path = pathlib.Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote JSON document to %s", data_path)
