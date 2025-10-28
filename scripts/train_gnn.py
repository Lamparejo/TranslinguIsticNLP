"""CLI entry point to train the GNN once data has been exported."""
from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.train_gnn import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GNN on exported snapshots")
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
