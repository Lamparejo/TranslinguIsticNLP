"""CLI entry point to execute the extraction and loading pipeline."""
from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.extract_and_load import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the translinguistic knowledge graph pipeline")
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
