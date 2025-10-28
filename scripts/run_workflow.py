"""CLI helper to execute the entire ML workflow with a single command."""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.extract_and_load import run_pipeline
from src.pipelines.train_gnn import train_from_config
from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _launch_dashboard(port: int) -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "apps/dashboard.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
    ]
    LOGGER.info("Launching Streamlit dashboard on http://localhost:%s", port)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        LOGGER.error(
            "Streamlit is not installed in the active environment. Install it before using --launch-dashboard."
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Streamlit exited with return code %s", exc.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction, training, and optionally launch the dashboard."
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training stage (only run data pipeline).",
    )
    parser.add_argument(
        "--launch-dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after processing.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port where the dashboard should listen (default: 8501).",
    )
    return parser.parse_args()


def _run_training(config_path: str, config: dict[str, Any]) -> None:
    dataset_path = config.get("gnn", {}).get("dataset_path")
    if not dataset_path:
        LOGGER.warning("gnn.dataset_path missing in config; skipping training stage.")
        return

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        LOGGER.warning(
            "Expected dataset at %s but it was not generated. Skipping training stage.",
            dataset_file,
        )
        return

    LOGGER.info("Starting GNN training.")
    try:
        history = train_from_config(config_path)
    except ImportError as exc:  # pragma: no cover - torch optional in tests
        LOGGER.error("Training skipped because required dependencies are missing: %s", exc)
        return
    except Exception:  # pragma: no cover - propagate unexpected errors after logging
        LOGGER.exception("Training failed with an unexpected error.")
        raise

    LOGGER.info(
        "Training finished. Final epoch metrics: %s",
        history[-1] if history else {},
    )


def main() -> None:
    args = _parse_args()

    LOGGER.info("Starting extraction pipeline using config %s", args.config)
    graph = run_pipeline(args.config)
    LOGGER.info(
        "Pipeline finished with %s entities and %s relations",
        len(graph.entities),
        len(graph.relations),
    )

    if args.skip_train:
        LOGGER.info("Training stage skipped by user request.")
    else:
        config = load_config(args.config)
        _run_training(args.config, config)

    if args.launch_dashboard:
        LOGGER.info("Launching dashboard as part of the workflow.")
        _launch_dashboard(args.dashboard_port)


if __name__ == "__main__":
    main()
