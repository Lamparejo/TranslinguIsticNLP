"""Run extraction, GNN training, and launch the dashboard sequentially."""
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data extraction, train the GNN, and launch the dashboard in one go.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dashboard-address",
        default="0.0.0.0",
        help="Address where the Streamlit server should bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port for the Streamlit dashboard (default: 8501).",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Run pipeline and training but skip launching the dashboard.",
    )
    return parser.parse_args()


def _ensure_dataset_exists(config: dict[str, Any]) -> bool:
    dataset_path = config.get("gnn", {}).get("dataset_path")
    if not dataset_path:
        LOGGER.error("gnn.dataset_path is missing in the configuration; aborting training.")
        return False

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        LOGGER.error(
            "Expected dataset at %s but it was not generated. Run the pipeline first.",
            dataset_file,
        )
        return False
    return True


def _launch_dashboard(address: str, port: int) -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "apps/dashboard.py",
        f"--server.address={address}",
        f"--server.port={port}",
    ]
    LOGGER.info(
        "Launching Streamlit dashboard at http://%s:%s (Ctrl+C to stop)",
        address,
        port,
    )
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        LOGGER.error(
            "Streamlit is not installed in the active environment. Install it before launching the dashboard.",
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Streamlit exited with return code %s", exc.returncode)


def main() -> None:
    args = _parse_args()
    LOGGER.info("Stage 1/3 – Running extraction pipeline (%s)", args.config)
    graph = run_pipeline(args.config)
    LOGGER.info(
        "Pipeline finished with %s entities and %s relations. Metrics and artifacts saved to the artifacts/ folder.",
        len(graph.entities),
        len(graph.relations),
    )

    LOGGER.info("Stage 2/3 – Training GNN")
    config = load_config(args.config)
    if not _ensure_dataset_exists(config):
        return

    try:
        history = train_from_config(args.config)
    except ImportError as exc:
        LOGGER.error("Training skipped because required dependencies are missing: %s", exc)
        return
    except Exception:  # pragma: no cover - propagate unexpected errors after logging
        LOGGER.exception("Training failed with an unexpected error.")
        raise

    LOGGER.info(
        "Training completed. Final epoch metrics: %s",
        history[-1] if history else {},
    )

    if args.skip_dashboard:
        LOGGER.info("Stage 3/3 – Dashboard launch skipped by user request.")
        return

    LOGGER.info("Stage 3/3 – Launching dashboard")
    _launch_dashboard(args.dashboard_address, args.dashboard_port)


if __name__ == "__main__":
    main()
