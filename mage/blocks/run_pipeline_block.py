"""Mage block that executes the extraction pipeline."""
from src.pipelines.extract_and_load import run_pipeline


def execute(config_path: str = "config/pipeline.yaml"):
    """Entry point expected by Mage's Python block API."""
    run_pipeline(config_path)
