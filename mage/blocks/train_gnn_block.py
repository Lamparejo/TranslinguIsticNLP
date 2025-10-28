"""Mage block that trains the GNN."""
from src.pipelines.train_gnn import train_from_config


def execute(config_path: str = "config/pipeline.yaml"):
    train_from_config(config_path)
