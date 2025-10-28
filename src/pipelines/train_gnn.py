"""Pipeline to train the link prediction GNN on the exported snapshot."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..gnn.datasets import build_link_prediction_dataset
from ..gnn.model import GraphModelConfig, build_model
from ..gnn.train import TrainingConfig, train_link_prediction
from ..utils.config import load_config
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def train_from_config(
    config_path: str | Path = "config/pipeline.yaml",
    overrides: Dict[str, Any] | None = None,
):
    overrides = overrides or {}
    config_path = Path(config_path)
    config = load_config(config_path)
    gnn_cfg = dict(config.get("gnn", {}))
    if overrides:
        LOGGER.info("Applying training overrides: %s", overrides)
        gnn_cfg.update(overrides)

    dataset_path_value = gnn_cfg.get("dataset_path")
    dataset_path = Path(dataset_path_value) if dataset_path_value else None
    if not dataset_path:
        raise ValueError("gnn.dataset_path must be set in the configuration")

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch must be installed to train the GNN") from exc

    data = torch.load(dataset_path)
    dataset = build_link_prediction_dataset(
        data,
        val_ratio=float(gnn_cfg.get("val_ratio", 0.1)),
        test_ratio=float(gnn_cfg.get("test_ratio", 0.1)),
    )

    model_config = GraphModelConfig(
        in_channels=data.num_features,
        hidden_channels=int(gnn_cfg.get("hidden_channels", 128)),
        out_channels=int(gnn_cfg.get("out_channels", 64)),
        dropout=float(gnn_cfg.get("dropout", 0.3)),
    )
    model, predictor = build_model(model_config)

    training_config = TrainingConfig(
        epochs=int(gnn_cfg.get("epochs", 5)),
        learning_rate=float(gnn_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(gnn_cfg.get("weight_decay", 1e-4)),
        device=str(gnn_cfg.get("device", "cpu")),
    )

    training_start = datetime.utcnow()
    history = train_link_prediction(model, predictor, dataset, training_config)
    training_end = datetime.utcnow()
    LOGGER.info("Finished training. Final metrics: %s", history[-1] if history else {})
    _persist_training_outputs(
        history=history,
        model=model,
        predictor=predictor,
        model_config=model_config,
        training_config=training_config,
        gnn_cfg=gnn_cfg,
        config_path=str(config_path),
        overrides=overrides,
        training_start=training_start,
        training_end=training_end,
    )
    return history


def _persist_training_outputs(
    history,
    model,
    predictor,
    model_config,
    training_config,
    gnn_cfg,
    config_path: str,
    overrides: Dict[str, Any],
    training_start: datetime,
    training_end: datetime,
):
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch must be installed to persist training artefacts") from exc

    generated_at = datetime.utcnow().isoformat()
    training_duration = (training_end - training_start).total_seconds()
    overrides_serialisable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in overrides.items()
    }

    model_param_total = sum(param.numel() for param in model.parameters())
    predictor_param_total = sum(param.numel() for param in predictor.parameters())
    model_param_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    predictor_param_trainable = sum(param.numel() for param in predictor.parameters() if param.requires_grad)
    total_parameters = model_param_total + predictor_param_total
    trainable_parameters = model_param_trainable + predictor_param_trainable

    metrics_payload = {
        "metadata": {
            "generated_at": generated_at,
            "epochs": len(history),
            "training_started_at": training_start.isoformat(),
            "training_completed_at": training_end.isoformat(),
            "training_duration_seconds": training_duration,
            "config_path": config_path,
            "dataset_path": gnn_cfg.get("dataset_path"),
            "metrics_path": gnn_cfg.get("metrics_path"),
            "model_artifact_path": gnn_cfg.get("model_artifact_path"),
            "val_ratio": gnn_cfg.get("val_ratio"),
            "test_ratio": gnn_cfg.get("test_ratio"),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "parameter_counts": {
                "model": model_param_total,
                "predictor": predictor_param_total,
                "model_trainable": model_param_trainable,
                "predictor_trainable": predictor_param_trainable,
            },
        },
        "history": history,
        "config": {
            "model": asdict(model_config),
            "training": asdict(training_config),
        },
    }

    if history:
        best = max(history, key=lambda item: item.get("val_auc", 0.0))
        metrics_payload["best_epoch"] = best
        metrics_payload["metadata"]["best_epoch"] = best.get("epoch")
        metrics_payload["final_epoch"] = history[-1]
        avg_loss = sum(float(item.get("loss", 0.0)) for item in history) / len(history)
        metrics_payload["metadata"]["avg_epoch_loss"] = avg_loss

    if overrides_serialisable:
        metrics_payload["metadata"]["overrides"] = overrides_serialisable

    metrics_path = gnn_cfg.get("metrics_path")
    if metrics_path:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved GNN training metrics to %s", metrics_file)

    model_artifact_path = gnn_cfg.get("model_artifact_path")
    if model_artifact_path:
        artifact_file = Path(model_artifact_path)
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
                "generated_at": generated_at,
                "epochs_trained": len(history),
            },
            artifact_file,
        )
        LOGGER.info("Saved trained GNN weights to %s", artifact_file)
