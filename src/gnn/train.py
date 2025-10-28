"""Training utilities for the GNN models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


def train_link_prediction(model, predictor, dataset, config: TrainingConfig):
    try:
        from torch import optim  # type: ignore[import-not-found]
        from torch.nn import BCEWithLogitsLoss  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch must be installed to train models") from exc

    from .evaluate import evaluate_link_prediction

    model.to(config.device)
    predictor.to(config.device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: List[Dict[str, float]] = []
    loss_fn = BCEWithLogitsLoss()

    edge_index = getattr(dataset.data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        LOGGER.warning("Training skipped: dataset does not contain edges for message passing.")
        return history

    if dataset.train_pos_edge_index.numel() == 0 or dataset.train_neg_edge_index.numel() == 0:
        LOGGER.warning("Training skipped: dataset does not contain positive/negative edges.")
        return history

    edge_index = edge_index.to(config.device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        x = dataset.data.x.to(config.device)
        train_pos_edge_index = dataset.train_pos_edge_index.to(config.device)
        train_neg_edge_index = dataset.train_neg_edge_index.to(config.device)

        z = model(x, edge_index)

        pos_scores = predictor(z[train_pos_edge_index[0]], z[train_pos_edge_index[1]])
        pos_labels = pos_scores.new_ones(pos_scores.size(0))
        pos_loss = loss_fn(pos_scores, pos_labels)

        neg_scores = predictor(z[train_neg_edge_index[0]], z[train_neg_edge_index[1]])
        neg_labels = neg_scores.new_zeros(neg_scores.size(0))
        neg_loss = loss_fn(neg_scores, neg_labels)

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        metrics = evaluate_link_prediction(model, predictor, dataset, config.device)
        metrics["loss"] = float(loss.detach().cpu())
        metrics["epoch"] = epoch
        history.append(metrics)
    return history
