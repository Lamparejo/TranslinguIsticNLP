"""Evaluation utilities for link prediction."""
from __future__ import annotations

from typing import Dict

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def evaluate_link_prediction(model, predictor, dataset, device: str) -> Dict[str, float]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch must be installed to evaluate models") from exc

    model.eval()
    predictor.eval()

    with torch.no_grad():
        x = dataset.data.x.to(device)
        full_edge_index = getattr(dataset.data, "edge_index", None)
        if full_edge_index is None or full_edge_index.numel() == 0:
            LOGGER.warning("Skipping evaluation: dataset has no edges for message passing.")
            return {"val_auc": 0.0, "val_ap": 0.0, "test_auc": 0.0, "test_ap": 0.0}
        full_edge_index = full_edge_index.to(device)
        z = model(x, full_edge_index)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scikit-learn must be installed to compute metrics") from exc

        def compute_scores(pos_edge_index, neg_edge_index):
            pos_scores = predictor(
                z[pos_edge_index[0].to(device)],
                z[pos_edge_index[1].to(device)],
            )
            neg_scores = predictor(
                z[neg_edge_index[0].to(device)],
                z[neg_edge_index[1].to(device)],
            )
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat(
                [
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores),
                ]
            ).cpu().numpy()
            return {
                "auc": float(roc_auc_score(labels, scores)),
                "ap": float(average_precision_score(labels, scores)),
            }

        if dataset.val_pos_edge_index.numel() == 0 or dataset.val_neg_edge_index.numel() == 0:
            LOGGER.warning("Validation metrics unavailable: not enough edges to compute scores.")
            val_metrics = {"auc": 0.0, "ap": 0.0}
        else:
            val_metrics = compute_scores(dataset.val_pos_edge_index, dataset.val_neg_edge_index)

        if dataset.test_pos_edge_index.numel() == 0 or dataset.test_neg_edge_index.numel() == 0:
            LOGGER.warning("Test metrics unavailable: not enough edges to compute scores.")
            test_metrics = {"auc": 0.0, "ap": 0.0}
        else:
            test_metrics = compute_scores(dataset.test_pos_edge_index, dataset.test_neg_edge_index)

    metrics = {
        "val_auc": val_metrics["auc"],
        "val_ap": val_metrics["ap"],
        "test_auc": test_metrics["auc"],
        "test_ap": test_metrics["ap"],
    }
    return metrics
