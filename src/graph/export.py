"""Export Neo4j graph snapshots into a PyTorch Geometric compatible format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..data.models import GraphRecord
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PyGExportConfig:
    output_path: str
    default_feature_dim: int = 32


def graph_record_to_pyg(graph: GraphRecord, default_feature_dim: int = 32):
    try:
        import torch  # type: ignore[import-not-found]
        from torch_geometric.data import Data  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch and torch-geometric must be installed to export PyG data"
        ) from exc

    node_indices: Dict[str, int] = {}
    features: List[np.ndarray] = []
    node_metadata: List[Dict[str, str]] = []
    for index, entity in enumerate(graph.entities):
        node_indices[entity.canonical_id] = index
        if entity.embedding:
            features.append(np.array(entity.embedding, dtype=np.float32))
        else:
            features.append(_fallback_embedding(entity, length=default_feature_dim))
        node_metadata.append(
            {
                "id": entity.canonical_id,
                "label": entity.name,
                "type": entity.entity_type,
            }
        )

    if not features:
        x = torch.zeros((0, 0), dtype=torch.float32)
    else:
        feature_dim = max(feat.shape[-1] for feat in features)
        padded_features = [
            np.pad(feat, (0, max(0, feature_dim - feat.shape[-1])), mode="constant")
            for feat in features
        ]
        x = torch.tensor(np.stack(padded_features), dtype=torch.float32)

    edge_pairs: List[Tuple[int, int]] = []
    edge_weights: List[float] = []
    for relation in graph.relations:
        if relation.source_id not in node_indices or relation.target_id not in node_indices:
            continue
        edge_pairs.append(
            (node_indices[relation.source_id], node_indices[relation.target_id])
        )
        edge_weights.append(relation.weight)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.node_metadata = node_metadata
    data.node_ids = [meta["id"] for meta in node_metadata]
    LOGGER.info(
        "Exported PyG data with %s nodes and %s edges", data.num_nodes, data.num_edges
    )
    return data


def export_to_disk(graph: GraphRecord, config: PyGExportConfig) -> None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch must be installed to export data to disk") from exc
    data = graph_record_to_pyg(graph, default_feature_dim=config.default_feature_dim)
    torch.save(data, config.output_path)
    LOGGER.info("Saved PyG snapshot to %s", config.output_path)


def _fallback_embedding(entity, length: int = 32) -> np.ndarray:
    # Construct a deterministic embedding based on the entity identifier.
    import hashlib

    seed_bytes = hashlib.sha1(entity.canonical_id.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_bytes[:8], "big", signed=False)
    rng = np.random.default_rng(seed_int)
    base = rng.random(length, dtype=np.float32)
    scale = min(len(entity.name) / 10.0, 1.0)
    return base * scale
