"""Utilities to load trained GNN artefacts and run link predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

from .model import GraphModelConfig, build_model
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class GNNInferenceArtifacts:
    model: Any
    predictor: Any
    embeddings: Any
    node_metadata: List[Dict[str, str]]
    id_to_index: Dict[str, int]
    device: str
    known_edges: set[tuple[int, int]]


def load_inference_artifacts(
    model_artifact_path: str | Path,
    dataset_path: str | Path,
    device: str = "cpu",
) -> Optional[GNNInferenceArtifacts]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        LOGGER.error("torch is required for GNN inference: %s", exc)
        return None

    model_file = Path(model_artifact_path)
    dataset_file = Path(dataset_path)

    if not model_file.exists() or not dataset_file.exists():
        LOGGER.warning("Inference artefacts missing: model=%s dataset=%s", model_file, dataset_file)
        return None

    artifact = torch.load(model_file, map_location=device)
    model_config = GraphModelConfig(**artifact["model_config"])
    model, predictor = build_model(model_config)
    model.load_state_dict(artifact["model_state_dict"])
    predictor.load_state_dict(artifact["predictor_state_dict"])

    model.to(device)
    predictor.to(device)
    model.eval()
    predictor.eval()

    data = torch.load(dataset_file, map_location=device)
    embeddings = _compute_embeddings(model, data, device)
    node_metadata = list(getattr(data, "node_metadata", []))
    id_to_index = {meta["id"]: idx for idx, meta in enumerate(node_metadata)}
    known_edges = _build_known_edge_set(data)

    return GNNInferenceArtifacts(
        model=model,
        predictor=predictor,
        embeddings=embeddings,
        node_metadata=node_metadata,
        id_to_index=id_to_index,
        device=device,
        known_edges=known_edges,
    )


def predict_link_probability(
    artefacts: GNNInferenceArtifacts,
    source_index: int,
    target_index: int,
) -> float:
    import torch  # type: ignore[import-not-found]

    with torch.no_grad():
        src = artefacts.embeddings[source_index].unsqueeze(0).to(artefacts.device)
        dst = artefacts.embeddings[target_index].unsqueeze(0).to(artefacts.device)
        score = artefacts.predictor(src, dst)
    return float(score.squeeze().cpu())


def recommend_targets(
    artefacts: GNNInferenceArtifacts,
    source_index: int,
    top_k: int = 5,
    exclude_existing: bool = True,
    candidates: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    import torch  # type: ignore[import-not-found]

    num_nodes = artefacts.embeddings.size(0)
    if num_nodes == 0:
        return []

    with torch.no_grad():
        if candidates is None:
            candidate_indices = torch.arange(num_nodes)
        else:
            candidate_indices = torch.tensor(list(candidates), dtype=torch.long)

        candidate_indices = candidate_indices[candidate_indices != source_index]
        if candidate_indices.numel() == 0:
            return []

        src_emb = artefacts.embeddings[source_index].to(artefacts.device)
        dst_emb = artefacts.embeddings[candidate_indices].to(artefacts.device)
        repeated_src = src_emb.unsqueeze(0).expand(dst_emb.size(0), -1)
    scores = artefacts.predictor(repeated_src, dst_emb).cpu()

    recommendations: List[Dict[str, Any]] = []
    for idx, score in zip(candidate_indices.tolist(), scores.tolist()):
        if exclude_existing and _edge_exists(artefacts.known_edges, source_index, idx):
            continue
        meta = artefacts.node_metadata[idx] if idx < len(artefacts.node_metadata) else {"id": str(idx)}
        recommendations.append(
            {
                "index": idx,
                "score": float(score),
                "id": meta.get("id", str(idx)),
                "label": meta.get("label", meta.get("id", str(idx))),
                "type": meta.get("type", "?"),
            }
        )

    recommendations.sort(key=lambda item: cast(float, item["score"]), reverse=True)
    return recommendations[:top_k]


def _compute_embeddings(model, data, device: str):
    import torch  # type: ignore[import-not-found]

    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        embeddings = model(x, edge_index)
    return embeddings.cpu()


def _build_known_edge_set(data) -> set[tuple[int, int]]:
    import torch  # type: ignore[import-not-found]

    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return set()
    if isinstance(edge_index, torch.Tensor):
        indices = edge_index.cpu().numpy()
    else:  # pragma: no cover - safety net for unexpected types
        return set()
    edges: set[tuple[int, int]] = set()
    for source, target in zip(indices[0], indices[1]):
        edges.add((int(source), int(target)))
        edges.add((int(target), int(source)))
    return edges


def _edge_exists(edges: set[tuple[int, int]], source: int, target: int) -> bool:
    return (source, target) in edges or (target, source) in edges
