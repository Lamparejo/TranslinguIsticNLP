"""Datasets utilities for GNN link prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data
else:  # pragma: no cover - fallback types for optional deps
    Tensor = Any
    Data = Any


@dataclass(slots=True)
class LinkPredictionDataset:
    data: "Data"
    train_pos_edge_index: "Tensor"
    val_pos_edge_index: "Tensor"
    test_pos_edge_index: "Tensor"
    train_neg_edge_index: "Tensor"
    val_neg_edge_index: "Tensor"
    test_neg_edge_index: "Tensor"


def build_link_prediction_dataset(
    data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> LinkPredictionDataset:
    try:
        import torch
        from torch_geometric.utils import coalesce  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch and torch-geometric must be installed to create datasets"
        ) from exc

    data = data.clone()
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        data.edge_index = empty
        data.train_pos_edge_index = empty
        data.val_pos_edge_index = empty
        data.val_neg_edge_index = empty
        data.test_pos_edge_index = empty
        data.test_neg_edge_index = empty
        data.train_neg_edge_index = empty
        return LinkPredictionDataset(
            data=data,
            train_pos_edge_index=empty,
            val_pos_edge_index=empty,
            test_pos_edge_index=empty,
            train_neg_edge_index=empty,
            val_neg_edge_index=empty,
            test_neg_edge_index=empty,
        )

    edge_index = edge_index.to(torch.long)

    def _format_pairs(pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return pairs.t().contiguous()

    def _make_bidirectional(pairs: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        forward = pairs.t().contiguous()
        reverse = pairs[:, [1, 0]].t().contiguous()
        merged = torch.cat([forward, reverse], dim=1)
        merged, _ = coalesce(merged, None, num_nodes=num_nodes)
        return merged

    # Consider edges como não direcionados para definir pares únicos
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    edges = edge_index.t().contiguous()
    if edges.size(0) == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        data.edge_index = empty
        data.train_pos_edge_index = empty
        data.val_pos_edge_index = empty
        data.test_pos_edge_index = empty
        data.val_neg_edge_index = empty
        data.test_neg_edge_index = empty
        data.train_neg_edge_index = empty
        return LinkPredictionDataset(
            data=data,
            train_pos_edge_index=empty,
            val_pos_edge_index=empty,
            test_pos_edge_index=empty,
            train_neg_edge_index=empty,
            val_neg_edge_index=empty,
            test_neg_edge_index=empty,
        )

    edges_sorted = torch.sort(edges, dim=1)[0]
    unique_pairs = torch.unique(edges_sorted, dim=0)
    num_pairs = unique_pairs.size(0)

    if num_pairs == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        data.edge_index = empty
        data.train_pos_edge_index = empty
        data.val_pos_edge_index = empty
        data.test_pos_edge_index = empty
        data.train_neg_edge_index = empty
        data.val_neg_edge_index = empty
        data.test_neg_edge_index = empty
        return LinkPredictionDataset(
            data=data,
            train_pos_edge_index=empty,
            val_pos_edge_index=empty,
            test_pos_edge_index=empty,
            train_neg_edge_index=empty,
            val_neg_edge_index=empty,
            test_neg_edge_index=empty,
        )

    # Definição determinística dos splits
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(num_pairs, generator=generator)

    val_count = min(int(round(num_pairs * val_ratio)), num_pairs)
    test_count = min(int(round(num_pairs * test_ratio)), num_pairs - val_count)
    train_count = num_pairs - val_count - test_count

    if train_count <= 0:
        if val_count > 0:
            val_count -= 1
            train_count += 1
        elif test_count > 0:
            test_count -= 1
            train_count += 1

    val_pairs = unique_pairs[perm[:val_count]] if val_count > 0 else unique_pairs[:0]
    test_pairs = (
        unique_pairs[perm[val_count : val_count + test_count]] if test_count > 0 else unique_pairs[:0]
    )
    train_pairs = (
        unique_pairs[perm[val_count + test_count :]] if train_count > 0 else unique_pairs[:0]
    )

    train_edge_index = _make_bidirectional(train_pairs, data.num_nodes)
    nodes = torch.arange(data.num_nodes, dtype=torch.long)
    all_candidate_pairs = (
        torch.combinations(nodes, r=2) if nodes.numel() >= 2 else torch.empty((0, 2), dtype=torch.long)
    )

    if unique_pairs.numel() > 0 and all_candidate_pairs.numel() > 0:
        matches = (
            all_candidate_pairs.unsqueeze(0) == unique_pairs.unsqueeze(1)
        ).all(dim=2)
        available_mask = ~matches.any(dim=0)
        available_pairs = all_candidate_pairs[available_mask]
    else:
        available_pairs = all_candidate_pairs

    available_pairs = available_pairs.clone()

    def _sample_negative_pairs(num_samples: int, rng: torch.Generator) -> torch.Tensor:
        nonlocal available_pairs
        if num_samples <= 0 or available_pairs.size(0) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        count = min(num_samples, available_pairs.size(0))
        perm = torch.randperm(available_pairs.size(0), generator=rng)
        selected = available_pairs[perm[:count]]
        available_pairs = available_pairs[perm[count:]]
        return _format_pairs(selected)

    train_neg = _sample_negative_pairs(train_pairs.size(0), generator)
    val_neg = _sample_negative_pairs(val_pairs.size(0), generator)
    test_neg = _sample_negative_pairs(test_pairs.size(0), generator)

    train_pos = _format_pairs(train_pairs)
    val_pos = _format_pairs(val_pairs)
    test_pos = _format_pairs(test_pairs)

    dataset = LinkPredictionDataset(
        data=data,
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        train_neg_edge_index=train_neg,
        val_neg_edge_index=val_neg,
        test_neg_edge_index=test_neg,
    )

    data.edge_index = train_edge_index
    data.train_pos_edge_index = train_pos
    data.train_neg_edge_index = train_neg
    data.val_pos_edge_index = val_pos
    data.val_neg_edge_index = val_neg
    data.test_pos_edge_index = test_pos
    data.test_neg_edge_index = test_neg

    return dataset
