import torch
from torch_geometric.data import Data

from src.gnn.datasets import build_link_prediction_dataset


def _count_unique_pairs(edge_index: torch.Tensor) -> int:
    if edge_index.numel() == 0:
        return 0
    pairs = torch.sort(edge_index.t(), dim=1)[0]
    return torch.unique(pairs, dim=0).size(0)


def test_build_link_prediction_dataset_produces_non_empty_splits():
    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 1, 2, 0, 3, 4, 5, 5],
            [1, 0, 3, 2, 5, 2, 1, 2, 4, 3, 0, 4],
        ],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index)

    dataset = build_link_prediction_dataset(data, val_ratio=0.2, test_ratio=0.2, seed=123)

    total_pos = (
        dataset.train_pos_edge_index.size(1)
        + dataset.val_pos_edge_index.size(1)
        + dataset.test_pos_edge_index.size(1)
    )
    expected_total = _count_unique_pairs(edge_index)

    assert dataset.train_pos_edge_index.numel() > 0
    assert dataset.train_neg_edge_index.size(1) == dataset.train_pos_edge_index.size(1)
    assert expected_total == total_pos
    assert dataset.data.edge_index is not None
    assert dataset.data.edge_index.numel() > 0

    # Arestas utilizadas na mensagem devem corresponder às arestas de treino
    train_pairs_from_adj = _count_unique_pairs(dataset.data.edge_index)
    assert train_pairs_from_adj == dataset.train_pos_edge_index.size(1)

    # Splits de validação/teste mantêm formato esperado (2, N)
    assert dataset.val_pos_edge_index.size(0) == 2
    assert dataset.val_neg_edge_index.size(0) == 2
    assert dataset.test_pos_edge_index.size(0) == 2
    assert dataset.test_neg_edge_index.size(0) == 2


def test_build_link_prediction_dataset_handles_small_graphs():
    x = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    dataset = build_link_prediction_dataset(data, val_ratio=0.5, test_ratio=0.0, seed=42)

    assert dataset.train_pos_edge_index.numel() > 0
    assert dataset.train_neg_edge_index.numel() == dataset.train_pos_edge_index.numel()
    assert dataset.data.edge_index is not None
    assert dataset.data.edge_index.numel() > 0
