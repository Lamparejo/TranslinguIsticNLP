"""Graph neural network models for link prediction."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GraphModelConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    dropout: float = 0.2


def build_model(config: GraphModelConfig):
    try:
        import torch  # type: ignore[import-not-found]
        from torch import nn  # type: ignore[import-not-found]
        from torch_geometric.nn import SAGEConv  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch and torch-geometric must be installed to build the model"
        ) from exc

    class GraphSAGEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(config.in_channels, config.hidden_channels)
            self.conv2 = SAGEConv(config.hidden_channels, config.out_channels)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x, edge_index):  # type: ignore[override]
            x = self.conv1(x, edge_index).relu()
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return x

    class LinkPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(config.out_channels * 2, config.hidden_channels)
            self.lin2 = nn.Linear(config.hidden_channels, 1)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, src_emb, dst_emb):  # type: ignore[override]
            h = torch.cat([src_emb, dst_emb], dim=-1)
            h = self.dropout(h.relu())
            h = self.lin1(h).relu()
            return torch.sigmoid(self.lin2(h)).view(-1)

    return GraphSAGEModel(), LinkPredictor()
