from datetime import datetime

import numpy as np
import pytest  # type: ignore[import-not-found]

from src.data.models import EntityMention, GraphRecord, Relation, ResolvedEntity
from src.graph.export import graph_record_to_pyg


def build_entity(entity_id: str, name: str) -> ResolvedEntity:
    mention = EntityMention(
        mention_id=f"m-{entity_id}",
        article_id="a1",
        text=name,
        entity_type="ORG",
        sentence="Leaders met in Berlin.",
        start_char=0,
        end_char=len(name),
        language="en",
    )
    return ResolvedEntity(
        canonical_id=entity_id,
        name=name,
        entity_type="ORG",
        mentions=[mention],
        embedding=np.array([0.1, 0.2, 0.3], dtype=float).tolist(),
    )


def test_graph_record_to_pyg_converts_entities_and_relations():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    entities = [build_entity("e1", "Entity One"), build_entity("e2", "Entity Two")]
    relation = Relation(
        relation_id="r1",
        source_id="e1",
        target_id="e2",
        relation_type="MENTIONED_TOGETHER",
        weight=1.0,
        article_id="a1",
        sentence="Leaders met in Berlin.",
        published_at=datetime.utcnow(),
    )

    graph = GraphRecord(entities=entities, relations=[relation])
    data = graph_record_to_pyg(graph)

    assert data.num_nodes == 2
    assert data.num_edges == 1
    assert data.edge_index.shape[1] == 1  # type: ignore[union-attr]
    assert data.x.shape[0] == 2  # type: ignore[union-attr]
    assert hasattr(data, "node_metadata")
    assert data.node_metadata[0]["id"] == "e1"
    assert data.node_ids == [meta["id"] for meta in data.node_metadata]