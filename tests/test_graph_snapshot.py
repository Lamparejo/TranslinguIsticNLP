from datetime import datetime

from src.data.models import EntityMention, GraphRecord, Relation, ResolvedEntity
from src.pipelines.extract_and_load import _write_graph_snapshot


def test_write_graph_snapshot(tmp_path):
    mention = EntityMention(
        mention_id="m1",
        article_id="a1",
        text="Alice",
        entity_type="PER",
        sentence="Alice met Bob",
        start_char=0,
        end_char=5,
        language="en",
    )
    entity_a = ResolvedEntity(
        canonical_id="entity-a",
        name="Alice",
        entity_type="PER",
        mentions=[mention],
        embedding=[0.1, 0.2],
    )
    entity_b = ResolvedEntity(
        canonical_id="entity-b",
        name="Bob",
        entity_type="PER",
        mentions=[],
        embedding=[0.3, 0.4],
    )
    relation = Relation(
        relation_id="rel-1",
        source_id="entity-a",
        target_id="entity-b",
        relation_type="MENTIONED_TOGETHER",
        weight=1.0,
        article_id="a1",
        sentence="Alice met Bob",
        published_at=datetime(2024, 1, 1, 12, 0, 0),
    )

    output_path = tmp_path / "snapshot.json"
    graph = GraphRecord(entities=[entity_a, entity_b], relations=[relation])

    _write_graph_snapshot(graph, output_path)

    assert output_path.exists()
    payload = output_path.read_text(encoding="utf-8")
    assert "entity-a" in payload
    assert "entity-b" in payload
    assert "MENTIONED_TOGETHER" in payload
    assert "2024-01-01T12:00:00" in payload
