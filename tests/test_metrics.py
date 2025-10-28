from datetime import datetime

from src.analytics.metrics import compute_graph_metrics
from src.data.models import EntityMention, GraphRecord, Relation, ResolvedEntity


def build_entity(
    canonical_id: str,
    name: str,
    entity_type: str,
    article_id: str,
    language: str,
) -> ResolvedEntity:
    mention = EntityMention(
        mention_id=f"m-{canonical_id}",
        article_id=article_id,
        text=name,
        entity_type=entity_type,
        sentence=f"{name} appeared in headline.",
        start_char=0,
        end_char=len(name),
        language=language,
    )
    return ResolvedEntity(
        canonical_id=canonical_id,
        name=name,
        entity_type=entity_type,
        mentions=[mention],
        embedding=[0.1, 0.2, 0.3],
    )


def test_compute_graph_metrics_counts_entities_and_relations():
    entity_a = build_entity("e1", "White House", "ORG", "a1", "en")
    entity_b = build_entity("e2", "Maison Blanche", "ORG", "a2", "fr")
    relation = Relation(
        relation_id="r1",
        source_id="e1",
        target_id="e2",
        relation_type="MENTIONED_TOGETHER",
        weight=1.0,
        article_id="a1",
        sentence="White House met Maison Blanche.",
        published_at=datetime(2024, 1, 15),
    )
    graph = GraphRecord(entities=[entity_a, entity_b], relations=[relation])

    metrics = compute_graph_metrics(graph)

    metadata = metrics.get("metadata")
    entity_summary = metrics.get("entity_summary")
    relation_summary = metrics.get("relation_summary")
    articles_summary = metrics.get("articles")
    top_entities = metrics.get("top_entities_by_degree")
    timeline = metrics.get("timeline")
    graph_summary = metrics.get("graph_summary")

    assert isinstance(metadata, dict)
    assert isinstance(entity_summary, dict)
    assert isinstance(relation_summary, dict)
    assert isinstance(articles_summary, dict)
    assert isinstance(top_entities, list)
    assert isinstance(timeline, dict)
    assert isinstance(graph_summary, dict)

    assert metadata["num_entities"] == 2
    assert metadata["num_relations"] == 1
    assert entity_summary["entity_type_counts"]["ORG"] == 2
    assert relation_summary["relation_type_counts"]["MENTIONED_TOGETHER"] == 1
    assert relation_summary["max_relation_weight"] == 1.0
    assert relation_summary["median_relation_weight"] == 1.0
    assert relation_summary["min_relation_weight"] == 1.0
    assert articles_summary["unique_articles"] == 2
    assert isinstance(top_entities[0], dict)
    assert top_entities[0]["degree"] == 1
    assert top_entities[0]["degree"] == 1
    assert timeline["total_days"] == 1
    assert len(timeline["daily"]) == 1
    assert timeline["daily"][0]["relations"] == 1
    assert timeline["daily"][0]["unique_entities"] == 2
    assert graph_summary["connected_components"] == 1
    assert graph_summary["largest_component_size"] == 2
    assert graph_summary["median_degree"] == 1
    assert graph_summary["isolated_nodes"] == 0
    assert timeline["peak_relations"] == 1
    assert timeline["peak_entities"] == 2