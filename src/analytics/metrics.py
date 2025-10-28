"""Aggregation helpers producing dashboard-ready metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
from statistics import median, pstdev, quantiles
from typing import DefaultDict, Dict, List, Set

from src.data.models import GraphRecord


def compute_graph_metrics(graph: GraphRecord) -> Dict[str, object]:
    """Compute aggregate metrics for a `GraphRecord`.

    The returned dictionary is JSON-serialisable and suitable for dashboard consumption.
    """
    metrics: Dict[str, object] = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "num_entities": len(graph.entities),
            "num_relations": len(graph.relations),
        },
        "timeline": {
            "daily": [],
            "total_days": 0,
        },
    }

    if not graph.entities:
        metrics.update(
            {
                "entity_summary": {
                    "entity_type_counts": {},
                    "avg_mentions_per_entity": 0.0,
                },
                "relation_summary": {
                    "relation_type_counts": {},
                    "avg_relation_weight": 0.0,
                    "graph_density": 0.0,
                },
                "language_distribution": {},
                "articles": {
                    "unique_articles": 0,
                    "time_span": None,
                },
                "top_entities_by_degree": [],
            }
        )
        return metrics

    entity_type_counter = Counter(entity.entity_type for entity in graph.entities)
    total_mentions = sum(len(entity.mentions) for entity in graph.entities)
    mentions_languages = Counter(
        mention.language for entity in graph.entities for mention in entity.mentions
    )
    article_ids = {
        mention.article_id for entity in graph.entities for mention in entity.mentions
    }
    language_distribution = dict(sorted(mentions_languages.items(), key=lambda item: item[1], reverse=True))

    relation_type_counter = Counter(relation.relation_type for relation in graph.relations)
    relation_weights = [relation.weight for relation in graph.relations]
    avg_relation_weight = (
        sum(relation_weights) / len(relation_weights)
        if relation_weights
        else 0.0
    )

    degree_counter = Counter()
    adjacency: DefaultDict[str, Set[str]] = defaultdict(set)
    for relation in graph.relations:
        degree_counter[relation.source_id] += 1
        degree_counter[relation.target_id] += 1
        adjacency[relation.source_id].add(relation.target_id)
        adjacency[relation.target_id].add(relation.source_id)

    for entity in graph.entities:
        adjacency.setdefault(entity.canonical_id, set())

    top_entities_by_degree: List[Dict[str, object]] = []
    entity_lookup = {entity.canonical_id: entity for entity in graph.entities}
    for canonical_id, degree in degree_counter.most_common(10):
        entity = entity_lookup.get(canonical_id)
        if not entity:
            continue
        top_entities_by_degree.append(
            {
                "canonical_id": canonical_id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "degree": degree,
            }
        )

    first_event: datetime | None = None
    last_event: datetime | None = None
    for relation in graph.relations:
        if first_event is None or relation.published_at < first_event:
            first_event = relation.published_at
        if last_event is None or relation.published_at > last_event:
            last_event = relation.published_at

    density = 0.0
    num_entities = len(graph.entities)
    if num_entities > 1:
        density = len(graph.relations) / (num_entities * (num_entities - 1))

    relations_per_day: DefaultDict[str, int] = defaultdict(int)
    entities_per_day: DefaultDict[str, Set[str]] = defaultdict(set)
    for relation in graph.relations:
        date_key = relation.published_at.date().isoformat()
        relations_per_day[date_key] += 1
        entities_per_day[date_key].update({relation.source_id, relation.target_id})

    daily_timeline: List[Dict[str, object]] = [
        {
            "date": date_key,
            "relations": relations_per_day[date_key],
            "unique_entities": len(entities_per_day[date_key]),
        }
        for date_key in sorted(relations_per_day.keys())
    ]

    total_degree = sum(degree_counter.values())
    avg_degree = total_degree / num_entities if num_entities else 0.0
    max_degree = max(degree_counter.values(), default=0)
    degree_values = [len(neighbours) for neighbours in adjacency.values()]
    median_degree = median(degree_values) if degree_values else 0.0
    degree_percentile_90 = (
        quantiles(degree_values, n=10, method="inclusive")[8]
        if len(degree_values) > 1
        else float(max_degree)
    )
    isolated_nodes = sum(1 for neighbours in adjacency.values() if not neighbours)

    def _average_clustering() -> float:
        clustering_values: List[float] = []
        for node, neighbours in adjacency.items():
            neighbour_list = list(neighbours)
            degree = len(neighbour_list)
            if degree < 2:
                continue
            possible_edges = degree * (degree - 1) / 2
            if possible_edges == 0:
                continue
            closed_edges = 0
            for left, right in combinations(neighbour_list, 2):
                if right in adjacency[left]:
                    closed_edges += 1
            clustering_values.append(closed_edges / possible_edges)
        if not clustering_values:
            return 0.0
        return sum(clustering_values) / len(clustering_values)

    average_clustering = _average_clustering()

    def _connected_components() -> List[Set[str]]:
        components: List[Set[str]] = []
        visited: Set[str] = set()
        for node in adjacency:
            if node in visited:
                continue
            stack = [node]
            component: Set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(neighbour for neighbour in adjacency[current] if neighbour not in visited)
            components.append(component)
        return components

    components = _connected_components()
    num_components = len(components)
    largest_component_size = max((len(component) for component in components), default=0)

    metrics.update(
        {
            "entity_summary": {
                "entity_type_counts": dict(entity_type_counter),
                "avg_mentions_per_entity": total_mentions / len(graph.entities),
            },
            "relation_summary": {
                "relation_type_counts": dict(relation_type_counter),
                "avg_relation_weight": avg_relation_weight,
                "graph_density": density,
                "max_relation_weight": max((relation.weight for relation in graph.relations), default=0.0),
                "min_relation_weight": min(relation_weights) if relation_weights else 0.0,
                "median_relation_weight": median(relation_weights) if relation_weights else 0.0,
                "std_relation_weight": pstdev(relation_weights) if len(relation_weights) > 1 else 0.0,
            },
            "language_distribution": language_distribution,
            "articles": {
                "unique_articles": len(article_ids),
                "time_span": {
                    "start": first_event.isoformat() if first_event else None,
                    "end": last_event.isoformat() if last_event else None,
                },
            },
            "top_entities_by_degree": top_entities_by_degree,
            "graph_summary": {
                "average_degree": avg_degree,
                "max_degree": max_degree,
                "connected_components": num_components,
                "largest_component_size": largest_component_size,
                "median_degree": median_degree,
                "degree_percentile_90": degree_percentile_90,
                "isolated_nodes": isolated_nodes,
                "average_clustering_coefficient": average_clustering,
            },
        }
    )

    metrics["timeline"] = {
        "daily": daily_timeline,
        "total_days": len(daily_timeline),
        "peak_relations": max(relations_per_day.values(), default=0),
        "peak_entities": max((len(entities) for entities in entities_per_day.values()), default=0),
    }

    return metrics
