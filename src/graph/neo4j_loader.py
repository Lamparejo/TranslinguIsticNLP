"""Utility for writing entities and relations into Neo4j."""
from __future__ import annotations

from dataclasses import dataclass
from ..data.models import GraphRecord, Relation, ResolvedEntity
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


class Neo4jLoader:
    """Wrapper around the official Neo4j driver to persist the knowledge graph."""

    def __init__(self, config: Neo4jConfig):
        self.config = config
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("neo4j must be installed to use the Neo4j loader") from exc
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    def close(self) -> None:
        self._driver.close()

    def sync_graph(self, graph: GraphRecord) -> None:
        """Persist the supplied entities and relations into the database."""
        with self._driver.session() as session:
            for entity in graph.entities:
                session.execute_write(self._merge_entity, entity)
            LOGGER.info("Persisted %s entities", len(graph.entities))
            for relation in graph.relations:
                session.execute_write(self._merge_relation, relation)
            LOGGER.info("Persisted %s relations", len(graph.relations))

    @staticmethod
    def _merge_entity(tx, entity: ResolvedEntity) -> None:
        labels = ["Entity", entity.entity_type.title()]
        label_clause = ":".join(labels)
        tx.run(
            f"MERGE (e:{label_clause} {{canonical_id: $canonical_id}}) "
            "SET e.name = $name, e.entity_type = $entity_type, e.embedding = $embedding",
            canonical_id=entity.canonical_id,
            name=entity.name,
            entity_type=entity.entity_type,
            embedding=entity.embedding,
        )

    @staticmethod
    def _merge_relation(tx, relation: Relation) -> None:
        tx.run(
            "MATCH (source {canonical_id: $source_id}) "
            "MATCH (target {canonical_id: $target_id}) "
            f"MERGE (source)-[rel:{relation.relation_type}]->(target) "
            "SET rel.weight = $weight, rel.article_id = $article_id, "
            "rel.sentence = $sentence, rel.published_at = $published_at",
            source_id=relation.source_id,
            target_id=relation.target_id,
            weight=relation.weight,
            article_id=relation.article_id,
            sentence=relation.sentence,
            published_at=relation.published_at.isoformat(),
        )


def persist_graph(config: Neo4jConfig, graph: GraphRecord) -> None:
    loader = Neo4jLoader(config)
    try:
        loader.sync_graph(graph)
    finally:
        loader.close()
