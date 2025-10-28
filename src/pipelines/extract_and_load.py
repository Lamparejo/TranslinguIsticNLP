"""End-to-end pipeline: ingest articles, extract entities, load graph, export snapshots."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np

from ..analytics.metrics import compute_graph_metrics
from ..data.csv_loader import CSVArticleLoader, CSVArticleSourceConfig
from ..data.gdelt_loader import GDELTConfig, GDELTLoader
from ..data.models import GraphRecord
from ..nlp.entity_resolution import EntityResolutionConfig, EntityResolver
from ..nlp.ner import NERConfig, NamedEntityExtractor
from ..nlp.relation_extraction import (
    RebelGraphRelationExtractor,
    RelationExtractionConfig,
    RelationExtractor,
)
from ..nlp.rebel_pipeline import RebelExtractorConfig
from ..nlp.sentence_embeddings import SentenceEmbeddingConfig, SentenceEmbeddingService
from ..graph.neo4j_loader import Neo4jConfig, persist_graph
from ..graph.export import PyGExportConfig, export_to_disk
from ..utils.config import load_config
from ..utils.io import write_json, write_jsonl
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def run_pipeline(config_path: str | Path = "config/pipeline.yaml") -> GraphRecord:
    config = load_config(config_path)

    articles = _load_articles(config)
    if not articles:
        LOGGER.warning("No articles were loaded; aborting pipeline")
        return GraphRecord()

    mentions = _run_ner(config, articles)
    if not mentions:
        LOGGER.warning("NER extracted no entities; aborting pipeline")
        return GraphRecord()

    embeddings = _generate_embeddings(config, mentions)
    resolver_cfg = EntityResolutionConfig(
        similarity_threshold=config["nlp"].get("similarity_threshold", 0.8)
    )
    resolved_entities = EntityResolver(resolver_cfg).resolve(mentions, embeddings)

    relations = _extract_relations(config, articles, resolved_entities)

    graph = GraphRecord(entities=resolved_entities, relations=relations)

    analytics_cfg = config.get("analytics", {})
    metrics_path = analytics_cfg.get("dashboard_metrics_path", "artifacts/dashboard_metrics.json")
    snapshot_path = analytics_cfg.get("graph_snapshot_path")

    if snapshot_path:
        _write_graph_snapshot(graph, snapshot_path)
    metrics_payload = compute_graph_metrics(graph)
    metadata = metrics_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        metrics_payload["metadata"] = metadata

    if snapshot_path:
        metadata["graph_snapshot_path"] = snapshot_path

    gnn_cfg = config.get("gnn", {})
    gnn_metrics_path = gnn_cfg.get("metrics_path")
    if gnn_metrics_path:
        metadata["gnn_metrics_path"] = gnn_metrics_path
    gnn_model_path = gnn_cfg.get("model_artifact_path")
    if gnn_model_path:
        metadata["gnn_model_path"] = gnn_model_path
    write_json(metrics_path, metrics_payload)
    LOGGER.info("Dashboard metrics saved to %s", metrics_path)

    neo4j_cfg = config.get("graph", {}).get("neo4j")
    if neo4j_cfg:
        try:
            persist_graph(
                Neo4jConfig(
                    uri=neo4j_cfg["uri"],
                    user=neo4j_cfg["user"],
                    password=neo4j_cfg["password"],
                ),
                graph,
            )
        except Exception as exc:  # pragma: no cover - requires live Neo4j service
            LOGGER.warning(
                "Failed to persist graph to Neo4j (%s). Continuing without database sync.",
                exc,
            )

    dataset_path = gnn_cfg.get("dataset_path")
    if dataset_path:
        metadata["gnn_dataset_path"] = dataset_path
        export_to_disk(
            graph,
            PyGExportConfig(
                output_path=dataset_path,
                default_feature_dim=gnn_cfg.get("default_feature_dim", 32),
            ),
        )
    return graph


def _load_articles(config):
    data_cfg = config.get("data", {})
    csv_cfg = data_cfg.get("csv")
    if csv_cfg and csv_cfg.get("path"):
        loader = CSVArticleLoader(
            CSVArticleSourceConfig(
                path=csv_cfg["path"],
                limit=csv_cfg.get("limit", 15),
                url_column_index=csv_cfg.get("url_column_index", 4),
                article_id_column_index=csv_cfg.get("article_id_column_index", 0),
                published_at_column_index=csv_cfg.get("published_at_column_index", 1),
                encoding=csv_cfg.get("encoding", "utf-8"),
                separator=csv_cfg.get("separator", "\t"),
                timeout=csv_cfg.get("timeout", 20),
            )
        )
        articles = list(loader.iter_articles())
        preview_path = csv_cfg.get("preview_path")
        if preview_path:
            _write_articles_preview(
                articles,
                preview_path,
                csv_cfg.get("preview_limit", csv_cfg.get("limit", 15)),
            )
    else:
        gdelt_cfg = data_cfg.get("gdelt", {})
        loader = GDELTLoader(
            GDELTConfig(
                gkg_url=gdelt_cfg.get("gkg_url"),
                local_path=gdelt_cfg.get("sample_path"),
            )
        )
        articles = list(loader.iter_articles())
    LOGGER.info("Loaded %s articles", len(articles))
    return articles


def _write_articles_preview(articles, path, limit):
    if not path:
        return
    preview_path = Path(path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    limit = limit or len(articles)
    payload = [
        {
            "article_id": article.article_id,
            "source_url": article.source_url,
            "published_at": article.published_at.isoformat(),
            "text": article.text,
        }
        for article in articles[:limit]
    ]
    write_jsonl(preview_path, payload)
    LOGGER.info("Saved article preview to %s", preview_path)


def _run_ner(config, articles):
    ner_cfg = config.get("nlp", {}).get("ner", {})
    raw_entity_types = ner_cfg.get("entity_types")
    entity_types = None
    if raw_entity_types:
        entity_types = [
            str(entity_type).strip()
            for entity_type in raw_entity_types
            if str(entity_type).strip()
        ]
    extractor = NamedEntityExtractor(
        NERConfig(
            model_name=ner_cfg.get("model", "xlm-roberta-large-finetuned-conll03-english"),
            entity_types=entity_types,
            device=ner_cfg.get("device", -1),
            aggregation_strategy=ner_cfg.get("aggregation_strategy", "simple"),
        )
    )
    mentions = extractor.extract(articles)
    return mentions


def _generate_embeddings(config, mentions):
    embed_cfg = config.get("nlp", {}).get("embeddings", {})
    service = SentenceEmbeddingService(
        SentenceEmbeddingConfig(
            model_name=embed_cfg.get(
                "model",
                "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            ),
            device=embed_cfg.get("device"),
        )
    )
    phrases = [
        f"{mention.text.strip()} [{mention.entity_type}]"
        if mention.text.strip()
        else mention.sentence
        for mention in mentions
    ]
    if not phrases:
        return np.zeros((0, 0), dtype=np.float32)
    return service.encode(phrases)


def _extract_relations(config, articles, resolved_entities):
    nlp_cfg = config.get("nlp", {})
    strategy = nlp_cfg.get("relation_strategy", "cooccurrence")
    if strategy == "rebel":
        rebel_cfg = nlp_cfg.get("rebel", {})
        LOGGER.info("Using REBEL relation extraction strategy")
        rebel_config = RebelExtractorConfig(
            model_name=rebel_cfg.get("model_name", "Babelscape/rebel-large"),
            device=rebel_cfg.get("device"),
            max_length=rebel_cfg.get("max_length", 256),
            num_beams=rebel_cfg.get("num_beams", 3),
            article_timeout=rebel_cfg.get("article_timeout", 30),
            min_sentence_length=rebel_cfg.get("min_sentence_length", 10),
        )
        extractor = RebelGraphRelationExtractor(rebel_config)
        return extractor.extract(articles, resolved_entities)

    LOGGER.info("Using co-occurrence relation extraction strategy")
    return RelationExtractor(
        RelationExtractionConfig(
            relation_type=nlp_cfg.get("relation_type", "MENTIONED_TOGETHER"),
            min_weight=nlp_cfg.get("relation_min_weight", 1.0),
        )
    ).extract(resolved_entities)


def _write_graph_snapshot(graph: GraphRecord, path: str | Path) -> None:
    if not path:
        return
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = [
        {
            "id": entity.canonical_id,
            "label": entity.name,
            "type": entity.entity_type,
            "mention_count": len(entity.mentions),
        }
        for entity in graph.entities
    ]
    edges = [
        {
            "source": relation.source_id,
            "target": relation.target_id,
            "weight": relation.weight,
            "relation_type": relation.relation_type,
            "sentence": relation.sentence,
            "published_at": _isoformat(relation.published_at),
        }
        for relation in graph.relations
    ]

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "nodes": nodes,
        "edges": edges,
    }
    write_json(snapshot_path, payload)
    LOGGER.info("Saved graph snapshot to %s", snapshot_path)


def _isoformat(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value
