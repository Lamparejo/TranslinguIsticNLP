"""Dataclasses describing the core data structures processed by the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(slots=True)
class NewsArticle:
    """Represents a news article harvested from the GDELT dataset."""

    article_id: str
    language: str
    text: str
    published_at: datetime
    source_url: str
    themes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class EntityMention:
    """A raw entity mention extracted by the NER component."""

    mention_id: str
    article_id: str
    text: str
    entity_type: str
    sentence: str
    start_char: int
    end_char: int
    language: str


@dataclass(slots=True)
class ResolvedEntity:
    """A canonical entity after cross-lingual resolution."""

    canonical_id: str
    name: str
    entity_type: str
    mentions: List[EntityMention] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass(slots=True)
class Relation:
    """Represents a detected relationship between two resolved entities."""

    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    article_id: str
    sentence: str
    published_at: datetime


@dataclass(slots=True)
class GraphRecord:
    """Container aggregating resolved entities and relations for persistence."""

    entities: List[ResolvedEntity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
