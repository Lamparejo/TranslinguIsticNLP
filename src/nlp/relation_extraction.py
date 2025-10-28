"""Simple heuristic relation extraction based on sentence-level co-occurrence."""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from ..data.models import EntityMention, NewsArticle, Relation, ResolvedEntity
from ..data.preprocess import split_sentences
from ..utils.logging import get_logger
from .rebel_pipeline import RebelExtractorConfig, RebelRelationExtractor

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RelationExtractionConfig:
    """Configuration for the relation extractor."""

    relation_type: str = "MENTIONED_TOGETHER"
    min_weight: float = 1.0


class RelationExtractor:
    """Create relations by pairing entities that co-occur in the same sentence."""

    def __init__(self, config: RelationExtractionConfig):
        self.config = config

    def extract(self, entities: Iterable[ResolvedEntity]) -> List[Relation]:
        relations: List[Relation] = []
        entities_list = list(entities)
        for entity_a, entity_b in combinations(entities_list, 2):
            shared_mentions = self._find_shared_mentions(entity_a, entity_b)
            for mention_a, mention_b in shared_mentions:
                relation = Relation(
                    relation_id=str(uuid4()),
                    source_id=entity_a.canonical_id,
                    target_id=entity_b.canonical_id,
                    relation_type=self.config.relation_type,
                    weight=self.config.min_weight,
                    article_id=mention_a.article_id,
                    sentence=mention_a.sentence,
                    published_at=self._resolve_publication(mention_a, mention_b),
                )
                relations.append(relation)
        LOGGER.info("Constructed %s relations", len(relations))
        return relations

    @staticmethod
    def _find_shared_mentions(entity_a: ResolvedEntity, entity_b: ResolvedEntity):
        for mention_a in entity_a.mentions:
            for mention_b in entity_b.mentions:
                if mention_a.article_id == mention_b.article_id and mention_a.sentence == mention_b.sentence:
                    yield mention_a, mention_b

    @staticmethod
    def _resolve_publication(mention_a, mention_b) -> datetime:
        # Use the mention_a context since they are in the same sentence/article.
        for mention in (mention_a, mention_b):
            if hasattr(mention, "published_at") and mention.published_at:
                return mention.published_at
        # If mention does not carry publication metadata, fall back to now.
        return datetime.utcnow()


_NON_WORD_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowercase = without_accents.lower()
    without_punct = _NON_WORD_PATTERN.sub(" ", lowercase)
    return " ".join(without_punct.split())


def _is_fuzzy_match(left: str, right: str, threshold: float = 0.82) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    ratio = difflib.SequenceMatcher(None, left, right).ratio()
    return ratio >= threshold


class RebelGraphRelationExtractor:
    """Derive graph relations from REBEL triples and resolved entities."""

    def __init__(
        self,
        config: RebelExtractorConfig,
        extractor: RebelRelationExtractor | None = None,
    ) -> None:
        self.config = config
        self._extractor = extractor or RebelRelationExtractor(config)

    def extract(
        self,
        articles: Iterable[NewsArticle],
        entities: Iterable[ResolvedEntity],
    ) -> List[Relation]:
        articles_list = list(articles)
        entities_list = list(entities)
        if not articles_list or not entities_list:
            LOGGER.info("No data available for REBEL relation extraction")
            return []

        mentions_by_sentence: Dict[Tuple[str, str], List[EntityMention]] = defaultdict(list)
        mentions_by_article: Dict[str, List[EntityMention]] = defaultdict(list)
        mention_to_entity: Dict[str, ResolvedEntity] = {}

        for entity in entities_list:
            for mention in entity.mentions:
                mentions_by_sentence[(mention.article_id, mention.sentence)].append(mention)
                mentions_by_article[mention.article_id].append(mention)
                mention_to_entity[mention.mention_id] = entity

        relation_records: List[Relation] = []
        seen_keys: set[Tuple[str, str, str, str, str]] = set()

        for article in articles_list:
            sentences = split_sentences(article.text)
            for sentence in sentences:
                if len(sentence.strip()) < self.config.min_sentence_length:
                    continue
                triples = self._extractor.extract_from_sentence(sentence)
                if not triples:
                    continue
                for triple in triples:
                    source = self._resolve_entity(
                        triple.subject,
                        article.article_id,
                        sentence,
                        mentions_by_sentence,
                        mentions_by_article,
                        mention_to_entity,
                        entities_list,
                    )
                    target = self._resolve_entity(
                        triple.object,
                        article.article_id,
                        sentence,
                        mentions_by_sentence,
                        mentions_by_article,
                        mention_to_entity,
                        entities_list,
                    )
                    if source is None or target is None:
                        continue
                    if source.canonical_id == target.canonical_id:
                        continue
                    dedupe_key = (
                        source.canonical_id,
                        target.canonical_id,
                        _normalize(triple.relation),
                        article.article_id,
                        sentence,
                    )
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    relation_records.append(
                        Relation(
                            relation_id=str(uuid4()),
                            source_id=source.canonical_id,
                            target_id=target.canonical_id,
                            relation_type=triple.relation,
                            weight=triple.score,
                            article_id=article.article_id,
                            sentence=sentence,
                            published_at=article.published_at,
                        )
                    )

        LOGGER.info("Constructed %s REBEL-backed relations", len(relation_records))
        return relation_records

    def _resolve_entity(
        self,
        surface: str,
        article_id: str,
        sentence: str,
        mentions_by_sentence: Dict[Tuple[str, str], List[EntityMention]],
        mentions_by_article: Dict[str, List[EntityMention]],
        mention_to_entity: Dict[str, ResolvedEntity],
        entities: List[ResolvedEntity],
    ) -> Optional[ResolvedEntity]:
        normalized_surface = _normalize(surface)
        if not normalized_surface:
            return None

        sentence_mentions = mentions_by_sentence.get((article_id, sentence), [])
        entity = self._match_mentions(normalized_surface, sentence_mentions, mention_to_entity)
        if entity is not None:
            return entity

        article_mentions = mentions_by_article.get(article_id, [])
        entity = self._match_mentions(normalized_surface, article_mentions, mention_to_entity)
        if entity is not None:
            return entity

        for resolved in entities:
            if not any(mention.article_id == article_id for mention in resolved.mentions):
                continue
            canonical = _normalize(resolved.name)
            if not canonical:
                continue
            if canonical == normalized_surface:
                return resolved
            if canonical in normalized_surface or normalized_surface in canonical:
                return resolved
            if _is_fuzzy_match(canonical, normalized_surface):
                return resolved
        return None

    @staticmethod
    def _match_mentions(
        normalized_surface: str,
        mentions: List[EntityMention],
        mention_to_entity: Dict[str, ResolvedEntity],
    ) -> Optional[ResolvedEntity]:
        for mention in mentions:
            mention_norm = _normalize(mention.text)
            if mention_norm == normalized_surface:
                return mention_to_entity.get(mention.mention_id)
        for mention in mentions:
            mention_norm = _normalize(mention.text)
            if not mention_norm:
                continue
            if normalized_surface in mention_norm or mention_norm in normalized_surface:
                return mention_to_entity.get(mention.mention_id)
        for mention in mentions:
            mention_norm = _normalize(mention.text)
            if _is_fuzzy_match(mention_norm, normalized_surface):
                return mention_to_entity.get(mention.mention_id)
        return None
