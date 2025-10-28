"""Named Entity Recognition utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
from uuid import uuid4

from ..data.models import EntityMention, NewsArticle
from ..data.preprocess import split_sentences
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class NERConfig:
    """Configuration for the named entity recogniser."""

    model_name: str
    entity_types: Sequence[str] | None = None
    device: int = -1
    aggregation_strategy: str = "simple"


class NamedEntityExtractor:
    """Wraps a Hugging Face pipeline to produce entity mentions."""

    def __init__(self, config: NERConfig, ner_pipeline=None):
        self.config = config
        self._pipeline = ner_pipeline

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "transformers must be installed to run the NER pipeline"
                ) from exc
            LOGGER.info("Loading NER model %s", self.config.model_name)
            self._pipeline = pipeline(
                "ner",
                model=self.config.model_name,
                aggregation_strategy=self.config.aggregation_strategy,
                device=self.config.device,
            )
        return self._pipeline

    def extract(self, articles: Iterable[NewsArticle]) -> List[EntityMention]:
        """Extract entity mentions from the provided articles."""
        pipeline = self._load_pipeline()
        mentions: List[EntityMention] = []
        for article in articles:
            mentions.extend(self._process_article(pipeline, article))
        LOGGER.info("Extracted %s mentions", len(mentions))
        return mentions

    def _process_article(self, pipeline, article: NewsArticle) -> List[EntityMention]:
        mentions: List[EntityMention] = []
        for sentence in split_sentences(article.text):
            mentions.extend(self._process_sentence(pipeline, article, sentence))
        return mentions

    def _process_sentence(self, pipeline, article: NewsArticle, sentence: str) -> List[EntityMention]:
        mentions: List[EntityMention] = []
        allowed_types = self.config.entity_types
        for item in pipeline(sentence):
            entity_group = item.get("entity_group") or item.get("entity")
            if allowed_types and entity_group not in allowed_types:
                continue
            mentions.append(
                EntityMention(
                    mention_id=str(uuid4()),
                    article_id=article.article_id,
                    text=item["word"],
                    entity_type=entity_group,
                    sentence=sentence,
                    start_char=item.get("start", 0),
                    end_char=item.get("end", 0),
                    language=article.language,
                )
            )
        return mentions
