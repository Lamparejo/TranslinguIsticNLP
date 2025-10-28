"""Sentence embedding utilities for cross-lingual similarity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class SentenceEmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str
    device: str | None = None


class SentenceEmbeddingService:
    """Wraps a sentence-transformers model to generate embeddings."""

    def __init__(self, config: SentenceEmbeddingConfig, model=None):
        self.config = config
        self._model = model

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers must be installed to use embeddings"
                ) from exc
            LOGGER.info("Loading sentence embedding model %s", self.config.model_name)
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        """Compute embeddings for the provided sentences."""
        model = self._load_model()
        sentence_list = list(sentences)
        if not sentence_list:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = model.encode(sentence_list, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)
