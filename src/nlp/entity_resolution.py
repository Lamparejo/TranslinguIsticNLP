"""Entity resolution across languages using cosine similarity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
from uuid import uuid4

import numpy as np
from numpy.linalg import norm

from ..data.models import EntityMention, ResolvedEntity
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EntityResolutionConfig:
    """Configuration for the entity resolver."""

    similarity_threshold: float = 0.8


class EntityResolver:
    """Group entity mentions into canonical entities based on cosine similarity."""

    def __init__(self, config: EntityResolutionConfig):
        self.config = config

    def resolve(self, mentions: Sequence[EntityMention], embeddings: np.ndarray) -> List[ResolvedEntity]:
        if len(mentions) != len(embeddings):
            raise ValueError("Mentions and embeddings must have matching lengths")
        clusters: list[list[int]] = []
        centroid_vectors: list[np.ndarray] = []
        for idx, embedding in enumerate(embeddings):
            assigned = False
            for cluster_index, centroid in enumerate(centroid_vectors):
                similarity = self._cosine_similarity(embedding, centroid)
                if similarity >= self.config.similarity_threshold:
                    clusters[cluster_index].append(idx)
                    centroid_vectors[cluster_index] = self._recompute_centroid(embeddings, clusters[cluster_index])
                    assigned = True
                    break
            if not assigned:
                clusters.append([idx])
                centroid_vectors.append(embedding.astype(np.float32))
        resolved = [self._build_entity(mentions, embeddings, cluster) for cluster in clusters]
        LOGGER.info("Resolved %s canonical entities from %s mentions", len(resolved), len(mentions))
        return resolved

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        denominator = norm(vec_a) * norm(vec_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denominator)

    @staticmethod
    def _recompute_centroid(embeddings: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        vectors = embeddings[indices]
        return vectors.mean(axis=0)

    def _build_entity(
        self,
        mentions: Sequence[EntityMention],
        embeddings: np.ndarray,
        indices: Sequence[int],
    ) -> ResolvedEntity:
        selected_mentions = [mentions[i] for i in indices]
        canonical_name = max(selected_mentions, key=lambda mention: len(mention.text)).text
        entity_type = selected_mentions[0].entity_type
        embedding = self._recompute_centroid(embeddings, indices).astype(float).tolist()
        return ResolvedEntity(
            canonical_id=str(uuid4()),
            name=canonical_name,
            entity_type=entity_type,
            mentions=list(selected_mentions),
            embedding=embedding,
        )
