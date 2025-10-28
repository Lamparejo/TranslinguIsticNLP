import numpy as np

from src.data.models import EntityMention
from src.nlp.entity_resolution import EntityResolutionConfig, EntityResolver


def build_mention(mention_id: str, text: str, sentence: str, article_id: str = "a1"):
    return EntityMention(
        mention_id=mention_id,
        article_id=article_id,
        text=text,
        entity_type="ORG",
        sentence=sentence,
        start_char=0,
        end_char=len(text),
        language="en",
    )


def test_entity_resolver_clusters_similar_mentions():
    mentions = [
        build_mention("m1", "White House", "Meeting at the White House"),
        build_mention("m2", "Maison Blanche", "Rencontre à la Maison Blanche"),
    ]
    embeddings = np.array([
        [0.9, 0.1, 0.0],
        [0.88, 0.12, 0.0],
    ], dtype=np.float32)

    resolver = EntityResolver(EntityResolutionConfig(similarity_threshold=0.8))
    entities = resolver.resolve(mentions, embeddings)
    assert len(entities) == 1
    assert entities[0].name in {"White House", "Maison Blanche"}
    assert len(entities[0].mentions) == 2


def test_entity_resolver_separates_dissimilar_mentions():
    mentions = [
        build_mention("m1", "White House", "Meeting at the White House"),
        build_mention("m2", "Kremlin", "Talks at the Kremlin"),
    ]
    embeddings = np.array([
        [0.9, 0.1, 0.0],
        [-0.9, 0.1, 0.0],
    ], dtype=np.float32)

    resolver = EntityResolver(EntityResolutionConfig(similarity_threshold=0.8))
    entities = resolver.resolve(mentions, embeddings)
    assert len(entities) == 2
