from datetime import datetime

from src.data.models import EntityMention, NewsArticle, ResolvedEntity
from src.nlp.rebel_pipeline import RebelExtractorConfig, RelationTriple
from src.nlp.relation_extraction import (
    RebelGraphRelationExtractor,
    RelationExtractionConfig,
    RelationExtractor,
)


def mention(mention_id: str, sentence: str, article_id: str = "a1", text: str = "Sample") -> EntityMention:
    return EntityMention(
        mention_id=mention_id,
        article_id=article_id,
        text=text,
        entity_type="PER",
        sentence=sentence,
        start_char=0,
        end_char=6,
        language="en",
    )


def test_relation_extractor_pairs_mentions_in_same_sentence():
    entity_a = ResolvedEntity(
        canonical_id="ca",
        name="Leader A",
        entity_type="PER",
        mentions=[mention("m1", "Leaders met in Berlin.")],
        embedding=[],
    )
    entity_b = ResolvedEntity(
        canonical_id="cb",
        name="Leader B",
        entity_type="PER",
        mentions=[mention("m2", "Leaders met in Berlin.")],
        embedding=[],
    )

    extractor = RelationExtractor(RelationExtractionConfig(relation_type="MEETS"))
    relations = extractor.extract([entity_a, entity_b])

    assert len(relations) == 1
    relation = relations[0]
    assert relation.source_id == "ca"
    assert relation.target_id == "cb"
    assert relation.relation_type == "MEETS"
    assert relation.sentence == "Leaders met in Berlin."


def test_relation_extractor_ignores_mentions_in_different_sentences():
    entity_a = ResolvedEntity(
        canonical_id="ca",
        name="Leader A",
        entity_type="PER",
        mentions=[mention("m1", "First sentence."), mention("m2", "Second sentence.")],
        embedding=[],
    )
    entity_b = ResolvedEntity(
        canonical_id="cb",
        name="Leader B",
        entity_type="PER",
        mentions=[mention("m3", "Third sentence.")],
        embedding=[],
    )

    extractor = RelationExtractor(RelationExtractionConfig())
    relations = extractor.extract([entity_a, entity_b])
    assert len(relations) == 0


class DummyRebelExtractor:
    def __init__(self, mapping):
        self.mapping = mapping

    def extract_from_sentence(self, sentence: str):
        return self.mapping.get(sentence, [])


def article(article_id: str, text: str) -> NewsArticle:
    return NewsArticle(
        article_id=article_id,
        language="en",
        text=text,
        published_at=datetime(2024, 1, 1),
        source_url="https://example.com",
    )


def test_rebel_graph_relation_extractor_maps_triples_to_resolved_entities():
    sentence = "Alice met Bob in Berlin."
    resolved_a = ResolvedEntity(
        canonical_id="ca",
        name="Alice",
        entity_type="PER",
        mentions=[mention("m1", sentence, text="Alice")],
        embedding=[],
    )
    resolved_b = ResolvedEntity(
        canonical_id="cb",
        name="Bob",
        entity_type="PER",
        mentions=[mention("m2", sentence, text="Bob")],
        embedding=[],
    )

    dummy = DummyRebelExtractor(
        {
            sentence: [
                RelationTriple(
                    subject="Alice",
                    relation="met_with",
                    object="Bob",
                    sentence=sentence,
                    score=0.9,
                )
            ]
        }
    )

    extractor = RebelGraphRelationExtractor(
        RebelExtractorConfig(min_sentence_length=0),
        extractor=dummy,  # type: ignore[arg-type]
    )

    relations = extractor.extract([article("a1", sentence)], [resolved_a, resolved_b])
    assert len(relations) == 1
    relation = relations[0]
    assert relation.source_id == "ca"
    assert relation.target_id == "cb"
    assert relation.relation_type == "met_with"
    assert relation.article_id == "a1"


def test_rebel_graph_relation_extractor_skips_unmatched_entities():
    sentence = "Alice praised Charlie in Berlin."
    resolved_a = ResolvedEntity(
        canonical_id="ca",
        name="Alice",
        entity_type="PER",
        mentions=[mention("m1", sentence, text="Alice")],
        embedding=[],
    )

    dummy = DummyRebelExtractor(
        {
            sentence: [
                RelationTriple(
                    subject="Alice",
                    relation="praised",
                    object="Charlie",
                    sentence=sentence,
                    score=0.8,
                )
            ]
        }
    )

    extractor = RebelGraphRelationExtractor(
        RebelExtractorConfig(min_sentence_length=0),
        extractor=dummy,  # type: ignore[arg-type]
    )

    relations = extractor.extract([article("a1", sentence)], [resolved_a])
    assert relations == []


def test_rebel_graph_relation_extractor_handles_accents_and_punctuation():
    sentence = "José María visited Bob at the summit."
    resolved_a = ResolvedEntity(
        canonical_id="ca",
        name="José María",
        entity_type="PER",
        mentions=[mention("m1", sentence, text="José María")],
        embedding=[],
    )
    resolved_b = ResolvedEntity(
        canonical_id="cb",
        name="Bob",
        entity_type="PER",
        mentions=[mention("m2", sentence, text="Bob")],
        embedding=[],
    )

    dummy = DummyRebelExtractor(
        {
            sentence: [
                RelationTriple(
                    subject="Jose Maria",
                    relation="met_with",
                    object="Bob.",
                    sentence=sentence,
                    score=0.7,
                )
            ]
        }
    )

    extractor = RebelGraphRelationExtractor(
        RebelExtractorConfig(min_sentence_length=0),
        extractor=dummy,  # type: ignore[arg-type]
    )

    relations = extractor.extract([article("a1", sentence)], [resolved_a, resolved_b])
    assert len(relations) == 1
    relation = relations[0]
    assert relation.source_id == "ca"
    assert relation.target_id == "cb"
