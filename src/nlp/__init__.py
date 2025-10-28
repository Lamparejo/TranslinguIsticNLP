"""Natural language processing components."""

from .rebel_pipeline import (
	RebelExtractorConfig,
	RebelRelationExtractor,
	RelationTriple,
	fetch_article_text,
	parse_rebel_output,
	split_sentences,
)

__all__ = [
	"RebelExtractorConfig",
	"RebelRelationExtractor",
	"RelationTriple",
	"fetch_article_text",
	"parse_rebel_output",
	"split_sentences",
]
