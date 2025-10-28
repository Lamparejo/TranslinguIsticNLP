"""Command-line helper to extract relation triples from article URLs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nlp import RebelExtractorConfig, RebelRelationExtractor
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract knowledge graph triples from news URLs")
    parser.add_argument("urls", nargs="+", help="One or more article URLs to process")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional device index for transformers pipeline (defaults to CPU)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file to store extracted triples",
    )
    parser.add_argument(
        "--min-sentence-length",
        type=int,
        default=10,
        help="Minimum sentence length (characters) to consider for extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = RebelRelationExtractor(
        RebelExtractorConfig(device=args.device, min_sentence_length=args.min_sentence_length)
    )
    results = {}
    for url in args.urls:
        try:
            triples = extractor.run(url)
            results[url] = [triple.__dict__ for triple in triples]
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("Failed to process %s: %s", url, exc)
            results[url] = []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        LOGGER.info("Saved triples to %s", args.output)
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
