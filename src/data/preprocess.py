"""Text preprocessing utilities."""
from __future__ import annotations

import re
from typing import Iterable, List



def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters."""
    return " ".join(text.split())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a naive regex heuristic."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [normalize_whitespace(sentence) for sentence in sentences if sentence.strip()]


def filter_articles_by_language(texts: Iterable[str], language_codes: Iterable[str]) -> List[str]:
    """Placeholder for language-aware filtering; returns the input as-is."""
    # In a production setting this would leverage language detection
    # to keep only the desired set of languages. We keep the signature so unit
    # tests can exercise the pipeline even without the heavy dependency.
    return list(texts)
