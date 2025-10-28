"""Relation extraction pipeline using the Babelscape REBEL model."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional

import requests

try:
    import spacy  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime safeguard
    spacy = None  # type: ignore[assignment]
from bs4 import BeautifulSoup
from requests import HTTPError
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ArticleContent:
    """Represents the raw textual content fetched from an article URL."""

    url: str
    text: str
    sentences: List[str]


@dataclass(slots=True)
class RelationTriple:
    """Structured representation of a relationship extracted by REBEL."""

    subject: str
    relation: str
    object: str
    sentence: str
    score: float


@dataclass(slots=True)
class RebelExtractorConfig:
    """Configuration options for the REBEL extraction pipeline."""

    model_name: str = "Babelscape/rebel-large"
    device: Optional[int] = None
    max_length: int = 256
    num_beams: int = 3
    article_timeout: int = 30
    min_sentence_length: int = 10


class RebelRelationExtractor:
    """End-to-end pipeline from article URL to extracted relation triples."""

    def __init__(self, config: RebelExtractorConfig | None = None) -> None:
        self.config = config or RebelExtractorConfig()

    def run(self, url: str) -> List[RelationTriple]:
        """Fetch the article content and extract triples for every sentence."""

        content = self.fetch_and_segment(url)
        triples: List[RelationTriple] = []
        for sentence in content.sentences:
            if len(sentence.strip()) < self.config.min_sentence_length:
                continue
            triples.extend(self.extract_from_sentence(sentence))
        LOGGER.info("Extracted %s triples from %s", len(triples), url)
        return triples

    def fetch_and_segment(self, url: str) -> ArticleContent:
        """Download article text from *url* and split it into clean sentences."""

        raw_text = fetch_article_text(url, timeout=self.config.article_timeout)
        sentences = list(split_sentences(raw_text))
        return ArticleContent(url=url, text=raw_text, sentences=sentences)

    def extract_from_sentence(self, sentence: str) -> List[RelationTriple]:
        """Extract triples from a single sentence using the REBEL pipeline."""
        model, tokenizer, device = _get_rebel_model(
            self.config.model_name,
            self.config.device,
        )
        encoded = tokenizer(sentence, return_tensors="pt", truncation=True)
        encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        triples: List[RelationTriple] = []
        for text in decoded:
            for triple in parse_rebel_output(text):
                triples.append(
                    RelationTriple(
                        subject=triple[0],
                        relation=triple[1],
                        object=triple[2],
                        sentence=sentence,
                        score=1.0,
                    )
                )
        return triples


def fetch_article_text(url: str, timeout: int = 30) -> str:
    """Retrieve article text from *url* using a shallow HTML extraction approach."""

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except HTTPError as exc:  # pragma: no cover - depends on network
        raise RuntimeError(f"Failed to fetch article from {url}: {exc}") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    paragraphs = [
        " ".join(paragraph.get_text(separator=" ", strip=True).split())
        for paragraph in soup.find_all("p")
    ]
    text = "\n".join([p for p in paragraphs if p])
    if not text:
        LOGGER.warning("No paragraph content found for %s", url)
    return text


@lru_cache(maxsize=2)
def _get_sentencizer(model: str | None = None):
    if spacy is None:  # pragma: no cover - runtime safeguard
        raise ImportError("spaCy is not installed. Install it with 'pip install spacy'.")
    if model:
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("xx")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str, model: str | None = None) -> Iterable[str]:
    """Yield sentences from *text* using spaCy's sentence boundary detection."""

    nlp = _get_sentencizer(model)
    doc = nlp(text)
    for sent in doc.sents:
        sentence = sent.text.strip()
        if sentence:
            yield sentence


@lru_cache(maxsize=2)
def _get_rebel_model(model_name: str, device_index: Optional[int]):
    LOGGER.info("Loading REBEL model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device_index is not None and device_index >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def parse_rebel_output(generated_text: str) -> List[tuple[str, str, str]]:
    """Convert the raw REBEL generation output into (subject, relation, object) triples."""

    if not generated_text:
        return []

    text = generated_text.strip()
    if text.startswith("<s>"):
        text = text[3:]
    text = text.replace("</s>", " ")

    triples: List[tuple[str, str, str]] = []
    segments = [segment.strip() for segment in text.split("<triplet>") if segment.strip()]

    for segment in segments:
        lower_segment = segment.lower()
        has_subj_close = "</subj>" in lower_segment
        has_rel_close = "</rel>" in lower_segment

        subject_tokens: List[str] = []
        relation_tokens: List[str] = []
        object_tokens: List[str] = []
        pointer: Optional[str] = "subject" if has_subj_close else None
        seen_subj_tag = False
        seen_rel_tag = False

        for raw_token in segment.split():
            lower = raw_token.lower()
            if lower in {"<s>", "</s>", "<pad>"}:
                continue
            if lower == "<subj>":
                seen_subj_tag = True
                pointer = "subject" if has_subj_close else "relation"
                continue
            if lower == "</subj>":
                pointer = None
                continue
            if lower == "<rel>":
                seen_rel_tag = True
                pointer = "relation" if has_rel_close else "object"
                continue
            if lower == "</rel>":
                pointer = None
                continue
            if lower == "<obj>":
                pointer = "object"
                continue
            if lower == "</obj>":
                pointer = None
                continue

            cleaned = raw_token.replace("▁", " ").strip()
            if not cleaned:
                continue

            if pointer == "subject":
                subject_tokens.append(cleaned)
                continue
            if pointer == "relation":
                relation_tokens.append(cleaned)
                continue
            if pointer == "object":
                object_tokens.append(cleaned)
                continue

            if not seen_subj_tag:
                subject_tokens.append(cleaned)
            elif not seen_rel_tag:
                relation_tokens.append(cleaned)
            else:
                object_tokens.append(cleaned)

        subj_clean = " ".join(" ".join(subject_tokens).split())
        rel_clean = " ".join(" ".join(relation_tokens).split())
        obj_clean = " ".join(" ".join(object_tokens).split())

        if subj_clean and rel_clean and obj_clean:
            triples.append((subj_clean, rel_clean, obj_clean))

    return triples


def run_batch(urls: Iterable[str], config: RebelExtractorConfig | None = None) -> dict[str, List[RelationTriple]]:
    """Process a batch of article URLs and return extracted triples keyed by URL."""

    extractor = RebelRelationExtractor(config)
    results: dict[str, List[RelationTriple]] = {}
    for url in urls:
        try:
            results[url] = extractor.run(url)
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("Failed to process %s: %s", url, exc)
            results[url] = []
    return results
