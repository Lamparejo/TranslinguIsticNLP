"""Utilities to load GDELT Global Knowledge Graph data."""
from __future__ import annotations

import csv
import io
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

import requests

from .models import NewsArticle
from ..utils.io import read_jsonl
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class GDELTConfig:
    """Configuration describing where to load GDELT data from."""

    gkg_url: Optional[str] = None
    local_path: Optional[str | pathlib.Path] = None

    def require_source(self) -> None:
        if not (self.gkg_url or self.local_path):
            raise ValueError("Either gkg_url or local_path must be provided")


class GDELTLoader:
    """Load and parse records from the GDELT Global Knowledge Graph (GKG)."""

    def __init__(self, config: GDELTConfig):
        self.config = config
        self.session = requests.Session()

    def iter_articles(self) -> Iterator[NewsArticle]:
        """Iterate over news articles based on the configured source."""
        self.config.require_source()
        if self.config.local_path:
            yield from self._from_jsonl(self.config.local_path)
        elif self.config.gkg_url:
            yield from self._from_remote(self.config.gkg_url)

    def _from_jsonl(self, path: str | pathlib.Path) -> Iterator[NewsArticle]:
        for record in read_jsonl(path):
            yield self._parse_article_from_json(record)

    def _from_remote(self, url: str) -> Iterator[NewsArticle]:
        LOGGER.info("Downloading GDELT data from %s", url)
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/plain" in content_type or urlparse(url).path.endswith(".CSV"):
            yield from self._parse_csv(response.text)
        else:
            raise ValueError(f"Unsupported content type for GDELT source: {content_type}")

    def _parse_csv(self, csv_text: str) -> Iterator[NewsArticle]:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            try:
                yield self._parse_article_from_csv(row)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to parse CSV row due to %s", exc, exc_info=False)

    @staticmethod
    def _parse_article_from_json(record: dict) -> NewsArticle:
        return NewsArticle(
            article_id=str(record["article_id"]),
            language=record.get("language", "unknown"),
            text=record.get("text", ""),
            published_at=datetime.fromisoformat(record["published_at"]),
            source_url=record.get("source_url", ""),
            themes=record.get("themes", []),
        )

    @staticmethod
    def _parse_article_from_csv(row: dict) -> NewsArticle:
        # GDELT columns: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
        article_id = row.get("GLOBALEVENTID", row.get("DocumentIdentifier", ""))
        published_at_raw = row.get("SQLDATE") or row.get("DATE")
        if published_at_raw:
            published_at = datetime.strptime(published_at_raw, "%Y%m%d")
        else:
            published_at = datetime.utcnow()
        text = row.get("SOURCECOMMONNAME", "")
        source_url = row.get("DocumentIdentifier", "")
        themes = row.get("Themes", "").split(";") if row.get("Themes") else []
        language = row.get("LANGUAGE", "unknown")
        return NewsArticle(
            article_id=str(article_id),
            language=language,
            text=text,
            published_at=published_at,
            source_url=source_url,
            themes=themes,
        )


def load_articles_from_source(config: GDELTConfig) -> Iterable[NewsArticle]:
    """Convenience wrapper returning a list of articles."""
    return list(GDELTLoader(config).iter_articles())
