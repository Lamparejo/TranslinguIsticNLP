"""Utilities to load article content from a CSV of URLs."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator, Optional

import polars as pl  # type: ignore[import-not-found]
import requests
from bs4 import BeautifulSoup  # type: ignore[import-not-found]

from .models import NewsArticle
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CSVArticleSourceConfig:
    """Configuration describing how to read and interpret the CSV source."""

    path: str | pathlib.Path
    limit: int = 15
    url_column_index: int = 4
    article_id_column_index: int = 0
    published_at_column_index: int = 1
    encoding: str = "utf-8"
    separator: str = "\t"
    timeout: int = 20


class CSVArticleLoader:
    """Load article bodies by scraping URLs listed inside a CSV/TSV file."""

    def __init__(
        self,
        config: CSVArticleSourceConfig,
        fetcher: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.config = config
        self._session = requests.Session()
        self._fetcher = fetcher or self._default_fetch

    def iter_articles(self) -> Iterator[NewsArticle]:
        table = self._read_table()
        if table is None or table.height == 0:
            LOGGER.warning("CSV source %s produced no rows", self.config.path)
            return

        limit = min(self.config.limit, table.height) if self.config.limit else table.height
        article_ids = table[:, self.config.article_id_column_index].to_list()
        published_raw = table[:, self.config.published_at_column_index].to_list()
        urls = table[:, self.config.url_column_index].to_list()

        produced = 0
        for idx in range(table.height):
            if limit and produced >= limit:
                break
            url = self._coerce_str(urls, idx)
            if not url:
                continue
            html = self._fetcher(url)
            if not html:
                continue
            text = self._extract_text(html)
            if not text:
                continue

            article_id = self._coerce_str(article_ids, idx) or f"csv-{idx}"
            published_at_value = published_raw[idx] if idx < len(published_raw) else None
            published_at = self._parse_datetime(published_at_value)

            yield NewsArticle(
                article_id=str(article_id),
                language="unknown",
                text=text,
                published_at=published_at,
                source_url=url,
                themes=[],
            )
            produced += 1

    def _read_table(self) -> Optional[pl.DataFrame]:
        path = pathlib.Path(self.config.path)
        if not path.exists():
            LOGGER.warning("CSV source path %s does not exist", path)
            return None

        try:
            return pl.read_csv(
                path,
                has_header=False,
                separator=self.config.separator,
                encoding=self.config.encoding,
                infer_schema_length=0,
                truncate_ragged_lines=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to read CSV %s: %s", path, exc)
            return None

    def _default_fetch(self, url: str) -> Optional[str]:  # pragma: no cover - network path
        try:
            response = self._session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
        except Exception as exc:
            LOGGER.warning("Failed to fetch article %s: %s", url, exc)
            return None
        return response.text

    @staticmethod
    def _extract_text(html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = [paragraph.get_text(strip=True) for paragraph in soup.find_all("p")]
        text = "\n\n".join(part for part in paragraphs if part)
        return text.strip() or None

    @staticmethod
    def _parse_datetime(raw_value) -> datetime:
        if raw_value is None:
            return datetime.utcnow()
        value = str(raw_value)
        if not value:
            return datetime.utcnow()

        try:
            if len(value) >= 14:
                return datetime.strptime(value[:14], "%Y%m%d%H%M%S")
            if len(value) >= 8:
                return datetime.strptime(value[:8], "%Y%m%d")
        except ValueError:
            LOGGER.debug("Failed to parse datetime from %s", value)
        return datetime.utcnow()

    @staticmethod
    def _coerce_str(seq, idx: int) -> str:
        if idx >= len(seq):
            return ""
        value = seq[idx]
        return "" if value is None else str(value).strip()
