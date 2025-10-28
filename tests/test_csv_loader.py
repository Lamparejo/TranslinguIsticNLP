from __future__ import annotations

from datetime import datetime

from src.data.csv_loader import CSVArticleLoader, CSVArticleSourceConfig


def test_csv_loader_extracts_articles(tmp_path):
    csv_path = tmp_path / "articles.tsv"
    csv_lines = [
        "20250101010101-A\t20250101010101\t1\tsource-a\thttp://example.com/a",
        "20250102020202-B\t20250102020202\t1\tsource-b\thttp://example.com/b",
    ]
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    html_map = {
        "http://example.com/a": "<html><body><p>First article.</p><p>Extra sentence.</p></body></html>",
        "http://example.com/b": "<html><body><p>Second article only.</p></body></html>",
    }

    def fake_fetch(url: str) -> str | None:
        return html_map.get(url)

    loader = CSVArticleLoader(
        CSVArticleSourceConfig(path=csv_path, limit=5),
        fetcher=fake_fetch,
    )

    articles = list(loader.iter_articles())

    assert len(articles) == 2
    assert articles[0].article_id == "20250101010101-A"
    assert articles[0].source_url == "http://example.com/a"
    assert "First article." in articles[0].text
    assert "Extra sentence." in articles[0].text
    assert isinstance(articles[0].published_at, datetime)
    assert articles[0].published_at.year == 2025
    assert articles[1].text.strip() == "Second article only."


def test_csv_loader_skips_failures_until_limit(tmp_path):
    csv_path = tmp_path / "articles.tsv"
    csv_lines = [
        "20250101010101-A\t20250101010101\t1\tsource-a\thttp://example.com/a",
        "20250102020202-B\t20250102020202\t1\tsource-b\thttp://example.com/b",
        "20250103030303-C\t20250103030303\t1\tsource-c\thttp://example.com/c",
    ]
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    html_map = {
        "http://example.com/c": "<html><body><p>Third article.</p></body></html>",
    }

    def flaky_fetch(url: str) -> str | None:
        return html_map.get(url)

    loader = CSVArticleLoader(
        CSVArticleSourceConfig(path=csv_path, limit=1),
        fetcher=flaky_fetch,
    )

    articles = list(loader.iter_articles())

    assert len(articles) == 1
    assert articles[0].article_id == "20250103030303-C"
    assert articles[0].text.strip() == "Third article."