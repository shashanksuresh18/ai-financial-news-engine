from typing import List
from datetime import datetime
import time
from urllib.parse import urlparse

import feedparser

from app.models.schema import NewsArticle


# Example RSS feeds – replace or extend with feeds you like.
# Some options you can consider:
# - Economic Times markets RSS
# - Moneycontrol buzzing stocks RSS
# - Business Standard markets RSS
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/buzzingstocks.xml",
]


def _safe_published_at(entry) -> datetime:
    """
    Try to parse the published date from the RSS entry.
    Fallback to current UTC time if not available.
    """
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            ts = time.mktime(entry.published_parsed)
            return datetime.fromtimestamp(ts)
        except Exception:
            pass
    return datetime.utcnow()


def _source_name(feed_url: str, parsed_feed_title: str | None) -> str:
    if parsed_feed_title:
        return parsed_feed_title
    try:
        netloc = urlparse(feed_url).netloc
        return netloc or "UnknownSource"
    except Exception:
        return "UnknownSource"


def fetch_rss_articles(max_per_feed: int = 10) -> List[NewsArticle]:
    """
    Fetch latest news from configured RSS feeds and convert them to NewsArticle objects.
    NOTE: tickers/sectors/regulators are left empty – your NER + impact mapping
    pipeline will infer them.
    """
    articles: List[NewsArticle] = []

    for feed_url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            # Skip this feed on error
            continue

        source = _source_name(feed_url, getattr(parsed.feed, "title", None))

        for entry in parsed.entries[:max_per_feed]:
            title = getattr(entry, "title", "").strip()
            if not title:
                continue

            body = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            url = getattr(entry, "link", None)

            published_at = _safe_published_at(entry)

            article = NewsArticle(
                source=source,
                published_at=published_at,
                title=title,
                body=body,
                url=url,
                tickers=[],
                sectors=[],
                regulators=[],
            )
            articles.append(article)

    return articles
