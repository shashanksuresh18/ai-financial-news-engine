import json
from datetime import datetime
from typing import List

from app.models.schema import NewsArticle


def load_mock_articles(path: str) -> List[NewsArticle]:
    articles: List[NewsArticle] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            article = NewsArticle(
                source=raw["source"],
                published_at=datetime.fromisoformat(raw["published_at"]),
                title=raw["title"],
                body=raw["body"],
                url=raw.get("url"),
                tickers=raw.get("tickers", []),
                sectors=raw.get("sectors", []),
                regulators=raw.get("regulators", []),
            )
            articles.append(article)
    return articles
