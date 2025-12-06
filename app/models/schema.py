from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
import uuid


def _uuid() -> str:
    return str(uuid.uuid4())


class NewsArticle(BaseModel):
    """
    Raw news article as loaded from JSONL or RSS.
    """
    id: str = Field(default_factory=_uuid)
    source: Optional[str] = None
    title: str
    body: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None

    # Optional metadata from mock file – we accept but don't rely on them.
    tickers: List[str] = []
    sectors: List[str] = []
    regulators: List[str] = []


class Story(BaseModel):
    """
    A deduplicated news 'story' that can group multiple related articles.
    """
    id: str = Field(default_factory=_uuid)
    title: str
    article_ids: List[str]

    # Needed by DeduplicationService (used to cache story embedding)
    embedding: Optional[List[float]] = None


class ImpactedStock(BaseModel):
    symbol: str
    confidence: float
    impact_type: str


class StoryWithImpact(Story):
    """
    Story enriched with NLP metadata and embeddings.
    Inherits `embedding` from Story.
    """
    summary: Optional[str] = None
    sectors: List[str] = []
    regulators: List[str] = []
    tickers: List[str] = []
    impacted_stocks: List[ImpactedStock] = []

    # NEW: list of underlying article sources (e.g. MockWire, EconomicTimes)
    sources: List[str] = []

    # NEW: simple sentiment annotation over the story text
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
