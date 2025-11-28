from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


def gen_id() -> str:
    return str(uuid.uuid4())


class NewsArticle(BaseModel):
    """
    Raw news article as ingested from any source (mock/live).
    """
    id: str = Field(default_factory=gen_id)
    source: str
    published_at: datetime
    title: str
    body: str
    url: Optional[str] = None

    tickers: List[str] = []
    sectors: List[str] = []
    regulators: List[str] = []


class Story(BaseModel):
    """
    Unique news event after deduplication.
    """
    id: str = Field(default_factory=gen_id)
    title: str
    summary: Optional[str] = None
    article_ids: List[str] = []

    # Embedding for semantic similarity
    embedding: Optional[List[float]] = None

    # These will be filled by NER + mapping later
    sectors: List[str] = []
    tickers: List[str] = []
    regulators: List[str] = []


class ImpactedStock(BaseModel):
    """
    Represents an individual stock impacted by a story,
    with a confidence score and impact type (direct/sector/regulatory).
    """
    symbol: str
    confidence: float
    impact_type: str  # e.g. "direct", "sector", "regulatory" or combinations


class StoryWithImpact(Story):
    """
    Story enriched with impact mapping information.
    """
    impacted_stocks: List[ImpactedStock] = []
