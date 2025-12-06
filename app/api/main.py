from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.ingestion.loader import load_mock_articles
from app.ingestion.live_rss import fetch_rss_articles
from app.dedup.dedup_service import DeduplicationService
from app.mapping.impact_mapping import ImpactMappingService
from app.ner.ner_service import EntityExtractionService
from app.query.query_service import QueryService
from app.models.schema import NewsArticle, StoryWithImpact


# -----------------------------------------------------------------------------
# Global in-memory state (simple for hackathon/demo)
# -----------------------------------------------------------------------------

articles_by_id: Dict[str, NewsArticle] = {}
enriched_stories: List[StoryWithImpact] = []
query_service: Optional[QueryService] = None

DATASET_PATH = "data/mock_news.jsonl"


# -----------------------------------------------------------------------------
# Pydantic response models for API
# -----------------------------------------------------------------------------

class ImpactedStockResponse(BaseModel):
    symbol: str
    confidence: float
    impact_type: str


class StoryResponse(BaseModel):
    id: str
    title: str
    summary: Optional[str]
    sectors: List[str]
    regulators: List[str]
    tickers: List[str]
    impacted_stocks: List[ImpactedStockResponse]

    # NEW: show where this story came from + sentiment
    sources: List[str] = []
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None


class QueryResultResponse(BaseModel):
    score: float
    story: StoryResponse


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResultResponse]


class StockStoryResponse(BaseModel):
    """
    Used for /stock/{symbol} endpoint: story + max confidence
    for that specific symbol in the story.
    """
    max_confidence: float
    story: StoryResponse


class AlertResponse(BaseModel):
    """
    Simple alert wrapper: why we think this story is important.
    """
    level: str   # e.g. "high", "medium"
    reason: str  # e.g. "max confidence ≥ 0.9, regulatory impact"
    story: StoryResponse


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="AI-Powered Financial News Intelligence API",
    description=(
        "Multi-agent LangGraph system for financial news "
        "deduplication, entity extraction, impact mapping, "
        "sentiment and context-aware querying."
    ),
    version="1.1.0",
)


# -----------------------------------------------------------------------------
# Simple Frontend (HTML + JS)
# -----------------------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AI Financial News Intelligence</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #e5e7eb;
    }
    header {
      padding: 16px 24px;
      background: #020617;
      border-bottom: 1px solid #1f2937;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    header h1 {
      margin: 0;
      font-size: 1.1rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #38bdf8;
    }
    header span {
      font-size: 0.8rem;
      color: #9ca3af;
    }
    main {
      max-width: 960px;
      margin: 24px auto;
      padding: 0 16px 32px 16px;
    }
    .search-box {
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }
    .search-box input {
      flex: 1;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: #020617;
      color: #e5e7eb;
      outline: none;
    }
    .search-box input:focus {
      border-color: #38bdf8;
    }
    .search-box button {
      padding: 10px 18px;
      border-radius: 999px;
      border: none;
      background: #38bdf8;
      color: #020617;
      font-weight: 600;
      cursor: pointer;
    }
    .search-box button:hover {
      background: #0ea5e9;
    }
    .hint {
      font-size: 0.8rem;
      color: #9ca3af;
      margin-bottom: 16px;
    }
    .results {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .story-card {
      background: #020617;
      border-radius: 12px;
      padding: 12px 14px;
      border: 1px solid #1f2937;
    }
    .story-title {
      margin: 0 0 4px 0;
      font-size: 0.98rem;
      color: #e5e7eb;
    }
    .story-meta {
      font-size: 0.75rem;
      color: #9ca3af;
      margin-bottom: 6px;
    }
    .score-pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #111827;
      font-size: 0.7rem;
      margin-right: 6px;
    }
    .impact-chip {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: #1d4ed8;
      font-size: 0.7rem;
      margin: 2px 4px 2px 0;
    }
    .summary {
      font-size: 0.82rem;
      color: #d1d5db;
      margin-top: 6px;
    }
    .badge {
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid #4b5563;
      margin-right: 6px;
    }
    .top-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      flex-wrap: wrap;
    }
    .section-title {
      font-size: 0.9rem;
      margin: 18px 0 8px 0;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }
    .live-btn {
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid #4b5563;
      background: transparent;
      color: #9ca3af;
      font-size: 0.75rem;
      cursor: pointer;
    }
    .live-btn:hover {
      border-color: #38bdf8;
      color: #e5e7eb;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AI Financial News Intelligence</h1>
      <span>LangGraph · Dedup · Entity Mapping · Sentiment · Query</span>
    </div>
    <button class="live-btn" onclick="ingestLive()">+ Ingest Live RSS</button>
  </header>
  <main>
    <div class="search-box">
      <input id="query-input" placeholder="Ask about stocks, sectors, regulators... e.g. 'HDFC Bank news', 'interest rate impact', 'INFY'" />
      <button onclick="runQuery()">Search</button>
    </div>
    <div class="hint">
      Try: <code>HDFC Bank news</code>, <code>Banking sector update</code>, <code>RBI policy changes</code>, <code>interest rate impact</code>, <code>INFY</code>
    </div>
    <div id="status" class="hint"></div>

    <div class="section-title">Results</div>
    <div id="results" class="results"></div>
  </main>

  <script>
    async function runQuery() {
      const input = document.getElementById('query-input');
      const q = input.value.trim();
      if (!q) return;

      const status = document.getElementById('status');
      const resultsDiv = document.getElementById('results');
      status.textContent = 'Searching...';
      resultsDiv.innerHTML = '';

      try {
        const res = await fetch('/query?q=' + encodeURIComponent(q));
        if (!res.ok) {
          throw new Error('HTTP ' + res.status);
        }
        const data = await res.json();

        status.textContent = '';
        if (!data.results || data.results.length === 0) {
          resultsDiv.innerHTML = '<p>No results.</p>';
          return;
        }

        for (const item of data.results) {
          const score = item.score;
          const story = item.story;

          const card = document.createElement('div');
          card.className = 'story-card';

          const impacted = (story.impacted_stocks || [])
            .map(s => s.symbol + ' (' + s.confidence.toFixed(2) + ')')
            .join(', ');

          const sectors = (story.sectors || []).join(', ');
          const regulators = (story.regulators || []).join(', ');
          const sources = (story.sources || []).join(', ');

          const sentiment = story.sentiment ? story.sentiment.toUpperCase() : '';
          const sentimentScore = (story.sentiment_score !== null && story.sentiment_score !== undefined)
            ? story.sentiment_score.toFixed(2)
            : '';

          let impactChips = '';
          for (const s of (story.impacted_stocks || [])) {
            impactChips += '<span class="impact-chip">' +
              s.symbol + ' · ' +
              s.impact_type +
              ' · ' + s.confidence.toFixed(2) +
              '</span>';
          }

          let sentimentBadge = '';
          if (sentiment) {
            sentimentBadge = '<span class="badge">Sentiment: ' + sentiment +
              (sentimentScore ? ' (' + sentimentScore + ')' : '') +
              '</span>';
          }

          card.innerHTML = `
            <div class="top-row">
              <h2 class="story-title">${story.title}</h2>
              <span class="score-pill">Score: ${score.toFixed(3)}</span>
            </div>
            <div class="story-meta">
              ${sectors ? '<span class="badge">Sectors: ' + sectors + '</span>' : ''}
              ${regulators ? '<span class="badge">Regulators: ' + regulators + '</span>' : ''}
              ${impacted ? '<span class="badge">Impacted: ' + impacted + '</span>' : ''}
              ${sources ? '<span class="badge">Source: ' + sources + '</span>' : ''}
              ${sentimentBadge}
            </div>
            <div>${impactChips}</div>
            <div class="summary">${story.summary || ''}</div>
          `;

          resultsDiv.appendChild(card);
        }
      } catch (err) {
        console.error(err);
        status.textContent = 'Error fetching results. See console for details.';
      }
    }

    async function ingestLive() {
      const status = document.getElementById('status');
      status.textContent = 'Ingesting live RSS feeds...';

      try {
        const res = await fetch('/ingest_live', { method: 'POST' });
        if (!res.ok) {
          throw new Error('HTTP ' + res.status);
        }
        const data = await res.json();
        status.textContent = 'Live ingestion complete. Articles: ' +
          data.total_articles + ', Stories: ' + data.stories_indexed + '. You can search again now.';
      } catch (err) {
        console.error(err);
        status.textContent = 'Error ingesting live RSS. See console for details.';
      }
    }

    document.getElementById('query-input').addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') {
        runQuery();
      }
    });
  </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Helper to (re)build index
# -----------------------------------------------------------------------------

def build_index_from_articles(articles: List[NewsArticle]) -> None:
    """
    Given a list of articles, rebuild:
      - articles_by_id
      - enriched_stories
      - query_service index
    """
    global articles_by_id, enriched_stories, query_service

    articles_by_id = {a.id: a for a in articles}

    # Deduplicate into stories
    dedup = DeduplicationService()
    for a in articles:
        dedup.process_article(a)
    stories = dedup.get_all_stories()

    # Impact mapping -> StoryWithImpact
    mapper = ImpactMappingService()
    enriched_stories = [mapper.map_story(s, articles_by_id) for s in stories]

    # Query service
    ner = EntityExtractionService()
    query_service = QueryService(ner=ner)
    query_service.index(enriched_stories)


def build_index(dataset_path: str = DATASET_PATH) -> None:
    """
    Default: load mock data from file and build index.
    """
    articles = load_mock_articles(dataset_path)
    build_index_from_articles(articles)


# -----------------------------------------------------------------------------
# Startup event
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    build_index()


# -----------------------------------------------------------------------------
# Utility converters
# -----------------------------------------------------------------------------

def to_story_response(story: StoryWithImpact) -> StoryResponse:
    return StoryResponse(
        id=story.id,
        title=story.title,
        summary=story.summary,
        sectors=story.sectors or [],
        regulators=story.regulators or [],
        tickers=story.tickers or [],
        impacted_stocks=[
            ImpactedStockResponse(
                symbol=is_.symbol,
                confidence=is_.confidence,
                impact_type=is_.impact_type,
            )
            for is_ in story.impacted_stocks
        ],
        sources=story.sources or [],
        sentiment=story.sentiment,
        sentiment_score=story.sentiment_score,
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def home():
    """
    Simple web UI for interactive querying.
    """
    return INDEX_HTML


@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "ok",
        "message": "AI Financial News Intelligence API is running",
        "articles_loaded": len(articles_by_id),
        "stories_indexed": len(enriched_stories),
    }


@app.get("/stories", response_model=List[StoryResponse], tags=["stories"])
async def list_stories():
    """
    Return all deduplicated stories with their impacted stocks.
    """
    return [to_story_response(s) for s in enriched_stories]


@app.get("/query", response_model=QueryResponse, tags=["query"])
async def query_news(
    q: str = Query(..., description="Natural language query, e.g. 'HDFC Bank news'"),
    top_k: int = Query(5, ge=1, le=20),
    min_score: float = Query(0.05, ge=0.0, le=1.0),
):
    """
    Run context-aware query over enriched stories.
    Uses:
      - Company/ticker impact
      - Sector and regulator relationships
      - Semantic similarity for thematic queries
    """
    global query_service

    if query_service is None or not enriched_stories:
        # Safety net: rebuild index if something went wrong
        build_index()

    results = query_service.search(q, top_k=top_k, min_score=min_score)

    return QueryResponse(
        query=q,
        results=[
            QueryResultResponse(
                score=score,
                story=to_story_response(story),
            )
            for (story, score) in results
        ],
    )


@app.post("/ingest_live", tags=["ingestion"])
async def ingest_live():
    """
    Fetch live RSS news, merge with current articles, and rebuild the index.
    Returns summary stats.
    """
    # existing articles from mock + any previous ingestion
    base_articles = list(articles_by_id.values())

    live_articles = fetch_rss_articles(max_per_feed=10)

    all_articles = base_articles + live_articles
    build_index_from_articles(all_articles)

    return {
        "status": "ok",
        "added_from_rss": len(live_articles),
        "total_articles": len(all_articles),
        "stories_indexed": len(enriched_stories),
    }


@app.get("/stock/{symbol}", response_model=List[StockStoryResponse], tags=["stocks"])
async def stock_impacts(
    symbol: str,
    min_confidence: float = Query(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum impact confidence to include a story",
    ),
):
    """
    Return all stories impacting a given stock symbol, sorted by max confidence.
    Example:
      - /stock/INFY
      - /stock/HDFCBANK
    """
    sym = symbol.upper()
    matches: List[Tuple[StoryWithImpact, float]] = []

    for story in enriched_stories:
        confs = [
            is_.confidence
            for is_ in story.impacted_stocks
            if is_.symbol.upper() == sym
        ]
        if confs:
            max_conf = max(confs)
            if max_conf >= min_confidence:
                matches.append((story, max_conf))

    matches.sort(key=lambda x: x[1], reverse=True)

    return [
        StockStoryResponse(
            max_confidence=max_conf,
            story=to_story_response(story),
        )
        for (story, max_conf) in matches
    ]


@app.get("/alerts", response_model=List[AlertResponse], tags=["alerts"])
async def get_alerts(
    min_confidence: float = Query(
        0.9,
        ge=0.0,
        le=1.0,
        description="Minimum max confidence for a story to be considered high impact",
    ),
):
    """
    Simple alert endpoint.

    Returns stories that:
      - have any impacted stock with confidence >= min_confidence, OR
      - have regulatory impacts (e.g. RBI) on Banking / Financial sectors.
    """
    alerts: List[AlertResponse] = []

    for story in enriched_stories:
        # max confidence across impacted stocks
        max_conf = 0.0
        has_regulatory = False
        for is_ in story.impacted_stocks:
            max_conf = max(max_conf, is_.confidence)
            if "regulatory" in (is_.impact_type or ""):
                has_regulatory = True

        if max_conf >= min_confidence or has_regulatory:
            level = "high" if max_conf >= min_confidence else "medium"
            reasons = []
            if max_conf >= min_confidence:
                reasons.append(f"max confidence ≥ {min_confidence}")
            if has_regulatory:
                reasons.append("regulatory impact present")
            reason = ", ".join(reasons) if reasons else "high impact"

            alerts.append(
                AlertResponse(
                    level=level,
                    reason=reason,
                    story=to_story_response(story),
                )
            )

    return alerts
