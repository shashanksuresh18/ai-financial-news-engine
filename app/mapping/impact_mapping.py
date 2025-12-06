from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.models.schema import NewsArticle, Story, StoryWithImpact, ImpactedStock
from app.ner.ner_service import EntityExtractionService


class ImpactMappingService:
    """
    Enriches deduplicated stories with:
      - sectors, regulators, tickers
      - impacted_stocks (per-stock confidence & type)
      - sentence-transformer embeddings
      - simple sentiment over concatenated article text
      - list of underlying sources (mock vs RSS feeds, etc.)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.ner = EntityExtractionService()
        self.model = SentenceTransformer(model_name)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Map sectors and regulators to tickers (demo-scale, can be extended)
        self.sector_to_tickers: Dict[str, List[str]] = {
            "Banking": ["HDFCBANK", "ICICIBANK"],
            "Financial Services": ["HDFCBANK", "ICICIBANK"],
            "IT": ["INFY"],
            "Economy": ["HDFCBANK", "ICICIBANK", "INFY"],
        }

        self.regulator_to_tickers: Dict[str, List[str]] = {
            "RBI": ["HDFCBANK", "ICICIBANK"],
        }

    # -------------------------- internal helpers -------------------------- #

    def _embed(self, text: str) -> List[float]:
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb[0].tolist()

    def _compute_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Compute a very simple sentiment label + score using VADER.
        """
        if not text:
            return "neutral", 0.0
        scores = self.sentiment_analyzer.polarity_scores(text)
        comp = scores.get("compound", 0.0)
        if comp >= 0.05:
            label = "positive"
        elif comp <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return label, float(comp)

    def _add_impact(
        self,
        impacts: Dict[str, ImpactedStock],
        symbol: str,
        confidence: float,
        impact_type: str,
    ) -> None:
        """
        Merge multiple impact signals for the same symbol:
        - keep max confidence
        - accumulate distinct impact_type tags (comma-separated)
        """
        if symbol in impacts:
            existing = impacts[symbol]
            existing.confidence = max(existing.confidence, confidence)

            existing_types = set(
                t.strip() for t in existing.impact_type.split(",") if t.strip()
            )
            for t in impact_type.split(","):
                t = t.strip()
                if t:
                    existing_types.add(t)
            existing.impact_type = ",".join(sorted(existing_types))
        else:
            impacts[symbol] = ImpactedStock(
                symbol=symbol,
                confidence=confidence,
                impact_type=impact_type,
            )

    # --------------------------- public methods --------------------------- #

    def map_story(
        self,
        story: Story,
        articles_by_id: Mapping[str, NewsArticle],
    ) -> StoryWithImpact:
        """
        Given a Story and a dict of NewsArticle by id, build StoryWithImpact.
        """
        article_texts: List[str] = []
        sources: List[str] = []

        for aid in story.article_ids:
            article = articles_by_id.get(aid)
            if not article:
                continue
            article_texts.append(f"{article.title}. {article.body}")
            if article.source:
                sources.append(article.source)

        full_text = " ".join(article_texts) if article_texts else story.title
        summary = article_texts[0][:480] if article_texts else None

        ents = self.ner.extract_from_text(full_text)
        companies = ents.get("companies", [])
        sectors = ents.get("sectors", [])
        regulators = ents.get("regulators", [])
        tickers_from_text = ents.get("tickers", [])

        # Map companies -> tickers, then union with any direct tickers in the text
        company_to_ticker = self.ner.companies_to_tickers(companies)
        tickers_set = set(company_to_ticker.values()) | set(tickers_from_text)

        impacts: Dict[str, ImpactedStock] = {}

        # Direct company impacts
        for _, ticker in company_to_ticker.items():
            self._add_impact(impacts, ticker, 1.0, "direct")

        # Sector-wide impacts
        for sector in sectors:
            for ticker in self.sector_to_tickers.get(sector, []):
                self._add_impact(impacts, ticker, 0.7, "sector")

        # Regulatory impacts
        for reg in regulators:
            for ticker in self.regulator_to_tickers.get(reg, []):
                # treat as regulatory + sector impact
                self._add_impact(impacts, ticker, 0.7, "regulatory,sector")

        # Sentiment + embedding
        sentiment_label, sentiment_score = self._compute_sentiment(full_text)
        embedding = self._embed(story.title)

        return StoryWithImpact(
            id=story.id,
            title=story.title,
            article_ids=story.article_ids,
            summary=summary,
            sectors=sorted(list(set(sectors))),
            regulators=sorted(list(set(regulators))),
            tickers=sorted(list(tickers_set)),
            impacted_stocks=list(impacts.values()),
            embedding=embedding,
            sources=sorted(list(set(sources))),
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
        )

    def map_stories(
        self,
        stories: List[Story],
        articles_by_id: Mapping[str, NewsArticle],
    ) -> List[StoryWithImpact]:
        """
        Convenience method to enrich a list of stories.
        """
        return [self.map_story(s, articles_by_id) for s in stories]
