from __future__ import annotations

from typing import Dict, List

from app.models.schema import (
    NewsArticle,
    Story,
    StoryWithImpact,
    ImpactedStock,
)


class ImpactMappingService:
    """
    Lightweight, rule-based impact mapping.

    For each story we look at:
      - Title + summary text
      - Underlying article text
      - Sectors, regulators, tickers

    Then we infer which stocks are impacted and how:
      - impact_type: ['direct', 'sector', 'regulatory']
      - confidence: 1.0 for direct mentions, 0.7 for sector/regulatory links
    """

    def __init__(self) -> None:
        # Simple knowledge of which stocks belong to which sector
        self.bank_stocks = ["HDFCBANK", "ICICIBANK"]
        self.it_stocks = ["INFY"]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _story_text(self, story: Story, articles_by_id: Dict[str, NewsArticle]) -> str:
        parts: List[str] = []

        if getattr(story, "title", None):
            parts.append(story.title)

        if getattr(story, "summary", None):
            parts.append(story.summary)

        # Include the first article's body/content if available
        for art_id in getattr(story, "articles", []):
            art = articles_by_id.get(art_id)
            if not art:
                continue
            if getattr(art, "title", None):
                parts.append(art.title)
            body = getattr(art, "body", None) or getattr(art, "content", None)
            if body:
                parts.append(body)
            break  # just one article is enough for rules

        return " ".join(parts).lower()

    def _add_impact(
        self,
        impacts: Dict[str, ImpactedStock],
        symbol: str,
        impact_type: str,
        confidence: float,
    ) -> None:
        if symbol not in impacts:
            impacts[symbol] = ImpactedStock(
                symbol=symbol,
                confidence=confidence,
                impact_type=[impact_type],
            )
        else:
            st = impacts[symbol]
            # add impact type if new
            if impact_type not in st.impact_type:
                st.impact_type.append(impact_type)
            # keep the highest confidence
            if confidence > st.confidence:
                st.confidence = confidence

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def map_story(
        self,
        story: Story,
        articles_by_id: Dict[str, NewsArticle],
    ) -> StoryWithImpact:
        """
        Enrich a Story with impacted_stocks using simple rules.

        Returns a StoryWithImpact instance.
        """
        text = self._story_text(story, articles_by_id)
        sectors = set(getattr(story, "sectors", []) or [])
        regulators = set(getattr(story, "regulators", []) or [])
        tickers = set(getattr(story, "tickers", []) or [])

        impacts: Dict[str, ImpactedStock] = {}

        # ---- Direct mentions in text or tickers ------------------------
        if "hdfc bank" in text or "hdfcbank" in text or "hdfc bank ltd" in text:
            self._add_impact(impacts, "HDFCBANK", "direct", 1.0)

        if "icici bank" in text or "icicibank" in text or "icici bank ltd" in text:
            self._add_impact(impacts, "ICICIBANK", "direct", 1.0)

        if "infosys" in text or "infy" in text or "infosys ltd" in text:
            self._add_impact(impacts, "INFY", "direct", 1.0)

        if "HDFCBANK" in tickers:
            self._add_impact(impacts, "HDFCBANK", "direct", 1.0)
        if "ICICIBANK" in tickers:
            self._add_impact(impacts, "ICICIBANK", "direct", 1.0)
        if "INFY" in tickers:
            self._add_impact(impacts, "INFY", "direct", 1.0)

        # ---- Sector-based impacts --------------------------------------
        if "Banking" in sectors or "banking" in sectors:
            for sym in self.bank_stocks:
                self._add_impact(impacts, sym, "sector", 0.7)

        if "IT" in sectors or "Technology" in sectors or "technology" in sectors:
            for sym in self.it_stocks:
                self._add_impact(impacts, sym, "sector", 0.7)

        # ---- Regulatory / macro impacts (RBI, etc.) --------------------
        if "RBI" in regulators or "rbi" in regulators or "reserve bank of india" in text:
            # RBI decisions mostly affect banks, but also IT via macro
            for sym in self.bank_stocks:
                self._add_impact(impacts, sym, "regulatory", 0.7)
            # IT sector indirectly via demand / currency / rates
            for sym in self.it_stocks:
                self._add_impact(impacts, sym, "sector", 0.7)

        # You can add more rule blocks here for other sectors, regulators, etc.

        impacted_list = list(impacts.values())

        # ---- Build StoryWithImpact without duplicating fields ----------
        return StoryWithImpact(
            id=story.id,
            title=story.title,
            summary=story.summary,
            articles=story.articles,
            sectors=story.sectors,
            regulators=story.regulators,
            tickers=story.tickers,
            sources=getattr(story, "sources", []),
            impacted_stocks=impacted_list,
        )
