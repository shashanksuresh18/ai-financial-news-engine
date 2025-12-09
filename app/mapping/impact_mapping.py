from __future__ import annotations

from typing import Dict, List, Union

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
      - impact_type: 'direct', 'sector', 'regulatory' or comma-joined combo
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
        """
        Build a lowercased text representation for rule checks:
        story title + summary + first article title/body.
        """
        parts: List[str] = []

        if getattr(story, "title", None):
            parts.append(story.title)

        if getattr(story, "summary", None):
            parts.append(story.summary)

        # Use article_ids (new) or articles (old) for compatibility
        article_ids = (
            getattr(story, "article_ids", None)
            or getattr(story, "articles", None)
            or []
        )

        for art_id in article_ids:
            art = articles_by_id.get(art_id)
            if not art:
                continue
            if getattr(art, "title", None):
                parts.append(art.title)
            body = getattr(art, "body", None) or getattr(art, "content", None)
            if body:
                parts.append(body)
            break  # one article is enough for rules

        return " ".join(parts).lower()

    def _add_impact(
        self,
        impacts: Dict[str, ImpactedStock],
        symbol: str,
        impact_type: Union[str, List[str]],
        confidence: float,
    ) -> None:
        """
        Merge impact info:

        - Accepts impact_type as str or List[str] (to be robust)
        - Stores it as a *string* on ImpactedStock (comma-joined if multiple)
        """
        # Normalise to string
        if isinstance(impact_type, list):
            impact_type_str = ",".join(impact_type)
        else:
            impact_type_str = impact_type

        if symbol not in impacts:
            impacts[symbol] = ImpactedStock(
                symbol=symbol,
                confidence=confidence,
                impact_type=impact_type_str,  # always a string here
            )
        else:
            st = impacts[symbol]
            # existing is also a string, possibly "direct,sector"
            current_types = [
                t.strip() for t in st.impact_type.split(",") if t.strip()
            ]
            new_types = [
                t.strip()
                for t in impact_type_str.split(",")
                if t.strip()
            ]
            for t in new_types:
                if t not in current_types:
                    current_types.append(t)
            st.impact_type = ",".join(current_types)
            # keep highest confidence
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
        if (
            "RBI" in regulators
            or "rbi" in regulators
            or "reserve bank of india" in text
        ):
            # RBI decisions mostly affect banks, but also IT via macro
            for sym in self.bank_stocks:
                self._add_impact(impacts, sym, "regulatory", 0.7)
            for sym in self.it_stocks:
                self._add_impact(impacts, sym, "sector", 0.7)

        impacted_list = list(impacts.values())

        # ---- Build StoryWithImpact from Story data ---------------------
        # StoryWithImpact likely extends Story, so reuse Story's fields
        story_data = story.model_dump()
        story_data["impacted_stocks"] = impacted_list

        return StoryWithImpact(**story_data)
