from typing import Dict, List
from collections import defaultdict

from app.models.schema import (
    Story,
    StoryWithImpact,
    ImpactedStock,
    NewsArticle,
)
from app.ner.ner_service import EntityExtractionService


class ImpactMappingService:
    """
    Takes a deduplicated Story + its underlying articles, runs NER aggregation,
    and computes impacted stocks with confidence scores.
    """

    def __init__(self):
        self.ner = EntityExtractionService()

        # For demo: which tickers belong to which sector
        # In a real system, you'd build this from NSE/BSE metadata.
        self.sector_to_tickers: Dict[str, List[str]] = {
            "Banking": ["HDFCBANK", "ICICIBANK"],
            "IT": ["INFY"],
        }

    def map_story(
        self,
        story: Story,
        articles_by_id: Dict[str, NewsArticle],
    ) -> StoryWithImpact:
        """
        Aggregate entities across all articles in the story,
        then compute impacted stocks with confidence.
        """
        all_companies = set()
        all_sectors = set()
        all_regulators = set()

        # 1) Aggregate entities over all articles in this story
        for aid in story.article_ids:
            article = articles_by_id[aid]
            ents = self.ner.extract_from_article(article)
            all_companies.update(ents["companies"])
            all_sectors.update(ents["sectors"])
            all_regulators.update(ents["regulators"])

        # 2) Map companies -> tickers (direct impact, confidence 1.0)
        company_to_ticker = self.ner.companies_to_tickers(list(all_companies))

        # ticker -> {confidence: float, types: set[str]}
        impact_scores = defaultdict(lambda: {"confidence": 0.0, "types": set()})

        # Direct company mentions
        for company, ticker in company_to_ticker.items():
            key = ticker
            impact_scores[key]["confidence"] = max(
                impact_scores[key]["confidence"], 1.0
            )
            impact_scores[key]["types"].add("direct")

        # 3) Sector-wide impact → confidence ~0.7 (within required 0.6–0.8)
        for sector in all_sectors:
            tickers = self.sector_to_tickers.get(sector, [])
            for ticker in tickers:
                sector_conf = 0.7
                impact_scores[ticker]["confidence"] = max(
                    impact_scores[ticker]["confidence"], sector_conf
                )
                impact_scores[ticker]["types"].add("sector")

        # 4) Regulatory impact (e.g. RBI affects all Banking stocks) → confidence ~0.6
        if "RBI" in all_regulators:
            for ticker in self.sector_to_tickers.get("Banking", []):
                reg_conf = 0.6
                impact_scores[ticker]["confidence"] = max(
                    impact_scores[ticker]["confidence"], reg_conf
                )
                impact_scores[ticker]["types"].add("regulatory")

        # 5) Build ImpactedStock objects
        impacted: List[ImpactedStock] = []
        for ticker, data in impact_scores.items():
            impact_type = ",".join(sorted(list(data["types"])))
            impacted.append(
                ImpactedStock(
                    symbol=ticker,
                    confidence=data["confidence"],
                    impact_type=impact_type,
                )
            )

        # Derive tickers list from impacted stocks
        all_tickers = sorted({is_.symbol for is_ in impacted})

        # 6) Start from the original story data, then override fields
        story_data = story.model_dump()
        story_data["sectors"] = sorted(list(all_sectors))
        story_data["regulators"] = sorted(list(all_regulators))
        story_data["tickers"] = all_tickers

        # 7) Build enriched StoryWithImpact
        return StoryWithImpact(
            **story_data,
            impacted_stocks=impacted,
        )
