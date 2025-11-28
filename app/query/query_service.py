from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.schema import StoryWithImpact
from app.ner.ner_service import EntityExtractionService


class QueryService:
    """
    Context-aware query service over StoryWithImpact objects.
    Combines:
      - Entity understanding (companies, tickers, sectors, regulators)
      - Stock impact metadata (impacted_stocks)
      - Semantic similarity (embeddings) for theme queries.
    """

    def __init__(
        self,
        ner: EntityExtractionService,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.ner = ner
        self.model = SentenceTransformer(model_name)
        self.stories: List[StoryWithImpact] = []

    def index(self, stories: List[StoryWithImpact]):
        """Store stories for querying."""
        self.stories = stories

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb[0]

    def _cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))

    # ------------------------------------------------------------------
    # Main search logic
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> List[Tuple[StoryWithImpact, float]]:
        """
        Returns list of (story, score) pairs sorted by score descending.

        Logic:
          1) Use NER to understand query entities:
             - companies / tickers
             - sectors
             - regulators
          2) Score stories using:
             - ticker overlap (direct > sector/regulatory)
             - sector match
             - regulator match
          3) Add semantic similarity bonus or use as fallback
             for vague queries like 'interest rate impact'.
          4) Filter out results below min_score.
        """
        if not self.stories:
            return []

        ents = self.ner.extract_from_text(query)
        companies = ents.get("companies", [])
        sectors = ents.get("sectors", [])
        regulators = ents.get("regulators", [])
        tickers_from_query = set(ents.get("tickers", []))

        company_to_ticker = self.ner.companies_to_tickers(companies)
        query_tickers = set(company_to_ticker.values()) | tickers_from_query

        # 1) Entity-based scoring
        scores = {id(s): 0.0 for s in self.stories}

        for story in self.stories:
            sid = id(story)
            score = 0.0

            # Company/ticker match
            if query_tickers:
                story_tickers = {is_.symbol for is_ in story.impacted_stocks}
                overlap = story_tickers & query_tickers
                if overlap:
                    # Prefer direct impact over sector/regulatory-only
                    for is_ in story.impacted_stocks:
                        if is_.symbol in overlap:
                            if "direct" in is_.impact_type:
                                score = max(score, 1.0)
                            else:
                                score = max(score, 0.8)

            # Sector match
            if sectors:
                if any(sec in story.sectors for sec in sectors):
                    score = max(score, 0.7)

            # Regulator match
            if regulators:
                if any(reg in story.regulators for reg in regulators):
                    score = max(score, 0.9)

            scores[sid] = score

        # 2) Semantic similarity bonus / fallback
        query_emb = self._embed(query)
        for story in self.stories:
            if story.embedding is None:
                continue
            sid = id(story)
            story_emb = np.array(story.embedding, dtype=np.float32)
            sem_sim = self._cosine(query_emb, story_emb)

            # If no entities detected at all, semantic drives ranking
            if (
                len(companies) == 0
                and len(sectors) == 0
                and len(regulators) == 0
                and len(query_tickers) == 0
            ):
                if sem_sim >= 0.3:
                    scores[sid] = max(scores[sid], sem_sim)
            else:
                # Otherwise, semantic acts as a small booster
                scores[sid] = scores[sid] + 0.2 * sem_sim

        # 3) Filter by min_score and sort
        scored = [
            (s, scores[id(s)])
            for s in self.stories
            if scores[id(s)] >= min_score
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]
