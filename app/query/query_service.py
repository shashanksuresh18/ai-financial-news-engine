from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schema import StoryWithImpact


@dataclass
class QueryResult:
    story: StoryWithImpact
    score: float


class QueryService:
    """
    Lightweight semantic-ish search over stories using TF-IDF.

    We index story title + summary + metadata (sectors, regulators, tickers,
    impacted stock symbols) and use cosine similarity for ranking.
    """

    def __init__(self, stories: Sequence[StoryWithImpact]):
        self.stories: List[StoryWithImpact] = list(stories)

        if not self.stories:
            self._vectorizer = None
            self._story_matrix = None
            return

        texts = [self._build_story_text(s) for s in self.stories]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._story_matrix = self._vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------ #

    def _build_story_text(self, s: StoryWithImpact) -> str:
        parts: List[str] = []

        if getattr(s, "title", None):
            parts.append(s.title)

        if getattr(s, "summary", None):
            parts.append(s.summary)

        sectors = getattr(s, "sectors", None)
        if sectors:
            parts.append(" ".join(sectors))

        regulators = getattr(s, "regulators", None)
        if regulators:
            parts.append(" ".join(regulators))

        tickers = getattr(s, "tickers", None)
        if tickers:
            parts.append(" ".join(tickers))

        impacted = getattr(s, "impacted_stocks", None)
        if impacted:
            syms = [
                st.symbol
                for st in impacted
                if getattr(st, "symbol", None) is not None
            ]
            if syms:
                parts.append(" ".join(syms))

        return " ".join(parts)

    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[QueryResult]:
        if (
            not self.stories
            or self._vectorizer is None
            or self._story_matrix is None
        ):
            return []

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(self._story_matrix, q_vec).ravel()
        order = scores.argsort()[::-1]

        results: List[QueryResult] = []
        for idx in order:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(QueryResult(story=self.stories[idx], score=score))
            if len(results) >= top_k:
                break

        return results
