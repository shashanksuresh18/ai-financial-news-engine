from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schema import NewsArticle, Story


class DeduplicationService:
    """
    Lightweight deduplication using TF-IDF + cosine similarity.

    - Each *story* represents a cluster of similar articles.
    - For every new article, we compare it to existing stories.
    - If similarity >= threshold → merge into that story.
    - Otherwise → create a new story.

    We re-fit a small TF-IDF model on all current stories each time a new
    story is created. Dataset sizes in this project are small, so this is
    fast and memory-friendly.
    """

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        # 0.6 works well for short headlines + snippets
        self.similarity_threshold = similarity_threshold
        self.stories: Dict[str, Story] = {}

    # ------------------------------------------------------------------ #
    # Text helpers
    # ------------------------------------------------------------------ #

    def _article_text(self, article: NewsArticle) -> str:
        parts: List[str] = []

        if getattr(article, "title", None):
            parts.append(article.title)

        summary = getattr(article, "summary", None)
        if summary:
            parts.append(summary)

        body = getattr(article, "body", None) or getattr(article, "content", None)
        if body:
            parts.append(body)

        sectors = getattr(article, "sectors", None)
        if sectors:
            parts.append(" ".join(sectors))

        regulators = getattr(article, "regulators", None)
        if regulators:
            parts.append(" ".join(regulators))

        tickers = getattr(article, "tickers", None)
        if tickers:
            parts.append(" ".join(tickers))

        return " ".join(parts)

    def _story_text(self, story: Story) -> str:
        parts: List[str] = []

        if getattr(story, "title", None):
            parts.append(story.title)

        if getattr(story, "summary", None):
            parts.append(story.summary)

        sectors = getattr(story, "sectors", None)
        if sectors:
            parts.append(" ".join(sectors))

        regulators = getattr(story, "regulators", None)
        if regulators:
            parts.append(" ".join(regulators))

        tickers = getattr(story, "tickers", None)
        if tickers:
            parts.append(" ".join(tickers))

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Core similarity logic
    # ------------------------------------------------------------------ #

    def _find_similar_story(self, article: NewsArticle) -> Optional[str]:
        """Return the ID of the most similar story, or None if no good match."""
        if not self.stories:
            return None

        # Build a tiny corpus: all story texts + the new article text
        corpus: List[str] = [self._story_text(s) for s in self.stories.values()]
        corpus.append(self._article_text(article))

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)

        story_matrix = matrix[:-1]  # existing stories
        article_vec = matrix[-1]    # new article

        sims = cosine_similarity(story_matrix, article_vec).ravel()
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score < self.similarity_threshold:
            return None

        # Map index back to story ID
        story_ids = list(self.stories.keys())
        return story_ids[best_idx]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_article(self, article: NewsArticle) -> Story:
        """
        Deduplicate a single article.

        Returns the Story the article belongs to (existing or newly created).
        """
        similar_story_id = self._find_similar_story(article)

        if similar_story_id is None:
            # ---- Create a new story -----------------------------
            story_id = str(uuid4())

            summary = getattr(article, "summary", None)
            if not summary:
                body = getattr(article, "body", None) or getattr(
                    article, "content", None
                ) or ""
                summary = body[:280]

            sectors = getattr(article, "sectors", None) or []
            regulators = getattr(article, "regulators", None) or []
            tickers = getattr(article, "tickers", None) or []

            sources: List[str] = []
            src = getattr(article, "source", None)
            if src:
                sources.append(src)

            story = Story(
                id=story_id,
                title=article.title,
                summary=summary,
                articles=[article.id],
                sectors=sectors,
                regulators=regulators,
                tickers=tickers,
                impacted_stocks=[],   # filled later by impact mapping
                sources=sources,      # custom field we added earlier
            )
            self.stories[story_id] = story
            return story

        # ---- Merge into an existing story ----------------------
        story = self.stories[similar_story_id]

        if article.id not in story.articles:
            story.articles.append(article.id)

        def _merge_list_attr(attr: str, new_vals: Optional[List[str]]) -> None:
            if not new_vals:
                return
            existing = getattr(story, attr, None)
            if existing is None:
                setattr(story, attr, list(dict.fromkeys(new_vals)))
                return
            for v in new_vals:
                if v not in existing:
                    existing.append(v)

        _merge_list_attr("sectors", getattr(article, "sectors", None))
        _merge_list_attr("regulators", getattr(article, "regulators", None))
        _merge_list_attr("tickers", getattr(article, "tickers", None))

        src = getattr(article, "source", None)
        if src:
            _merge_list_attr("sources", [src])

        return story
