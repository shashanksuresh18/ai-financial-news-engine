from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schema import NewsArticle, Story


class DeduplicationService:
    """
    Lightweight deduplication using TF-IDF + cosine similarity.

    - Each Story represents a cluster of similar articles.
    - For every new article, we compare it against existing stories.
    - If similarity >= threshold -> merge into that story.
    - Otherwise -> create a new story.

    This is intentionally simple and memory-friendly for small/medium datasets.
    """

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        # 0.6 works reasonably for short financial news
        self.similarity_threshold = similarity_threshold
        # story_id -> Story
        self.stories: Dict[str, Story] = {}

    # ------------------------------------------------------------------ #
    # Helpers to build text for similarity
    # ------------------------------------------------------------------ #

    def _article_text(self, article: NewsArticle) -> str:
        """Concatenate useful fields from an article into one text string."""
        parts: List[str] = []

        title = getattr(article, "title", None)
        if title:
            parts.append(title)

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

        source = getattr(article, "source", None)
        if source:
            parts.append(source)

        return " ".join(parts)

    def _story_text(self, story: Story) -> str:
        """Concatenate useful fields from a story into one text string."""
        parts: List[str] = []

        title = getattr(story, "title", None)
        if title:
            parts.append(title)

        summary = getattr(story, "summary", None)
        if summary:
            parts.append(summary)

        sectors = getattr(story, "sectors", None)
        if sectors:
            parts.append(" ".join(sectors))

        regulators = getattr(story, "regulators", None)
        if regulators:
            parts.append(" ".join(regulators))

        tickers = getattr(story, "tickers", None)
        if tickers:
            parts.append(" ".join(tickers))

        sources = getattr(story, "sources", None)
        if sources:
            parts.append(" ".join(sources))

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Core similarity logic
    # ------------------------------------------------------------------ #

    def _find_similar_story(self, article: NewsArticle) -> Optional[str]:
        """
        Return ID of the most similar story, or None if no match above threshold.
        """
        if not self.stories:
            return None

        # Build corpus: all current stories + the new article
        story_list: List[Story] = list(self.stories.values())
        corpus: List[str] = [self._story_text(s) for s in story_list]
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
        Deduplicate a single article and return the Story it belongs to.
        Creates a new Story or merges into an existing one.
        """
        similar_story_id = self._find_similar_story(article)

        if similar_story_id is None:
            # -------- Create a fresh story -------------------------------
            story_id = str(uuid4())

            # Simple summary fallback if Story has a summary field
            summary: Optional[str] = getattr(article, "summary", None)
            if not summary:
                body = getattr(article, "body", None) or getattr(
                    article, "content", None
                ) or ""
                summary = body[:280]

            sectors = getattr(article, "sectors", None) or []
            regulators = getattr(article, "regulators", None) or []
            tickers = getattr(article, "tickers", None) or []

            source = getattr(article, "source", None)
            sources: List[str] = [source] if source else []

            # Note: Story requires article_ids; other fields may be optional.
            story = Story(
                id=story_id,
                title=article.title,
                article_ids=[article.id],
                sectors=sectors,
                regulators=regulators,
                tickers=tickers,
                sources=sources,
                summary=summary,  # ignored if Story doesn't define it
            )
            self.stories[story_id] = story
            return story

        # -------- Merge into an existing story --------------------------
        story = self.stories[similar_story_id]

        # Add article id if not present
        if article.id not in getattr(story, "article_ids", []):
            story.article_ids.append(article.id)

        # Merge list-type attributes without duplicates
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
