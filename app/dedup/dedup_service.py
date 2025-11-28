from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import re

from app.models.schema import NewsArticle, Story


class DeduplicationService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_similarity_threshold: float = 0.78,
        fallback_similarity_threshold: float = 0.70,
        lexical_overlap_threshold: float = 0.30,
    ):
        self.model = SentenceTransformer(model_name)
        # main threshold for “strong semantic duplicate”
        self.base_similarity_threshold = base_similarity_threshold
        # lower semantic threshold when lexical overlap is high
        self.fallback_similarity_threshold = fallback_similarity_threshold
        self.lexical_overlap_threshold = lexical_overlap_threshold

        # In-memory store of stories {story_id: Story}
        self.stories: dict[str, Story] = {}

    # ---------- Text prep ----------

    def _article_to_text(self, article: NewsArticle) -> str:
        # Use title + small snippet of body to capture event context
        snippet = article.body[:200] if article.body else ""
        return f"{article.title}. {snippet}"

    def _tokenize(self, text: str) -> set[str]:
        # basic tokenization: split on non-word chars & lowercase
        return set(re.findall(r"\w+", text.lower()))

    def _lexical_overlap(self, t1: str, t2: str) -> float:
        s1 = self._tokenize(t1)
        s2 = self._tokenize(t2)
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union > 0 else 0.0

    # ---------- Embeddings ----------

    def _embed(self, text: str) -> List[float]:
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb[0].tolist()

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        # embeddings are normalized, so dot product == cosine similarity
        return float(np.dot(v1, v2))

    # ---------- Core matching logic ----------

    def _find_similar_story(
        self,
        article: NewsArticle,
        emb: List[float],
    ) -> Optional[str]:
        """
        Returns story_id of the best matching story if it qualifies as a duplicate,
        else None.
        Uses both semantic similarity and lexical title overlap.
        """
        if not self.stories:
            return None

        emb_arr = np.array(emb, dtype=np.float32)

        best_id = None
        best_sem_sim = -1.0
        best_lex_overlap = 0.0

        for story_id, story in self.stories.items():
            if story.embedding is None:
                continue

            story_emb = np.array(story.embedding, dtype=np.float32)
            sem_sim = self._cosine_similarity(emb_arr, story_emb)
            lex_overlap = self._lexical_overlap(article.title, story.title)

            # Track the best by semantic similarity (primary)
            if sem_sim > best_sem_sim:
                best_sem_sim = sem_sim
                best_id = story_id
                best_lex_overlap = lex_overlap

        if best_id is None:
            return None

        # Decision rule: strong semantic match OR weaker semantic + strong lexical overlap
        if best_sem_sim >= self.base_similarity_threshold:
            return best_id
        if (
            best_sem_sim >= self.fallback_similarity_threshold
            and best_lex_overlap >= self.lexical_overlap_threshold
        ):
            return best_id

        return None

    # ---------- Public API ----------

    def process_article(self, article: NewsArticle) -> Story:
        """
        Main entry: either attach to an existing story (duplicate)
        or create a new story.
        """
        text = self._article_to_text(article)
        emb = self._embed(text)

        similar_story_id = self._find_similar_story(article, emb)

        if similar_story_id:
            story = self.stories[similar_story_id]
            if article.id not in story.article_ids:
                story.article_ids.append(article.id)
            # (Optional) update summary/embedding here if you want
            self.stories[story.id] = story
            return story

        # Create new story
        new_story = Story(
            title=article.title,
            summary=article.body[:300],
            article_ids=[article.id],
            embedding=emb,
        )
        self.stories[new_story.id] = new_story
        return new_story

    def get_all_stories(self) -> List[Story]:
        return list(self.stories.values())
