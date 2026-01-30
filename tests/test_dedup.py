import pytest
from app.dedup.dedup_service import DeduplicationService
from app.models.schema import NewsArticle

def test_initial_story_creation(sample_article):
    service = DeduplicationService(similarity_threshold=0.6)
    story = service.process_article(sample_article)
    assert story.title == sample_article.title
    assert len(service.stories) == 1

def test_article_merging(sample_article):
    service = DeduplicationService(similarity_threshold=0.4)
    service.process_article(sample_article)
    
    similar_article = NewsArticle(
        title="HDFC Bank shares surge on strong dividend news",
        body="HDFC Bank announced a higher-than-expected dividend this morning. The banking sector responded positively to the news.",
        source="EconomicTimes"
    )
    
    story = service.process_article(similar_article)
    assert len(service.stories) == 1
    assert len(story.article_ids) == 2
    assert "MockWire" in story.sources
    assert "EconomicTimes" in story.sources

def test_dissimilar_articles(sample_article, tech_article):
    service = DeduplicationService(similarity_threshold=0.6)
    service.process_article(sample_article)
    service.process_article(tech_article)
    
    assert len(service.stories) == 2
