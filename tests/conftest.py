import pytest
from app.models.schema import NewsArticle

@pytest.fixture
def sample_article():
    return NewsArticle(
        title="HDFC Bank shares surge on strong dividend news",
        body="HDFC Bank announced a higher-than-expected dividend this morning. The banking sector responded positively to the news.",
        source="MockWire",
        url="http://example.com/article1"
    )

@pytest.fixture
def tech_article():
    return NewsArticle(
        title="Infosys announces new AI division",
        body="Infosys, a leading IT services company, is expanding its technology offerings. The software industry is rapidly evolving.",
        source="TechNews",
        url="http://example.com/article2"
    )
