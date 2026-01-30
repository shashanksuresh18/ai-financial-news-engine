import pytest
from app.ner.ner_service import EntityExtractionService

def test_extract_companies(sample_article):
    service = EntityExtractionService()
    results = service.extract_from_article(sample_article)
    assert "HDFC Bank" in results["companies"]

def test_extract_sectors(sample_article):
    service = EntityExtractionService()
    results = service.extract_from_article(sample_article)
    assert "Banking" in results["sectors"]

def test_extract_it_sector(tech_article):
    service = EntityExtractionService()
    results = service.extract_from_article(tech_article)
    assert "IT" in results["sectors"]
    assert "Infosys" in results["companies"]

def test_extract_tickers_from_text():
    service = EntityExtractionService()
    text = "Check the performance of INFY and HDFCBANK today."
    results = service.extract_from_text(text)
    assert "INFY" in results["tickers"]
    assert "HDFCBANK" in results["tickers"]
