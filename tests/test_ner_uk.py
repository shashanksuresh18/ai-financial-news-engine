import pytest
from app.ner.ner_service import EntityExtractionService
from app.models.schema import NewsArticle

def test_extract_uk_companies():
    service = EntityExtractionService()
    text = "HSBC and Barclays are leading the gains in the FTSE 100 today."
    results = service.extract_from_text(text)
    assert "HSBC" in results["companies"]
    assert "Barclays" in results["companies"]

def test_extract_uk_regulators():
    service = EntityExtractionService()
    text = "The FCA and the Bank of England are monitoring the situation."
    results = service.extract_from_text(text)
    assert "FCA" in results["regulators"]
    assert "BoE" in results["regulators"]

def test_extract_uk_tickers():
    service = EntityExtractionService()
    text = "Check the performance of HSBA and BARC."
    results = service.extract_from_text(text)
    assert "HSBA" in results["tickers"]
    assert "BARC" in results["tickers"]
