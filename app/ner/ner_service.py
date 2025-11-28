import spacy
from typing import List, Dict, Set
from app.models.schema import NewsArticle


class EntityExtractionService:
    def __init__(self, model_name: str = "en_core_web_sm"):
        # Load spaCy model
        self.nlp = spacy.load(model_name)

        # Simple alias dictionaries for hackathon demo
        self.company_aliases: Dict[str, str] = {
            "HDFC Bank": "HDFC Bank",
            "HDFC": "HDFC Bank",
            "ICICI Bank": "ICICI Bank",
            "ICICI": "ICICI Bank",
            "Infosys": "Infosys",
        }

        self.company_to_ticker: Dict[str, str] = {
            "HDFC Bank": "HDFCBANK",
            "ICICI Bank": "ICICIBANK",
            "Infosys": "INFY",
        }

        # NEW: recognized tickers directly in text / queries
        self.ticker_aliases: Dict[str, str] = {
            "HDFCBANK": "HDFCBANK",
            "ICICIBANK": "ICICIBANK",
            "INFY": "INFY",
        }

        self.regulator_aliases: Dict[str, str] = {
            "RBI": "RBI",
            "Reserve Bank of India": "RBI",
            "central bank": "RBI",  # heuristic, okay for demo
        }

        # Sector keywords used for simple rule-based detection
        self.sector_keywords: Dict[str, List[str]] = {
            "Banking": ["bank", "banking", "banks"],
            # We will handle "IT" as a special token to avoid matching pronoun "it"
            "IT": ["IT", "technology", "software"],
            "Financial Services": ["nbfc", "lending", "finance"],
            "Economy": ["economy", "macro", "inflation"],
        }

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _extract_orgs(self, text: str) -> Set[str]:
        """Extract ORG entities using spaCy."""
        doc = self.nlp(text)
        orgs: Set[str] = set()
        for ent in doc.ents:
            if ent.label_ == "ORG":
                orgs.add(ent.text.strip())
        return orgs

    def _detect_regulators(self, text: str) -> Set[str]:
        """Detect regulator mentions using simple substring matching."""
        lowered = text.lower()
        regs: Set[str] = set()
        for alias, canon in self.regulator_aliases.items():
            if alias.lower() in lowered:
                regs.add(canon)
        return regs

    def _detect_sectors(self, text: str) -> Set[str]:
        """
        Detect sectors using token-based matching to avoid false positives like the pronoun 'it'.
        """
        doc = self.nlp(text)
        sectors: Set[str] = set()

        for sector, keywords in self.sector_keywords.items():
            for kw in keywords:
                kw_lower = kw.lower()

                # Special handling for "IT": require exact uppercase token "IT"
                if kw == "IT":
                    if any(token.text == "IT" for token in doc):
                        sectors.add(sector)
                        break
                else:
                    # Match lemma for general words like 'bank', 'technology', 'software', etc.
                    if any(token.lemma_.lower() == kw_lower for token in doc):
                        sectors.add(sector)
                        break

        return sectors

    def _detect_tickers(self, text: str) -> Set[str]:
        """
        Detect known tickers (INFY, HDFCBANK, ICICIBANK, ...) in the text.
        We simply look for tokens whose uppercase form matches our ticker_aliases.
        """
        doc = self.nlp(text)
        found: Set[str] = set()
        for token in doc:
            sym = token.text.upper()
            if sym in self.ticker_aliases:
                found.add(self.ticker_aliases[sym])
        return found

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities (companies, sectors, regulators, tickers) from arbitrary text.
        Used for both queries and generic strings.
        """
        orgs = self._extract_orgs(text)
        regulators = self._detect_regulators(text)
        sectors = self._detect_sectors(text)
        tickers = self._detect_tickers(text)

        companies: Set[str] = set()
        for org in orgs:
            for alias, canon in self.company_aliases.items():
                if alias.lower() in org.lower():
                    companies.add(canon)

        return {
            "companies": sorted(list(companies)),
            "sectors": sorted(list(sectors)),
            "regulators": sorted(list(regulators)),
            "tickers": sorted(list(tickers)),
        }

    def extract_from_article(self, article: NewsArticle) -> Dict[str, List[str]]:
        """
        Extract entities from a NewsArticle object.
        """
        text = f"{article.title}. {article.body}"
        return self.extract_from_text(text)

    def companies_to_tickers(self, companies: List[str]) -> Dict[str, str]:
        """
        Map canonical company names to stock tickers (if known).
        """
        mapping: Dict[str, str] = {}
        for c in companies:
            ticker = self.company_to_ticker.get(c)
            if ticker:
                mapping[c] = ticker
        return mapping
