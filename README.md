# AI Financial News Engine

An AI-powered system that reads financial news, removes duplicate articles, figures out which stocks are impacted, and lets you ask questions like:

- `HDFC Bank news`
- `Banking sector update`
- `RBI policy changes`
- `interest rate impact`
- `INFY`, `HDFCBANK`, `ICICIBANK`

Built with a **LangGraph multi-agent pipeline** and exposed via a **FastAPI** API + minimal web UI.

---

## ✨ Features

- **Intelligent Deduplication**
  - Groups similar articles into a single *story* using sentence-transformer embeddings + lexical overlap.
  - Example: multiple HDFC Bank dividend headlines → one consolidated story.

- **Entity Extraction & Impact Mapping**
  - spaCy + custom rules to detect:
    - Companies (HDFC Bank, ICICI Bank, Infosys)
    - Sectors (Banking, IT, Economy)
    - Regulators (RBI)
  - Maps to *impacted stocks* with confidence:
    - Direct company mention → **1.0**
    - Sector-wide impact → **0.7**
    - Regulatory impact (e.g. RBI on banks) → **0.6**

- **Context-Aware Querying**
  - Understands:
    - Company and ticker queries: `HDFC Bank`, `INFY`, `HDFCBANK`
    - Sector queries: `Banking sector update`
    - Regulator queries: `RBI policy changes`
    - Thematic queries: `interest rate impact`
  - Combines entity signals + semantic similarity to rank relevant stories.

- **LangGraph Multi-Agent Pipeline**
  - Nodes:
    - `ingest` → load articles
    - `dedup` → build unique stories
    - `enrich` → NER + impact mapping
    - `query` → ranking & retrieval
  - Implemented as a `StateGraph` with shared typed state.

- **FastAPI Backend + Web UI**
  - REST endpoints for `/health`, `/stories`, `/query`, `/ingest_live`
  - Simple HTML+JS UI at `/` for interactive search.
  - Live RSS ingestion to augment mock data.

---

## 🧱 Project Structure

```text
app/
  api/
    main.py              # FastAPI app + web UI + live RSS ingest
  dedup/
    dedup_service.py     # Intelligent deduplication
  ingestion/
    loader.py            # Load mock dataset
    live_rss.py          # Fetch live RSS articles
  mapping/
    impact_mapping.py    # Aggregate entities + compute impacted stocks
  models/
    schema.py            # NewsArticle, Story, StoryWithImpact, ImpactedStock
  ner/
    ner_service.py       # NER + sector/regulator/ticker rules
  query/
    query_service.py     # Context-aware query engine
  graph/
    langgraph_flow.py    # LangGraph StateGraph (multi-agent pipeline)

run_dedup_demo.py        # Demo: dedup only
run_impact_demo.py       # Demo: dedup + impact mapping
run_query_demo.py        # Demo: query without LangGraph
run_langgraph_demo.py    # Demo: full LangGraph pipeline
