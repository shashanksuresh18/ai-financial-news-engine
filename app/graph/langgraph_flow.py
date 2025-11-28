from typing import List, Dict, Tuple
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from app.models.schema import NewsArticle, Story, StoryWithImpact
from app.ingestion.loader import load_mock_articles
from app.dedup.dedup_service import DeduplicationService
from app.mapping.impact_mapping import ImpactMappingService
from app.ner.ner_service import EntityExtractionService
from app.query.query_service import QueryService


class PipelineState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    total=False means all keys are optional and we only update parts.
    """
    # config
    dataset_path: str

    # data
    articles: List[NewsArticle]
    articles_by_id: Dict[str, NewsArticle]

    stories: List[Story]
    enriched_stories: List[StoryWithImpact]

    # query / results
    query: str
    results: List[Tuple[StoryWithImpact, float]]


# ----------------------------------------------------------------------
# Nodes (Agents)
# ----------------------------------------------------------------------


def ingest_node(state: PipelineState) -> PipelineState:
    """
    Ingestion agent:
    - loads articles from dataset_path (or default mock file)
    """
    path = state.get("dataset_path") or "data/mock_news.jsonl"
    articles = load_mock_articles(path)
    articles_by_id = {a.id: a for a in articles}

    return {
        "articles": articles,
        "articles_by_id": articles_by_id,
    }


def dedup_node(state: PipelineState) -> PipelineState:
    """
    Dedup agent:
    - takes raw articles
    - groups them into unique 'stories' using DeduplicationService
    """
    articles = state.get("articles", [])
    svc = DeduplicationService()
    for a in articles:
        svc.process_article(a)
    stories = svc.get_all_stories()

    return {"stories": stories}


def enrich_node(state: PipelineState) -> PipelineState:
    """
    NER + Impact Mapping agent:
    - takes stories + articles_by_id
    - runs entity extraction + impact mapping
    - outputs enriched stories (StoryWithImpact)
    """
    stories = state.get("stories", [])
    articles_by_id = state.get("articles_by_id", {})

    mapper = ImpactMappingService()
    enriched: List[StoryWithImpact] = [
        mapper.map_story(s, articles_by_id) for s in stories
    ]

    return {"enriched_stories": enriched}


def query_node(state: PipelineState) -> PipelineState:
    """
    Query agent:
    - takes enriched_stories and natural language query
    - returns ranked (story, score) results
    """
    enriched_stories = state.get("enriched_stories", [])
    query = state.get("query", "")

    if not query:
        return {"results": []}

    ner = EntityExtractionService()
    qs = QueryService(ner=ner)
    qs.index(enriched_stories)

    results = qs.search(query=query, top_k=5, min_score=0.05)
    return {"results": results}


# ----------------------------------------------------------------------
# Graph builder
# ----------------------------------------------------------------------


def create_app():
    """
    Build and compile the LangGraph StateGraph representing the full pipeline.

    Flow:
        START -> ingest -> dedup -> enrich -> query -> END
    """
    workflow = StateGraph(PipelineState)

    # Register nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("dedup", dedup_node)
    workflow.add_node("enrich", enrich_node)
    workflow.add_node("query", query_node)

    # Wire edges
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "dedup")
    workflow.add_edge("dedup", "enrich")
    workflow.add_edge("enrich", "query")
    workflow.add_edge("query", END)

    # Compile graph
    app = workflow.compile()
    return app


def run_query_through_graph(
    query: str,
    dataset_path: str = "data/mock_news.jsonl",
) -> PipelineState:
    """
    Convenience helper:
    - builds the graph
    - runs full pipeline on the given query
    - returns the final PipelineState (with 'results')
    """
    app = create_app()

    initial_state: PipelineState = {
        "dataset_path": dataset_path,
        "query": query,
    }

    final_state: PipelineState = app.invoke(initial_state)
    return final_state
