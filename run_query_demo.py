from app.ingestion.loader import load_mock_articles
from app.dedup.dedup_service import DeduplicationService
from app.mapping.impact_mapping import ImpactMappingService
from app.ner.ner_service import EntityExtractionService
from app.query.query_service import QueryService


def build_index():
    # 1. Load articles
    articles = load_mock_articles("data/mock_news.jsonl")
    articles_by_id = {a.id: a for a in articles}

    # 2. Deduplicate into stories
    dedup = DeduplicationService()
    for a in articles:
        dedup.process_article(a)
    stories = dedup.get_all_stories()

    # 3. Impact mapping -> StoryWithImpact
    mapper = ImpactMappingService()
    enriched_stories = [mapper.map_story(s, articles_by_id) for s in stories]

    return enriched_stories


def run_query_examples(qs: QueryService):
    test_queries = [
        "HDFC Bank news",
        "Banking sector update",
        "RBI policy changes",
        "interest rate impact",
        "Infosys deal",
    ]

    for q in test_queries:
        print("=" * 100)
        print(f"Query: {q}")
        print("-" * 100)
        results = qs.search(q, top_k=5, min_score=0.05)
        if not results:
            print("No results.")
        else:
            for story, score in results:
                print(f"[score={score:.3f}] {story.title}")
        print("=" * 100)
        print()


def main():
    stories = build_index()
    ner = EntityExtractionService()
    qs = QueryService(ner=ner)
    qs.index(stories)

    run_query_examples(qs)


if __name__ == "__main__":
    main()
