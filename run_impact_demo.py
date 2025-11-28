from app.ingestion.loader import load_mock_articles
from app.dedup.dedup_service import DeduplicationService
from app.mapping.impact_mapping import ImpactMappingService


def main():
    # 1. Load articles
    articles = load_mock_articles("data/mock_news.jsonl")
    articles_by_id = {a.id: a for a in articles}

    # 2. Deduplicate into stories
    dedup = DeduplicationService()
    stories = []

    for a in articles:
        story = dedup.process_article(a)
        # avoid duplicates in list: collect final set later
        # here we just continue
    stories = dedup.get_all_stories()

    # 3. Map stories to impacted stocks
    mapper = ImpactMappingService()

    print("=" * 100)
    print("Stories with Impact Mapping")
    print("=" * 100)

    for s in stories:
        enriched = mapper.map_story(s, articles_by_id)
        print(f"Story: {enriched.title}")
        print(f"  Articles in story: {len(enriched.article_ids)}")
        print(f"  Impacted stocks:")
        if not enriched.impacted_stocks:
            print("    (none detected)")
        else:
            for is_ in enriched.impacted_stocks:
                print(
                    f"    - {is_.symbol} | confidence={is_.confidence:.2f} | types={is_.impact_type}"
                )
        print("-" * 100)


if __name__ == "__main__":
    main()
