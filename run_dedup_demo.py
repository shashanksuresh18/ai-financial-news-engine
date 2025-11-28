from app.ingestion.loader import load_mock_articles
from app.dedup.dedup_service import DeduplicationService


def main():
    svc = DeduplicationService()
    articles = load_mock_articles("data/mock_news.jsonl")

    for a in articles:
        story = svc.process_article(a)
        print(f"Article: {a.title}")
        print(f" -> Story ID: {story.id}")
        print(f"    Story title: {story.title}")
        print(f"    Story articles: {story.article_ids}")
        print("-" * 80)

    all_stories = svc.get_all_stories()
    print(f"Total unique stories: {len(all_stories)}")
    print("Stories:")
    for s in all_stories:
        print(f"- {s.id} | {s.title} | {len(s.article_ids)} articles")


if __name__ == "__main__":
    main()
