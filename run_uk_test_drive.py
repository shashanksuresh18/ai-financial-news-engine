from app.graph.langgraph_flow import run_query_through_graph

def main():
    test_queries = [
        "HSBC news",
        "Barclays update",
        "FCA artificial intelligence review",
        "Bank of England rates",
        "Shell earnings",
        "AstraZeneca pharmaceuticals",
    ]

    print("STARTING UK TEST DRIVE")
    print("Using dataset: data/news_uk.jsonl")
    
    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"SEARCHING FOR: {q}")
        print("-" * 80)

        state = run_query_through_graph(query=q, dataset_path="data/news_uk.jsonl")
        results = state.get("results", [])

        if not results:
            print("No matching stories found.")
        else:
            for story, score in results:
                print(f"[Match Score: {score:.3f}] {story.title}")
                print(f"   Sectors: {', '.join(story.sectors) if hasattr(story, 'sectors') and story.sectors else 'None'}")
                print(f"   Companies: {', '.join(story.tickers) if hasattr(story, 'tickers') and story.tickers else 'None'}")

    print("\n" + "=" * 80)
    print("TEST DRIVE COMPLETE")

if __name__ == "__main__":
    main()
