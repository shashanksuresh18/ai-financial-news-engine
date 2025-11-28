from app.graph.langgraph_flow import run_query_through_graph


def main():
    test_queries = [
        "HDFC Bank news",
        "Banking sector update",
        "RBI policy changes",
        "interest rate impact",
        "Infosys deal",
    ]

    for q in test_queries:
        print("=" * 100)
        print(f"LangGraph pipeline – Query: {q}")
        print("-" * 100)

        state = run_query_through_graph(query=q)
        results = state.get("results", [])

        if not results:
            print("No results.")
        else:
            for story, score in results:
                print(f"[score={score:.3f}] {story.title}")

        print("=" * 100)
        print()


if __name__ == "__main__":
    main()
