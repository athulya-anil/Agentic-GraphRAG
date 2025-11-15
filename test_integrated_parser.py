#!/usr/bin/env python3
"""Test integrated query parser in retrieval pipeline"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_retrieval_pipeline
from src.agents import RetrievalStrategy

def main():
    print("=" * 70)
    print("  TESTING INTEGRATED QUERY PARSER")
    print("=" * 70)

    retrieval = get_retrieval_pipeline(use_reflection=False)  # Disable evaluation to save tokens

    # Test reverse queries that previously failed
    test_queries = [
        "Which drugs treat diabetes?",
        "Who manufactures Aspirin?",
        "Which diseases are treated by Metformin?",
    ]

    print(f"\n🔍 Testing {len(test_queries)} reverse queries with integrated parser...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: \"{query}\"")
        print(f"{'─' * 70}")

        try:
            result = retrieval.query(
                query,
                top_k=3,
                strategy=RetrievalStrategy.GRAPH,
                evaluate=False  # Skip evaluation to save API calls
            )

            print(f"✅ Retrieved {result['num_contexts']} contexts")

            if result['context']:
                print(f"\n📄 Top Context:")
                ctx = result['context'][0]
                print(f"   Source: {ctx['source']}, Score: {ctx['score']:.3f}")
                print(f"   {ctx['text'][:200]}...")

                print(f"\n💬 Response:")
                print(f"   {result['response'][:300]}...")
            else:
                print(f"❌ No contexts found")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
