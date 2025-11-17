#!/usr/bin/env python3
"""
Debug script to check OrchestratorAgent routing decisions for various queries.
Shows detailed LLM classification to understand why queries route to specific strategies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents import get_orchestrator_agent


def test_query_routing():
    """Test routing for various query types."""

    orchestrator = get_orchestrator_agent()

    # Test queries that should use different strategies
    test_queries = [
        # Should use GRAPH (explicit relationships)
        "What medications treat diabetes?",
        "What drugs are used for hypertension?",
        "Who founded Apple?",
        "Where is Tesla headquartered?",

        # Should use VECTOR (conceptual/semantic)
        "What is machine learning?",
        "Explain neural networks",
        "How does photosynthesis work?",

        # Should use HYBRID (exploratory)
        "Tell me about artificial intelligence",
        "What are the applications of blockchain?",

        # Ambiguous - could go either way
        "Who is Elon Musk?",
        "What is MIT?",
    ]

    print("=" * 80)
    print("ORCHESTRATOR ROUTING DEBUG")
    print("=" * 80)

    for query in test_queries:
        print(f"\n{'─' * 80}")
        print(f"Query: {query}")
        print(f"{'─' * 80}")

        routing = orchestrator.route_query(query)
        analysis = routing['analysis']

        print(f"  Query Type:          {routing['query_type'].value.upper()}")
        print(f"  Selected Strategy:   {routing['strategy'].value.upper()}")
        print(f"  Confidence:          {routing['confidence']:.2f}")
        print(f"\n  LLM Analysis:")
        print(f"    needs_relationships: {analysis.get('needs_relationships', 'N/A')}")
        print(f"    needs_semantic:      {analysis.get('needs_semantic', 'N/A')}")
        print(f"    needs_entities:      {analysis.get('needs_entities', 'N/A')}")
        print(f"    suggested_strategy:  {analysis.get('suggested_strategy', 'N/A')}")
        print(f"    reasoning:           {analysis.get('reasoning', 'N/A')}")

    print(f"\n{'=' * 80}")
    print("✅ Routing analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_query_routing()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
