"""
Test Failure-Aware Routing

Tests that the OrchestratorAgent correctly avoids graph for high-risk queries.
"""

from src.agents.orchestrator_agent import get_orchestrator_agent


def test_failure_aware_routing():
    """Test that failure-aware routing works correctly."""

    print("🔄 Testing Failure-Aware Routing\n")
    print("="*80)

    orchestrator = get_orchestrator_agent()

    # Test queries that should AVOID graph (temporal, contact)
    high_risk_queries = [
        ("weather in new york", "temporal"),
        ("current stock price of apple", "temporal"),
        ("phone number for customer service", "contact"),
        ("what is the address of the white house", "contact"),
    ]

    # Test queries that should USE graph (relationship)
    low_risk_queries = [
        ("what drugs treat diabetes", "relationship"),
        ("who founded Apple Inc", "relationship"),
        ("what is aripiprazole used for", "relationship"),
    ]

    # Test queries that should USE vector (conceptual)
    vector_queries = [
        ("explain photosynthesis", "conceptual"),
        ("what is machine learning", "conceptual"),
    ]

    print("\n🔴 HIGH RISK QUERIES (should avoid graph)")
    print("="*80)
    for query, expected_reason in high_risk_queries:
        routing = orchestrator.route_query(query)
        strategy = routing['strategy'].value

        if strategy == 'vector':
            print(f"✅ {query}")
            print(f"   → Correctly avoided graph (using {strategy})")
        else:
            print(f"❌ {query}")
            print(f"   → Should avoid graph but used {strategy}")
        print()

    print("\n🟢 LOW RISK QUERIES (should use graph)")
    print("="*80)
    for query, expected_reason in low_risk_queries:
        routing = orchestrator.route_query(query)
        strategy = routing['strategy'].value

        if strategy == 'graph':
            print(f"✅ {query}")
            print(f"   → Correctly used graph")
        else:
            print(f"⚠️  {query}")
            print(f"   → Used {strategy} instead of graph")
        print()

    print("\n🔵 CONCEPTUAL QUERIES (should use vector)")
    print("="*80)
    for query, expected_reason in vector_queries:
        routing = orchestrator.route_query(query)
        strategy = routing['strategy'].value

        if strategy == 'vector':
            print(f"✅ {query}")
            print(f"   → Correctly used vector")
        else:
            print(f"⚠️  {query}")
            print(f"   → Used {strategy} instead of vector")
        print()

    print("="*80)
    print("✅ Failure-Aware Routing Test Complete!")


if __name__ == '__main__':
    test_failure_aware_routing()
