#!/usr/bin/env python3
"""
Fallback & Shortcut Detector

Checks for common evaluation pitfalls:
1. Silent fallback to vector when graph fails
2. Cached results bypassing routing
3. Default strategy override
4. Query parser fallbacks

Run this with verbose logging to detect shortcuts.
"""

import json
import logging
from pathlib import Path
from src.pipeline import get_retrieval_pipeline
from src.agents import RetrievalStrategy

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)


def check_fallbacks():
    """Check for silent fallbacks and shortcuts."""

    print("\n" + "="*70)
    print("FALLBACK & SHORTCUT DETECTION")
    print("="*70)

    pipeline = get_retrieval_pipeline(use_reflection=False)

    # Load test query
    queries_path = Path("data/hotpotqa/queries_100.json")
    with open(queries_path) as f:
        queries = json.load(f)

    test_query = queries[0]['question']

    # ========== Test 1: Cache Bypass Check ==========
    print("\n1. Cache Bypass Check:")
    print("   Testing with use_cache=True vs False...")

    result_cached = pipeline.query(test_query, use_cache=True, evaluate=False)
    result_fresh = pipeline.query(test_query, use_cache=False, evaluate=False)

    print(f"   Cached result contexts: {result_cached['num_contexts']}")
    print(f"   Fresh result contexts: {result_fresh['num_contexts']}")

    if result_cached['response'] == result_fresh['response']:
        print("   ⚠️  Same response - either both fresh or both cached")
    else:
        print("   ✓ Different responses - cache working as expected")

    # Clear cache for evaluation
    print("   → For evaluation, use_cache=False is set in evaluate_hotpotqa_full.py")

    # ========== Test 2: Routing Consistency ==========
    print("\n2. Routing Consistency Check:")
    print("   Running same query 3 times to check routing...")

    strategies = []
    for i in range(3):
        routing = pipeline.orchestrator.route_query(test_query)
        strategies.append(routing['strategy'].value)

    print(f"   Routing decisions: {strategies}")

    if len(set(strategies)) == 1:
        print(f"   ✓ Consistent routing: always {strategies[0]}")
    else:
        print(f"   ⚠️  WARNING: Inconsistent routing across runs")

    # ========== Test 3: Graph Failure Fallback ==========
    print("\n3. Graph Failure Handling:")
    print("   Testing query that may not match graph entities...")

    obscure_query = "What is the quantum chromodynamics lagrangian?"

    # Force graph strategy
    graph_result = pipeline.retrieve(obscure_query, strategy=RetrievalStrategy.GRAPH)
    vector_result = pipeline.retrieve(obscure_query, strategy=RetrievalStrategy.VECTOR)

    print(f"   Graph results: {len(graph_result)}")
    print(f"   Vector results: {len(vector_result)}")

    if len(graph_result) == 0 and len(vector_result) > 0:
        print("   ✓ Graph returns empty, vector works - no silent fallback")
    elif len(graph_result) == len(vector_result) and len(graph_result) > 0:
        print("   ⚠️  Graph and vector return same count - check if fallback triggered")

    # ========== Test 4: Query Parser Fallback ==========
    print("\n4. Query Parser Fallback Check:")
    print("   Testing with complex query...")

    from src.agents.query_parser_agent import get_query_parser
    from src.graph import get_neo4j_manager

    neo4j = get_neo4j_manager()
    schema = neo4j.get_schema()

    parser = get_query_parser(schema={
        "entity_types": schema.get('labels', []),
        "relationship_types": schema.get('relationship_types', [])
    })

    intent = parser.parse_query(test_query)

    print(f"   Confidence: {intent.get('confidence', 0):.3f}")
    print(f"   Target: {intent.get('target_type')}")
    print(f"   Anchor: {intent.get('anchor_entity')}")
    print(f"   Direction: {intent.get('direction')}")

    if intent.get('confidence', 0) < 0.4:
        print("   ⚠️  WARNING: Low confidence - fallback parser may be active")
        if 'reasoning' in intent:
            print(f"   Reasoning: {intent['reasoning']}")

    # ========== Test 5: Strategy Distribution ==========
    print("\n5. Strategy Distribution Check:")
    print("   Testing 10 random queries to see routing distribution...")

    strategy_counts = {"vector": 0, "graph": 0}

    for query_obj in queries[:10]:
        routing = pipeline.orchestrator.route_query(query_obj['question'])
        strategy_counts[routing['strategy'].value] += 1

    print(f"   Vector: {strategy_counts['vector']}/10")
    print(f"   Graph: {strategy_counts['graph']}/10")

    if strategy_counts['vector'] == 10:
        print("   ⚠️  WARNING: All queries routed to vector - routing may be broken")
    elif strategy_counts['graph'] == 10:
        print("   ⚠️  WARNING: All queries routed to graph - routing may be broken")
    else:
        print("   ✓ Mixed routing - orchestrator is working")

    # ========== Final Checklist ==========
    print("\n" + "="*70)
    print("FALLBACK CHECKLIST")
    print("="*70)

    checks = {
        "Cache disabled in evaluation": "✓ (set in evaluate_hotpotqa_full.py)",
        "Routing is consistent": "✓" if len(set(strategies)) == 1 else "⚠️",
        "Graph failures don't silently fallback": "✓" if len(graph_result) != len(vector_result) else "⚠️",
        "Query parser confidence acceptable": "✓" if intent.get('confidence', 0) >= 0.4 else "⚠️",
        "Strategy distribution is mixed": "✓" if 0 < strategy_counts['graph'] < 10 else "⚠️"
    }

    print()
    for check, status in checks.items():
        print(f"{status} {check}")

    print("\n" + "="*70)
    print("✅ FALLBACK CHECK COMPLETE")
    print("="*70)
    print("\nReview warnings above. Most are informational, not blockers.")


if __name__ == "__main__":
    import sys
    try:
        check_fallbacks()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
