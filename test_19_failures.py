"""
Test how many of the 19 graph failures would be fixed by failure-aware routing.
"""

import json
from src.agents.orchestrator_agent import get_orchestrator_agent
from src.agents.failure_predictor import get_failure_predictor

def main():
    print("="*80)
    print("TESTING FAILURE-AWARE ROUTING ON 19 GRAPH FAILURES")
    print("="*80)

    # Load the 19 failed queries
    with open('data/evaluation/graph_failure_analysis.json', 'r') as f:
        data = json.load(f)

    failures = data['analysis']['categorized']

    orchestrator = get_orchestrator_agent()
    failure_predictor = get_failure_predictor()

    fixed_by_routing = []
    still_using_graph = []

    print(f"\nAnalyzing {len(failures)} failed queries...\n")

    for item in failures:
        query = item['query']
        category = item['category']

        # Get new routing decision
        routing = orchestrator.route_query(query)
        new_strategy = routing['strategy'].value

        # Get failure risk
        risk_score, reasoning = failure_predictor.predict_failure_risk(query)
        risk_level = reasoning['risk_level']

        if new_strategy == 'vector':
            fixed_by_routing.append({
                'query': query,
                'category': category,
                'risk_level': risk_level,
                'risk_factors': reasoning.get('risk_factors', [])
            })
        else:
            still_using_graph.append({
                'query': query,
                'category': category,
                'risk_level': risk_level
            })

    # Print results
    print("="*80)
    print(f"QUERIES FIXED BY ROUTING TO VECTOR: {len(fixed_by_routing)}/19")
    print("="*80)

    if fixed_by_routing:
        for item in fixed_by_routing:
            print(f"\n✅ \"{item['query']}\"")
            print(f"   Category: {item['category']}")
            print(f"   Risk: {item['risk_level']}")
            if item['risk_factors']:
                print(f"   Reason: {', '.join(item['risk_factors'])}")

    print(f"\n{'='*80}")
    print(f"QUERIES STILL USING GRAPH: {len(still_using_graph)}/19")
    print("="*80)

    if still_using_graph:
        for item in still_using_graph:
            print(f"\n❌ \"{item['query']}\"")
            print(f"   Category: {item['category']}")
            print(f"   Risk: {item['risk_level']}")

    # Calculate impact
    print(f"\n{'='*80}")
    print("ESTIMATED IMPACT")
    print("="*80)

    # From NEXT_STEPS.md: Baseline was 68.9% on 100 queries
    # 19 failures at 0.0 score, 81 successes averaging higher
    baseline_accuracy = 0.689
    total_queries = 100

    # If we fix N failures, assuming they go from 0.0 to ~0.7 (vector avg)
    queries_fixed = len(fixed_by_routing)
    improvement_per_fix = 0.7 / total_queries  # Each fix adds 0.007 to overall score

    new_accuracy = baseline_accuracy + (queries_fixed * improvement_per_fix)

    print(f"\nBaseline: 68.9% (19 graph failures)")
    print(f"Queries fixed by routing: {queries_fixed}")
    print(f"Estimated new accuracy: {new_accuracy:.1%}")
    print(f"Improvement: +{(new_accuracy - baseline_accuracy):.1%}")

    # Breakdown by category
    print(f"\n{'='*80}")
    print("FIXES BY CATEGORY")
    print("="*80)

    category_counts = {}
    for item in fixed_by_routing:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
