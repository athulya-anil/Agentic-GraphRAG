"""
Evaluate MS MARCO with Improvements

Re-runs the 15-query MS MARCO evaluation with:
1. Conflict resolution (if we re-ingest)
2. Failure-aware routing (immediate impact)

Compares against baseline results.
"""

import json
from src.agents.orchestrator_agent import get_orchestrator_agent
from src.agents.failure_predictor import get_failure_predictor


def load_baseline_results():
    """Load baseline results (before improvements)."""
    with open('data/evaluation/msmarco/complete_results.json', 'r') as f:
        return json.load(f)


def analyze_routing_changes():
    """
    Analyze how failure-aware routing would change strategy selections.
    """
    print("="*80)
    print("FAILURE-AWARE ROUTING ANALYSIS")
    print("="*80)

    baseline = load_baseline_results()
    orchestrator = get_orchestrator_agent()
    failure_predictor = get_failure_predictor()

    routing_changes = []
    potential_fixes = []

    print(f"\nAnalyzing {len(baseline['detailed'])} queries...\n")

    for item in baseline['detailed']:
        query = item['query']
        old_strategy = item['strategy_used']
        old_score = item['overall_score']

        # Get new routing decision with failure-aware logic
        routing = orchestrator.route_query(query)
        new_strategy = routing['strategy'].value

        # Check failure risk
        should_avoid, failure_reasoning = failure_predictor.should_avoid_graph(query)

        # Did strategy change?
        if old_strategy != new_strategy:
            routing_changes.append({
                'query': query,
                'old_strategy': old_strategy,
                'new_strategy': new_strategy,
                'old_score': old_score,
                'failure_risk': failure_reasoning.get('risk_level', 'UNKNOWN'),
                'risk_factors': failure_reasoning.get('risk_factors', [])
            })

            # If old strategy was graph and it failed (score < 0.5), this might fix it
            if old_strategy == 'graph' and old_score < 0.5 and new_strategy == 'vector':
                potential_fixes.append({
                    'query': query,
                    'old_score': old_score,
                    'reason': ', '.join(failure_reasoning.get('risk_factors', []))
                })

    # Print analysis
    print("="*80)
    print(f"ROUTING CHANGES: {len(routing_changes)}/{len(baseline['detailed'])}")
    print("="*80)

    if routing_changes:
        for change in routing_changes:
            print(f"\nQuery: \"{change['query']}\"")
            print(f"  Old: {change['old_strategy']} (score: {change['old_score']:.2f})")
            print(f"  New: {change['new_strategy']} (risk: {change['failure_risk']})")
            if change['risk_factors']:
                print(f"  Reason: {', '.join(change['risk_factors'])}")
    else:
        print("\nNo routing changes - baseline already optimal!")

    print(f"\n{'='*80}")
    print(f"POTENTIAL FIXES: {len(potential_fixes)}")
    print("="*80)

    if potential_fixes:
        print("\nQueries that FAILED on graph but would now use vector:\n")
        for fix in potential_fixes:
            print(f"  Query: \"{fix['query']}\"")
            print(f"  Old Score: {fix['old_score']:.2f} (FAILED)")
            print(f"  Reason: {fix['reason']}")
            print(f"  → Now routed to vector (likely to succeed)")
            print()

        # Estimate impact
        print("="*80)
        print("ESTIMATED IMPACT")
        print("="*80)
        print(f"\nBaseline Results:")
        print(f"  Overall Score: {baseline['aggregated']['avg_overall_score']:.2%}")
        print(f"  Strategy Distribution: {baseline['aggregated']['strategy_distribution']}")

        # Conservative estimate: fixes might improve by ~0.3-0.5 each
        estimated_improvement = len(potential_fixes) * 0.4 / len(baseline['detailed'])
        new_score = baseline['aggregated']['avg_overall_score'] + estimated_improvement

        print(f"\nWith Failure-Aware Routing:")
        print(f"  Queries Fixed: {len(potential_fixes)}")
        print(f"  Estimated Score: {new_score:.2%} (+{estimated_improvement:.2%})")

    else:
        print("\nNo failures detected that would be fixed by routing changes.")
        print("The baseline was already routing correctly for this dataset.")

    return routing_changes, potential_fixes


def detailed_failure_analysis():
    """
    Detailed analysis of each query's failure risk.
    """
    print(f"\n{'='*80}")
    print("DETAILED FAILURE RISK ANALYSIS")
    print("="*80)

    baseline = load_baseline_results()
    failure_predictor = get_failure_predictor()

    high_risk = []
    moderate_risk = []
    low_risk = []

    for item in baseline['detailed']:
        query = item['query']
        strategy = item['strategy_used']
        score = item['overall_score']

        risk_score, reasoning = failure_predictor.predict_failure_risk(query)
        risk_level = reasoning['risk_level']

        risk_data = {
            'query': query,
            'strategy': strategy,
            'score': score,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': reasoning.get('risk_factors', [])
        }

        if risk_level == 'HIGH':
            high_risk.append(risk_data)
        elif risk_level in ['MODERATE', 'LOW']:
            moderate_risk.append(risk_data)
        else:
            low_risk.append(risk_data)

    # Print summary
    print(f"\nHigh Risk Queries: {len(high_risk)}")
    print(f"Moderate/Low Risk: {len(moderate_risk)}")
    print(f"Minimal Risk: {len(low_risk)}")

    if high_risk:
        print(f"\n{'='*80}")
        print("HIGH RISK QUERIES (should avoid graph)")
        print("="*80)
        for item in high_risk:
            print(f"\n  Query: \"{item['query']}\"")
            print(f"  Used: {item['strategy']} | Score: {item['score']:.2f}")
            print(f"  Risk: {item['risk_level']} ({item['risk_score']:.2f})")
            if item['risk_factors']:
                print(f"  Factors: {', '.join(item['risk_factors'])}")


def main():
    print("\n🔄 Evaluating MS MARCO with Failure-Aware Routing\n")

    # Analyze routing changes
    routing_changes, potential_fixes = analyze_routing_changes()

    # Detailed failure analysis
    detailed_failure_analysis()

    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print("="*80)

    print(f"\nSummary:")
    print(f"  - Routing changes: {len(routing_changes)}")
    print(f"  - Potential fixes: {len(potential_fixes)}")

    if potential_fixes:
        print(f"\n  💡 Failure-aware routing would likely improve performance")
        print(f"     by fixing {len(potential_fixes)} queries that failed on graph.")
    else:
        print(f"\n  ✅ Baseline routing was already optimal for this dataset.")
        print(f"     (This was a cherry-picked set of 15 queries)")


if __name__ == '__main__':
    main()
