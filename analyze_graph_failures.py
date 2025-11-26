"""
Analyze Graph Retrieval Failures

This script analyzes the 19 graph failures from MS MARCO real evaluation
to identify patterns and categorize failure types.

This will help build a failure prediction model for failure-aware routing.
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Any


def load_results():
    """Load MS MARCO real evaluation results."""
    with open('data/evaluation/msmarco_real/results.json', 'r') as f:
        return json.load(f)


def extract_graph_failures(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all queries where graph retrieval failed (score = 0.0)."""
    failures = []
    for item in results['results']:
        if item['strategy'] == 'graph' and item['score'] == 0.0:
            failures.append(item)
    return failures


def categorize_query(query: str) -> Dict[str, Any]:
    """
    Categorize a query to understand why graph might fail.

    Categories:
    1. Temporal/Real-time: Asking for current/live data
    2. Location: "Where is X?" queries
    3. Contact Info: Phone numbers, addresses
    4. Relationship: "What/Who does X?" (should work on graph)
    5. Factual: Simple fact lookup
    6. Definition: "What is X?" queries
    """
    query_lower = query.lower()

    category = {
        'type': None,
        'has_temporal': False,
        'has_location': False,
        'has_relationship': False,
        'has_entity_lookup': False,
        'keywords': []
    }

    # Temporal indicators
    temporal_patterns = [
        r'\bweather\b', r'\bcurrent\b', r'\btoday\b', r'\bnow\b',
        r'\blatest\b', r'\brecent\b', r'\bwhen\b'
    ]
    for pattern in temporal_patterns:
        if re.search(pattern, query_lower):
            category['has_temporal'] = True
            category['keywords'].append(pattern.strip('\\b'))
            break

    # Location indicators
    location_patterns = [
        r'\bwhere is\b', r'\bwhere does\b', r'\blocation\b',
        r'\b(in|at) \w+\b'
    ]
    for pattern in location_patterns:
        if re.search(pattern, query_lower):
            category['has_location'] = True
            category['keywords'].append('location')
            break

    # Contact info
    if 'phone' in query_lower or 'number' in query_lower or 'address' in query_lower:
        category['type'] = 'contact_info'
        category['keywords'].append('contact')

    # Relationship queries (what/who/which)
    relationship_patterns = [
        r'^what (does|is|treats|causes|prevents|belongs)',
        r'^who (is|was|authored|founded)',
        r'^which ',
        r'what.*(treat|cause|prevent|produce)'
    ]
    for pattern in relationship_patterns:
        if re.search(pattern, query_lower):
            category['has_relationship'] = True
            category['keywords'].append('relationship')
            break

    # Entity lookup (asking about specific entity attributes)
    if re.search(r'\b(who|what|where) is \w+', query_lower):
        category['has_entity_lookup'] = True
        category['keywords'].append('entity_lookup')

    # Determine primary category
    if category['has_temporal']:
        category['type'] = 'temporal'
    elif category['type'] == 'contact_info':
        pass  # Already set
    elif category['has_location'] and category['has_entity_lookup']:
        category['type'] = 'location_lookup'
    elif category['has_relationship']:
        category['type'] = 'relationship'  # Should work on graph!
    elif category['has_entity_lookup']:
        category['type'] = 'entity_attribute'
    else:
        category['type'] = 'other'

    return category


def analyze_failures(failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze failure patterns."""

    # Categorize each failure
    categorized = []
    category_counts = defaultdict(int)

    for failure in failures:
        query = failure['query']
        category = categorize_query(query)

        categorized.append({
            'query': query,
            'query_id': failure['query_id'],
            'category': category['type'],
            'has_temporal': category['has_temporal'],
            'has_location': category['has_location'],
            'has_relationship': category['has_relationship'],
            'has_entity_lookup': category['has_entity_lookup'],
            'keywords': category['keywords']
        })

        category_counts[category['type']] += 1

    # Sort by category
    categorized.sort(key=lambda x: x['category'])

    return {
        'total_failures': len(failures),
        'categorized': categorized,
        'category_counts': dict(category_counts)
    }


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis results."""
    print("="*80)
    print(f"GRAPH RETRIEVAL FAILURE ANALYSIS")
    print("="*80)
    print(f"\nTotal Failures: {analysis['total_failures']}")

    print(f"\n{'='*80}")
    print("FAILURE CATEGORIES")
    print("="*80)
    for category, count in sorted(analysis['category_counts'].items(), key=lambda x: -x[1]):
        pct = (count / analysis['total_failures']) * 100
        print(f"{category:20s}: {count:2d} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWN")
    print("="*80)

    current_category = None
    for item in analysis['categorized']:
        if item['category'] != current_category:
            current_category = item['category']
            print(f"\n--- {current_category.upper().replace('_', ' ')} ---")

        print(f"\n  Query: \"{item['query']}\"")
        if item['keywords']:
            print(f"  Keywords: {', '.join(item['keywords'])}")
        flags = []
        if item['has_temporal']: flags.append('temporal')
        if item['has_location']: flags.append('location')
        if item['has_relationship']: flags.append('relationship')
        if item['has_entity_lookup']: flags.append('entity')
        if flags:
            print(f"  Flags: {', '.join(flags)}")


def generate_failure_rules(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate rules for predicting graph failures."""

    rules = []

    # Rule 1: Temporal queries almost always fail on static KG
    temporal_failures = sum(1 for item in analysis['categorized'] if item['has_temporal'])
    if temporal_failures > 0:
        rules.append({
            'name': 'temporal_query',
            'condition': 'has_temporal',
            'confidence': temporal_failures / analysis['total_failures'],
            'action': 'avoid_graph',
            'reason': 'Temporal queries require real-time data, static KG will fail'
        })

    # Rule 2: Contact info queries fail
    contact_failures = analysis['category_counts'].get('contact_info', 0)
    if contact_failures > 0:
        rules.append({
            'name': 'contact_info',
            'condition': 'contains phone/number/address',
            'confidence': contact_failures / analysis['total_failures'],
            'action': 'avoid_graph',
            'reason': 'Contact info rarely in KG, use vector for document search'
        })

    # Rule 3: Some relationship queries fail (missing entities/relationships)
    relationship_failures = analysis['category_counts'].get('relationship', 0)
    if relationship_failures > 0:
        rules.append({
            'name': 'relationship_with_risk',
            'condition': 'relationship query but unknown entity',
            'confidence': relationship_failures / analysis['total_failures'],
            'action': 'check_entity_exists',
            'reason': 'Relationship queries good for graph, but only if entities exist'
        })

    # Rule 4: Entity lookups for obscure entities fail
    entity_failures = analysis['category_counts'].get('entity_attribute', 0) + \
                     analysis['category_counts'].get('location_lookup', 0)
    if entity_failures > 0:
        rules.append({
            'name': 'entity_lookup',
            'condition': 'asks about entity attribute',
            'confidence': entity_failures / analysis['total_failures'],
            'action': 'check_entity_coverage',
            'reason': 'Entity lookups fail if entity not in KG'
        })

    return rules


def main():
    print("Loading MS MARCO real results...")
    results = load_results()

    print("Extracting graph failures...")
    failures = extract_graph_failures(results)
    print(f"Found {len(failures)} graph failures\n")

    print("Analyzing failure patterns...")
    analysis = analyze_failures(failures)

    print_analysis(analysis)

    print(f"\n{'='*80}")
    print("FAILURE PREDICTION RULES")
    print("="*80)

    rules = generate_failure_rules(analysis)
    for i, rule in enumerate(rules, 1):
        print(f"\nRule {i}: {rule['name']}")
        print(f"  Condition: {rule['condition']}")
        print(f"  Confidence: {rule['confidence']:.2%}")
        print(f"  Action: {rule['action']}")
        print(f"  Reason: {rule['reason']}")

    # Save analysis
    with open('data/evaluation/graph_failure_analysis.json', 'w') as f:
        json.dump({
            'analysis': analysis,
            'rules': rules
        }, f, indent=2)

    print(f"\n✅ Analysis saved to data/evaluation/graph_failure_analysis.json")


if __name__ == '__main__':
    main()
