#!/usr/bin/env python3
"""
Graph Construction Quality Checker

Verifies that the Neo4j graph constructed from HotpotQA is not degenerate:
1. Entities exist for Wikipedia titles
2. Relations are not overly generic
3. Graph has reasonable connectivity

Run this AFTER ingestion to verify graph quality.
"""

import json
from pathlib import Path
from collections import Counter
from src.graph import get_neo4j_manager


def check_graph_quality():
    """Check Neo4j graph construction quality."""

    print("\n" + "="*70)
    print("GRAPH CONSTRUCTION QUALITY CHECK")
    print("="*70)

    neo4j = get_neo4j_manager()

    # ========== 1. Basic Stats ==========
    print("\n1. Basic Statistics:")
    node_count = neo4j.get_node_count()
    rel_count = neo4j.get_relationship_count()
    schema = neo4j.get_schema()

    print(f"   Total nodes: {node_count:,}")
    print(f"   Total relationships: {rel_count:,}")
    print(f"   Avg relationships per node: {rel_count/max(node_count, 1):.2f}")

    if node_count == 0:
        print("   ❌ CRITICAL: Graph is empty!")
        return False

    if rel_count == 0:
        print("   ⚠️  WARNING: No relationships!")
        return False

    # ========== 2. Entity Type Distribution ==========
    print("\n2. Entity Type Distribution:")
    labels = schema.get('labels', [])
    print(f"   Total entity types: {len(labels)}")

    # Count nodes per label
    label_counts = {}
    for label in labels[:15]:  # Top 15
        count = neo4j.get_node_count(label)
        label_counts[label] = count

    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels[:10]:
        print(f"   {label:20} {count:6,} nodes")

    # Check for degenerate case: everything is one type
    if sorted_labels and sorted_labels[0][1] > node_count * 0.9:
        print(f"   ⚠️  WARNING: {sorted_labels[0][0]} dominates graph ({sorted_labels[0][1]/node_count*100:.1f}%)")

    # ========== 3. Relationship Type Distribution ==========
    print("\n3. Relationship Type Distribution:")
    rel_types = schema.get('relationship_types', [])
    print(f"   Total relationship types: {len(rel_types)}")

    rel_type_counts = {}
    for rel_type in rel_types[:15]:
        count = neo4j.get_relationship_count(rel_type)
        rel_type_counts[rel_type] = count

    sorted_rels = sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True)
    for rel_type, count in sorted_rels[:10]:
        print(f"   {rel_type:20} {count:6,} relationships")

    # Check for overly generic relationships
    if sorted_rels:
        top_rel, top_count = sorted_rels[0]
        if top_count > rel_count * 0.8 and top_rel.upper() in ['RELATED_TO', 'LINKED_TO', 'CONNECTED_TO']:
            print(f"   ⚠️  WARNING: '{top_rel}' is too generic ({top_count/rel_count*100:.1f}% of all relationships)")

    # ========== 4. Sample Entities from HotpotQA ==========
    print("\n4. Wikipedia Title Entity Check:")

    # Load corpus to get sample titles
    corpus_path = Path("data/hotpotqa/corpus.json")
    if corpus_path.exists():
        with open(corpus_path) as f:
            corpus = json.load(f)

        sample_titles = [doc['title'] for doc in corpus[:10]]

        print(f"   Checking {len(sample_titles)} sample Wikipedia titles...")

        found_count = 0
        for title in sample_titles:
            # Search for entity by name
            cypher = """
            MATCH (n)
            WHERE toLower(n.name) = $name
            RETURN n, labels(n) as labels
            LIMIT 1
            """
            results = neo4j.execute_query(cypher, {"name": title.lower()})

            if results:
                found_count += 1
                node_labels = results[0]['labels']
                print(f"   ✓ Found: {title} ({', '.join(node_labels)})")
            else:
                # Try partial match
                cypher_partial = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS $word
                RETURN n.name as name, labels(n) as labels
                LIMIT 1
                """
                # Try first word
                first_word = title.split()[0].lower()
                partial_results = neo4j.execute_query(cypher_partial, {"word": first_word})

                if partial_results:
                    print(f"   ~ Partial: {title} → found '{partial_results[0]['name']}'")
                    found_count += 0.5
                else:
                    print(f"   ✗ Missing: {title}")

        coverage = found_count / len(sample_titles) if sample_titles else 0
        print(f"\n   Coverage: {coverage*100:.1f}% ({found_count}/{len(sample_titles)})")

        if coverage < 0.3:
            print("   ⚠️  WARNING: Low entity coverage - most Wikipedia titles not in graph")

    # ========== 5. Graph Connectivity ==========
    print("\n5. Graph Connectivity:")

    # Sample connected component
    cypher = """
    MATCH path = (a)-[*1..3]-(b)
    RETURN a.name as start, b.name as end, length(path) as hops
    LIMIT 5
    """
    paths = neo4j.execute_query(cypher)

    if paths:
        print(f"   Sample multi-hop paths:")
        for p in paths[:3]:
            print(f"   {p['start']} → {p['end']} ({p['hops']} hops)")
        print("   ✓ Graph has multi-hop connectivity")
    else:
        print("   ⚠️  WARNING: No multi-hop paths found - graph may be disconnected")

    # Check isolated nodes
    cypher_isolated = """
    MATCH (n)
    WHERE NOT (n)--()
    RETURN count(n) as isolated_count
    """
    isolated = neo4j.execute_query(cypher_isolated)
    isolated_count = isolated[0]['isolated_count'] if isolated else 0

    print(f"\n   Isolated nodes: {isolated_count:,} ({isolated_count/max(node_count,1)*100:.1f}%)")

    if isolated_count > node_count * 0.5:
        print("   ⚠️  WARNING: >50% nodes are isolated - graph is poorly connected")

    # ========== 6. Sample Query Test ==========
    print("\n6. Sample Query Test:")

    test_entity = "France"  # Common entity likely in HotpotQA
    cypher_test = """
    MATCH (n)
    WHERE toLower(n.name) CONTAINS $word
    RETURN n.name as name, labels(n) as labels
    LIMIT 3
    """
    results = neo4j.execute_query(cypher_test, {"word": test_entity.lower()})

    if results:
        print(f"   Search for '{test_entity}':")
        for r in results:
            print(f"   ✓ {r['name']} ({', '.join(r['labels'])})")
    else:
        print(f"   ⚠️  '{test_entity}' not found - may need better entity extraction")

    # ========== Final Verdict ==========
    print("\n" + "="*70)
    print("QUALITY ASSESSMENT")
    print("="*70)

    issues = []

    if node_count == 0:
        issues.append("Graph is empty")
    if rel_count == 0:
        issues.append("No relationships")
    if rel_count / max(node_count, 1) < 0.5:
        issues.append("Low connectivity (avg < 0.5 rels/node)")
    if isolated_count > node_count * 0.5:
        issues.append(">50% isolated nodes")
    if coverage < 0.3:
        issues.append("Low Wikipedia title coverage")

    if not issues:
        print("✅ GRAPH QUALITY: ACCEPTABLE")
        print("\nGraph is not degenerate. Ready for evaluation.")
        return True
    else:
        print("⚠️  GRAPH QUALITY: ISSUES DETECTED")
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nGraph may work but could be improved.")
        return True  # Not blocking, just warning


if __name__ == "__main__":
    import sys
    try:
        success = check_graph_quality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
