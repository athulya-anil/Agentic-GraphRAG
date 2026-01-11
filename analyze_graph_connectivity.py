#!/usr/bin/env python3
"""Analyze graph connectivity issues."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.graph.neo4j_manager import Neo4jManager

def main():
    print("\n" + "="*70)
    print("GRAPH CONNECTIVITY ANALYSIS")
    print("="*70)

    manager = Neo4jManager()

    # 1. Check connectivity
    print("\n1. CONNECTIVITY STATISTICS:")

    # Total nodes
    total_nodes = manager.execute_query("MATCH (n) RETURN count(n) as count")[0]['count']

    # Isolated nodes (no connections)
    isolated = manager.execute_query("""
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) as count
    """)[0]['count']

    # Connected nodes
    connected = total_nodes - isolated

    print(f"   Total nodes: {total_nodes:,}")
    print(f"   Connected nodes: {connected:,} ({100*connected/total_nodes:.1f}%)")
    print(f"   Isolated nodes: {isolated:,} ({100*isolated/total_nodes:.1f}%)")

    # 2. Relationship types that DO exist
    print("\n2. ACTUAL RELATIONSHIP TYPES IN GRAPH:")
    rel_types = manager.execute_query("""
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
        LIMIT 20
    """)

    if rel_types:
        for rt in rel_types:
            print(f"   {rt['rel_type']:<50} {rt['count']:>5}")
    else:
        print("   ❌ NO RELATIONSHIPS FOUND!")

    # 3. Sample connected subgraphs
    print("\n3. SAMPLE CONNECTED COMPONENTS:")
    components = manager.execute_query("""
        MATCH (a)-[r]-(b)
        RETURN a.name as node1, type(r) as relationship, b.name as node2
        LIMIT 20
    """)

    if components:
        for comp in components:
            print(f"   {comp['node1']} --[{comp['relationship']}]-- {comp['node2']}")
    else:
        print("   ❌ No connected nodes found!")

    # 4. Check specific entities
    print("\n4. SAMPLE ENTITY CONNECTIONS:")

    # Check a few entities
    entities_to_check = [
        "United States",
        "English",
        "American",
        "New York"
    ]

    for entity_name in entities_to_check:
        result = manager.execute_query(f"""
            MATCH (n)
            WHERE n.name =~ '(?i).*{entity_name}.*'
            WITH n LIMIT 1
            MATCH (n)-[r]-(connected)
            RETURN n.name as entity, count(r) as connections
        """)

        if result:
            print(f"   '{result[0]['entity']}': {result[0]['connections']} connections")
        else:
            print(f"   '{entity_name}': Not found or no connections")

    # 5. Check relationship extraction quality
    print("\n5. RELATIONSHIP EXTRACTION QUALITY:")
    total_rels = manager.execute_query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']

    print(f"   Total relationships: {total_rels:,}")
    print(f"   Nodes: {total_nodes:,}")
    print(f"   Ratio: {total_rels/total_nodes:.2f} relationships per node")
    print(f"   Expected for dense graph: 3-10 relationships per node")

    if total_rels / total_nodes < 1.0:
        print(f"   ⚠️  Graph is VERY SPARSE - relationship extraction may have issues")

    print("\n" + "="*70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
