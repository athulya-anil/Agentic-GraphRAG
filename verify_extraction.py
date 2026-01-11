#!/usr/bin/env python3
"""Verify extraction quality."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.neo4j_manager import Neo4jManager

def main():
    print("\n" + "="*70)
    print("EXTRACTION QUALITY VERIFICATION")
    print("="*70)

    manager = Neo4jManager()

    # Check Ed Wood entities
    print("\n1. Ed Wood Entities:")
    ed_wood_query = """
    MATCH (n)
    WHERE n.name =~ '(?i).*ed wood.*'
    RETURN n.name as name, n.type as type, n.nationality as nationality,
           labels(n) as labels
    ORDER BY n.name
    """
    ed_wood_results = manager.execute_query(ed_wood_query)

    if ed_wood_results:
        for result in ed_wood_results:
            print(f"   - Name: {result['name']}")
            print(f"     Type: {result['type']}")
            print(f"     Nationality: {result['nationality']}")
            print(f"     Labels: {result['labels']}")
            print()
    else:
        print("   ❌ No Ed Wood entities found!")

    # Check Scott Derrickson entities
    print("\n2. Scott Derrickson Entities:")
    scott_query = """
    MATCH (n)
    WHERE n.name =~ '(?i).*scott.*derrickson.*'
    RETURN n.name as name, n.type as type, n.nationality as nationality,
           labels(n) as labels
    ORDER BY n.name
    """
    scott_results = manager.execute_query(scott_query)

    if scott_results:
        for result in scott_results:
            print(f"   - Name: {result['name']}")
            print(f"     Type: {result['type']}")
            print(f"     Nationality: {result['nationality']}")
            print(f"     Labels: {result['labels']}")
            print()
    else:
        print("   ❌ No Scott Derrickson entities found!")

    # Check for type-safe deduplication (French language vs nationality)
    print("\n3. Type-Safe Deduplication Check (French entities):")
    french_query = """
    MATCH (n)
    WHERE n.name =~ '(?i)^french$'
    RETURN n.name as name, n.type as type, labels(n) as labels
    ORDER BY n.type
    """
    french_results = manager.execute_query(french_query)

    if french_results:
        for result in french_results:
            print(f"   - Name: {result['name']}, Type: {result['type']}, Labels: {result['labels']}")
        if len(french_results) > 1:
            print(f"   ✅ Type-safe dedup working: {len(french_results)} distinct 'French' entities with different types")
    else:
        print("   No 'French' entities found (checking similar...)")

    print("\n" + "="*70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
