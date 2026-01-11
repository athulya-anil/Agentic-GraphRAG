#!/usr/bin/env python3
"""Verify Neo4j and FAISS databases are cleared."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.neo4j_manager import Neo4jManager

def main():
    print("\n" + "="*50)
    print("DATABASE VERIFICATION")
    print("="*50)

    # Check Neo4j
    print("\n1. Neo4j Database:")
    manager = Neo4jManager()

    # Count nodes
    node_result = manager.execute_query("MATCH (n) RETURN count(n) as count")
    node_count = node_result[0]['count']
    print(f"   - Nodes: {node_count}")

    # Count relationships
    rel_result = manager.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
    rel_count = rel_result[0]['count']
    print(f"   - Relationships: {rel_count}")

    if node_count == 0 and rel_count == 0:
        print("   ✓ Neo4j is EMPTY")
    else:
        print("   ✗ Neo4j still has data!")

    # Check FAISS
    print("\n2. FAISS Index:")
    faiss_path = Path("data/hotpotqa/faiss_index")
    if not faiss_path.exists():
        print(f"   - Directory: Not found")
        print("   ✓ FAISS is CLEARED")
    else:
        files = list(faiss_path.glob("*"))
        print(f"   - Directory: Exists")
        print(f"   - Files: {len(files)}")
        if len(files) == 0:
            print("   ✓ FAISS directory empty")
        else:
            print("   ✗ FAISS still has files!")

    print("\n" + "="*50)

    return 0

if __name__ == "__main__":
    sys.exit(main())
