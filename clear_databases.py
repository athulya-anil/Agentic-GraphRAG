#!/usr/bin/env python3
"""Clear Neo4j and FAISS databases."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.neo4j_manager import Neo4jManager

def main():
    print("Clearing Neo4j database...")
    manager = Neo4jManager()
    manager.clear_database()
    print("✓ Neo4j database cleared")

    return 0

if __name__ == "__main__":
    sys.exit(main())
