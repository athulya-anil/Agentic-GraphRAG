#!/usr/bin/env python3
"""Test Query Parser Agent"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.query_parser_agent import get_query_parser

def main():
    print("🔄 Testing QueryParserAgent...\n")

    parser = get_query_parser(schema={
        "entity_types": ["Drug", "Disease", "Organization", "Person", "Location"],
        "relationship_types": ["TREATS", "MANUFACTURES", "FOUNDED", "HEADQUARTERED_IN"]
    })

    test_queries = [
        "What does Aspirin treat?",
        "Which drugs treat diabetes?",
        "Who manufactures Aspirin?",
        "Which diseases are treated by Metformin?",
        "Where is Tesla headquartered?",
        "Who founded Apple?"
    ]

    for query in test_queries:
        print(f"📝 Query: {query}")
        result = parser.parse_query(query)
        print(f"   Target: {result.get('target_type')}")
        print(f"   Anchor: {result.get('anchor_entity')} ({result.get('anchor_type')})")
        print(f"   Relationship: {result.get('relationship_type')}")
        print(f"   Direction: {result.get('direction')}")
        print(f"   Confidence: {result.get('confidence')}")

        cypher, params = parser.construct_cypher_query(result, max_results=3)
        print(f"   Cypher preview: {cypher.strip()[:120]}...")
        print()

if __name__ == "__main__":
    main()
