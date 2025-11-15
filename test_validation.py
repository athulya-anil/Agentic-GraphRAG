#!/usr/bin/env python3
"""
Test Entity Validation Fix

This script tests that the validation system now catches entity type errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import get_neo4j_manager
from src.pipeline import get_ingestion_pipeline

def main():
    print("=" * 70)
    print("  ENTITY VALIDATION TEST")
    print("=" * 70)

    # Clear database
    print("\n🗑️  Clearing Neo4j database...")
    neo4j = get_neo4j_manager()
    neo4j.clear_database()
    print("   ✓ Database cleared")

    # Test documents with known entity type issues
    documents = [
        """Aspirin is a medication used to reduce pain, fever, and inflammation.
        It is manufactured by Bayer. Aspirin treats headaches and muscle pain.""",

        """Metformin is a medication commonly prescribed to treat type 2 diabetes.
        It helps control blood sugar levels.""",
    ]

    print("\n📚 Ingesting test documents...")
    print("   Expected entity corrections:")
    print("   • Aspirin: should be Drug (not Organization/Product)")
    print("   • Metformin: should be Drug (not Organization/Product)")
    print("   • Bayer: should be Organization")
    print("   • Diabetes: should be Disease")

    schema_path = Path("data/processed/schema.json")
    pipeline = get_ingestion_pipeline(schema_path=schema_path)

    result = pipeline.ingest_documents(
        documents,
        infer_schema=True,
        enrich_metadata=True
    )

    print(f"\n✅ Ingestion complete:")
    print(f"   • Entities: {result['entities_extracted']}")
    print(f"   • Relations: {result['relations_extracted']}")
    print(f"   • Nodes: {result['nodes_created']}")
    print(f"   • Edges: {result['edges_created']}")

    # Check entity types in graph
    print("\n🔍 Checking entity types in graph...")

    # Check if Metformin is Drug or Organization
    metformin_drug = neo4j.execute_query("MATCH (n:Drug {name: 'Metformin'}) RETURN n")
    metformin_org = neo4j.execute_query("MATCH (n:Organization {name: 'Metformin'}) RETURN n")

    aspirin_drug = neo4j.execute_query("MATCH (n:Drug {name: 'Aspirin'}) RETURN n")
    aspirin_org = neo4j.execute_query("MATCH (n:Organization {name: 'Aspirin'}) RETURN n")

    bayer_org = neo4j.execute_query("MATCH (n:Organization {name: 'Bayer'}) RETURN n")

    print("\n📊 Validation Results:")

    success = True

    if metformin_drug:
        print("   ✅ Metformin correctly classified as Drug")
    elif metformin_org:
        print("   ❌ Metformin INCORRECTLY classified as Organization")
        success = False
    else:
        print("   ⚠️  Metformin not found in graph")
        success = False

    if aspirin_drug:
        print("   ✅ Aspirin correctly classified as Drug")
    elif aspirin_org:
        print("   ❌ Aspirin INCORRECTLY classified as Organization")
        success = False
    else:
        print("   ⚠️  Aspirin not found in graph")
        success = False

    if bayer_org:
        print("   ✅ Bayer correctly classified as Organization")
    else:
        print("   ⚠️  Bayer not found as Organization")

    # Check relationships
    stats = pipeline.get_statistics()
    print(f"\n📈 Graph Statistics:")
    print(f"   • Total nodes: {stats['neo4j_stats']['total_nodes']}")
    print(f"   • Total relationships: {stats['neo4j_stats']['total_relationships']}")

    print("\n" + "=" * 70)
    if success:
        print("  ✅ VALIDATION WORKING - All entities correctly classified!")
    else:
        print("  ❌ VALIDATION FAILED - Some entities misclassified")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
