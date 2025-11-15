#!/usr/bin/env python3
"""
Graph-Only Retrieval Test

This script tests the graph retrieval strategy exclusively to verify:
1. Relationships are being created correctly
2. Graph traversal can find entities
3. Multi-hop queries work
4. Relationship-based queries succeed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
from src.agents import RetrievalStrategy


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a formatted section title."""
    print(f"\n{'─' * 70}")
    print(f"📌 {title}")
    print(f"{'─' * 70}")


def main():
    """Run graph-only retrieval test."""
    print_header("GRAPH-ONLY RETRIEVAL TEST")
    print("\nTesting Neo4j graph traversal and relationship queries.\n")

    # ========================================================================
    # STAGE 1: VERIFY DATA INGESTION
    # ========================================================================
    print_section("Stage 1: Verify Graph Data")

    # Initialize ingestion pipeline to check stats
    schema_path = Path("data/processed/schema.json")
    ingestion_pipeline = get_ingestion_pipeline(schema_path=schema_path)

    stats = ingestion_pipeline.get_statistics()
    print(f"\n📊 Current Graph State:")
    print(f"   • Total nodes: {stats['neo4j_stats']['total_nodes']}")
    print(f"   • Total relationships: {stats['neo4j_stats']['total_relationships']}")
    print(f"   • Entity types: {stats['schema_summary']['entity_types']}")
    print(f"   • Relation types: {stats['schema_summary']['relation_types']}")

    if stats['neo4j_stats']['total_relationships'] == 0:
        print("\n⚠️  WARNING: No relationships found in graph!")
        print("   Re-running ingestion to create relationships...")

        # Re-ingest with small dataset
        documents = [
            """Aspirin is a medication used to reduce pain, fever, and inflammation.
            It is manufactured by Bayer. Aspirin treats headaches and muscle pain.""",

            """Metformin is a medication commonly prescribed to treat type 2 diabetes.
            It helps control blood sugar levels.""",
        ]

        result = ingestion_pipeline.ingest_documents(
            documents,
            infer_schema=True,
            enrich_metadata=True
        )

        print(f"\n✅ Re-ingestion complete:")
        print(f"   • Entities: {result['entities_extracted']}")
        print(f"   • Relations: {result['relations_extracted']}")
        print(f"   • Nodes created: {result['nodes_created']}")
        print(f"   • Edges created: {result['edges_created']}")

        # Get updated stats
        stats = ingestion_pipeline.get_statistics()
        print(f"\n📊 Updated Graph State:")
        print(f"   • Total nodes: {stats['neo4j_stats']['total_nodes']}")
        print(f"   • Total relationships: {stats['neo4j_stats']['total_relationships']}")

    # ========================================================================
    # STAGE 2: GRAPH-ONLY QUERIES
    # ========================================================================
    print_section("Stage 2: Graph-Only Queries")

    # Initialize retrieval pipeline
    retrieval_pipeline = get_retrieval_pipeline(use_reflection=True)

    # Test queries that MUST use graph traversal
    graph_queries = [
        {
            "query": "What medications treat diabetes?",
            "description": "Relational query requiring TREATS relationship traversal",
            "expected": "Should find Metformin->TREATS->diabetes"
        },
        {
            "query": "What does Aspirin treat?",
            "description": "Relational query for drug effects",
            "expected": "Should find Aspirin->TREATS->headaches/pain"
        },
        {
            "query": "Who manufactures Aspirin?",
            "description": "Relational query for manufacturer",
            "expected": "Should find Bayer->MANUFACTURES->Aspirin"
        },
    ]

    print(f"\n🔍 Testing {len(graph_queries)} graph-only queries...\n")

    results_summary = []

    for i, test_case in enumerate(graph_queries, 1):
        query = test_case["query"]
        description = test_case["description"]

        print(f"\n{'─' * 70}")
        print(f"Query {i}: \"{query}\"")
        print(f"Description: {description}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'─' * 70}")

        # Force GRAPH strategy
        result = retrieval_pipeline.query(
            query,
            top_k=5,
            strategy=RetrievalStrategy.GRAPH,  # Force graph-only
            evaluate=True
        )

        # Display results
        print(f"\n💬 Response:")
        print(f"   {result['response'][:400]}...")

        print(f"\n📊 Retrieval Stats:")
        print(f"   • Strategy used: {result['strategy']}")
        print(f"   • Contexts retrieved: {result['num_contexts']}")

        if result.get('metrics'):
            metrics = result['metrics']
            print(f"\n📈 RAGAS Metrics:")
            print(f"   • Faithfulness:      {metrics.get('faithfulness', 0):.3f}")
            print(f"   • Answer Relevancy:  {metrics.get('answer_relevancy', 0):.3f}")
            print(f"   • Context Precision: {metrics.get('context_precision', 0):.3f}")
            print(f"   • Overall Score:     {metrics.get('overall', 0):.3f}")

            results_summary.append({
                "query": query,
                "success": result['num_contexts'] > 0,
                "score": metrics.get('overall', 0)
            })
        else:
            results_summary.append({
                "query": query,
                "success": result['num_contexts'] > 0,
                "score": 0.0
            })

        if result['context']:
            print(f"\n📄 Retrieved Contexts:")
            for j, ctx in enumerate(result['context'][:3], 1):
                print(f"   [{j}] Source: {ctx['source']}, Score: {ctx['score']:.3f}")
                print(f"       {ctx['text'][:150]}...")
        else:
            print(f"\n❌ NO CONTEXTS RETRIEVED - Graph query failed!")

    # ========================================================================
    # STAGE 3: RESULTS SUMMARY
    # ========================================================================
    print_section("Stage 3: Test Results Summary")

    success_count = sum(1 for r in results_summary if r['success'])
    avg_score = sum(r['score'] for r in results_summary) / len(results_summary)

    print(f"\n📊 Overall Performance:")
    print(f"   • Queries tested: {len(graph_queries)}")
    print(f"   • Successful: {success_count}/{len(graph_queries)} ({success_count/len(graph_queries)*100:.1f}%)")
    print(f"   • Average score: {avg_score:.3f}")

    print(f"\n📋 Individual Results:")
    for r in results_summary:
        status = "✅" if r['success'] else "❌"
        print(f"   {status} {r['query']}: {r['score']:.3f}")

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print_header("TEST COMPLETE")

    if success_count == len(graph_queries):
        print("\n✅ ALL GRAPH QUERIES SUCCEEDED!")
        print("   Graph retrieval is working correctly.")
    elif success_count > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {success_count}/{len(graph_queries)} queries succeeded")
        print("   Graph retrieval is partially working.")
    else:
        print("\n❌ ALL GRAPH QUERIES FAILED!")
        print("   Graph retrieval needs debugging.")

    print(f"\n🎯 Key Findings:")
    if stats['neo4j_stats']['total_relationships'] > 0:
        print(f"   • Graph has {stats['neo4j_stats']['total_relationships']} relationships")
        print(f"   • Success rate: {success_count/len(graph_queries)*100:.1f}%")
    else:
        print("   • No relationships in graph - relationship creation failed")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
