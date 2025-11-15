#!/usr/bin/env python3
"""
Comprehensive Graph-Only Retrieval Test

Tests graph traversal with:
- Simple relational queries (1-hop)
- Multi-hop queries (2-hop paths)
- Different relationship types
- Entity existence checks
- Performance comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.graph import get_neo4j_manager
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
    """Run comprehensive graph-only test."""
    print_header("COMPREHENSIVE GRAPH-ONLY TEST")
    print("\nTesting Neo4j graph traversal with expanded query set.\n")

    # ========================================================================
    # STAGE 1: FRESH DATA INGESTION
    # ========================================================================
    print_section("Stage 1: Fresh Data Ingestion")

    print("\n🗑️  Clearing Neo4j database...")
    neo4j = get_neo4j_manager()
    neo4j.clear_database()
    print("   ✓ Database cleared")

    # Expanded dataset with more relationships
    documents = [
        # Medical domain
        """Aspirin is a medication used to reduce pain, fever, and inflammation.
        It is manufactured by Bayer, a German pharmaceutical company.
        Aspirin treats headaches, muscle pain, and arthritis.""",

        """Metformin is a medication commonly prescribed to treat type 2 diabetes.
        It helps control blood sugar levels. Metformin is produced by several
        pharmaceutical companies including Merck.""",

        """Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to
        treat pain, fever, and inflammation. It is manufactured by various
        companies including Pfizer. Ibuprofen treats headaches and arthritis.""",

        # Technology domain
        """Tesla is an American electric vehicle company founded by Elon Musk.
        The company is headquartered in Austin, Texas. Tesla manufactures
        electric cars and battery systems. Elon Musk serves as CEO.""",

        """Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak,
        and Ronald Wayne. Tim Cook is the current CEO of Apple. The company is
        headquartered in Cupertino, California. Apple manufactures the iPhone,
        iPad, and Mac computers.""",

        # Academic domain
        """MIT (Massachusetts Institute of Technology) is a research university
        located in Cambridge, Massachusetts. The university has a strong
        Computer Science department. Many researchers work at MIT's CSAIL
        laboratory studying artificial intelligence and robotics.""",
    ]

    print(f"\n📚 Ingesting {len(documents)} documents...")
    print("   Expected relationships:")
    print("   • Drug->TREATS->Disease")
    print("   • Organization->MANUFACTURES->Product")
    print("   • Person->FOUNDED->Organization")
    print("   • Organization->HEADQUARTERED_IN->Location")

    schema_path = Path("data/processed/schema.json")
    pipeline = get_ingestion_pipeline(schema_path=schema_path)

    result = pipeline.ingest_documents(
        documents,
        infer_schema=True,
        enrich_metadata=True
    )

    print(f"\n✅ Ingestion complete in {result['duration_seconds']:.2f}s")
    print(f"   • Entities: {result['entities_extracted']}")
    print(f"   • Relations: {result['relations_extracted']}")
    print(f"   • Nodes: {result['nodes_created']}")
    print(f"   • Edges: {result['edges_created']}")

    # Verify graph state
    stats = pipeline.get_statistics()
    print(f"\n📊 Graph State:")
    print(f"   • Total nodes: {stats['neo4j_stats']['total_nodes']}")
    print(f"   • Total relationships: {stats['neo4j_stats']['total_relationships']}")

    if stats['neo4j_stats']['total_relationships'] == 0:
        print("\n❌ ERROR: No relationships in graph! Cannot test.")
        return

    # ========================================================================
    # STAGE 2: COMPREHENSIVE GRAPH QUERIES
    # ========================================================================
    print_section("Stage 2: Comprehensive Graph Queries")

    retrieval_pipeline = get_retrieval_pipeline(use_reflection=True)

    # Comprehensive test queries
    test_queries = [
        # === Simple Relational Queries (1-hop) ===
        {
            "category": "Simple Relational",
            "query": "What medications treat diabetes?",
            "description": "Drug->TREATS->Disease (1-hop)",
            "expected": "Metformin",
        },
        {
            "category": "Simple Relational",
            "query": "What does Aspirin treat?",
            "description": "Drug->TREATS->Disease (1-hop)",
            "expected": "headaches, pain, arthritis",
        },
        {
            "category": "Simple Relational",
            "query": "What does Ibuprofen treat?",
            "description": "Drug->TREATS->Disease (1-hop)",
            "expected": "headaches, arthritis",
        },
        {
            "category": "Simple Relational",
            "query": "Who manufactures Aspirin?",
            "description": "Organization->MANUFACTURES->Drug (1-hop)",
            "expected": "Bayer",
        },
        {
            "category": "Simple Relational",
            "query": "Who manufactures Tesla cars?",
            "description": "Organization->MANUFACTURES->Product (1-hop)",
            "expected": "Tesla",
        },

        # === Entity Existence Queries ===
        {
            "category": "Entity Existence",
            "query": "Tell me about Elon Musk",
            "description": "Person entity retrieval",
            "expected": "CEO, Tesla, founder",
        },
        {
            "category": "Entity Existence",
            "query": "What is MIT?",
            "description": "Organization entity retrieval",
            "expected": "university, research, Cambridge",
        },

        # === Reverse Relationship Queries ===
        {
            "category": "Reverse Relational",
            "query": "Which drugs are manufactured by Bayer?",
            "description": "Reverse: Drug<-MANUFACTURES-Organization",
            "expected": "Aspirin",
        },
        {
            "category": "Reverse Relational",
            "query": "Which diseases are treated by Metformin?",
            "description": "Reverse: Disease<-TREATS-Drug",
            "expected": "diabetes",
        },

        # === Multi-hop Queries (if relationships exist) ===
        {
            "category": "Multi-hop",
            "query": "Where is Tesla headquartered?",
            "description": "Organization->HEADQUARTERED_IN->Location (1-hop)",
            "expected": "Austin, Texas",
        },
        {
            "category": "Multi-hop",
            "query": "Who founded Apple?",
            "description": "Person->FOUNDED->Organization (1-hop)",
            "expected": "Steve Jobs, Steve Wozniak",
        },

        # === Complex/Comparative Queries ===
        {
            "category": "Complex",
            "query": "What medications treat headaches?",
            "description": "Multiple Drug->TREATS->Disease matches",
            "expected": "Aspirin, Ibuprofen",
        },
    ]

    print(f"\n🔍 Testing {len(test_queries)} graph queries across {len(set(q['category'] for q in test_queries))} categories...\n")

    results_by_category = {}
    all_results = []

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        category = test_case["category"]
        description = test_case["description"]

        if category not in results_by_category:
            results_by_category[category] = []

        print(f"\n{'─' * 70}")
        print(f"Query {i}/{len(test_queries)}: \"{query}\"")
        print(f"Category: {category}")
        print(f"Description: {description}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'─' * 70}")

        # Force GRAPH strategy
        result = retrieval_pipeline.query(
            query,
            top_k=5,
            strategy=RetrievalStrategy.GRAPH,
            evaluate=True
        )

        # Analyze results
        success = result['num_contexts'] > 0
        score = result.get('metrics', {}).get('overall', 0.0) if success else 0.0

        print(f"\n{'✅' if success else '❌'} Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"   • Contexts retrieved: {result['num_contexts']}")

        if success:
            print(f"   • Overall score: {score:.3f}")
            print(f"\n💬 Response Preview:")
            print(f"   {result['response'][:200]}...")

            if result['context']:
                print(f"\n📄 Top Context:")
                ctx = result['context'][0]
                print(f"   Source: {ctx['source']}, Score: {ctx['score']:.3f}")
                print(f"   {ctx['text'][:150]}...")
        else:
            print(f"\n❌ No contexts retrieved - query failed")

        # Record result
        result_summary = {
            "query": query,
            "category": category,
            "success": success,
            "score": score,
            "num_contexts": result['num_contexts']
        }
        all_results.append(result_summary)
        results_by_category[category].append(result_summary)

    # ========================================================================
    # STAGE 3: CATEGORY ANALYSIS
    # ========================================================================
    print_section("Stage 3: Category Performance Analysis")

    print(f"\n📊 Results by Category:\n")

    for category in sorted(results_by_category.keys()):
        cat_results = results_by_category[category]
        successes = sum(1 for r in cat_results if r['success'])
        total = len(cat_results)
        avg_score = sum(r['score'] for r in cat_results) / total

        print(f"  {category}:")
        print(f"    • Success: {successes}/{total} ({successes/total*100:.1f}%)")
        print(f"    • Avg Score: {avg_score:.3f}")

        for r in cat_results:
            status = "✅" if r['success'] else "❌"
            print(f"      {status} {r['query']}: {r['score']:.3f}")
        print()

    # ========================================================================
    # STAGE 4: OVERALL SUMMARY
    # ========================================================================
    print_section("Stage 4: Overall Summary")

    total_queries = len(all_results)
    total_successes = sum(1 for r in all_results if r['success'])
    total_failures = total_queries - total_successes
    avg_score = sum(r['score'] for r in all_results) / total_queries
    success_rate = total_successes / total_queries * 100

    print(f"\n📊 Overall Performance:")
    print(f"   • Total queries: {total_queries}")
    print(f"   • Successes: {total_successes} ({success_rate:.1f}%)")
    print(f"   • Failures: {total_failures} ({100-success_rate:.1f}%)")
    print(f"   • Average score: {avg_score:.3f}")

    print(f"\n📈 Graph Statistics:")
    print(f"   • Nodes: {stats['neo4j_stats']['total_nodes']}")
    print(f"   • Relationships: {stats['neo4j_stats']['total_relationships']}")
    print(f"   • Entity types: {stats['schema_summary']['entity_types']}")
    print(f"   • Relation types: {stats['schema_summary']['relation_types']}")

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print_header("TEST COMPLETE")

    if success_rate >= 80:
        print("\n🎉 EXCELLENT: Graph retrieval performing very well!")
    elif success_rate >= 50:
        print("\n✅ GOOD: Graph retrieval working but has room for improvement")
    elif success_rate >= 30:
        print("\n⚠️  MODERATE: Graph retrieval partially working")
    else:
        print("\n❌ POOR: Graph retrieval needs significant improvement")

    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    print(f"📊 Average Score: {avg_score:.3f}")

    # Identify problem areas
    failed_queries = [r for r in all_results if not r['success']]
    if failed_queries:
        print(f"\n🔍 Failed Queries ({len(failed_queries)}):")
        for r in failed_queries:
            print(f"   ❌ [{r['category']}] {r['query']}")

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
