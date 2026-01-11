#!/usr/bin/env python3
"""
Pre-evaluation validation script.

Run this BEFORE HotpotQA evaluation to ensure system is ready.
Validates that:
1. Neo4j has indexed data
2. FAISS has vectors
3. Query routing works
4. Both retrieval strategies return results

Usage:
    python validate_system.py
"""

import sys
from pathlib import Path
from src.graph import get_neo4j_manager
from src.vector import get_faiss_index
from src.pipeline import get_retrieval_pipeline
from src.agents import RetrievalStrategy


def validate_system():
    """Run full system validation."""
    print("=" * 70)
    print("SYSTEM VALIDATION CHECK")
    print("=" * 70)

    # 1. Neo4j validation
    print("\n1. Checking Neo4j...")
    try:
        neo4j = get_neo4j_manager()
        node_count = neo4j.get_node_count()
        rel_count = neo4j.get_relationship_count()
        schema = neo4j.get_schema()

        print(f"   Nodes: {node_count:,}")
        print(f"   Relationships: {rel_count:,}")
        print(f"   Labels: {schema['labels'][:10]}")
        print(f"   Relationship types: {schema['relationship_types'][:10]}")

        if node_count == 0:
            print("   ❌ CRITICAL: Neo4j is empty!")
            print("   → Run: python ingest_hotpotqa.py")
            return False

        if rel_count == 0:
            print("   ⚠️  WARNING: No relationships in Neo4j!")
            print("   → Graph retrieval may not work effectively")

        print("   ✅ Neo4j OK")

    except Exception as e:
        print(f"   ❌ CRITICAL: Neo4j connection failed: {e}")
        print("   → Make sure Neo4j is running: ./scripts/start_neo4j.sh")
        return False

    # 2. FAISS validation
    print("\n2. Checking FAISS...")
    try:
        faiss = get_faiss_index()
        vector_count = faiss.get_count()

        print(f"   Vectors: {vector_count:,}")
        print(f"   Dimension: {faiss.dimension}")
        print(f"   Model: {faiss.embedding_config.model_name}")

        if vector_count == 0:
            print("   ❌ CRITICAL: FAISS is empty!")
            print("   → Run: python ingest_hotpotqa.py")
            return False

        print("   ✅ FAISS OK")

    except Exception as e:
        print(f"   ❌ CRITICAL: FAISS initialization failed: {e}")
        return False

    # 3. Test query routing
    print("\n3. Testing query routing...")
    try:
        pipeline = get_retrieval_pipeline(use_reflection=False)

        test_query = "What is the capital of France?"
        result = pipeline.query(test_query, use_cache=False, evaluate=False)

        print(f"   Query: {test_query}")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Contexts: {result['num_contexts']}")
        print(f"   Response preview: {result['response'][:80]}...")

        if result['num_contexts'] == 0:
            print("   ⚠️  WARNING: No contexts retrieved for test query")
            print("   → This may indicate retrieval issues")
        else:
            print("   ✅ Routing OK")

    except Exception as e:
        print(f"   ❌ CRITICAL: Query pipeline failed: {e}")
        return False

    # 4. Test vector strategy
    print("\n4. Testing vector strategy...")
    try:
        vec_result = pipeline.retrieve(test_query, strategy=RetrievalStrategy.VECTOR)
        print(f"   Vector results: {len(vec_result)}")

        if len(vec_result) == 0:
            print("   ❌ CRITICAL: Vector retrieval returned 0 results")
            return False

        print(f"   Top result score: {vec_result[0]['score']:.4f}")
        print("   ✅ Vector OK")

    except Exception as e:
        print(f"   ❌ CRITICAL: Vector retrieval failed: {e}")
        return False

    # 5. Test graph strategy
    print("\n5. Testing graph strategy...")
    try:
        graph_result = pipeline.retrieve(test_query, strategy=RetrievalStrategy.GRAPH)
        print(f"   Graph results: {len(graph_result)}")

        if len(graph_result) == 0:
            print("   ⚠️  WARNING: Graph retrieval returned 0 results")
            print("   → This may be expected if test query doesn't match entities")
            print("   → Graph retrieval will be tested on HotpotQA queries")
        else:
            print(f"   Top result score: {graph_result[0]['score']:.4f}")
            print("   ✅ Graph OK")

    except Exception as e:
        print(f"   ⚠️  WARNING: Graph retrieval error: {e}")

    # 6. Check HotpotQA data
    print("\n6. Checking HotpotQA data...")
    corpus_path = Path("data/hotpotqa/corpus.json")
    queries_path = Path("data/hotpotqa/queries_100.json")

    if not corpus_path.exists():
        print(f"   ⚠️  WARNING: {corpus_path} not found")
        print("   → HotpotQA corpus missing")
    else:
        print(f"   ✅ Corpus found: {corpus_path}")

    if not queries_path.exists():
        print(f"   ❌ CRITICAL: {queries_path} not found")
        print("   → Cannot run HotpotQA evaluation")
        return False
    else:
        import json
        with open(queries_path) as f:
            queries = json.load(f)
        print(f"   ✅ Queries found: {len(queries)} queries")

    # Final verdict
    print("\n" + "=" * 70)
    print("✅ SYSTEM VALIDATION PASSED")
    print("=" * 70)
    print("\nYou can now run HotpotQA evaluation:")
    print("  python evaluate_hotpotqa_full.py")
    print()
    return True


if __name__ == "__main__":
    try:
        success = validate_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
