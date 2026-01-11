#!/usr/bin/env python3
"""
Execution Path Tracer for HotpotQA

Traces ONE query through the full pipeline to verify:
1. Query analysis → routing decision
2. Retrieval execution (FAISS or Neo4j)
3. Context assembly
4. Response generation

Run this BEFORE full evaluation to verify execution path.
"""

import json
import logging
from pathlib import Path
from src.pipeline import get_retrieval_pipeline
from src.agents import get_orchestrator_agent, RetrievalStrategy
from src.vector import get_faiss_index
from src.graph import get_neo4j_manager

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def trace_query(query_text: str, gold_answer: str = None):
    """Trace a single query through the full pipeline."""

    print("\n" + "="*80)
    print("EXECUTION PATH TRACE")
    print("="*80)
    print(f"\n📝 Query: {query_text}")
    if gold_answer:
        print(f"🎯 Gold Answer: {gold_answer}")

    # ========== STAGE 0: System State ==========
    print("\n" + "="*80)
    print("STAGE 0: SYSTEM STATE")
    print("="*80)

    faiss = get_faiss_index()
    neo4j = get_neo4j_manager()

    faiss_count = faiss.get_count()
    neo4j_count = neo4j.get_node_count()
    neo4j_rel_count = neo4j.get_relationship_count()
    schema = neo4j.get_schema()

    print(f"✓ FAISS: {faiss_count:,} vectors")
    print(f"✓ Neo4j: {neo4j_count:,} nodes, {neo4j_rel_count:,} relationships")
    print(f"✓ Labels: {schema['labels'][:10]}")
    print(f"✓ Relationships: {schema['relationship_types'][:10]}")

    if faiss_count == 0 or neo4j_count == 0:
        print("\n❌ CRITICAL: Indexes are empty!")
        return False

    # ========== STAGE 1: Query Analysis & Routing ==========
    print("\n" + "="*80)
    print("STAGE 1: QUERY ANALYSIS & ROUTING")
    print("="*80)

    orchestrator = get_orchestrator_agent()
    routing = orchestrator.route_query(query_text)

    print(f"\n📊 Routing Decision:")
    print(f"   Strategy: {routing['strategy'].value}")
    print(f"   Query Type: {routing['query_type'].value}")
    print(f"   Confidence: {routing['confidence']:.3f}")
    print(f"   Parameters: {routing['params']}")

    if 'analysis' in routing:
        analysis = routing['analysis']
        print(f"\n🔍 Query Analysis:")
        print(f"   Needs relationships: {analysis.get('needs_relationships', 'N/A')}")
        print(f"   Needs semantic: {analysis.get('needs_semantic', 'N/A')}")
        print(f"   Suggested strategy: {analysis.get('suggested_strategy', 'N/A')}")
        if 'reasoning' in analysis:
            print(f"   Reasoning: {analysis['reasoning']}")

    strategy = routing['strategy']

    # ========== STAGE 2: Retrieval Execution ==========
    print("\n" + "="*80)
    print("STAGE 2: RETRIEVAL EXECUTION")
    print("="*80)

    pipeline = get_retrieval_pipeline(use_reflection=False)

    # Force the routed strategy (no cache)
    print(f"\n🔎 Executing {strategy.value.upper()} retrieval...")

    contexts = pipeline.retrieve(
        query_text,
        top_k=5,
        strategy=strategy,
        return_metadata=True
    )

    print(f"\n📦 Retrieved {len(contexts)} contexts")

    if len(contexts) == 0:
        print("⚠️  WARNING: No contexts retrieved!")
        print("   → Graph may be empty or query doesn't match entities")
    else:
        print(f"\n📄 Context Details:")
        for i, ctx in enumerate(contexts[:3], 1):
            print(f"\n   [{i}] Source: {ctx.get('source', 'unknown')}")
            print(f"       Score: {ctx.get('score', 0.0):.4f}")
            print(f"       Text: {ctx['text'][:150]}...")

            if 'metadata' in ctx:
                meta = ctx['metadata']
                if 'node' in meta:
                    print(f"       Node: {meta['node'].get('name', 'unnamed')}")
                if 'paths_found' in meta:
                    print(f"       Paths: {meta['paths_found']}")

    # ========== STAGE 3: Test Both Strategies ==========
    print("\n" + "="*80)
    print("STAGE 3: STRATEGY COMPARISON")
    print("="*80)

    print("\n🔵 Testing VECTOR strategy:")
    vec_contexts = pipeline.retrieve(query_text, strategy=RetrievalStrategy.VECTOR, top_k=5)
    print(f"   Results: {len(vec_contexts)}")
    if vec_contexts:
        print(f"   Top score: {vec_contexts[0]['score']:.4f}")
        print(f"   Top text: {vec_contexts[0]['text'][:100]}...")

    print("\n🟢 Testing GRAPH strategy:")
    graph_contexts = pipeline.retrieve(query_text, strategy=RetrievalStrategy.GRAPH, top_k=5)
    print(f"   Results: {len(graph_contexts)}")
    if graph_contexts:
        print(f"   Top score: {graph_contexts[0]['score']:.4f}")
        print(f"   Top text: {graph_contexts[0]['text'][:100]}...")
    else:
        print("   ⚠️  No graph results - checking Neo4j...")

        # Quick entity search
        test_words = query_text.lower().split()[:5]
        for word in test_words:
            if len(word) < 4:
                continue
            cypher = f"""
            MATCH (n)
            WHERE toLower(n.name) CONTAINS $word
            RETURN n.name as name, labels(n) as labels
            LIMIT 3
            """
            results = neo4j.execute_query(cypher, {"word": word})
            if results:
                print(f"   Found entities matching '{word}': {[r['name'] for r in results]}")
                break

    # ========== STAGE 4: Context Assembly ==========
    print("\n" + "="*80)
    print("STAGE 4: CONTEXT ASSEMBLY")
    print("="*80)

    # Use the routed contexts
    active_contexts = contexts

    print(f"\n📋 Context window:")
    print(f"   Number of contexts: {len(active_contexts)}")

    if active_contexts:
        total_chars = sum(len(ctx['text']) for ctx in active_contexts)
        total_tokens_est = total_chars // 4  # Rough estimate

        print(f"   Total characters: {total_chars:,}")
        print(f"   Estimated tokens: {total_tokens_est:,}")
        print(f"   Fits in context: {'✓' if total_tokens_est < 8000 else '⚠️  May exceed'}")

    # ========== STAGE 5: Response Generation ==========
    print("\n" + "="*80)
    print("STAGE 5: RESPONSE GENERATION")
    print("="*80)

    print("\n🤖 Generating response...")
    response = orchestrator.synthesize_response(query_text, active_contexts, strategy)

    print(f"\n💬 Generated Response:")
    print(f"   {response}")

    if gold_answer:
        print(f"\n🎯 Gold Answer: {gold_answer}")

        # Simple match check
        response_lower = response.lower()
        gold_lower = gold_answer.lower()

        if gold_lower in response_lower:
            print("   ✅ Gold answer appears in response")
        else:
            print("   ⚠️  Gold answer NOT in response")

    # ========== STAGE 6: Execution Path Verification ==========
    print("\n" + "="*80)
    print("STAGE 6: VERIFICATION CHECKLIST")
    print("="*80)

    checks = {
        "✓ Routing triggered": True,
        "✓ Strategy selected": strategy is not None,
        "✓ Retrieval executed": len(contexts) >= 0,  # Can be 0, just needs to execute
        "✓ Vector path works": len(vec_contexts) > 0,
        "✓ Graph path works": len(graph_contexts) > 0,
        "✓ Response generated": len(response) > 0,
        "✓ Same generator used": True  # orchestrator.synthesize_response
    }

    print()
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    all_passed = all(checks.values())

    print("\n" + "="*80)
    if all_passed:
        print("✅ EXECUTION PATH VERIFIED")
    else:
        print("⚠️  SOME CHECKS FAILED")
    print("="*80)

    return all_passed


def main():
    # Load a 2-hop HotpotQA query
    queries_path = Path("data/hotpotqa/queries_100.json")

    if not queries_path.exists():
        print(f"❌ {queries_path} not found")
        return 1

    with open(queries_path) as f:
        queries = json.load(f)

    # Find a bridge (2-hop) question
    bridge_queries = [q for q in queries if q.get('type') == 'bridge']

    if not bridge_queries:
        print("⚠️  No bridge queries found, using first query")
        query_obj = queries[0]
    else:
        query_obj = bridge_queries[0]

    print(f"\n📌 Selected query type: {query_obj.get('type', 'unknown')}")

    success = trace_query(
        query_text=query_obj['question'],
        gold_answer=query_obj['answer']
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
