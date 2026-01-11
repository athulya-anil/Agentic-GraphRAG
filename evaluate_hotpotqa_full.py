#!/usr/bin/env python3
"""
Full Agentic GraphRAG Evaluation on HotpotQA

Uses the complete retrieval pipeline with:
- Automatic query routing (OrchestratorAgent)
- Vector retrieval (FAISS)
- Graph traversal (Neo4j)
- Failure-aware routing

CRITICAL: Cache is DISABLED to ensure routing is exercised.
"""

import json
import time
import re
import string
from pathlib import Path
from tqdm import tqdm
from src.pipeline import get_retrieval_pipeline
from src.vector import get_faiss_index
from src.graph import get_neo4j_manager
from src.agents import RetrievalStrategy
from src.utils.llm_client import get_llm_client


def extract_answer(llm_response: str) -> str:
    """Extract answer from LLM response."""
    answer = llm_response.strip().split('\n')[0]

    # Remove common prefixes
    prefixes = [
        r'the answer is:?\s*',
        r'answer:?\s*',
        r'based on the (context|passage|document),?\s*',
        r'according to the (text|passage|document|context),?\s*',
        r'the (text|passage|document) (says|states) that\s*',
        r'from the (context|passage|document),?\s*',
    ]

    for prefix in prefixes:
        answer = re.sub(f'^{prefix}', '', answer, flags=re.IGNORECASE)

    # Check for refusal patterns
    refusal_patterns = [
        r"i (don't|do not|dont) know",
        r"i (can't|cannot|can not) (answer|determine|find)",
        r"(there is )?no (clear|sufficient|enough) (answer|information)",
        r"not enough information",
        r"unable to (answer|determine)",
        r"cannot be determined",
        r"insufficient (information|context)",
    ]

    answer_lower = answer.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, answer_lower):
            return ""

    return answer.strip()


def normalize_answer(s: str) -> str:
    """Normalize answer for EM (HotpotQA standard)."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def compute_em(prediction: str, gold: str) -> int:
    """Compute Exact Match."""
    if not prediction or not prediction.strip():
        return 0
    return int(normalize_answer(prediction) == normalize_answer(gold))


def main():
    print("\n" + "=" * 70)
    print("AGENTIC GRAPHRAG EVALUATION ON HOTPOTQA")
    print("=" * 70)

    # Validate system first
    print("\nValidating system...")
    faiss = get_faiss_index()
    neo4j = get_neo4j_manager()

    faiss_count = faiss.get_count()
    neo4j_count = neo4j.get_node_count()

    print(f"✓ FAISS: {faiss_count:,} vectors")
    print(f"✓ Neo4j: {neo4j_count:,} nodes, {neo4j.get_relationship_count():,} relationships")

    if faiss_count == 0 or neo4j_count == 0:
        print("\n❌ CRITICAL: Indexes are empty!")
        print("→ Run: python validate_system.py")
        return 1

    # Load queries
    queries_path = Path("data/hotpotqa/queries_100.json")
    output_path = Path("data/hotpotqa/results_full_graphrag.json")

    with open(queries_path, 'r') as f:
        queries = json.load(f)
    print(f"✓ Loaded {len(queries)} queries")

    # Initialize pipeline
    print("\nInitializing retrieval pipeline...")
    pipeline = get_retrieval_pipeline(use_reflection=False)
    print("✓ Pipeline ready")

    # Run evaluation
    results = []
    failures = []
    strategy_counts = {"vector": 0, "graph": 0}
    start_time = time.time()

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION (CACHE DISABLED)")
    print("=" * 70)

    for query_obj in tqdm(queries, desc="Evaluating"):
        query_text = query_obj['question']
        gold_answer = query_obj['answer']

        # CRITICAL: use_cache=False ensures routing is exercised
        result = pipeline.query(
            query_text,
            use_cache=False,  # ← FIX 1: Disable cache
            evaluate=False     # Don't compute RAGAS (too slow)
        )

        # Track strategy usage
        strategy = result.get('strategy', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Track empty context cases
        if result['num_contexts'] == 0:
            failures.append({
                'query_id': query_obj['query_id'],
                'question': query_text,
                'strategy': strategy,
                'reason': 'no_contexts'
            })

        # Extract answer from response
        predicted_answer = extract_answer(result['response'])

        # Compute EM
        em = compute_em(predicted_answer, gold_answer)

        results.append({
            'query_id': query_obj['query_id'],
            'question': query_text,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'em': em,
            'type': query_obj['type'],
            'strategy': strategy,
            'num_contexts': result['num_contexts']
        })

    duration = time.time() - start_time

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Compute metrics
    overall_em = sum(r['em'] for r in results) / len(results)

    comparison_results = [r for r in results if r['type'] == 'comparison']
    bridge_results = [r for r in results if r['type'] == 'bridge']

    comparison_em = sum(r['em'] for r in comparison_results) / len(comparison_results) if comparison_results else 0
    bridge_em = sum(r['em'] for r in bridge_results) / len(bridge_results) if bridge_results else 0

    # Strategy-specific metrics
    vector_results = [r for r in results if r['strategy'] == 'vector']
    graph_results = [r for r in results if r['strategy'] == 'graph']

    vector_em = sum(r['em'] for r in vector_results) / len(vector_results) if vector_results else 0
    graph_em = sum(r['em'] for r in graph_results) / len(graph_results) if graph_results else 0

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n📊 Overall Performance:")
    print(f"   EM: {overall_em:.3f} ({overall_em * 100:.1f}%)")
    print(f"   Correct: {sum(r['em'] for r in results)}/{len(results)}")

    print(f"\n📊 By Question Type:")
    print(f"   Comparison EM: {comparison_em:.3f} ({len(comparison_results)} queries)")
    print(f"   Bridge EM: {bridge_em:.3f} ({len(bridge_results)} queries)")

    print(f"\n📊 By Retrieval Strategy:")
    print(f"   Vector: {vector_em:.3f} ({len(vector_results)} queries, {strategy_counts.get('vector', 0)} routed)")
    print(f"   Graph: {graph_em:.3f} ({len(graph_results)} queries, {strategy_counts.get('graph', 0)} routed)")

    print(f"\n⚠️  Empty Context Cases: {len(failures)}/{len(results)}")
    if failures:
        print(f"   Vector failures: {sum(1 for f in failures if f['strategy'] == 'vector')}")
        print(f"   Graph failures: {sum(1 for f in failures if f['strategy'] == 'graph')}")

    print(f"\n⏱️  Duration: {duration:.1f}s ({duration / 60:.1f} minutes)")
    print(f"   Avg per query: {duration / len(results):.2f}s")

    print(f"\n✓ Results saved to: {output_path}")

    # Save summary
    summary = {
        "overall_em": overall_em,
        "comparison_em": comparison_em,
        "bridge_em": bridge_em,
        "vector_em": vector_em,
        "graph_em": graph_em,
        "strategy_counts": strategy_counts,
        "empty_contexts": len(failures),
        "total_queries": len(results),
        "duration_seconds": duration
    }

    summary_path = Path("data/hotpotqa/summary_full_graphrag.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
