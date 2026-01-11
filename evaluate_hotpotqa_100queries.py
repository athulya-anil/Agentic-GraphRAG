#!/usr/bin/env python3
"""
HotpotQA 100-Query Evaluation (Shared Graph Approach)

Evaluates 100 queries against a pre-built shared graph containing all required documents.

Key differences from KG²RAG:
- KG²RAG: Rebuilds index per query (10 docs each) - simple vector index, fast
- Us: Single shared graph with all docs - complex graph extraction, expensive to rebuild
- Both: 100% coverage guaranteed (all needed docs present)

This approach is more efficient for our architecture while maintaining fair evaluation.
"""

import json
import time
import re
import string
from pathlib import Path
from typing import Dict, Any
from src.pipeline import get_retrieval_pipeline


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (HotpotQA official)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(prediction: str, gold: str) -> int:
    """Compute Exact Match."""
    if not prediction or not prediction.strip():
        return 0
    return int(normalize_answer(prediction) == normalize_answer(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    if not prediction or not prediction.strip():
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common_tokens = set(pred_tokens) & set(gold_tokens)
    num_common = len(common_tokens)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def main():
    print("\n" + "="*70)
    print("HOTPOTQA 100-QUERY EVALUATION")
    print("Shared Graph Approach (Efficient for Our Architecture)")
    print("="*70)

    # Load query metadata
    metadata_path = Path("data/hotpotqa/queries_100_metadata.json")

    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found!")
        print(f"   Please run ingest_hotpotqa_100queries.py first")
        return 1

    with open(metadata_path, 'r') as f:
        queries = json.load(f)

    print(f"\n📊 Evaluation Setup:")
    print(f"  Dataset: HotpotQA dev distractor")
    print(f"  Queries: {len(queries)}")
    print(f"  Approach: Shared graph (all docs pre-ingested)")
    print(f"  Coverage: 100% guaranteed (all needed docs present)")

    # Query type distribution
    types = {}
    for q in queries:
        t = q['type']
        types[t] = types.get(t, 0) + 1

    print(f"\n  Query types:")
    for t, count in types.items():
        print(f"    {t}: {count} ({count/len(queries)*100:.1f}%)")

    # Verify system state
    print(f"\n🔍 Checking system state...")
    from src.graph.neo4j_manager import get_neo4j_manager
    from src.vector.faiss_index import get_faiss_index

    neo4j = get_neo4j_manager()
    faiss = get_faiss_index()

    neo4j_nodes = neo4j.get_node_count()
    neo4j_rels = neo4j.get_relationship_count()
    faiss_vectors = faiss.get_count()

    print(f"  Neo4j nodes: {neo4j_nodes}")
    print(f"  Neo4j relationships: {neo4j_rels}")
    print(f"  FAISS vectors: {faiss_vectors}")

    if neo4j_nodes == 0:
        print(f"\n❌ Error: Neo4j is empty! Run ingest_hotpotqa_100queries.py first.")
        return 1

    if faiss_vectors == 0:
        print(f"\n❌ Error: FAISS is empty! Run ingest_hotpotqa_100queries.py first.")
        return 1

    # Estimate time
    estimated_time = len(queries) * 1.5  # ~1.5 sec per query
    print(f"\n⏱️  Estimated time: {estimated_time/60:.1f} minutes")

    input(f"\nPress Enter to start evaluation...")

    # Initialize pipeline
    print(f"\n🔧 Initializing retrieval pipeline...")
    retrieval_pipeline = get_retrieval_pipeline()

    results = []
    metrics = {
        'em_scores': [],
        'f1_scores': [],
        'strategy_counts': {},
        'empty_contexts': 0,
        'query_times': [],
    }

    start_time = time.time()

    # Evaluate each query
    for i, query_obj in enumerate(queries, 1):
        question = query_obj['question']
        gold_answer = query_obj['answer']
        query_id = query_obj['_id']
        query_type = query_obj['type']

        print(f"\n{'='*70}")
        print(f"[{i}/{len(queries)}] {question[:60]}...")
        print(f"{'='*70}")

        # Query
        print(f"  🔍 Querying...")
        query_start = time.time()

        result = retrieval_pipeline.query(
            question,
            use_cache=False,
            evaluate=False
        )

        query_time = time.time() - query_start
        metrics['query_times'].append(query_time)

        # Extract results
        predicted_answer = result.get('response', '') or ''
        predicted_answer = predicted_answer.strip()
        strategy = result.get('strategy', 'unknown')
        num_contexts = result.get('num_contexts', 0)

        # Compute metrics
        em = compute_em(predicted_answer, gold_answer)
        f1 = compute_f1(predicted_answer, gold_answer)

        metrics['em_scores'].append(em)
        metrics['f1_scores'].append(f1)
        metrics['strategy_counts'][strategy] = metrics['strategy_counts'].get(strategy, 0) + 1

        if num_contexts == 0:
            metrics['empty_contexts'] += 1

        # Store result
        results.append({
            'query_id': query_id,
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'em': em,
            'f1': f1,
            'type': query_type,
            'strategy': strategy,
            'num_contexts': num_contexts,
            'query_time': query_time,
        })

        # Display progress
        print(f"  📊 Results:")
        print(f"     Strategy: {strategy} | Contexts: {num_contexts}")
        print(f"     Gold: {gold_answer}")
        print(f"     Pred: {predicted_answer[:80]}...")
        print(f"     EM: {em} | F1: {f1:.3f} | Time: {query_time:.1f}s")

        # Running average
        current_em = sum(metrics['em_scores']) / len(metrics['em_scores']) * 100
        current_f1 = sum(metrics['f1_scores']) / len(metrics['f1_scores']) * 100
        print(f"\n  📈 Running Avg: EM={current_em:.2f}% ({sum(metrics['em_scores'])}/{i}), F1={current_f1:.2f}%")

    duration = time.time() - start_time

    # Aggregate metrics
    print(f"\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    overall_em = sum(metrics['em_scores']) / len(metrics['em_scores']) * 100
    overall_f1 = sum(metrics['f1_scores']) / len(metrics['f1_scores']) * 100

    # Metrics by type
    bridge_results = [r for r in results if r['type'] == 'bridge']
    comparison_results = [r for r in results if r['type'] == 'comparison']

    bridge_em = sum(r['em'] for r in bridge_results) / len(bridge_results) * 100 if bridge_results else 0
    comparison_em = sum(r['em'] for r in comparison_results) / len(comparison_results) * 100 if comparison_results else 0

    bridge_f1 = sum(r['f1'] for r in bridge_results) / len(bridge_results) * 100 if bridge_results else 0
    comparison_f1 = sum(r['f1'] for r in comparison_results) / len(comparison_results) * 100 if comparison_results else 0

    # Metrics by strategy
    vector_results = [r for r in results if r['strategy'] == 'vector']
    graph_results = [r for r in results if r['strategy'] == 'graph']

    vector_em = sum(r['em'] for r in vector_results) / len(vector_results) * 100 if vector_results else 0
    graph_em = sum(r['em'] for r in graph_results) / len(graph_results) * 100 if graph_results else 0

    print(f"\n📊 OVERALL METRICS:")
    print(f"  EM: {overall_em:.2f}%")
    print(f"  F1: {overall_f1:.2f}%")

    print(f"\n📊 METRICS BY QUERY TYPE:")
    print(f"  Bridge queries ({len(bridge_results)}):")
    print(f"    EM: {bridge_em:.2f}%")
    print(f"    F1: {bridge_f1:.2f}%")
    print(f"  Comparison queries ({len(comparison_results)}):")
    print(f"    EM: {comparison_em:.2f}%")
    print(f"    F1: {comparison_f1:.2f}%")

    print(f"\n📊 METRICS BY STRATEGY:")
    print(f"  Vector ({len(vector_results)} queries): EM {vector_em:.2f}%")
    print(f"  Graph ({len(graph_results)} queries): EM {graph_em:.2f}%")

    print(f"\n📊 RETRIEVAL STATISTICS:")
    print(f"  Empty contexts: {metrics['empty_contexts']}/{len(queries)} ({metrics['empty_contexts']/len(queries)*100:.1f}%)")
    print(f"  Strategy distribution:")
    for strategy, count in metrics['strategy_counts'].items():
        print(f"    {strategy}: {count} ({count/len(queries)*100:.1f}%)")

    print(f"\n⏱️  TIMING:")
    avg_query = sum(metrics['query_times']) / len(metrics['query_times'])
    print(f"  Total duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"  Avg per query: {avg_query:.2f}s")

    # Save results
    results_path = Path("data/hotpotqa/results_100queries.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # Save summary
    summary = {
        'overall_em': overall_em,
        'overall_f1': overall_f1,
        'bridge_em': bridge_em,
        'bridge_f1': bridge_f1,
        'comparison_em': comparison_em,
        'comparison_f1': comparison_f1,
        'vector_em': vector_em,
        'graph_em': graph_em,
        'strategy_counts': metrics['strategy_counts'],
        'empty_contexts': metrics['empty_contexts'],
        'total_queries': len(queries),
        'duration_seconds': duration,
        'avg_query_time': avg_query,
    }

    summary_path = Path("data/hotpotqa/summary_100queries.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")

    print(f"\n" + "="*70)
    print("COMPARISON TO KG²RAG (NAACL 2025)")
    print("="*70)
    print(f"\nAgentic GraphRAG (Your System):")
    print(f"  Approach: Shared graph with all docs ({neo4j_nodes} nodes)")
    print(f"  EM: {overall_em:.2f}%")
    print(f"  F1: {overall_f1:.2f}%")
    print(f"\nKG²RAG (NAACL 2025):")
    print(f"  Approach: Per-query vector index (10 docs/query)")
    print(f"  EM: ~60%* (reported in paper)")
    print(f"  F1: ~70%* (estimated)")
    print(f"\n* Note: Different indexing approach (per-query vs shared)")
    print(f"  Both have 100% coverage, but methodology differs")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
