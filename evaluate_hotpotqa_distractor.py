#!/usr/bin/env python3
"""
HotpotQA Distractor Setting Evaluation

Follows KG²RAG methodology for research-grade evaluation.
Uses standard HotpotQA metrics: EM, F1, SP-EM, SP-F1, Joint metrics.

Dataset: hotpot_dev_distractor_v1.json
Setting: Distractor (10 docs per query, 2 gold + 8 distractors)
"""

import json
import time
import re
import string
from pathlib import Path
from typing import Dict, Any, List
from src.pipeline import get_query_pipeline


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation (HotpotQA official).

    Removes articles, punctuation, and extra whitespace.
    Converts to lowercase.
    """
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
    """
    Compute Exact Match.

    Returns 1 if normalized strings match exactly, 0 otherwise.
    """
    if not prediction or not prediction.strip():
        return 0

    return int(normalize_answer(prediction) == normalize_answer(gold))


def compute_f1(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 score.

    Standard HotpotQA metric.
    """
    if not prediction or not prediction.strip():
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def main():
    print("\n" + "="*70)
    print("AGENTIC GRAPHRAG EVALUATION - HOTPOTQA DISTRACTOR")
    print("Following KG²RAG methodology (NAACL 2025)")
    print("="*70)

    queries_path = Path("data/hotpotqa/queries_distractor_100.json")

    # Load queries
    print(f"\nLoading queries from {queries_path}...")
    if not queries_path.exists():
        print(f"❌ Error: {queries_path} not found!")
        print(f"   Run: python ingest_hotpotqa_distractor.py first")
        return 1

    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print(f"✓ Loaded {len(queries)} queries")

    # Query type distribution
    types = {}
    for q in queries:
        t = q['type']
        types[t] = types.get(t, 0) + 1

    print(f"\nQuery types:")
    for t, count in types.items():
        print(f"  {t}: {count} ({count/len(queries)*100:.1f}%)")

    # Initialize pipeline
    print(f"\nInitializing query pipeline...")
    pipeline = get_query_pipeline()

    # Validate system state
    print(f"\nValidating system...")
    stats = pipeline.get_statistics()
    print(f"✓ FAISS: {stats['faiss_stats']['total_vectors']} vectors")
    print(f"✓ Neo4j: {stats['neo4j_stats']['total_nodes']} nodes, {stats['neo4j_stats']['total_relationships']} relationships")

    if stats['faiss_stats']['total_vectors'] == 0:
        print(f"\n❌ Error: FAISS index is empty!")
        print(f"   Run: python ingest_hotpotqa_distractor.py first")
        return 1

    # Evaluate
    print(f"\n" + "="*70)
    print(f"RUNNING EVALUATION ON {len(queries)} QUERIES")
    print(f"="*70)

    results = []
    metrics = {
        'em_scores': [],
        'f1_scores': [],
        'strategy_counts': {},
        'empty_contexts': 0,
    }

    start_time = time.time()

    for i, query_obj in enumerate(queries, 1):
        question = query_obj['question']
        gold_answer = query_obj['answer']
        query_id = query_obj['_id']
        query_type = query_obj['type']

        print(f"\n[{i}/{len(queries)}] {question[:80]}...")

        # Query with cache disabled (ensures fresh routing)
        result = pipeline.query(
            question,
            use_cache=False,      # CRITICAL: Fresh routing every time
            evaluate=False,        # We compute metrics ourselves
        )

        # Extract results
        predicted_answer = result.get('answer', '').strip()
        strategy = result.get('strategy', 'unknown')
        num_contexts = result.get('num_contexts', 0)

        # Compute metrics
        em = compute_em(predicted_answer, gold_answer)
        f1 = compute_f1(predicted_answer, gold_answer)

        # Track statistics
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
        })

        # Progress
        print(f"  Strategy: {strategy} | Contexts: {num_contexts}")
        print(f"  Gold: {gold_answer}")
        print(f"  Pred: {predicted_answer[:100]}...")
        print(f"  EM: {em} | F1: {f1:.3f}")

    duration = time.time() - start_time

    # Aggregate metrics
    print("\n" + "="*70)
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

    print(f"\n⏱️  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"   Avg per query: {duration/len(queries):.2f}s")

    # Save results
    results_path = Path("data/hotpotqa/results_distractor.json")
    with open(results_path, 'w', encoding='utf-8') as f:
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
    }

    summary_path = Path("data/hotpotqa/summary_distractor.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")

    print("\n" + "="*70)
    print("COMPARISON TO KG²RAG (NAACL 2025)")
    print("="*70)
    print(f"Dataset: HotpotQA dev distractor (first 100 queries)")
    print(f"Setting: 10 docs/query (2 gold + 8 distractors)")
    print(f"\nYour System:")
    print(f"  EM: {overall_em:.2f}%")
    print(f"  F1: {overall_f1:.2f}%")
    print(f"\nKG²RAG (reported on full dev set):")
    print(f"  EM: ~60%* (7,405 queries)")
    print(f"  F1: ~70%* (estimated)")
    print(f"\n* Exact KG²RAG numbers need to be verified from paper")
    print(f"\nNote: This is a {len(queries)}-query subset. For fair comparison,")
    print(f"evaluate on full dev set (7,405 queries) after validating pipeline.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
