#!/usr/bin/env python3
"""
Mini-5 HotpotQA Evaluation

Runs 5 queries against the ingested mini_5 corpus and evaluates:
- Query routing accuracy
- Context retrieval (supporting doc coverage)
- Exact Match (EM) score
"""

import json
import sys
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.retrieval import get_retrieval_pipeline


def normalize_answer(answer: str) -> str:
    """Normalize answer for EM comparison (HotpotQA standard)."""
    # Lowercase
    answer = answer.lower()
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    return answer


def exact_match(prediction: str, gold: str) -> bool:
    """Check if prediction matches gold answer (normalized)."""
    return normalize_answer(prediction) == normalize_answer(gold)


def check_supporting_coverage(contexts: list, supporting_titles: list) -> dict:
    """Check if retrieved contexts include supporting documents."""
    # Extract titles from contexts
    retrieved_titles = set()
    for ctx in contexts:
        if isinstance(ctx, dict):
            # Check for doc_title or title field
            title = ctx.get('doc_title') or ctx.get('title') or ctx.get('metadata', {}).get('title', '')
            if title:
                retrieved_titles.add(title)

    # Check coverage
    found = []
    missing = []
    for title in supporting_titles:
        if title in retrieved_titles:
            found.append(title)
        else:
            missing.append(title)

    return {
        "found": found,
        "missing": missing,
        "coverage": len(found) / len(supporting_titles) if supporting_titles else 0
    }


def main():
    print("\n" + "="*70)
    print("MINI-5 HOTPOTQA EVALUATION")
    print("="*70)

    # Load queries
    queries_path = Path("data/hotpotqa/mini_5/queries.json")
    with open(queries_path, 'r') as f:
        queries = json.load(f)

    print(f"\nLoaded {len(queries)} queries")

    # Initialize pipeline
    print("\nInitializing retrieval pipeline...")
    pipeline = get_retrieval_pipeline(use_reranking=True)

    # Results storage
    results = []
    em_scores = []

    print("\n" + "-"*70)
    print("RUNNING QUERIES")
    print("-"*70)

    for i, q in enumerate(queries, 1):
        query_id = q['query_id']
        question = q['question']
        gold_answer = q['answer']
        query_type = q['type']
        supporting_titles = q['supporting_titles']

        print(f"\n[{i}/5] Query: {query_id}")
        print(f"  Question: {question[:60]}...")
        print(f"  Expected: {gold_answer}")
        print(f"  Type: {query_type}")

        # Run query
        try:
            result = pipeline.query(question, use_cache=False, top_k=10)

            predicted = result.get('response', '')
            strategy = result.get('strategy', 'unknown')
            routed_type = result.get('query_type', 'unknown')
            num_contexts = result.get('num_contexts', 0)
            contexts = result.get('context', [])

            # Calculate EM
            em = exact_match(predicted, gold_answer)
            em_scores.append(1 if em else 0)

            # Check supporting document coverage
            coverage = check_supporting_coverage(contexts, supporting_titles)

            print(f"  Strategy: {strategy}")
            print(f"  Routed type: {routed_type}")
            print(f"  Contexts: {num_contexts}")
            print(f"  Predicted: {predicted}")
            print(f"  EM: {'✓' if em else '✗'}")
            print(f"  Supporting coverage: {coverage['coverage']:.0%} ({len(coverage['found'])}/{len(supporting_titles)})")
            if coverage['missing']:
                print(f"  Missing: {coverage['missing']}")

            results.append({
                "query_id": query_id,
                "question": question,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "em": em,
                "query_type": query_type,
                "routed_type": routed_type,
                "strategy": strategy,
                "num_contexts": num_contexts,
                "supporting_coverage": coverage
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            em_scores.append(0)
            results.append({
                "query_id": query_id,
                "error": str(e),
                "em": False
            })

    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    total_em = sum(em_scores)
    em_rate = total_em / len(em_scores) if em_scores else 0

    print(f"\nOverall EM: {total_em}/{len(em_scores)} = {em_rate:.1%}")

    # By query type
    comparison_em = [r['em'] for r in results if r.get('query_type') == 'comparison' and 'em' in r]
    bridge_em = [r['em'] for r in results if r.get('query_type') == 'bridge' and 'em' in r]

    if comparison_em:
        print(f"Comparison EM: {sum(comparison_em)}/{len(comparison_em)} = {sum(comparison_em)/len(comparison_em):.1%}")
    if bridge_em:
        print(f"Bridge EM: {sum(bridge_em)}/{len(bridge_em)} = {sum(bridge_em)/len(bridge_em):.1%}")

    # Strategy distribution
    strategies = {}
    for r in results:
        s = r.get('strategy', 'unknown')
        strategies[s] = strategies.get(s, 0) + 1
    print(f"\nStrategy distribution: {strategies}")

    # Supporting coverage
    coverages = [r['supporting_coverage']['coverage'] for r in results if 'supporting_coverage' in r]
    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        print(f"Avg supporting coverage: {avg_coverage:.1%}")

    # Save results
    output_path = Path("data/hotpotqa/mini_5/evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {
                "total_queries": len(queries),
                "em_score": em_rate,
                "em_correct": total_em,
                "strategy_distribution": strategies,
                "avg_supporting_coverage": avg_coverage if coverages else 0
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
