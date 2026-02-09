#!/usr/bin/env python3
"""
Mini-25 Gold-Only Evaluation - All Strategies + Orchestrator

Evaluates all 25 queries against the gold-only corpus (50 docs, no distractors).
Tests VECTOR, GRAPH, HYBRID (forced), and ORCHESTRATOR (auto-routed) modes.

Compares results against the full-corpus (242 docs) baseline.
"""

import json
import sys
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.retrieval import get_retrieval_pipeline


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def normalize_numbers(text):
        return re.sub(r'(\d),(\d)', r'\1\2', text)

    return white_space_fix(
        remove_articles(
            remove_punc(
                normalize_numbers(
                    lower(s)
                )
            )
        )
    )


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def subem_check(prediction: str, gold: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    pattern = r'\b' + re.escape(normalized_gold) + r'\b'
    return bool(re.search(pattern, normalized_prediction))


def em_check(prediction: str, gold: str) -> tuple[bool, str]:
    if exact_match(prediction, gold):
        return True, 'strict'
    elif subem_check(prediction, gold):
        return True, 'subem'
    else:
        return False, 'none'


def check_supporting_coverage(contexts: list, supporting_titles: list) -> dict:
    retrieved_titles = set()
    for ctx in contexts:
        if isinstance(ctx, dict):
            title = ctx.get('doc_title') or ctx.get('title') or ctx.get('metadata', {}).get('title', '')
            if title:
                retrieved_titles.add(title)

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


def run_strategy(pipeline, queries, strategy_name, strategy_enum=None):
    """Run all queries with a given strategy (or orchestrator if strategy_enum is None)."""
    results = []
    em_scores = []
    strict_em_scores = []
    strategy_usage = {}

    for i, q in enumerate(queries, 1):
        query_id = q['query_id']
        question = q['question']
        gold_answer = q['answer']
        query_type = q['type']
        supporting_titles = q['supporting_titles']

        print(f"\n[{i}/{len(queries)}] {query_id}: {question[:60]}...")
        print(f"  Expected: {gold_answer}")

        try:
            if strategy_enum is not None:
                result = pipeline.query(question, use_cache=False, top_k=10, strategy=strategy_enum)
            else:
                result = pipeline.query(question, use_cache=False, top_k=10)

            predicted = result.get('response', '')
            strategy = result.get('strategy', strategy_name)
            routed_type = result.get('query_type', 'unknown')
            num_contexts = result.get('num_contexts', 0)
            contexts = result.get('context', [])

            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

            em, match_type = em_check(predicted, gold_answer)
            em_scores.append(1 if em else 0)
            strict_em_scores.append(1 if match_type == 'strict' else 0)

            coverage = check_supporting_coverage(contexts, supporting_titles)

            print(f"  Strategy: {strategy.upper()}")
            print(f"  Contexts: {num_contexts}")
            print(f"  Predicted: {predicted}")
            print(f"  EM: {'Y' if em else 'N'} ({match_type})")
            print(f"  Coverage: {coverage['coverage']:.0%}")

            results.append({
                "query_id": query_id,
                "question": question,
                "gold_answer": gold_answer,
                "predicted": predicted,
                "em": em,
                "match_type": match_type,
                "query_type": query_type,
                "strategy": strategy,
                "routed_type": routed_type if strategy_enum is None else None,
                "num_contexts": num_contexts,
                "supporting_coverage": coverage
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            em_scores.append(0)
            strict_em_scores.append(0)
            results.append({
                "query_id": query_id,
                "error": str(e),
                "em": False,
                "match_type": "none",
                "strategy": strategy_name
            })

    scores = {
        "em_score": sum(em_scores) / len(em_scores) if em_scores else 0,
        "em_correct": sum(em_scores),
        "strict_em_score": sum(strict_em_scores) / len(strict_em_scores) if strict_em_scores else 0,
        "strict_em_correct": sum(strict_em_scores),
        "total": len(queries),
        "strategy_distribution": strategy_usage
    }

    return results, scores


def main():
    print("\n" + "="*70)
    print("MINI-25 GOLD-ONLY EVALUATION")
    print("50 supporting docs, 0 distractors")
    print("="*70)

    # Load queries
    queries_path = Path("data/hotpotqa/mini_25_gold_only/queries.json")
    if not queries_path.exists():
        queries_path = Path("data/hotpotqa/mini_25/queries.json")
    with open(queries_path, 'r') as f:
        queries = json.load(f)

    print(f"\nLoaded {len(queries)} queries")

    # Initialize pipeline
    print("\nInitializing retrieval pipeline...")
    pipeline = get_retrieval_pipeline(use_reranking=True)

    # Import RetrievalStrategy enum
    from src.agents.orchestrator_agent import RetrievalStrategy

    # Test all strategies + orchestrator
    strategies_to_test = [
        ("vector", RetrievalStrategy.VECTOR),
        ("graph", RetrievalStrategy.GRAPH),
        ("hybrid", RetrievalStrategy.HYBRID),
    ]

    all_results = {}
    all_scores = {}

    for strategy_name, strategy_enum in strategies_to_test:
        print("\n" + "="*70)
        print(f"TESTING STRATEGY: {strategy_name.upper()}")
        print("="*70)

        results, scores = run_strategy(pipeline, queries, strategy_name, strategy_enum)
        all_results[strategy_name] = results
        all_scores[strategy_name] = scores

    # Orchestrator mode
    print("\n" + "="*70)
    print("TESTING: ORCHESTRATOR (auto-routing)")
    print("="*70)

    orch_results, orch_scores = run_strategy(pipeline, queries, "orchestrator", strategy_enum=None)
    all_results["orchestrator"] = orch_results
    all_scores["orchestrator"] = orch_scores

    # Load baseline results for comparison
    baseline_path = Path("data/hotpotqa/mini_25/evaluation_results_all_strategies.json")
    baseline = None
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

    baseline_orch_path = Path("data/hotpotqa/mini_25/evaluation_results_orchestrator.json")
    baseline_orch = None
    if baseline_orch_path.exists():
        with open(baseline_orch_path, 'r') as f:
            baseline_orch = json.load(f)

    # Summary
    print("\n" + "="*70)
    print("GOLD-ONLY EVALUATION SUMMARY")
    print("="*70)

    print(f"\n{'Strategy':<15} {'EM (SubEM)':<15} {'Strict EM':<15}", end="")
    if baseline:
        print(f" {'Baseline EM':<15} {'Delta':<10}", end="")
    print()
    print("-" * 70)

    for strategy_name in ["vector", "graph", "hybrid", "orchestrator"]:
        scores = all_scores[strategy_name]
        em_str = f"{scores['em_correct']}/{scores['total']} = {scores['em_score']:.1%}"
        strict_str = f"{scores['strict_em_correct']}/{scores['total']} = {scores['strict_em_score']:.1%}"
        print(f"{strategy_name.upper():<15} {em_str:<15} {strict_str:<15}", end="")

        if baseline and strategy_name in baseline.get('summary', {}):
            b_score = baseline['summary'][strategy_name]['em_score']
            delta = scores['em_score'] - b_score
            b_str = f"{b_score:.1%}"
            d_str = f"{delta:+.1%}"
            print(f" {b_str:<15} {d_str}", end="")
        elif baseline_orch and strategy_name == "orchestrator":
            b_score = baseline_orch['summary']['em_score']
            delta = scores['em_score'] - b_score
            b_str = f"{b_score:.1%}"
            d_str = f"{delta:+.1%}"
            print(f" {b_str:<15} {d_str}", end="")
        print()

    # Orchestrator strategy distribution
    print(f"\nOrchestrator Strategy Distribution:")
    for strategy, count in sorted(all_scores["orchestrator"]["strategy_distribution"].items()):
        print(f"  {strategy.upper()}: {count}/{len(queries)} ({count/len(queries):.1%})")

    # Per-query comparison (gold-only vs baseline)
    if baseline:
        print(f"\nPer-Query Comparison (VECTOR strategy):")
        print(f"{'Query':<8} {'Gold-Only':<12} {'Baseline':<12} {'Change':<10}")
        print("-" * 42)

        baseline_vector = {r['query_id']: r for r in baseline['results_by_strategy']['vector']}
        gold_vector = {r['query_id']: r for r in all_results['vector']}

        improved = []
        regressed = []
        for q in queries:
            qid = q['query_id']
            gold_em = gold_vector.get(qid, {}).get('em', False)
            base_em = baseline_vector.get(qid, {}).get('em', False)

            gold_mark = "Y" if gold_em else "N"
            base_mark = "Y" if base_em else "N"

            if gold_em and not base_em:
                change = "+FIXED"
                improved.append(qid)
            elif not gold_em and base_em:
                change = "-REGRESSED"
                regressed.append(qid)
            else:
                change = "="

            print(f"{qid:<8} {gold_mark:<12} {base_mark:<12} {change}")

        print(f"\nImproved: {len(improved)} queries ({', '.join(improved) if improved else 'none'})")
        print(f"Regressed: {len(regressed)} queries ({', '.join(regressed) if regressed else 'none'})")

    # Find best strategy
    best_strategy = max(all_scores.items(), key=lambda x: x[1]['em_score'])
    print(f"\nBest Strategy: {best_strategy[0].upper()} ({best_strategy[1]['em_score']:.1%} EM)")

    # Save results
    output_path = Path("data/hotpotqa/mini_25_gold_only/evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "experiment": "gold_only",
            "description": "50 supporting docs, 0 distractors",
            "summary": all_scores,
            "best_strategy": best_strategy[0],
            "results_by_strategy": all_results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
