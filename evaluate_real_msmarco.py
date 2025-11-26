#!/usr/bin/env python3
"""
Evaluate Agentic GraphRAG on real MS MARCO dataset.
"""

import json
import time
from pathlib import Path
from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
from src.agents import ReflectionAgent

def main():
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - REAL MS MARCO EVALUATION")
    print("=" * 70)

    # Load evaluation set
    eval_file = Path("data/msmarco/real_evaluation_set.json")
    if not eval_file.exists():
        print("❌ Run msmarco_real_loader.py first to create evaluation set")
        return

    with open(eval_file) as f:
        eval_set = json.load(f)

    passages = eval_set['passages']
    queries = eval_set['queries']
    qrels = eval_set['qrels']

    print(f"\n📊 Loaded: {len(passages)} passages, {len(queries)} queries")

    # Stage 1: Ingestion
    print("\n" + "=" * 70)
    print("  STAGE 1: PASSAGE INGESTION")
    print("=" * 70)

    ingestion = get_ingestion_pipeline()

    # Convert passages to text list
    documents = [p['text'] for p in passages]

    print(f"\n📚 Ingesting {len(documents)} passages...")
    start_time = time.time()

    # Clear existing data first
    ingestion.neo4j_manager.clear_database()

    result = ingestion.ingest_documents(documents)

    duration = time.time() - start_time
    print(f"\n✅ Ingestion Complete ({duration:.1f}s)")
    print(f"   • Entities: {result.get('entities_extracted', 0)}")
    print(f"   • Relations: {result.get('relations_extracted', 0)}")

    # Stage 2: Evaluation
    print("\n" + "=" * 70)
    print("  STAGE 2: QUERY EVALUATION")
    print("=" * 70)

    retrieval = get_retrieval_pipeline(use_reranking=True, use_reflection=True)
    reflection = ReflectionAgent()

    results = []
    total_score = 0

    print(f"\n📊 Evaluating {len(queries)} queries...\n")

    for i, (qid, query_text) in enumerate(queries.items(), 1):
        start = time.time()

        # Get response
        response = retrieval.query(query_text, top_k=5)
        latency = (time.time() - start) * 1000

        # Get ground truth
        ground_truth = None
        if qid in qrels:
            # Find the passage text for ground truth
            relevant_ids = qrels[qid]
            for p in passages:
                if p['id'] in relevant_ids:
                    ground_truth = p['text']
                    break

        # Evaluate with RAGAS metrics
        metrics = reflection.evaluate_response(
            query=query_text,
            response=response['response'],
            retrieved_context=response['context'],
            ground_truth=ground_truth
        )

        score = (
            metrics.get('faithfulness', 0) +
            metrics.get('answer_relevancy', 0) +
            metrics.get('context_precision', 0) +
            metrics.get('context_recall', 0)
        ) / 4

        total_score += score

        results.append({
            'query_id': qid,
            'query': query_text,
            'score': score,
            'latency_ms': latency,
            'strategy': response.get('strategy', 'unknown'),
            'metrics': metrics
        })

        # Progress
        status = "✓" if score >= 0.7 else "⚠"
        print(f"[{i}/{len(queries)}] {status} Score: {score:.3f} | {latency:.0f}ms | {query_text[:50]}...")

    # Stage 3: Summary
    print("\n" + "=" * 70)
    print("  STAGE 3: EVALUATION SUMMARY")
    print("=" * 70)

    avg_score = total_score / len(queries)
    avg_latency = sum(r['latency_ms'] for r in results) / len(results)

    # Calculate average metrics
    avg_metrics = {
        'faithfulness': sum(r['metrics'].get('faithfulness', 0) for r in results) / len(results),
        'answer_relevancy': sum(r['metrics'].get('answer_relevancy', 0) for r in results) / len(results),
        'context_precision': sum(r['metrics'].get('context_precision', 0) for r in results) / len(results),
        'context_recall': sum(r['metrics'].get('context_recall', 0) for r in results) / len(results),
    }

    print(f"\n📊 Overall Performance:")
    print(f"   • Total queries: {len(queries)}")
    print(f"   • Average score: {avg_score:.3f}")
    print(f"   • Average latency: {avg_latency:.1f}ms")

    print(f"\n📈 RAGAS Metrics:")
    print(f"   • Faithfulness:      {avg_metrics['faithfulness']:.3f}")
    print(f"   • Answer Relevancy:  {avg_metrics['answer_relevancy']:.3f}")
    print(f"   • Context Precision: {avg_metrics['context_precision']:.3f}")
    print(f"   • Context Recall:    {avg_metrics['context_recall']:.3f}")

    # Strategy distribution
    strategies = {}
    for r in results:
        s = r['strategy']
        strategies[s] = strategies.get(s, 0) + 1

    print(f"\n🎯 Strategy Distribution:")
    for strategy, count in sorted(strategies.items()):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 5)
        print(f"   • {strategy:15} : {count:3} ({pct:5.1f}%) {bar}")

    # Save results
    output_dir = Path("data/evaluation/msmarco_real")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'avg_score': avg_score,
                'avg_latency': avg_latency,
                'metrics': avg_metrics,
                'num_queries': len(queries),
                'num_passages': len(passages)
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("  ✅ REAL MS MARCO EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
