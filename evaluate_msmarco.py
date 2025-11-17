#!/usr/bin/env python3
"""
MS MARCO Evaluation for Agentic GraphRAG

Comprehensive evaluation using MS MARCO-style dataset with:
- Passage ingestion and knowledge graph construction
- Query evaluation with ground truth
- RAGAS metrics computation
- Comparison across retrieval strategies
- Publication-ready results and visualizations
"""

import sys
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
from msmarco_loader import MSMARCOLoader


@dataclass
class QueryResult:
    """Single query evaluation result."""
    query_id: str
    query: str
    response: str
    ground_truth_passages: List[str]
    retrieved_passages: List[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    latency_ms: float
    strategy_used: str
    success: bool
    error: str = ""


@dataclass
class AggregatedMetrics:
    """Aggregated evaluation metrics."""
    num_queries: int
    success_rate: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall_score: float
    avg_latency_ms: float
    strategy_distribution: Dict[str, int]


def ingest_passages(passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ingest MS MARCO passages into knowledge graph.

    Args:
        passages: List of passage dictionaries with id and text

    Returns:
        Ingestion results
    """
    print("\n" + "=" * 70)
    print("  STAGE 1: PASSAGE INGESTION")
    print("=" * 70)

    # Extract passage texts
    documents = [p['text'] for p in passages]
    print(f"\n📚 Ingesting {len(documents)} passages...")

    schema_path = Path("data/processed/schema_msmarco.json")
    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=True
    )

    results = pipeline.ingest_documents(
        documents,
        infer_schema=True,
        enrich_metadata=True
    )

    print(f"\n✅ Ingestion Complete:")
    print(f"   • Documents processed: {results['documents_processed']}")
    print(f"   • Entities extracted: {results['entities_extracted']}")
    print(f"   • Relations extracted: {results['relations_extracted']}")
    print(f"   • Neo4j nodes created: {results['nodes_created']}")
    print(f"   • Neo4j edges created: {results['edges_created']}")
    print(f"   • FAISS vectors stored: {results.get('vectors_stored', 'N/A')}")
    print(f"   • Duration: {results['duration_seconds']:.2f}s")

    # Save schema
    if results.get('schema'):
        print(f"\n📊 Inferred Schema:")
        print(f"   • Entity types: {len(results['schema'].get('entity_types', []))}")
        print(f"   • Relation types: {len(results['schema'].get('relation_types', []))}")

    return results


def evaluate_query(pipeline, query_id: str, query: str,
                   ground_truth_passages: List[str],
                   ground_truth_text: str = "") -> QueryResult:
    """Evaluate a single query.

    Args:
        pipeline: Retrieval pipeline
        query_id: Query identifier
        query: Query text
        ground_truth_passages: List of relevant passage IDs
        ground_truth_text: Actual text from ground truth passages

    Returns:
        Query evaluation result
    """
    try:
        start_time = time.time()

        # Query the system with ground truth text for Context Recall
        result = pipeline.query(
            query,
            top_k=5,
            evaluate=True,
            ground_truth=ground_truth_text
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract metrics
        metrics = result.get('metrics', {})

        # Get retrieved passage info
        retrieved = []
        if result.get('context'):
            for ctx in result['context']:
                retrieved.append(ctx.get('source', 'unknown'))

        return QueryResult(
            query_id=query_id,
            query=query,
            response=result['response'],
            ground_truth_passages=ground_truth_passages,
            retrieved_passages=retrieved,
            faithfulness=metrics.get('faithfulness', 0.0),
            answer_relevancy=metrics.get('answer_relevancy', 0.0),
            context_precision=metrics.get('context_precision', 0.0),
            context_recall=metrics.get('context_recall', 0.0),
            overall_score=metrics.get('overall', 0.0),
            latency_ms=latency_ms,
            strategy_used=result.get('strategy', 'unknown'),
            success=True
        )

    except Exception as e:
        return QueryResult(
            query_id=query_id,
            query=query,
            response="",
            ground_truth_passages=ground_truth_passages,
            retrieved_passages=[],
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
            overall_score=0.0,
            latency_ms=0.0,
            strategy_used="error",
            success=False,
            error=str(e)
        )


def evaluate_all_queries(queries: Dict[str, str],
                        qrels: Dict[str, List[str]],
                        passages: List[Dict[str, Any]]) -> List[QueryResult]:
    """Evaluate all queries.

    Args:
        queries: Dictionary of query_id -> query_text
        qrels: Dictionary of query_id -> relevant_passage_ids
        passages: List of passage dictionaries with id and text

    Returns:
        List of query results
    """
    print("\n" + "=" * 70)
    print("  STAGE 2: QUERY EVALUATION")
    print("=" * 70)

    # Create passage lookup dict for ground truth text
    passage_lookup = {p['id']: p['text'] for p in passages}

    pipeline = get_retrieval_pipeline(
        use_reranking=True,  # Enable cross-encoder reranking for better precision
        use_reflection=True
    )

    results = []
    total = len(queries)

    print(f"\n📊 Evaluating {total} queries...\n")

    for i, (query_id, query_text) in enumerate(queries.items(), 1):
        print(f"[{i}/{total}] {query_id}: {query_text}")

        ground_truth_passage_ids = qrels.get(query_id, [])

        # Convert passage IDs to actual passage text for ground truth
        ground_truth_text = ""
        if ground_truth_passage_ids:
            ground_truth_texts = [
                passage_lookup.get(pid, "")
                for pid in ground_truth_passage_ids
            ]
            ground_truth_text = "\n\n".join(gt for gt in ground_truth_texts if gt)

        result = evaluate_query(
            pipeline,
            query_id,
            query_text,
            ground_truth_passage_ids,
            ground_truth_text
        )

        results.append(result)

        if result.success:
            print(f"   ✓ Score: {result.overall_score:.3f} | "
                  f"Latency: {result.latency_ms:.0f}ms | "
                  f"Strategy: {result.strategy_used}")
        else:
            print(f"   ✗ Error: {result.error}")

        print()

    return results


def calculate_aggregated_metrics(results: List[QueryResult]) -> AggregatedMetrics:
    """Calculate aggregated metrics across all queries.

    Args:
        results: List of query results

    Returns:
        Aggregated metrics
    """
    successful = [r for r in results if r.success]

    if not successful:
        return AggregatedMetrics(
            num_queries=len(results),
            success_rate=0.0,
            avg_faithfulness=0.0,
            avg_answer_relevancy=0.0,
            avg_context_precision=0.0,
            avg_context_recall=0.0,
            avg_overall_score=0.0,
            avg_latency_ms=0.0,
            strategy_distribution={}
        )

    # Count strategy usage
    strategy_counts = {}
    for r in successful:
        strategy_counts[r.strategy_used] = strategy_counts.get(r.strategy_used, 0) + 1

    return AggregatedMetrics(
        num_queries=len(results),
        success_rate=len(successful) / len(results),
        avg_faithfulness=statistics.mean([r.faithfulness for r in successful]),
        avg_answer_relevancy=statistics.mean([r.answer_relevancy for r in successful]),
        avg_context_precision=statistics.mean([r.context_precision for r in successful]),
        avg_context_recall=statistics.mean([r.context_recall for r in successful]),
        avg_overall_score=statistics.mean([r.overall_score for r in successful]),
        avg_latency_ms=statistics.mean([r.latency_ms for r in successful]),
        strategy_distribution=strategy_counts
    )


def save_results(results: List[QueryResult],
                aggregated: AggregatedMetrics,
                output_dir: Path):
    """Save evaluation results.

    Args:
        results: List of query results
        aggregated: Aggregated metrics
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as CSV
    csv_file = output_dir / "detailed_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'query_id', 'query', 'response', 'faithfulness', 'answer_relevancy',
            'context_precision', 'context_recall', 'overall_score',
            'latency_ms', 'strategy_used', 'success'
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'query_id': result.query_id,
                'query': result.query,
                'response': result.response[:200] + '...' if len(result.response) > 200 else result.response,
                'faithfulness': f"{result.faithfulness:.4f}",
                'answer_relevancy': f"{result.answer_relevancy:.4f}",
                'context_precision': f"{result.context_precision:.4f}",
                'context_recall': f"{result.context_recall:.4f}",
                'overall_score': f"{result.overall_score:.4f}",
                'latency_ms': f"{result.latency_ms:.2f}",
                'strategy_used': result.strategy_used,
                'success': result.success
            })

    print(f"\n✅ Saved detailed results: {csv_file}")

    # Save aggregated metrics
    agg_file = output_dir / "aggregated_metrics.json"
    with open(agg_file, 'w') as f:
        json.dump(asdict(aggregated), f, indent=2)

    print(f"✅ Saved aggregated metrics: {agg_file}")

    # Save complete results as JSON
    json_file = output_dir / "complete_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            'detailed': [asdict(r) for r in results],
            'aggregated': asdict(aggregated)
        }, f, indent=2)

    print(f"✅ Saved complete results: {json_file}")


def print_summary(results: List[QueryResult], aggregated: AggregatedMetrics):
    """Print evaluation summary.

    Args:
        results: List of query results
        aggregated: Aggregated metrics
    """
    print("\n" + "=" * 70)
    print("  STAGE 3: EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n📊 Overall Performance:")
    print(f"   • Total queries: {aggregated.num_queries}")
    print(f"   • Success rate: {aggregated.success_rate:.1%}")
    print(f"   • Average overall score: {aggregated.avg_overall_score:.3f}")
    print(f"   • Average latency: {aggregated.avg_latency_ms:.1f}ms")

    print(f"\n📈 RAGAS Metrics:")
    print(f"   • Faithfulness:      {aggregated.avg_faithfulness:.3f}")
    print(f"   • Answer Relevancy:  {aggregated.avg_answer_relevancy:.3f}")
    print(f"   • Context Precision: {aggregated.avg_context_precision:.3f}")
    print(f"   • Context Recall:    {aggregated.avg_context_recall:.3f}")

    print(f"\n🎯 Strategy Distribution:")
    for strategy, count in aggregated.strategy_distribution.items():
        percentage = (count / aggregated.num_queries) * 100
        bar = '█' * int(percentage / 5)
        print(f"   • {strategy:15s}: {count:2d} ({percentage:5.1f}%) {bar}")

    # Best and worst queries
    successful = [r for r in results if r.success]
    if successful:
        successful.sort(key=lambda x: x.overall_score, reverse=True)

        print(f"\n🏆 Top 3 Best Queries:")
        for i, r in enumerate(successful[:3], 1):
            print(f"   {i}. [{r.query_id}] {r.query}")
            print(f"      Score: {r.overall_score:.3f} | Strategy: {r.strategy_used}")

        print(f"\n⚠️  Top 3 Worst Queries:")
        for i, r in enumerate(successful[-3:], 1):
            print(f"   {i}. [{r.query_id}] {r.query}")
            print(f"      Score: {r.overall_score:.3f} | Strategy: {r.strategy_used}")


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - MS MARCO EVALUATION")
    print("=" * 70)

    # Load MS MARCO data
    loader = MSMARCOLoader()
    eval_set = loader.prepare_evaluation_set(
        num_passages=12,
        num_queries=15
    )

    loader.get_statistics(eval_set)

    # Stage 1: Ingest passages
    ingestion_results = ingest_passages(eval_set['passages'])

    # Stage 2: Evaluate queries
    query_results = evaluate_all_queries(
        eval_set['queries'],
        eval_set['qrels'],
        eval_set['passages']
    )

    # Stage 3: Calculate metrics and save
    aggregated = calculate_aggregated_metrics(query_results)

    print_summary(query_results, aggregated)

    # Save results
    output_dir = Path("data/evaluation/msmarco")
    save_results(query_results, aggregated, output_dir)

    print("\n" + "=" * 70)
    print("  ✅ MS MARCO EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Results saved in: {output_dir.absolute()}")
    print("\n💡 Next steps:")
    print("   1. Review results: cat data/evaluation/msmarco/aggregated_metrics.json")
    print("   2. Generate visualizations: python visualization.py")
    print("   3. Compare with baselines: python compare_methods.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
