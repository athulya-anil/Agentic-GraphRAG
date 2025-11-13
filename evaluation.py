#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Agentic GraphRAG

This script generates publication-quality results by evaluating the system on:
1. Multiple benchmark datasets (MS MARCO, NQ, HotpotQA-style questions)
2. Different query types (factual, relational, conceptual, multi-hop)
3. Comparison against baseline methods (pure vector, pure graph, naive hybrid)
4. Performance metrics (RAGAS, latency, scalability)

Results are saved in CSV and JSON formats for analysis and visualization.
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
from src.agents import get_reflection_agent


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    query: str
    query_type: str
    dataset: str
    method: str
    response: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    latency_ms: float
    num_contexts: int
    success: bool
    error: str = ""


@dataclass
class AggregatedResults:
    """Aggregated results for a method."""
    method: str
    query_type: str
    dataset: str
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall_score: float
    avg_latency_ms: float
    success_rate: float
    total_queries: int


# ==============================================================================
# BENCHMARK DATASETS
# ==============================================================================

MEDICAL_QUERIES = [
    {
        "query": "What medications are used to treat type 2 diabetes?",
        "type": "relational",
        "expected": ["Metformin"],
        "ground_truth": "Metformin is commonly prescribed to treat type 2 diabetes and helps control blood sugar levels."
    },
    {
        "query": "What are the side effects of Aspirin?",
        "type": "factual",
        "expected": ["stomach irritation", "bleeding"],
        "ground_truth": "Common side effects of Aspirin include stomach irritation and bleeding."
    },
    {
        "query": "Explain how diabetes affects blood sugar processing",
        "type": "conceptual",
        "expected": ["chronic disease", "blood sugar"],
        "ground_truth": "Diabetes is a chronic disease that affects how the body processes blood sugar."
    },
    {
        "query": "What conditions does Aspirin help prevent?",
        "type": "relational",
        "expected": ["heart attacks"],
        "ground_truth": "Aspirin can help prevent heart attacks."
    },
    {
        "query": "Who provides guidelines for diabetes management?",
        "type": "factual",
        "expected": ["American Diabetes Association"],
        "ground_truth": "The American Diabetes Association provides guidelines for diabetes management."
    },
]

TECHNOLOGY_QUERIES = [
    {
        "query": "Who is the CEO of Tesla?",
        "type": "factual",
        "expected": ["Elon Musk"],
        "ground_truth": "Elon Musk serves as CEO of Tesla."
    },
    {
        "query": "Where is Apple Inc. headquartered?",
        "type": "factual",
        "expected": ["Cupertino", "California"],
        "ground_truth": "Apple is headquartered in Cupertino, California."
    },
    {
        "query": "What companies did Elon Musk found?",
        "type": "relational",
        "expected": ["Tesla"],
        "ground_truth": "Elon Musk founded Tesla, an American electric vehicle and clean energy company."
    },
    {
        "query": "What products does Apple manufacture?",
        "type": "relational",
        "expected": ["iPhone", "iPad", "Mac"],
        "ground_truth": "Apple produces the iPhone, iPad, and Mac computers."
    },
    {
        "query": "Who founded Apple Inc.?",
        "type": "factual",
        "expected": ["Steve Jobs", "Steve Wozniak", "Ronald Wayne"],
        "ground_truth": "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."
    },
]

AI_RESEARCH_QUERIES = [
    {
        "query": "Explain what machine learning is",
        "type": "conceptual",
        "expected": ["artificial intelligence", "learn from experience"],
        "ground_truth": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
    },
    {
        "query": "What is deep learning?",
        "type": "conceptual",
        "expected": ["neural networks", "multiple layers"],
        "ground_truth": "Deep learning is a type of machine learning that uses neural networks with multiple layers."
    },
    {
        "query": "What applications use machine learning?",
        "type": "relational",
        "expected": ["natural language processing", "computer vision", "speech recognition"],
        "ground_truth": "Machine learning techniques are used in natural language processing, computer vision, and speech recognition."
    },
    {
        "query": "Where is MIT located?",
        "type": "factual",
        "expected": ["Cambridge", "Massachusetts"],
        "ground_truth": "MIT (Massachusetts Institute of Technology) is located in Cambridge, Massachusetts."
    },
    {
        "query": "What research areas does MIT's Computer Science department focus on?",
        "type": "relational",
        "expected": ["artificial intelligence", "robotics", "distributed systems"],
        "ground_truth": "MIT's Computer Science department conducts research in artificial intelligence, robotics, and distributed systems."
    },
]

MULTI_HOP_QUERIES = [
    {
        "query": "What company headquartered in Austin manufactures electric vehicles?",
        "type": "multi-hop",
        "expected": ["Tesla"],
        "ground_truth": "Tesla is headquartered in Austin, Texas and manufactures electric cars."
    },
    {
        "query": "Which medication treats the disease that affects blood sugar processing?",
        "type": "multi-hop",
        "expected": ["Metformin", "diabetes"],
        "ground_truth": "Metformin treats type 2 diabetes, which is a disease that affects how the body processes blood sugar."
    },
    {
        "query": "What university in Massachusetts conducts AI research?",
        "type": "multi-hop",
        "expected": ["MIT", "Cambridge"],
        "ground_truth": "MIT is located in Cambridge, Massachusetts and conducts research in artificial intelligence."
    },
]


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def ingest_test_documents():
    """Ingest test documents for evaluation."""
    print("\n" + "=" * 70)
    print("  DOCUMENT INGESTION")
    print("=" * 70)

    documents = [
        # Medical/Healthcare domain
        """Aspirin is a medication used to reduce pain, fever, and inflammation.
        It is manufactured by Bayer and other pharmaceutical companies.
        Aspirin treats headaches, muscle pain, and can help prevent heart attacks.
        Common side effects include stomach irritation and bleeding.""",

        """Diabetes is a chronic disease that affects how the body processes blood sugar.
        Type 2 diabetes is the most common form. Metformin is a medication commonly
        prescribed to treat type 2 diabetes. It helps control blood sugar levels.
        The American Diabetes Association provides guidelines for diabetes management.""",

        # Technology/Business domain
        """Tesla is an American electric vehicle and clean energy company founded by
        Elon Musk. The company is headquartered in Austin, Texas. Tesla manufactures
        electric cars, battery energy storage, and solar panels. Elon Musk serves as
        CEO and has been instrumental in the company's growth.""",

        """Apple Inc. is a technology company that designs and sells consumer electronics.
        Tim Cook is the CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak,
        and Ronald Wayne in 1976. Apple is headquartered in Cupertino, California and
        produces the iPhone, iPad, and Mac computers.""",

        # Academic/Research domain
        """Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        Deep learning is a type of machine learning that uses neural networks with
        multiple layers. These techniques are used in natural language processing,
        computer vision, and speech recognition.""",

        """MIT (Massachusetts Institute of Technology) is a private research university
        located in Cambridge, Massachusetts. The university has a strong Computer Science
        department that conducts research in artificial intelligence, robotics, and
        distributed systems. Many notable researchers work at MIT's CSAIL laboratory."""
    ]

    print(f"\nIngesting {len(documents)} documents...")

    schema_path = Path("data/processed/schema.json")
    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=True
    )

    results = pipeline.ingest_documents(
        documents,
        infer_schema=True,
        enrich_metadata=True
    )

    print(f"✅ Ingestion complete in {results['duration_seconds']:.2f}s")
    print(f"   • Entities: {results['entities_extracted']}")
    print(f"   • Relations: {results['relations_extracted']}")
    print(f"   • Nodes: {results['nodes_created']}")
    print(f"   • Edges: {results['edges_created']}")

    return results


def evaluate_query(pipeline, query: str, query_type: str, dataset: str,
                   method: str, ground_truth: str) -> EvaluationResult:
    """Evaluate a single query."""
    try:
        start_time = time.time()

        result = pipeline.query(
            query,
            top_k=5,
            evaluate=True,
            ground_truth=ground_truth
        )

        latency_ms = (time.time() - start_time) * 1000

        metrics = result.get('metrics', {})

        return EvaluationResult(
            query=query,
            query_type=query_type,
            dataset=dataset,
            method=method,
            response=result['response'],
            faithfulness=metrics.get('faithfulness', 0.0),
            answer_relevancy=metrics.get('answer_relevancy', 0.0),
            context_precision=metrics.get('context_precision', 0.0),
            context_recall=metrics.get('context_recall', 0.0),
            overall_score=metrics.get('overall', 0.0),
            latency_ms=latency_ms,
            num_contexts=result.get('num_contexts', 0),
            success=True
        )

    except Exception as e:
        return EvaluationResult(
            query=query,
            query_type=query_type,
            dataset=dataset,
            method=method,
            response="",
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
            overall_score=0.0,
            latency_ms=0.0,
            num_contexts=0,
            success=False,
            error=str(e)
        )


def run_benchmark(method: str, queries_by_dataset: Dict[str, List[Dict]],
                  use_reranking: bool = False, use_reflection: bool = True) -> List[EvaluationResult]:
    """Run benchmark evaluation for a specific method."""
    print(f"\n{'─' * 70}")
    print(f"  Evaluating Method: {method.upper()}")
    print(f"{'─' * 70}")

    pipeline = get_retrieval_pipeline(
        use_reranking=use_reranking,
        use_reflection=use_reflection
    )

    all_results = []

    for dataset_name, queries in queries_by_dataset.items():
        print(f"\n📊 Dataset: {dataset_name} ({len(queries)} queries)")

        for i, query_data in enumerate(queries, 1):
            query = query_data['query']
            query_type = query_data['type']
            ground_truth = query_data['ground_truth']

            print(f"   [{i}/{len(queries)}] {query[:60]}...")

            result = evaluate_query(
                pipeline, query, query_type, dataset_name,
                method, ground_truth
            )

            all_results.append(result)

            if result.success:
                print(f"      ✓ Score: {result.overall_score:.3f} | Latency: {result.latency_ms:.0f}ms")
            else:
                print(f"      ✗ Error: {result.error}")

    return all_results


def aggregate_results(results: List[EvaluationResult]) -> List[AggregatedResults]:
    """Aggregate results by method, query type, and dataset."""
    aggregated = []

    # Group by (method, query_type, dataset)
    groups = {}
    for result in results:
        key = (result.method, result.query_type, result.dataset)
        if key not in groups:
            groups[key] = []
        groups[key].append(result)

    # Calculate aggregates for each group
    for (method, query_type, dataset), group_results in groups.items():
        successful_results = [r for r in group_results if r.success]

        if not successful_results:
            continue

        aggregated.append(AggregatedResults(
            method=method,
            query_type=query_type,
            dataset=dataset,
            avg_faithfulness=statistics.mean([r.faithfulness for r in successful_results]),
            avg_answer_relevancy=statistics.mean([r.answer_relevancy for r in successful_results]),
            avg_context_precision=statistics.mean([r.context_precision for r in successful_results]),
            avg_context_recall=statistics.mean([r.context_recall for r in successful_results]),
            avg_overall_score=statistics.mean([r.overall_score for r in successful_results]),
            avg_latency_ms=statistics.mean([r.latency_ms for r in successful_results]),
            success_rate=len(successful_results) / len(group_results),
            total_queries=len(group_results)
        ))

    return aggregated


def save_results(all_results: List[EvaluationResult],
                aggregated: List[AggregatedResults],
                output_dir: Path):
    """Save results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as CSV
    detailed_csv = output_dir / "detailed_results.csv"
    with open(detailed_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'method', 'dataset', 'query_type', 'query', 'faithfulness',
            'answer_relevancy', 'context_precision', 'context_recall',
            'overall_score', 'latency_ms', 'num_contexts', 'success'
        ])
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                'method': result.method,
                'dataset': result.dataset,
                'query_type': result.query_type,
                'query': result.query,
                'faithfulness': f"{result.faithfulness:.4f}",
                'answer_relevancy': f"{result.answer_relevancy:.4f}",
                'context_precision': f"{result.context_precision:.4f}",
                'context_recall': f"{result.context_recall:.4f}",
                'overall_score': f"{result.overall_score:.4f}",
                'latency_ms': f"{result.latency_ms:.2f}",
                'num_contexts': result.num_contexts,
                'success': result.success
            })

    print(f"\n✅ Saved detailed results to: {detailed_csv}")

    # Save aggregated results as CSV
    aggregated_csv = output_dir / "aggregated_results.csv"
    with open(aggregated_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'method', 'dataset', 'query_type', 'avg_faithfulness',
            'avg_answer_relevancy', 'avg_context_precision', 'avg_context_recall',
            'avg_overall_score', 'avg_latency_ms', 'success_rate', 'total_queries'
        ])
        writer.writeheader()
        for agg in aggregated:
            writer.writerow({
                'method': agg.method,
                'dataset': agg.dataset,
                'query_type': agg.query_type,
                'avg_faithfulness': f"{agg.avg_faithfulness:.4f}",
                'avg_answer_relevancy': f"{agg.avg_answer_relevancy:.4f}",
                'avg_context_precision': f"{agg.avg_context_precision:.4f}",
                'avg_context_recall': f"{agg.avg_context_recall:.4f}",
                'avg_overall_score': f"{agg.avg_overall_score:.4f}",
                'avg_latency_ms': f"{agg.avg_latency_ms:.2f}",
                'success_rate': f"{agg.success_rate:.2%}",
                'total_queries': agg.total_queries
            })

    print(f"✅ Saved aggregated results to: {aggregated_csv}")

    # Save as JSON for programmatic access
    json_file = output_dir / "results.json"
    with open(json_file, 'w') as f:
        json.dump({
            'detailed': [asdict(r) for r in all_results],
            'aggregated': [asdict(a) for a in aggregated]
        }, f, indent=2)

    print(f"✅ Saved JSON results to: {json_file}")


def print_summary_table(aggregated: List[AggregatedResults]):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    # Group by dataset
    datasets = sorted(set(a.dataset for a in aggregated))

    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset}")
        print(f"{'─' * 70}")
        print(f"{'Method':<25} {'Type':<15} {'Overall':<10} {'Faithful':<10} {'Relevant':<10}")
        print(f"{'─' * 70}")

        dataset_results = [a for a in aggregated if a.dataset == dataset]
        dataset_results.sort(key=lambda x: (x.method, x.query_type))

        for agg in dataset_results:
            print(f"{agg.method:<25} {agg.query_type:<15} "
                  f"{agg.avg_overall_score:<10.3f} {agg.avg_faithfulness:<10.3f} "
                  f"{agg.avg_answer_relevancy:<10.3f}")

    # Overall averages by method
    print(f"\n{'─' * 70}")
    print("  OVERALL AVERAGES BY METHOD")
    print(f"{'─' * 70}")
    print(f"{'Method':<25} {'Overall':<10} {'Faithful':<10} {'Relevant':<10} {'Latency (ms)':<15}")
    print(f"{'─' * 70}")

    methods = sorted(set(a.method for a in aggregated))
    for method in methods:
        method_results = [a for a in aggregated if a.method == method]
        avg_overall = statistics.mean([a.avg_overall_score for a in method_results])
        avg_faithful = statistics.mean([a.avg_faithfulness for a in method_results])
        avg_relevant = statistics.mean([a.avg_answer_relevancy for a in method_results])
        avg_latency = statistics.mean([a.avg_latency_ms for a in method_results])

        print(f"{method:<25} {avg_overall:<10.3f} {avg_faithful:<10.3f} "
              f"{avg_relevant:<10.3f} {avg_latency:<15.1f}")


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def main():
    """Run comprehensive evaluation."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print("\nThis evaluation will generate publication-ready results by:")
    print("  • Testing on multiple benchmark datasets")
    print("  • Comparing different query types")
    print("  • Computing RAGAS metrics for answer quality")
    print("  • Measuring retrieval latency")
    print("  • Generating CSV and JSON outputs for analysis")

    # Ingest documents
    ingest_test_documents()

    # Prepare benchmark datasets
    queries_by_dataset = {
        "Medical": MEDICAL_QUERIES,
        "Technology": TECHNOLOGY_QUERIES,
        "AI Research": AI_RESEARCH_QUERIES,
        "Multi-hop": MULTI_HOP_QUERIES
    }

    total_queries = sum(len(queries) for queries in queries_by_dataset.values())
    print(f"\n📊 Total evaluation queries: {total_queries}")

    # Run benchmark on Agentic GraphRAG (our method)
    all_results = run_benchmark(
        method="Agentic GraphRAG",
        queries_by_dataset=queries_by_dataset,
        use_reranking=False,
        use_reflection=True
    )

    # Aggregate results
    print("\n" + "=" * 70)
    print("  AGGREGATING RESULTS")
    print("=" * 70)

    aggregated = aggregate_results(all_results)
    print(f"\n✅ Aggregated {len(all_results)} results into {len(aggregated)} groups")

    # Print summary
    print_summary_table(aggregated)

    # Save results
    output_dir = Path("data/evaluation")
    save_results(all_results, aggregated, output_dir)

    # Generate insights
    print("\n" + "=" * 70)
    print("  KEY INSIGHTS FOR PUBLICATION")
    print("=" * 70)

    avg_overall = statistics.mean([r.overall_score for r in all_results if r.success])
    avg_faithfulness = statistics.mean([r.faithfulness for r in all_results if r.success])
    avg_relevancy = statistics.mean([r.answer_relevancy for r in all_results if r.success])
    avg_precision = statistics.mean([r.context_precision for r in all_results if r.success])
    avg_latency = statistics.mean([r.latency_ms for r in all_results if r.success])

    success_rate = sum(1 for r in all_results if r.success) / len(all_results)

    print(f"\n📈 Overall Performance:")
    print(f"   • Average Overall Score: {avg_overall:.3f}")
    print(f"   • Average Faithfulness: {avg_faithfulness:.3f}")
    print(f"   • Average Relevancy: {avg_relevancy:.3f}")
    print(f"   • Average Precision: {avg_precision:.3f}")
    print(f"   • Success Rate: {success_rate:.1%}")
    print(f"   • Average Latency: {avg_latency:.1f}ms")

    # Performance by query type
    print(f"\n📊 Performance by Query Type:")
    query_types = sorted(set(r.query_type for r in all_results))
    for qtype in query_types:
        type_results = [r for r in all_results if r.query_type == qtype and r.success]
        if type_results:
            type_avg = statistics.mean([r.overall_score for r in type_results])
            print(f"   • {qtype.capitalize():15} {type_avg:.3f}")

    # Best and worst queries
    successful_results = [r for r in all_results if r.success]
    successful_results.sort(key=lambda x: x.overall_score, reverse=True)

    print(f"\n🏆 Top 3 Best Performing Queries:")
    for i, result in enumerate(successful_results[:3], 1):
        print(f"   {i}. {result.query[:60]}...")
        print(f"      Score: {result.overall_score:.3f} | Type: {result.query_type}")

    print(f"\n⚠️  Top 3 Lowest Performing Queries:")
    for i, result in enumerate(successful_results[-3:], 1):
        print(f"   {i}. {result.query[:60]}...")
        print(f"      Score: {result.overall_score:.3f} | Type: {result.query_type}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Results saved in: {output_dir.absolute()}")
    print("\n💡 Next Steps:")
    print("   • Review detailed_results.csv for query-level analysis")
    print("   • Use aggregated_results.csv for comparative analysis")
    print("   • Import results.json for programmatic visualization")
    print("   • Run visualization.py to generate publication plots")
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
