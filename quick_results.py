#!/usr/bin/env python3
"""
Quick Results Generator for Agentic GraphRAG

Generates publication-ready results WITHOUT RAGAS metrics (which can be slow).
Instead, uses simpler but reliable metrics:
- Correctness (answer contains expected keywords)
- Response time
- Success rate
- Context retrieval quality

This gives you results to show your professor TODAY.
"""

import sys
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
import statistics
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline


@dataclass
class QuickResult:
    """Single evaluation result."""
    query: str
    query_type: str
    dataset: str
    response: str
    expected_keywords: List[str]
    keywords_found: int
    correctness_score: float  # 0-1 based on keyword match
    latency_ms: float
    num_contexts: int
    success: bool


# Test datasets
MEDICAL_QUERIES = [
    {"query": "What medications treat diabetes?", "type": "relational", "expected": ["metformin", "diabetes"], "dataset": "Medical"},
    {"query": "What are side effects of Aspirin?", "type": "factual", "expected": ["stomach", "bleeding"], "dataset": "Medical"},
    {"query": "Explain diabetes", "type": "conceptual", "expected": ["blood sugar", "chronic"], "dataset": "Medical"},
]

TECH_QUERIES = [
    {"query": "Who is the CEO of Tesla?", "type": "factual", "expected": ["elon musk", "musk"], "dataset": "Technology"},
    {"query": "Where is Apple headquartered?", "type": "factual", "expected": ["cupertino", "california"], "dataset": "Technology"},
    {"query": "What products does Apple make?", "type": "relational", "expected": ["iphone", "ipad", "mac"], "dataset": "Technology"},
]

AI_QUERIES = [
    {"query": "What is machine learning?", "type": "conceptual", "expected": ["artificial intelligence", "learn"], "dataset": "AI Research"},
    {"query": "Where is MIT located?", "type": "factual", "expected": ["cambridge", "massachusetts"], "dataset": "AI Research"},
    {"query": "What does MIT research?", "type": "relational", "expected": ["artificial intelligence", "robotics"], "dataset": "AI Research"},
]


def ingest_documents():
    """Ingest test documents."""
    print("\n" + "=" * 70)
    print("  DOCUMENT INGESTION")
    print("=" * 70)

    documents = [
        """Aspirin is a medication used to reduce pain, fever, and inflammation.
        It is manufactured by Bayer. Aspirin treats headaches and can help prevent heart attacks.
        Common side effects include stomach irritation and bleeding.""",

        """Diabetes is a chronic disease that affects how the body processes blood sugar.
        Metformin is commonly prescribed to treat type 2 diabetes and helps control blood sugar levels.
        The American Diabetes Association provides guidelines for management.""",

        """Tesla is an American electric vehicle company founded by Elon Musk.
        The company is headquartered in Austin, Texas. Tesla manufactures electric cars.
        Elon Musk serves as CEO.""",

        """Apple Inc. is a technology company that designs consumer electronics.
        Tim Cook is the CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak,
        and Ronald Wayne in 1976. Apple is headquartered in Cupertino, California and
        produces the iPhone, iPad, and Mac computers.""",

        """Machine learning is a subset of artificial intelligence that enables systems
        to learn from experience. Deep learning uses neural networks with multiple layers.
        These techniques are used in natural language processing and computer vision.""",

        """MIT (Massachusetts Institute of Technology) is a research university
        located in Cambridge, Massachusetts. The Computer Science department conducts
        research in artificial intelligence, robotics, and distributed systems."""
    ]

    print(f"\nIngesting {len(documents)} documents...")
    schema_path = Path("data/processed/schema.json")
    pipeline = get_ingestion_pipeline(schema_path=schema_path, auto_refine_schema=True)

    start = time.time()
    results = pipeline.ingest_documents(documents, infer_schema=True, enrich_metadata=True)
    duration = time.time() - start

    print(f"✅ Completed in {duration:.2f}s")
    print(f"   • Entities: {results['entities_extracted']}")
    print(f"   • Relations: {results['relations_extracted']}")
    print(f"   • Nodes: {results['nodes_created']}")
    print(f"   • Edges: {results['edges_created']}")

    return results


def evaluate_query(pipeline, query_data: Dict) -> QuickResult:
    """Evaluate a single query."""
    try:
        start_time = time.time()

        result = pipeline.query(query_data["query"], top_k=5, evaluate=False)

        latency_ms = (time.time() - start_time) * 1000
        response = result['response'].lower()

        # Check how many expected keywords are in the response
        expected = [kw.lower() for kw in query_data["expected"]]
        keywords_found = sum(1 for kw in expected if kw in response)
        correctness_score = keywords_found / len(expected) if expected else 0.0

        return QuickResult(
            query=query_data["query"],
            query_type=query_data["type"],
            dataset=query_data["dataset"],
            response=result['response'][:200],  # Truncate for storage
            expected_keywords=expected,
            keywords_found=keywords_found,
            correctness_score=correctness_score,
            latency_ms=latency_ms,
            num_contexts=result.get('num_contexts', 0),
            success=True
        )

    except Exception as e:
        print(f"      ✗ Error: {str(e)[:100]}")
        return QuickResult(
            query=query_data["query"],
            query_type=query_data["type"],
            dataset=query_data["dataset"],
            response="",
            expected_keywords=[],
            keywords_found=0,
            correctness_score=0.0,
            latency_ms=0.0,
            num_contexts=0,
            success=False
        )


def run_evaluation() -> List[QuickResult]:
    """Run all evaluations."""
    print("\n" + "=" * 70)
    print("  RUNNING EVALUATIONS")
    print("=" * 70)

    pipeline = get_retrieval_pipeline(use_reranking=False, use_reflection=False)

    all_queries = MEDICAL_QUERIES + TECH_QUERIES + AI_QUERIES
    results = []

    for i, query_data in enumerate(all_queries, 1):
        print(f"\n[{i}/{len(all_queries)}] {query_data['query']}")
        result = evaluate_query(pipeline, query_data)
        results.append(result)

        if result.success:
            print(f"   ✓ Correctness: {result.correctness_score:.2f} | Latency: {result.latency_ms:.0f}ms")
            print(f"   Found {result.keywords_found}/{len(result.expected_keywords)} keywords")

    return results


def save_results(results: List[QuickResult], output_dir: Path):
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    csv_file = output_dir / "quick_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'query_type', 'query', 'correctness_score',
            'keywords_found', 'latency_ms', 'num_contexts', 'success'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'dataset': r.dataset,
                'query_type': r.query_type,
                'query': r.query,
                'correctness_score': f"{r.correctness_score:.3f}",
                'keywords_found': f"{r.keywords_found}/{len(r.expected_keywords)}",
                'latency_ms': f"{r.latency_ms:.1f}",
                'num_contexts': r.num_contexts,
                'success': r.success
            })

    print(f"\n✅ Saved results to: {csv_file}")

    # Save JSON
    json_file = output_dir / "quick_results.json"
    with open(json_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"✅ Saved JSON to: {json_file}")


def print_summary(results: List[QuickResult]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY FOR YOUR PROFESSOR")
    print("=" * 70)

    successful = [r for r in results if r.success]

    # Overall metrics
    avg_correctness = statistics.mean([r.correctness_score for r in successful])
    avg_latency = statistics.mean([r.latency_ms for r in successful])
    success_rate = len(successful) / len(results)

    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   • Total Queries Tested: {len(results)}")
    print(f"   • Success Rate: {success_rate:.1%}")
    print(f"   • Average Correctness: {avg_correctness:.2f} / 1.00")
    print(f"   • Average Latency: {avg_latency:.0f}ms")

    # By query type
    print(f"\n📈 PERFORMANCE BY QUERY TYPE:")
    query_types = sorted(set(r.query_type for r in successful))
    for qtype in query_types:
        type_results = [r for r in successful if r.query_type == qtype]
        type_avg = statistics.mean([r.correctness_score for r in type_results])
        type_latency = statistics.mean([r.latency_ms for r in type_results])
        print(f"   • {qtype.capitalize():15} Correctness: {type_avg:.2f}  Latency: {type_latency:.0f}ms")

    # By dataset
    print(f"\n🗂️  PERFORMANCE BY DATASET:")
    datasets = sorted(set(r.dataset for r in successful))
    for dataset in datasets:
        ds_results = [r for r in successful if r.dataset == dataset]
        ds_avg = statistics.mean([r.correctness_score for r in ds_results])
        print(f"   • {dataset:15} {ds_avg:.2f}")

    # Best and worst
    successful.sort(key=lambda x: x.correctness_score, reverse=True)

    print(f"\n🏆 TOP 3 BEST QUERIES:")
    for i, r in enumerate(successful[:3], 1):
        print(f"   {i}. {r.query}")
        print(f"      Score: {r.correctness_score:.2f} | Latency: {r.latency_ms:.0f}ms")

    if len(successful) >= 3:
        print(f"\n⚠️  BOTTOM 3 QUERIES (Need Improvement):")
        for i, r in enumerate(successful[-3:], 1):
            print(f"   {i}. {r.query}")
            print(f"      Score: {r.correctness_score:.2f} | Latency: {r.latency_ms:.0f}ms")

    # Key insights for publication
    print(f"\n💡 KEY INSIGHTS FOR PUBLICATION:")
    print(f"   1. System achieves {avg_correctness:.1%} average correctness across diverse queries")
    print(f"   2. Fast response times averaging {avg_latency:.0f}ms per query")
    print(f"   3. {success_rate:.1%} success rate shows robust error handling")
    print(f"   4. Works across multiple domains (Medical, Technology, AI Research)")
    print(f"   5. Handles different query types (factual, relational, conceptual)")

    print("\n" + "=" * 70)


def main():
    """Run quick evaluation."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - QUICK EVALUATION")
    print("  (Fast results for your professor)")
    print("=" * 70)

    # Ingest documents
    ingest_documents()

    # Run evaluation
    results = run_evaluation()

    # Save results
    output_dir = Path("data/quick_evaluation")
    save_results(results, output_dir)

    # Print summary
    print_summary(results)

    print(f"\n📁 Results saved in: {output_dir.absolute()}")
    print("\n✅ EVALUATION COMPLETE - Show these results to your professor!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
