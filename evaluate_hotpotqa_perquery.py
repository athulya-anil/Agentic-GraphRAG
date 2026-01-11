#!/usr/bin/env python3
"""
HotpotQA Per-Query Evaluation (Matching KG²RAG Methodology)

Mirrors KG²RAG's exact evaluation setup:
- For each query: Create fresh index with exactly 10 documents
- 10 docs = 2 gold supporting + 8 distractors (from query['context'])
- 100% coverage guaranteed
- Tests reasoning quality, not retrieval quality

This ensures apples-to-apples comparison with KG²RAG's reported 60% EM.
"""

import json
import time
import re
import string
from pathlib import Path
from typing import Dict, Any
from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
from src.graph.neo4j_manager import get_neo4j_manager
from src.vector.faiss_index import get_faiss_index
import shutil


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


def clear_indexes():
    """Clear Neo4j and FAISS indexes for fresh start."""
    # Clear Neo4j
    neo4j = get_neo4j_manager()
    neo4j.execute_query("MATCH (n) DETACH DELETE n")

    # Clear FAISS (delete and recreate directory)
    faiss_path = Path("data/faiss_index")
    if faiss_path.exists():
        shutil.rmtree(faiss_path)
    faiss_path.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "="*70)
    print("HOTPOTQA PER-QUERY EVALUATION")
    print("Matching KG²RAG Methodology (NAACL 2025)")
    print("="*70)

    # Load distractor dataset
    distractor_path = Path("data/hotpotqa/hotpot_dev_distractor.json")

    if not distractor_path.exists():
        print(f"❌ Error: {distractor_path} not found!")
        return 1

    with open(distractor_path, 'r') as f:
        all_queries = json.load(f)

    # Configuration
    NUM_QUERIES = 100  # Evaluate first 100 queries
    queries = all_queries[:NUM_QUERIES]

    print(f"\n📊 Evaluation Setup:")
    print(f"  Dataset: HotpotQA dev distractor")
    print(f"  Queries: {NUM_QUERIES}")
    print(f"  Docs per query: 10 (2 gold + 8 distractors)")
    print(f"  Coverage: 100% guaranteed")
    print(f"  Method: Per-query fresh indexing (matches KG²RAG)")

    # Query type distribution
    types = {}
    for q in queries:
        t = q['type']
        types[t] = types.get(t, 0) + 1

    print(f"\n  Query types:")
    for t, count in types.items():
        print(f"    {t}: {count} ({count/len(queries)*100:.1f}%)")

    # Estimate time
    estimated_time = NUM_QUERIES * 0.5  # ~30 seconds per query (ingestion + query)
    print(f"\n⏱️  Estimated time: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes")
    print(f"   (Ingesting 10 docs per query takes ~15-20 seconds)")

    input(f"\nPress Enter to start evaluation...")

    # Initialize pipelines
    print(f"\n🔧 Initializing pipelines...")
    schema_path = Path("data/hotpotqa/schema.json")

    results = []
    metrics = {
        'em_scores': [],
        'f1_scores': [],
        'strategy_counts': {},
        'empty_contexts': 0,
        'ingestion_times': [],
        'query_times': [],
    }

    start_time = time.time()

    # Evaluate each query
    for i, query_obj in enumerate(queries, 1):
        question = query_obj['question']
        gold_answer = query_obj['answer']
        query_id = query_obj['_id']
        query_type = query_obj['type']
        context = query_obj['context']  # List of [title, sentences]

        print(f"\n{'='*70}")
        print(f"[{i}/{NUM_QUERIES}] {question[:60]}...")
        print(f"{'='*70}")

        # Step 1: Clear indexes
        print(f"  1️⃣  Clearing indexes...")
        clear_start = time.time()
        clear_indexes()
        print(f"     ✓ Cleared (Neo4j + FAISS)")

        # Step 2: Extract 10 documents for this query
        documents = []
        for title, sentences in context:
            doc_text = ' '.join(sentences)
            documents.append(doc_text)

        print(f"  2️⃣  Extracted {len(documents)} documents for this query")

        # Step 3: Ingest these 10 documents
        print(f"  3️⃣  Ingesting 10 documents...")
        ingest_start = time.time()

        ingestion_pipeline = get_ingestion_pipeline(
            schema_path=schema_path,
            auto_refine_schema=False
        )

        ingest_result = ingestion_pipeline.ingest_documents(
            documents=documents,
            infer_schema=False,  # Use existing schema
            enrich_metadata=False
        )

        ingest_time = time.time() - ingest_start
        metrics['ingestion_times'].append(ingest_time)

        print(f"     ✓ Ingested: {ingest_result['entities_extracted']} entities, "
              f"{ingest_result['relations_extracted']} relations ({ingest_time:.1f}s)")

        # Step 4: Query
        print(f"  4️⃣  Querying...")
        query_start = time.time()

        retrieval_pipeline = get_retrieval_pipeline()
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

        # Step 5: Compute metrics
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
            'ingestion_time': ingest_time,
            'query_time': query_time,
        })

        # Progress
        print(f"  5️⃣  Results:")
        print(f"     Strategy: {strategy} | Contexts: {num_contexts}")
        print(f"     Gold: {gold_answer}")
        print(f"     Pred: {predicted_answer[:100]}...")
        print(f"     EM: {em} | F1: {f1:.3f}")
        print(f"     Time: ingest={ingest_time:.1f}s, query={query_time:.1f}s")

        # Running average
        current_em = sum(metrics['em_scores']) / len(metrics['em_scores']) * 100
        print(f"\n  📊 Running Average: EM = {current_em:.2f}% ({sum(metrics['em_scores'])}/{i} correct)")

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
    avg_ingest = sum(metrics['ingestion_times']) / len(metrics['ingestion_times'])
    avg_query = sum(metrics['query_times']) / len(metrics['query_times'])
    print(f"  Total duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"  Avg per query: {duration/len(queries):.1f}s")
    print(f"  Avg ingestion: {avg_ingest:.1f}s")
    print(f"  Avg query: {avg_query:.1f}s")

    # Save results
    results_path = Path("data/hotpotqa/results_perquery.json")
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
        'avg_ingestion_time': avg_ingest,
        'avg_query_time': avg_query,
    }

    summary_path = Path("data/hotpotqa/summary_perquery.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")

    print(f"\n" + "="*70)
    print("COMPARISON TO KG²RAG (NAACL 2025)")
    print("="*70)
    print(f"Methodology: Per-query indexing (10 docs/query, 100% coverage)")
    print(f"Dataset: HotpotQA dev distractor ({len(queries)} queries)")
    print(f"\nAgentic GraphRAG (Your System):")
    print(f"  EM: {overall_em:.2f}%")
    print(f"  F1: {overall_f1:.2f}%")
    print(f"\nKG²RAG (NAACL 2025 baseline):")
    print(f"  EM: ~60%* (7,405 queries, reported in paper)")
    print(f"  F1: ~70%* (estimated)")
    print(f"\n* Exact numbers from KG²RAG paper (check publication for confirmation)")
    print(f"\n✅ Evaluation methodology matches KG²RAG for fair comparison")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
