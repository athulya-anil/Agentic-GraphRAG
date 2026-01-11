#!/usr/bin/env python3
"""
HotpotQA DISTRACTOR Setting Ingestion

Follows KG²RAG methodology for research-grade evaluation:
1. Uses hotpot_dev_distractor_v1.json (standard benchmark)
2. Each query comes with 10 documents (2 gold + 8 distractors)
3. Ingests all documents for N queries
4. Guarantees 100% coverage for fair evaluation

Strategy:
- Load first N queries from distractor dataset
- Extract all 10 documents per query from context field
- Deduplicate documents across queries (same title = same doc)
- Ingest unique documents
- Report coverage statistics
"""

import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from src.pipeline import get_ingestion_pipeline


def main():
    print("\n" + "="*70)
    print("HOTPOTQA DISTRACTOR SETTING INGESTION")
    print("Following KG²RAG methodology (NAACL 2025)")
    print("="*70)

    distractor_path = Path("data/hotpotqa/hotpot_dev_distractor.json")
    schema_path = Path("data/hotpotqa/schema.json")

    # Configuration
    NUM_QUERIES = 100  # Start with 100 queries for validation
    # Full evaluation would use all 7,405 queries

    # Load distractor dataset
    print(f"\n1. Loading distractor dataset...")
    with open(distractor_path, 'r', encoding='utf-8') as f:
        all_queries = json.load(f)
    print(f"✓ Loaded {len(all_queries)} total queries")

    # Sample first N queries
    queries_subset = all_queries[:NUM_QUERIES]
    print(f"✓ Selected first {NUM_QUERIES} queries for evaluation")

    # Extract documents from context field
    print(f"\n2. Extracting documents from query contexts...")
    title_to_text = {}  # Deduplicate by title
    title_to_queries = defaultdict(list)  # Track which queries use each doc

    for i, query in enumerate(queries_subset):
        q_id = query['_id']
        context = query['context']  # List of [title, sentences] pairs

        for title, sentences in context:
            # Reconstruct full document text
            doc_text = ' '.join(sentences)

            # Store unique documents
            if title not in title_to_text:
                title_to_text[title] = doc_text

            # Track query usage
            title_to_queries[title].append(i)

    unique_docs = len(title_to_text)
    total_doc_slots = NUM_QUERIES * 10

    print(f"✓ Query contexts: {NUM_QUERIES} queries × 10 docs = {total_doc_slots} total")
    print(f"✓ Unique documents: {unique_docs} (after deduplication)")
    print(f"✓ Deduplication rate: {(1 - unique_docs/total_doc_slots)*100:.1f}%")

    # Show most common documents
    doc_counts = {title: len(queries) for title, queries in title_to_queries.items()}
    most_common = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nMost shared documents:")
    for title, count in most_common:
        print(f"  {title}: used in {count} queries")

    # Verify coverage
    print(f"\n3. Verifying coverage...")
    coverage_stats = {
        'full_coverage': 0,      # All supporting docs present
        'partial_coverage': 0,   # Some supporting docs present
        'no_coverage': 0         # No supporting docs present
    }

    for query in queries_subset:
        supporting_titles = set(fact[0] for fact in query['supporting_facts'])
        available_titles = set(title_to_text.keys())

        overlap = supporting_titles & available_titles

        if len(overlap) == len(supporting_titles):
            coverage_stats['full_coverage'] += 1
        elif len(overlap) > 0:
            coverage_stats['partial_coverage'] += 1
        else:
            coverage_stats['no_coverage'] += 1

    print(f"✓ Full coverage: {coverage_stats['full_coverage']}/{NUM_QUERIES} queries ({coverage_stats['full_coverage']/NUM_QUERIES*100:.1f}%)")
    print(f"  Partial coverage: {coverage_stats['partial_coverage']}/{NUM_QUERIES} queries")
    print(f"  No coverage: {coverage_stats['no_coverage']}/{NUM_QUERIES} queries")

    if coverage_stats['full_coverage'] < NUM_QUERIES:
        print(f"\n⚠️  WARNING: Expected 100% coverage in distractor setting!")
        print(f"   This suggests data processing issue - investigating...")

    # Prepare documents for ingestion
    documents = list(title_to_text.values())
    doc_titles = list(title_to_text.keys())

    print(f"\n4. Documents prepared for ingestion:")
    print(f"  Total unique documents: {len(documents)}")
    print(f"  Average doc length: {sum(len(d) for d in documents) / len(documents):.0f} chars")

    # Initialize pipeline
    print(f"\n5. Initializing ingestion pipeline...")
    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=False
    )
    print("✓ Pipeline initialized")

    # Estimate time
    estimated_time = len(documents) * 1.5  # ~1.5 min per doc
    print(f"\n6. Ingesting {len(documents)} documents...")
    print(f"(Estimated time: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes)")
    print(f"Rate limiting: Groq allows 30 req/min, expect some delays\n")

    # Ingest
    start_time = time.time()

    result = pipeline.ingest_documents(
        documents=documents,
        infer_schema=True,
        enrich_metadata=False
    )

    duration = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("INGESTION COMPLETE")
    print("="*70)
    print(f"\nStatistics:")
    print(f"  Documents processed: {result['documents_processed']}")
    print(f"  Entities extracted: {result['entities_extracted']}")
    print(f"  Entities after dedup: {result['entities_after_dedup']}")
    print(f"  Relations extracted: {result['relations_extracted']}")
    print(f"  Relations after resolution: {result['relations_after_resolution']}")
    print(f"  Neo4j nodes created: {result['nodes_created']}")
    print(f"  Neo4j edges created: {result['edges_created']}")
    print(f"  Conflicts resolved: {result['conflicts_resolved']}")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")

    # Verify
    stats = pipeline.get_statistics()
    print(f"\nFinal counts:")
    print(f"  Neo4j nodes: {stats['neo4j_stats']['total_nodes']}")
    print(f"  Neo4j relationships: {stats['neo4j_stats']['total_relationships']}")
    print(f"  FAISS vectors: {stats['faiss_stats']['total_vectors']}")

    print(f"\n✓ Schema saved to: {schema_path}")

    # Save query subset for evaluation
    eval_queries_path = Path(f"data/hotpotqa/queries_distractor_{NUM_QUERIES}.json")
    with open(eval_queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries_subset, f, indent=2)
    print(f"✓ Evaluation queries saved to: {eval_queries_path}")

    print("\n" + "="*70)
    print("✅ READY FOR EVALUATION")
    print("="*70)
    print(f"\nDataset: HotpotQA dev distractor (first {NUM_QUERIES} queries)")
    print(f"Coverage: {coverage_stats['full_coverage']}/{NUM_QUERIES} queries (100% expected)")
    print(f"Documents: {len(documents)} unique documents ingested")
    print(f"\nRun evaluation:")
    print(f"  python evaluate_hotpotqa_distractor.py")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
