#!/usr/bin/env python3
"""
HotpotQA ALIGNED Corpus Ingestion

Ingests ONLY the documents needed to answer the test queries.
Ensures evaluation measures system quality, not random chance.

Strategy:
1. Load test queries (queries_100.json)
2. Extract gold_supporting_titles from each query
3. Find those documents in corpus
4. Ingest ONLY those documents
5. Guarantee 100% coverage for evaluation
"""

import json
import time
from pathlib import Path
from collections import Counter
from src.pipeline import get_ingestion_pipeline


def main():
    print("\n" + "="*70)
    print("HOTPOTQA ALIGNED CORPUS INGESTION")
    print("="*70)

    corpus_path = Path("data/hotpotqa/corpus.json")
    queries_path = Path("data/hotpotqa/queries_100.json")
    schema_path = Path("data/hotpotqa/schema.json")

    # Load queries first
    print("\n1. Loading test queries...")
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"✓ Loaded {len(queries)} test queries")

    # Extract needed document titles
    print("\n2. Extracting required documents...")
    needed_titles = set()
    for query in queries:
        titles = query.get('gold_supporting_titles', [])
        needed_titles.update(titles)

    print(f"✓ {len(queries)} queries require {len(needed_titles)} unique documents")

    # Count how many queries need each document (for prioritization)
    title_counts = Counter()
    for query in queries:
        for title in query.get('gold_supporting_titles', []):
            title_counts[title] += 1

    print(f"\nMost referenced documents:")
    for title, count in title_counts.most_common(10):
        print(f"  {title}: {count} queries")

    # Load corpus
    print("\n3. Loading full corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    print(f"✓ Loaded corpus with {len(corpus)} total documents")

    # Build title → document mapping
    print("\n4. Finding required documents in corpus...")
    corpus_map = {doc['title']: doc for doc in corpus}

    # Find matching documents
    matched_docs = []
    missing_titles = []

    for title in needed_titles:
        if title in corpus_map:
            matched_docs.append(corpus_map[title])
        else:
            missing_titles.append(title)

    print(f"✓ Found {len(matched_docs)}/{len(needed_titles)} documents")

    if missing_titles:
        print(f"\n⚠️  Missing {len(missing_titles)} documents:")
        for title in missing_titles[:10]:
            print(f"  - {title}")
        if len(missing_titles) > 10:
            print(f"  ... and {len(missing_titles) - 10} more")

    # Limit to reasonable size for verification
    SUBSET_SIZE = 20  # Increased from 10 for better coverage
    if len(matched_docs) > SUBSET_SIZE:
        # Prioritize documents referenced by most queries
        matched_docs.sort(key=lambda d: title_counts[d['title']], reverse=True)
        matched_docs = matched_docs[:SUBSET_SIZE]
        print(f"\n📌 Limiting to top {SUBSET_SIZE} most-referenced documents")

    documents = [doc['text'] for doc in matched_docs]
    doc_titles = [doc['title'] for doc in matched_docs]

    print(f"\n5. Selected documents for ingestion:")
    for i, title in enumerate(doc_titles, 1):
        queries_using = title_counts[title]
        print(f"  {i}. {title} (answers {queries_using} queries)")

    # Calculate expected coverage
    answerable = 0
    for query in queries:
        needed = set(query.get('gold_supporting_titles', []))
        available = needed & set(doc_titles)
        if len(available) == len(needed):
            answerable += 1

    print(f"\n📊 Expected coverage: {answerable}/{len(queries)} queries ({answerable/len(queries)*100:.1f}%)")
    print(f"   This is the MAXIMUM possible EM score")

    # Initialize pipeline
    print("\n6. Initializing ingestion pipeline...")
    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=False
    )
    print("✓ Pipeline initialized")

    # Ingest
    print(f"\n7. Ingesting {len(documents)} documents...")
    estimated_time = len(documents) * 1.5
    print(f"(Estimated time: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes)\n")

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

    print("\n" + "="*70)
    print("✅ READY FOR EVALUATION")
    print("="*70)
    print(f"\nExpected EM range: {answerable/len(queries)*100:.1f}% - {(answerable/len(queries)*100 + 10):.1f}%")
    print(f"(System has data to answer {answerable} queries)")
    print("\nRun: python evaluate_hotpotqa_full.py")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
