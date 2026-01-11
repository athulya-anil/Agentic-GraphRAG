#!/usr/bin/env python3
"""
HotpotQA 100-Query Corpus Ingestion

Extracts all unique documents from 100 test queries and ingests them ONCE.
This provides 100% coverage for evaluation while avoiding repeated graph building.

Approach:
1. Load 100 queries from hotpot_dev_distractor.json
2. Extract all documents from their contexts (deduplicate by title)
3. Ingest once into shared graph
4. Evaluation can then query this graph 100 times (no re-ingestion)

This is more efficient than per-query clearing/rebuilding.
"""

import json
from pathlib import Path
from src.pipeline import get_ingestion_pipeline

def main():
    print("\n" + "="*70)
    print("HOTPOTQA 100-QUERY CORPUS INGESTION")
    print("="*70)

    # Load distractor dataset
    distractor_path = Path("data/hotpotqa/hotpot_dev_distractor.json")

    if not distractor_path.exists():
        print(f"❌ Error: {distractor_path} not found!")
        return 1

    with open(distractor_path, 'r') as f:
        all_queries = json.load(f)

    # Take first 100 queries
    NUM_QUERIES = 100
    queries = all_queries[:NUM_QUERIES]

    print(f"\n📊 Dataset Info:")
    print(f"  Queries: {NUM_QUERIES}")
    print(f"  Docs per query: 10 (2 gold + 8 distractors)")

    # Extract all unique documents
    print(f"\n📥 Extracting documents from query contexts...")

    title_to_doc = {}  # Deduplicate by title

    for i, query_obj in enumerate(queries, 1):
        context = query_obj['context']  # List of [title, sentences]

        for title, sentences in context:
            if title not in title_to_doc:
                doc_text = ' '.join(sentences)
                title_to_doc[title] = {
                    'text': doc_text,
                    'title': title,
                    'first_seen_query': i
                }

    documents = [doc['text'] for doc in title_to_doc.values()]

    print(f"\n📚 Document Statistics:")
    print(f"  Total document slots: {NUM_QUERIES * 10} (100 queries × 10 docs)")
    print(f"  Unique documents: {len(documents)} (after deduplication)")
    print(f"  Deduplication rate: {(1 - len(documents)/(NUM_QUERIES*10))*100:.1f}%")

    # Show coverage guarantee
    print(f"\n✅ Coverage Guarantee:")
    print(f"  All {NUM_QUERIES} queries have their required documents in this corpus")
    print(f"  (Both gold supporting docs + distractors)")

    # Estimate ingestion time
    avg_time_per_doc = 2.6  # From 991-doc ingestion: 43.5 min / 991 docs ≈ 2.6 sec/doc
    estimated_time = len(documents) * avg_time_per_doc / 60
    print(f"\n⏱️  Estimated ingestion time: {estimated_time:.1f} minutes")
    print(f"   (Based on {avg_time_per_doc:.1f} sec/doc from previous ingestion)")

    # Confirm before starting
    response = input(f"\nProceed with ingestion of {len(documents)} documents? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Aborted by user")
        return 1

    # Initialize ingestion pipeline
    print(f"\n🔧 Initializing ingestion pipeline...")
    schema_path = Path("data/hotpotqa/schema.json")

    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=False  # Use existing schema
    )

    # Ingest all documents
    print(f"\n🚀 Starting ingestion...")
    print(f"   This is a ONE-TIME ingestion (no clearing between queries)")

    result = pipeline.ingest_documents(
        documents=documents,
        infer_schema=False,  # Use existing schema
        enrich_metadata=False
    )

    print(f"\n" + "="*70)
    print("INGESTION COMPLETE")
    print("="*70)

    # Display statistics
    print(f"\n📊 Ingestion Results:")
    print(f"  Documents processed: {result['documents_processed']}")
    print(f"  Entities extracted: {result['entities_extracted']}")
    print(f"  Relations extracted: {result['relations_extracted']}")
    print(f"  Processing time: {result['processing_time']:.1f}s ({result['processing_time']/60:.1f} min)")

    # Get graph statistics
    from src.graph.neo4j_manager import get_neo4j_manager
    from src.vector.faiss_index import get_faiss_index

    neo4j = get_neo4j_manager()
    faiss = get_faiss_index()

    print(f"\n📈 Graph Statistics:")
    print(f"  Neo4j nodes: {neo4j.get_node_count()}")
    print(f"  Neo4j relationships: {neo4j.get_relationship_count()}")
    print(f"  FAISS vectors: {faiss.get_count()}")

    # Save query metadata
    queries_metadata = []
    for query_obj in queries:
        queries_metadata.append({
            '_id': query_obj['_id'],
            'question': query_obj['question'],
            'answer': query_obj['answer'],
            'type': query_obj['type'],
            'supporting_facts': query_obj.get('supporting_facts', [])
        })

    metadata_path = Path("data/hotpotqa/queries_100_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(queries_metadata, f, indent=2)

    print(f"\n✓ Saved query metadata to: {metadata_path}")

    print(f"\n" + "="*70)
    print("READY FOR EVALUATION")
    print("="*70)
    print(f"\n✅ The graph now contains all documents needed for 100 queries")
    print(f"✅ You can now run evaluation WITHOUT re-ingestion")
    print(f"✅ 100% coverage guaranteed for all {NUM_QUERIES} queries")
    print(f"\nNext step: Run evaluate_hotpotqa_100queries.py")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
