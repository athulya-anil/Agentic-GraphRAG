#!/usr/bin/env python3
"""
HotpotQA Corpus Ingestion into Neo4j + FAISS
"""

import json
import time
from pathlib import Path
from src.pipeline import get_ingestion_pipeline


def main():
    print("\n" + "="*70)
    print("HOTPOTQA CORPUS INGESTION")
    print("="*70)

    corpus_path = Path("data/hotpotqa/corpus.json")
    schema_path = Path("data/hotpotqa/schema.json")

    # Load corpus
    print("\nLoading corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    documents = [doc['text'] for doc in corpus]
    print(f"✓ Loaded {len(documents)} documents")

    # Initialize pipeline
    print("\nInitializing ingestion pipeline...")
    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=True
    )
    print("✓ Pipeline initialized")

    # Ingest
    print(f"\nIngesting {len(documents)} documents...")
    print("(This may take 30-60 minutes due to LLM API rate limits)\n")

    start_time = time.time()

    result = pipeline.ingest_documents(
        documents=documents,
        infer_schema=True,
        enrich_metadata=True
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
    print(f"\n✅ Ready for evaluation")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
