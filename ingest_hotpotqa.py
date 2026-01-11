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

    corpus_path = Path("data/hotpotqa/test_dataset/test_corpus.json")  # 991 unique docs (2 supporting + 8 distractors per question)
    schema_path = Path("data/hotpotqa/schema.json")

    # Load test corpus
    print("\nLoading test corpus (991 docs: 2 supporting + 8 distractors per 100 questions)...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # Process all test corpus documents (991 unique docs: 2 supporting + 8 distractors per question)
    # Using Groq llama-3.3-70b-versatile for complete JSON extraction
    documents = [doc['text'] for doc in corpus]
    print(f"✓ Loaded {len(documents)} documents (test corpus)")
    print(f"   Using Groq llama-3.3-70b-versatile (~20-30 min for 991 docs)")

    # Initialize pipeline
    print("\nInitializing ingestion pipeline...")
    pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=False  # Use pre-defined schema without refinement
    )
    print("✓ Pipeline initialized")

    # Ingest
    print(f"\nIngesting {len(documents)} documents...")
    estimated_time = len(documents) * 1.5  # ~1.5 min per document (LLM API calls)
    print(f"(Estimated time: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes due to LLM API rate limits)\n")

    start_time = time.time()

    result = pipeline.ingest_documents(
        documents=documents,
        infer_schema=True,
        enrich_metadata=False  # Disabled for performance
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
