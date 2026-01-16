#!/usr/bin/env python3
"""
Mini-5 HotpotQA Corpus Ingestion

Ingests the 50 documents from mini_5 test dataset into Neo4j + FAISS.
This is a ONE-TIME ingestion for debugging/sanity testing.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline
from src.graph import get_neo4j_manager
from src.vector import get_faiss_index


def main():
    print("\n" + "="*70)
    print("MINI-5 HOTPOTQA CORPUS INGESTION")
    print("="*70)

    # Load corpus
    corpus_path = Path("data/hotpotqa/mini_5/corpus.json")

    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found!")
        return 1

    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    print(f"\nDataset Info:")
    print(f"  Documents: {len(corpus)}")
    print(f"  Supporting: {sum(1 for d in corpus if d['is_supporting'])}")
    print(f"  Distractors: {sum(1 for d in corpus if not d['is_supporting'])}")

    # Extract document texts AND metadata (titles)
    documents = [doc['text'] for doc in corpus]
    document_metadata = [{'title': doc['title']} for doc in corpus]

    # Count sentences
    total_sentences = sum(len(doc.get('sentences', [])) for doc in corpus)
    print(f"  Total sentences: {total_sentences}")

    # Initialize ingestion pipeline
    print(f"\nInitializing ingestion pipeline...")
    schema_path = Path("data/hotpotqa/schema.json")

    pipeline = get_ingestion_pipeline(
        schema_path=schema_path if schema_path.exists() else None,
        auto_refine_schema=False
    )

    # Ingest documents WITH metadata
    print(f"\nStarting ingestion of {len(documents)} documents...")

    result = pipeline.ingest_documents(
        documents=documents,
        document_metadata=document_metadata,  # Pass titles!
        infer_schema=not schema_path.exists(),
        enrich_metadata=False
    )

    print(f"\n" + "="*70)
    print("INGESTION COMPLETE")
    print("="*70)

    # Display statistics
    print(f"\nIngestion Results:")
    print(f"  Documents processed: {result['documents_processed']}")
    print(f"  Entities extracted: {result['entities_extracted']}")
    print(f"  Relations extracted: {result['relations_extracted']}")
    print(f"  Processing time: {result['processing_time']:.1f}s")

    # Get graph statistics
    neo4j = get_neo4j_manager()
    faiss = get_faiss_index()

    print(f"\nStorage Statistics:")
    print(f"  Neo4j nodes: {neo4j.get_node_count()}")
    print(f"  Neo4j relationships: {neo4j.get_relationship_count()}")
    print(f"  FAISS vectors: {faiss.get_count()}")

    print(f"\n" + "="*70)
    print("READY FOR EVALUATION")
    print("="*70)
    print(f"\nNext step: Run evaluate_mini5.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
