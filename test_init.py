#!/usr/bin/env python3
"""Test pipeline initialization to find bottleneck."""

import time

print("=" * 50)
print("TESTING PIPELINE INITIALIZATION")
print("=" * 50)

print("\n[1/5] Importing modules...")
start = time.time()
from src.pipeline import get_ingestion_pipeline
print(f"✓ Imports done ({time.time() - start:.1f}s)")

print("\n[2/5] Loading schema...")
start = time.time()
from pathlib import Path
schema_path = Path("data/hotpotqa/schema.json")
print(f"✓ Schema path: {schema_path} ({'exists' if schema_path.exists() else 'not found'})")
print(f"✓ Schema path loaded ({time.time() - start:.1f}s)")

print("\n[3/5] Initializing pipeline...")
start = time.time()
pipeline = get_ingestion_pipeline(
    schema_path=schema_path,
    auto_refine_schema=True
)
print(f"✓ Pipeline initialized ({time.time() - start:.1f}s)")

print("\n[4/5] Getting statistics...")
start = time.time()
stats = pipeline.get_statistics()
print(f"✓ Stats retrieved ({time.time() - start:.1f}s)")
print(f"  Neo4j nodes: {stats['neo4j_stats']['total_nodes']}")
print(f"  FAISS vectors: {stats['faiss_stats']['total_vectors']}")

print("\n[5/5] Loading corpus...")
start = time.time()
import json
with open("data/hotpotqa/corpus.json", 'r') as f:
    corpus = json.load(f)
documents = [doc['text'] for doc in corpus[:10]]  # Just 10 docs for testing
print(f"✓ Loaded {len(documents)} test documents ({time.time() - start:.1f}s)")

print("\n" + "=" * 50)
print("ALL INITIALIZATION STEPS COMPLETED")
print("=" * 50)
