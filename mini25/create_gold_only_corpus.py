#!/usr/bin/env python3
"""
Create a gold-only corpus for Mini-25 evaluation.

Filters corpus.json to keep only supporting (gold) documents,
removing all 192 distractors. Results in 50 documents (2 per query).
"""

import json
from pathlib import Path


def main():
    # Load full corpus
    corpus_path = Path("data/hotpotqa/mini_25/corpus.json")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    # Filter to gold-only
    gold_corpus = [doc for doc in corpus if doc['is_supporting']]

    # Re-index doc_ids sequentially
    for i, doc in enumerate(gold_corpus):
        doc['original_doc_id'] = doc['doc_id']
        doc['doc_id'] = i

    print(f"Original corpus: {len(corpus)} documents")
    print(f"  Supporting: {sum(1 for d in corpus if d['is_supporting'])}")
    print(f"  Distractors: {sum(1 for d in corpus if not d['is_supporting'])}")
    print(f"\nGold-only corpus: {len(gold_corpus)} documents")

    # Verify 2 gold docs per query
    from collections import Counter
    query_counts = Counter()
    for doc in gold_corpus:
        for qid in doc['for_queries']:
            query_counts[qid] += 1

    print(f"\nDocs per query:")
    for qid in sorted(query_counts.keys(), key=lambda x: int(x[1:])):
        print(f"  {qid}: {query_counts[qid]} gold docs")

    assert all(c == 2 for c in query_counts.values()), "Expected exactly 2 gold docs per query!"

    # Build new mapping
    mapping = {}
    for doc in gold_corpus:
        for qid in doc['for_queries']:
            if qid not in mapping:
                mapping[qid] = []
            mapping[qid].append(doc['doc_id'])

    # Save outputs
    out_dir = Path("data/hotpotqa/mini_25_gold_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "corpus.json", 'w') as f:
        json.dump(gold_corpus, f, indent=2)

    with open(out_dir / "mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)

    # Copy queries (unchanged)
    queries_path = Path("data/hotpotqa/mini_25/queries.json")
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    with open(out_dir / "queries.json", 'w') as f:
        json.dump(queries, f, indent=2)

    print(f"\nSaved gold-only corpus to: {out_dir}/")
    print(f"  corpus.json: {len(gold_corpus)} docs")
    print(f"  mapping.json: {len(mapping)} query mappings")
    print(f"  queries.json: {len(queries)} queries (copied)")


if __name__ == "__main__":
    main()
