#!/usr/bin/env python3
"""
Test Single Query on HotpotQA Distractor Setting

Tests the first query from the distractor dataset to verify:
1. System can retrieve documents from 991-doc corpus
2. Routing works (VECTOR vs GRAPH)
3. Answer generation works
4. Can compare to gold answer
"""

import json
from pathlib import Path
from src.pipeline import get_retrieval_pipeline


def main():
    print("\n" + "="*70)
    print("SINGLE QUERY TEST - HOTPOTQA DISTRACTOR")
    print("="*70)

    # Load queries
    queries_path = Path("data/hotpotqa/queries_distractor_100.json")

    if not queries_path.exists():
        print(f"❌ Error: {queries_path} not found!")
        print("   Make sure ingestion completed and saved this file.")
        return 1

    with open(queries_path, 'r') as f:
        queries = json.load(f)

    # Test first query
    query = queries[0]

    print(f"\n📋 Query Details:")
    print(f"  ID: {query['_id']}")
    print(f"  Type: {query['type']}")
    print(f"  Question: {query['question']}")
    print(f"  Gold Answer: {query['answer']}")

    # Show supporting facts
    if 'supporting_facts' in query:
        print(f"\n  Supporting Facts:")
        for fact in query['supporting_facts'][:3]:
            print(f"    - {fact[0]}: sentence {fact[1]}")
        if len(query['supporting_facts']) > 3:
            print(f"    ... and {len(query['supporting_facts']) - 3} more")

    # Initialize pipeline
    print(f"\n🔧 Initializing retrieval pipeline...")
    pipeline = get_retrieval_pipeline()

    # Verify system state
    print(f"\n🔍 Checking system state...")
    from src.graph.neo4j_manager import get_neo4j_manager
    from src.vector.faiss_index import get_faiss_index

    neo4j = get_neo4j_manager()
    faiss = get_faiss_index()

    neo4j_nodes = neo4j.get_node_count()
    neo4j_rels = neo4j.get_relationship_count()
    faiss_vectors = faiss.get_count()

    print(f"  Neo4j nodes: {neo4j_nodes}")
    print(f"  Neo4j relationships: {neo4j_rels}")
    print(f"  FAISS vectors: {faiss_vectors}")

    if neo4j_nodes == 0:
        print(f"\n❌ Error: Neo4j is empty! Ingestion may not have completed.")
        return 1

    if faiss_vectors == 0:
        print(f"\n❌ Error: FAISS is empty! Ingestion may not have completed.")
        return 1

    # Run query
    print(f"\n🚀 Running query...")
    print(f"  (Using cache=False to test fresh routing)")

    result = pipeline.query(
        query['question'],
        use_cache=False,
        evaluate=False
    )

    # Display results
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n📊 Retrieval:")
    print(f"  Strategy: {result.get('strategy', 'unknown')}")
    print(f"  Contexts retrieved: {result.get('num_contexts', 0)}")

    if result.get('num_contexts', 0) > 0:
        print(f"\n  Retrieved contexts:")
        contexts = result.get('contexts', [])
        for i, ctx in enumerate(contexts[:3], 1):
            preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
            print(f"    {i}. {preview}")
        if len(contexts) > 3:
            print(f"    ... and {len(contexts) - 3} more contexts")
    else:
        print(f"  ⚠️  No contexts retrieved!")

    print(f"\n🎯 Answer:")
    predicted = result.get('answer', '').strip()
    if predicted:
        print(f"  Predicted: {predicted}")
    else:
        print(f"  ⚠️  Empty prediction!")

    print(f"  Gold: {query['answer']}")

    # Check if correct
    import re
    import string

    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            return text.translate(str.maketrans('', '', string.punctuation))
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_em(prediction, gold):
        if not prediction or not prediction.strip():
            return 0
        return int(normalize_answer(prediction) == normalize_answer(gold))

    em = compute_em(predicted, query['answer'])
    print(f"\n✅ Exact Match: {'YES' if em == 1 else 'NO'} (EM={em})")

    # Detailed comparison
    print(f"\n📝 Normalized Comparison:")
    print(f"  Predicted (normalized): '{normalize_answer(predicted)}'")
    print(f"  Gold (normalized): '{normalize_answer(query['answer'])}'")

    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    status = "✅ PASS" if em == 1 else "❌ FAIL"
    print(f"\nStatus: {status}")
    print(f"  - System retrieved {result.get('num_contexts', 0)} contexts")
    print(f"  - Used {result.get('strategy', 'unknown')} strategy")
    print(f"  - Generated {'an' if predicted else 'no'} answer")
    print(f"  - EM score: {em}")

    if em == 0 and result.get('num_contexts', 0) > 0:
        print(f"\n💡 Note: System retrieved contexts but answered incorrectly.")
        print(f"   This suggests reasoning issue, not retrieval issue.")

    if result.get('num_contexts', 0) == 0:
        print(f"\n💡 Note: System retrieved no contexts.")
        print(f"   This suggests retrieval issue (can't find docs in 991-doc corpus).")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
