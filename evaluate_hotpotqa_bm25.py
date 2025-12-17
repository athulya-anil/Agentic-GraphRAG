#!/usr/bin/env python3
"""
BM25 Baseline Evaluation on HotpotQA (EM metric)
"""

import json
import time
import re
import string
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.utils.llm_client import get_llm_client
from tqdm import tqdm


def extract_answer(llm_response: str) -> str:
    """Extract answer from LLM response."""
    answer = llm_response.strip().split('\n')[0]

    prefixes = [
        r'the answer is:?\s*',
        r'answer:?\s*',
        r'based on the (context|passage|document),?\s*',
        r'according to the (text|passage|document|context),?\s*',
        r'the (text|passage|document) (says|states) that\s*',
        r'from the (context|passage|document),?\s*',
    ]

    for prefix in prefixes:
        answer = re.sub(f'^{prefix}', '', answer, flags=re.IGNORECASE)

    refusal_patterns = [
        r"i (don't|do not|dont) know",
        r"i (can't|cannot|can not) (answer|determine|find)",
        r"(there is )?no (clear|sufficient|enough) (answer|information)",
        r"not enough information",
        r"unable to (answer|determine)",
        r"cannot be determined",
        r"insufficient (information|context)",
    ]

    answer_lower = answer.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, answer_lower):
            return ""

    return answer.strip()


def normalize_answer(s: str) -> str:
    """Normalize answer for EM (HotpotQA standard)."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


def compute_em(prediction: str, gold: str) -> int:
    """Compute Exact Match."""
    if not prediction or not prediction.strip():
        return 0
    return int(normalize_answer(prediction) == normalize_answer(gold))


def main():
    print("\n" + "="*70)
    print("BM25 BASELINE EVALUATION")
    print("="*70)

    corpus_path = Path("data/hotpotqa/corpus.json")
    queries_path = Path("data/hotpotqa/queries_100.json")
    output_path = Path("data/hotpotqa/results_bm25.json")

    # Load corpus
    print("\nLoading corpus...")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    corpus_texts = [doc['text'] for doc in corpus]
    print(f"✓ Loaded {len(corpus)} documents")

    # Build BM25 index
    print("\nBuilding BM25 index...")
    tokenized = [text.split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    print("✓ BM25 index built")

    # Load queries
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    print(f"✓ Loaded {len(queries)} queries")

    # Initialize LLM
    llm = get_llm_client()

    # Evaluate
    results = []
    start_time = time.time()

    for query_obj in tqdm(queries, desc="BM25 Evaluation"):
        query_text = query_obj['question']
        gold_answer = query_obj['answer']

        # Retrieve with BM25
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

        retrieved_docs = [corpus_texts[idx] for idx in top_k_idx]
        retrieved_titles = [corpus[idx]['title'] for idx in top_k_idx]

        # Generate answer
        context = "\n\n".join([f"[{i+1}] {doc[:500]}..." for i, doc in enumerate(retrieved_docs)])

        prompt = f"""Answer the following question using the provided context. Give a concise answer (1-10 words).

Context:
{context}

Question: {query_text}

Answer:"""

        llm_response = llm.generate(prompt, temperature=0.0)
        predicted_answer = extract_answer(llm_response)

        # Compute EM
        em = compute_em(predicted_answer, gold_answer)

        results.append({
            'query_id': query_obj['query_id'],
            'question': query_text,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'em': em,
            'type': query_obj['type'],
            'retrieved_titles': retrieved_titles
        })

    duration = time.time() - start_time

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Compute metrics
    overall_em = sum(r['em'] for r in results) / len(results)
    comparison_em = sum(r['em'] for r in results if r['type'] == 'comparison') / sum(1 for r in results if r['type'] == 'comparison')
    bridge_em = sum(r['em'] for r in results if r['type'] == 'bridge') / sum(1 for r in results if r['type'] == 'bridge')

    # Summary
    print("\n" + "="*70)
    print("BM25 BASELINE RESULTS")
    print("="*70)
    print(f"\nOverall EM: {overall_em:.3f} ({overall_em*100:.1f}%)")
    print(f"  Comparison EM: {comparison_em:.3f}")
    print(f"  Bridge EM: {bridge_em:.3f}")
    print(f"\nCorrect: {sum(r['em'] for r in results)}/{len(results)}")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"\n✓ Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
