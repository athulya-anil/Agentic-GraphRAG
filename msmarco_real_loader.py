#!/usr/bin/env python3
"""
Real MS MARCO Dataset Loader for Agentic GraphRAG

Loads the actual MS MARCO dataset from BEIR format and samples
a subset for evaluation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm


class RealMSMARCOLoader:
    """Load and sample from the real MS MARCO dataset."""

    def __init__(self, data_dir: Path = Path("data/raw/msmarco_full_dataset")):
        """Initialize loader.

        Args:
            data_dir: Directory containing MS MARCO data
        """
        self.data_dir = Path(data_dir)
        self.corpus_file = self.data_dir / "corpus.jsonl"
        self.queries_file = self.data_dir / "queries.jsonl"
        self.qrels_file = self.data_dir / "qrels" / "dev.tsv"

        # Verify files exist
        for f in [self.corpus_file, self.queries_file, self.qrels_file]:
            if not f.exists():
                raise FileNotFoundError(f"Missing file: {f}")

    def load_qrels(self) -> Dict[str, List[str]]:
        """Load relevance judgments.

        Returns:
            Dict mapping query_id to list of relevant passage_ids
        """
        print("   Loading relevance judgments...")
        qrels = {}

        with open(self.qrels_file, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id, passage_id, score = parts[0], parts[1], parts[2]
                    if int(score) > 0:  # Only positive relevance
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(passage_id)

        print(f"   ✓ Loaded {len(qrels)} queries with relevance judgments")
        return qrels

    def load_queries(self, query_ids: Set[str]) -> Dict[str, str]:
        """Load query texts for given IDs.

        Args:
            query_ids: Set of query IDs to load

        Returns:
            Dict mapping query_id to query_text
        """
        print(f"   Loading {len(query_ids)} queries...")
        queries = {}

        with open(self.queries_file, 'r') as f:
            for line in tqdm(f, desc="   Scanning queries"):
                data = json.loads(line)
                if data['_id'] in query_ids:
                    queries[data['_id']] = data['text']
                    if len(queries) == len(query_ids):
                        break

        print(f"   ✓ Loaded {len(queries)} queries")
        return queries

    def load_passages(self, passage_ids: Set[str]) -> List[Dict[str, Any]]:
        """Load passage texts for given IDs.

        Args:
            passage_ids: Set of passage IDs to load

        Returns:
            List of passage dictionaries
        """
        print(f"   Loading {len(passage_ids)} passages from corpus...")
        passages = []
        found_ids = set()

        with open(self.corpus_file, 'r') as f:
            for line in tqdm(f, desc="   Scanning corpus", total=8841823):
                data = json.loads(line)
                if data['_id'] in passage_ids:
                    passages.append({
                        'id': data['_id'],
                        'text': data['text'],
                        'title': data.get('title', '')
                    })
                    found_ids.add(data['_id'])
                    if len(found_ids) == len(passage_ids):
                        break

        print(f"   ✓ Loaded {len(passages)} passages")
        return passages

    def sample_evaluation_set(
        self,
        num_queries: int = 5000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Sample a subset of MS MARCO for evaluation.

        Strategy:
        1. Load all qrels (query-passage relevance)
        2. Sample queries that have relevance judgments
        3. Load ALL passages needed for those queries (no limiting)
        4. This ensures every query has its ground truth in the corpus

        Args:
            num_queries: Number of queries to sample
            seed: Random seed for reproducibility

        Returns:
            Evaluation set dictionary
        """
        random.seed(seed)

        print("\n📚 Preparing real MS MARCO evaluation set...")

        # Step 1: Load relevance judgments
        qrels = self.load_qrels()

        # Step 2: Sample queries
        all_query_ids = list(qrels.keys())
        sampled_query_ids = random.sample(
            all_query_ids,
            min(num_queries, len(all_query_ids))
        )

        # Step 3: Get ALL relevant passage IDs for sampled queries
        # No limiting - every query must have its relevant passages
        relevant_passage_ids = set()
        sampled_qrels = {}
        for qid in sampled_query_ids:
            sampled_qrels[qid] = qrels[qid]
            relevant_passage_ids.update(qrels[qid])

        print(f"   Sampled {len(sampled_query_ids)} queries")
        print(f"   Loading ALL {len(relevant_passage_ids)} relevant passages (no limit)")

        # Step 4: Load actual texts - ALL passages needed
        passages = self.load_passages(relevant_passage_ids)
        queries = self.load_queries(set(sampled_query_ids))

        # Build evaluation set
        eval_set = {
            'passages': passages,
            'queries': queries,
            'qrels': sampled_qrels,
            'metadata': {
                'num_passages': len(passages),
                'num_queries': len(queries),
                'source': 'MS MARCO (real dataset)',
                'seed': seed,
            }
        }

        # Save to file
        output_dir = Path("data/msmarco")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "real_evaluation_set.json"

        with open(output_file, 'w') as f:
            json.dump(eval_set, f, indent=2)

        print(f"\n✓ Prepared {len(passages)} passages and {len(queries)} queries")
        print(f"✓ Saved to: {output_file}")

        return eval_set

    def get_statistics(self, eval_set: Dict[str, Any]) -> None:
        """Print statistics about the evaluation set."""
        passages = eval_set['passages']
        queries = eval_set['queries']
        qrels = eval_set['qrels']

        avg_passage_len = sum(len(p['text'].split()) for p in passages) / len(passages)
        avg_query_len = sum(len(q.split()) for q in queries.values()) / len(queries)

        print("\n📊 Evaluation Set Statistics:")
        print(f"   • Passages: {len(passages)}")
        print(f"   • Queries: {len(queries)}")
        print(f"   • Avg passage length: {avg_passage_len:.1f} words")
        print(f"   • Avg query length: {avg_query_len:.1f} words")
        print(f"   • Queries with relevance judgments: {len(qrels)}")

        # Show sample
        print("\n📝 Sample query:")
        sample_qid = list(queries.keys())[0]
        print(f"   Q: {queries[sample_qid]}")
        if sample_qid in qrels:
            print(f"   Relevant passages: {qrels[sample_qid]}")


def main():
    """Main function to prepare real MS MARCO evaluation set."""
    print("=" * 70)
    print("  REAL MS MARCO DATASET LOADER")
    print("=" * 70)

    loader = RealMSMARCOLoader()

    # Prepare evaluation set with 5000 queries
    # All relevant passages will be loaded (no limiting)
    eval_set = loader.sample_evaluation_set(
        num_queries=5000,
        seed=42
    )

    # Show statistics
    loader.get_statistics(eval_set)

    print("\n" + "=" * 70)
    print("  ✅ REAL MS MARCO EVALUATION SET READY")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Run evaluation with: python evaluate_real_msmarco.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
