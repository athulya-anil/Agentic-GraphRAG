#!/usr/bin/env python3
"""
HotpotQA Dataset Loader for Agentic GraphRAG Evaluation

Loads HotpotQA dev distractor set, builds deduplicated corpus,
samples stratified queries, and validates no leakage.

Author: Agentic GraphRAG Evaluation
Date: 2025-12-17
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import hashlib


class HotpotQALoader:
    """Load and process HotpotQA dev distractor set."""

    def __init__(self, data_path: Path):
        """Initialize loader.

        Args:
            data_path: Path to hotpot_dev_distractor_v1.json
        """
        self.data_path = Path(data_path)
        self.examples = []
        self.corpus = []
        self.sampled_queries = []

        # Validate input file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"HotpotQA file not found: {data_path}")

    def load_dev_set(self) -> List[Dict[str, Any]]:
        """Load HotpotQA dev distractor set."""
        print("\n" + "="*70)
        print("STEP 1: LOADING HOTPOTQA DEV SET")
        print("="*70)

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)

        print(f"✓ Loaded {len(self.examples)} examples")

        # Validate structure
        sample = self.examples[0]
        required_fields = ['_id', 'question', 'answer', 'context', 'supporting_facts', 'type']
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"

        print(f"✓ Validated example structure")

        # Show statistics
        types = [ex['type'] for ex in self.examples]
        comparison_count = sum(1 for t in types if t == 'comparison')
        bridge_count = sum(1 for t in types if t == 'bridge')

        print(f"  - Comparison questions: {comparison_count}")
        print(f"  - Bridge questions: {bridge_count}")

        return self.examples

    def build_corpus(self) -> List[Dict[str, Any]]:
        """Build deduplicated corpus from context documents."""
        print("\n" + "="*70)
        print("STEP 2: BUILDING CORPUS (DEDUPLICATION)")
        print("="*70)

        doc_map = {}
        doc_to_examples = defaultdict(list)

        for example in self.examples:
            example_id = example['_id']
            context = example['context']

            for title, sentences in context:
                doc_text = " ".join(sentences)
                content_hash = hashlib.md5(doc_text.encode('utf-8')).hexdigest()[:8]
                doc_key = (title, content_hash)

                if doc_key not in doc_map:
                    doc_id = f"doc_{len(doc_map):06d}"
                    doc_map[doc_key] = {
                        'doc_id': doc_id,
                        'title': title,
                        'text': doc_text,
                        'source': 'hotpotqa',
                        'num_sentences': len(sentences)
                    }

                doc_to_examples[doc_key].append(example_id)

        self.corpus = list(doc_map.values())

        for (title, content_hash), doc in doc_map.items():
            doc['example_ids'] = doc_to_examples[(title, content_hash)]
            doc['num_examples'] = len(doc['example_ids'])

        total_docs_before = sum(len(ex['context']) for ex in self.examples)
        print(f"✓ Built corpus: {len(self.corpus)} unique documents")
        print(f"  - Before deduplication: {total_docs_before} total docs")
        print(f"  - After deduplication: {len(self.corpus)} unique docs")
        print(f"  - Deduplication rate: {(1 - len(self.corpus)/total_docs_before)*100:.1f}%")

        if len(self.corpus) < 10000:
            print(f"⚠️  WARNING: Corpus smaller than expected ({len(self.corpus)} < 10K docs)")
        elif len(self.corpus) > 20000:
            print(f"⚠️  WARNING: Corpus larger than expected ({len(self.corpus)} > 20K docs)")
        else:
            print(f"✓ Corpus size within expected range (10K-20K docs)")

        return self.corpus

    def validate_no_leakage(self) -> bool:
        """Ensure no answer text or question text appears as corpus labels."""
        print("\n" + "="*70)
        print("STEP 3: LEAKAGE VALIDATION")
        print("="*70)

        leakage_found = False

        corpus_fields = set()
        for doc in self.corpus:
            corpus_fields.update(doc.keys())

        forbidden_fields = {'question', 'answer', 'supporting_facts', 'type', 'level'}
        leaked_fields = corpus_fields & forbidden_fields

        if leaked_fields:
            print(f"❌ LEAKAGE: Forbidden fields in corpus: {leaked_fields}")
            leakage_found = True
        else:
            print(f"✓ No forbidden fields in corpus")

        expected_fields = {'doc_id', 'title', 'text', 'source', 'num_sentences', 'example_ids', 'num_examples'}
        actual_fields = corpus_fields

        if actual_fields == expected_fields:
            print(f"✓ Corpus structure is clean (only document metadata)")
        else:
            unexpected = actual_fields - expected_fields
            if unexpected:
                print(f"⚠️  Unexpected fields in corpus: {unexpected}")

        sample_doc = self.corpus[0]
        sample_example = self.examples[0]

        found_in_context = False
        for title, sentences in sample_example['context']:
            if title == sample_doc['title']:
                found_in_context = True
                break

        if found_in_context:
            print(f"✓ Spot check: corpus docs match example context")
        else:
            print(f"⚠️  Spot check failed (might be normal due to deduplication)")

        if not leakage_found:
            print(f"\n✓ LEAKAGE VALIDATION PASSED")

        return not leakage_found

    def sample_queries(self, n_total: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
        """Sample queries stratified by question type."""
        print("\n" + "="*70)
        print("STEP 4: STRATIFIED QUERY SAMPLING")
        print("="*70)

        random.seed(seed)

        comparison = [ex for ex in self.examples if ex['type'] == 'comparison']
        bridge = [ex for ex in self.examples if ex['type'] == 'bridge']

        print(f"Available questions:")
        print(f"  - Comparison: {len(comparison)}")
        print(f"  - Bridge: {len(bridge)}")

        n_per_type = n_total // 2

        sampled_comparison = random.sample(comparison, min(n_per_type, len(comparison)))
        sampled_bridge = random.sample(bridge, min(n_per_type, len(bridge)))

        sampled = sampled_comparison + sampled_bridge
        random.shuffle(sampled)

        self.sampled_queries = []
        for ex in sampled:
            gold_titles = [title for title, sent_idx in ex['supporting_facts']]

            query = {
                'query_id': ex['_id'],
                'question': ex['question'],
                'answer': ex['answer'],
                'type': ex['type'],
                'gold_supporting_titles': gold_titles,
                'all_context_titles': [title for title, sents in ex['context']],
                'level': ex.get('level', 'unknown')
            }
            self.sampled_queries.append(query)

        print(f"\n✓ Sampled {len(self.sampled_queries)} queries (seed={seed}):")
        print(f"  - Comparison: {sum(1 for q in self.sampled_queries if q['type'] == 'comparison')}")
        print(f"  - Bridge: {sum(1 for q in self.sampled_queries if q['type'] == 'bridge')}")

        sample = self.sampled_queries[0]
        print(f"\nSample query:")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Type: {sample['type']}")

        return self.sampled_queries

    def save(self, output_dir: Path):
        """Save corpus and sampled queries."""
        print("\n" + "="*70)
        print("STEP 5: SAVING OUTPUTS")
        print("="*70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = output_dir / "corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved corpus to {corpus_path}")
        print(f"  Size: {corpus_path.stat().st_size / 1024 / 1024:.1f} MB")

        queries_path = output_dir / "queries_100.json"
        with open(queries_path, 'w', encoding='utf-8') as f:
            json.dump(self.sampled_queries, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved queries to {queries_path}")
        print(f"  Size: {queries_path.stat().st_size / 1024:.1f} KB")

        metadata = {
            'dataset': 'HotpotQA dev distractor v1',
            'total_examples': len(self.examples),
            'corpus_size': len(self.corpus),
            'sampled_queries': len(self.sampled_queries),
            'random_seed': 42,
            'created': '2025-12-17'
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")


def main():
    """Main execution function."""

    print("\n" + "="*70)
    print("HOTPOTQA LOADER - AGENTIC GRAPHRAG EVALUATION")
    print("="*70)

    data_path = Path("data/hotpotqa/hotpot_dev_distractor_v1.json")
    output_dir = Path("data/hotpotqa")

    loader = HotpotQALoader(data_path=data_path)

    try:
        loader.load_dev_set()
        loader.build_corpus()

        if not loader.validate_no_leakage():
            raise ValueError("Leakage validation failed!")

        loader.sample_queries(n_total=100, seed=42)
        loader.save(output_dir=output_dir)

        print("\n" + "="*70)
        print("✅ HOTPOTQA INTEGRATION COMPLETE")
        print("="*70)
        print(f"\nOutputs:")
        print(f"  • Corpus: {len(loader.corpus)} documents → data/hotpotqa/corpus.json")
        print(f"  • Queries: {len(loader.sampled_queries)} (50 comparison + 50 bridge) → data/hotpotqa/queries_100.json")
        print(f"  • Metadata: → data/hotpotqa/metadata.json")

        print(f"\n✅ Ready for next step: corpus ingestion into Neo4j + FAISS")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
