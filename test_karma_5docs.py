#!/usr/bin/env python3
"""
Test KARMA-inspired enhancements on 5 documents

This script tests:
1. Summarizer Agent
2. Multi-label Relationship Extraction (KARMA RelationAgent)
3. Schema-guided extraction

Compare results vs baseline.
"""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.summarizer_agent import SummarizerAgent
from src.agents.relation_agent_karma import KARMARelationAgent
from src.agents.entity_agent import EntityAgent
from src.agents.conflict_resolution_agent import ConflictResolutionAgent
from src.graph.neo4j_manager import Neo4jManager
from src.vector.faiss_index import FAISSIndex
from src.utils.llm_client import get_llm_client


def main():
    print("\n" + "="*70)
    print("KARMA-INSPIRED ENHANCEMENTS - TEST ON 5 DOCUMENTS")
    print("="*70)

    # Load test corpus
    corpus_path = Path("data/hotpotqa/test_dataset/test_corpus.json")
    print(f"\nLoading corpus from: {corpus_path}")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # Take first 5 documents
    test_docs = corpus[:5]
    documents = [doc['text'] for doc in test_docs]

    print(f"✓ Loaded {len(documents)} test documents")

    # Initialize agents
    print("\nInitializing KARMA agents...")
    llm_client = get_llm_client()
    entity_agent = EntityAgent()
    summarizer_agent = SummarizerAgent(llm_client)
    relation_agent_karma = KARMARelationAgent()

    print("✓ Agents initialized")

    # Phase 1: Entity Extraction
    print("\n" + "="*70)
    print("PHASE 1: ENTITY EXTRACTION")
    print("="*70)

    print("\nExtracting entities from 5 documents...")
    start_time = time.time()

    all_entities = []
    for i, doc in enumerate(documents):
        print(f"  Processing document {i+1}/5...")
        entities = entity_agent.extract_entities_from_text(doc)
        all_entities.append(entities)
        print(f"    → Extracted {len(entities)} entities")

    entity_time = time.time() - start_time
    total_entities = sum(len(e) for e in all_entities)
    print(f"\n✓ Total entities extracted: {total_entities}")
    print(f"  Time: {entity_time:.1f}s")

    # Phase 2: Summarization (KARMA approach)
    print("\n" + "="*70)
    print("PHASE 2: CONTEXT SUMMARIZATION (KARMA)")
    print("="*70)

    print("\nGenerating summaries...")
    start_time = time.time()

    summaries = []
    for i, (doc, entities) in enumerate(zip(documents, all_entities)):
        print(f"  Summarizing document {i+1}/5...")
        summary = summarizer_agent.summarize_for_relationships(doc, entities)
        summaries.append(summary)
        print(f"    {len(doc)} chars → {len(summary)} chars")

    summary_time = time.time() - start_time
    print(f"\n✓ Summaries generated")
    print(f"  Time: {summary_time:.1f}s")

    # Phase 3: Multi-label Relationship Extraction (KARMA)
    print("\n" + "="*70)
    print("PHASE 3: MULTI-LABEL RELATIONSHIP EXTRACTION (KARMA)")
    print("="*70)

    print("\nExtracting relationships (multi-label mode)...")
    start_time = time.time()

    relations = relation_agent_karma.extract_relations_from_documents(
        documents=documents,
        entities_per_doc=all_entities,
        summaries=summaries  # Use summaries for better context
    )

    relation_time = time.time() - start_time
    print(f"\n✓ Relationships extracted: {len(relations)}")
    print(f"  Time: {relation_time:.1f}s")

    # Analyze relationship types
    rel_types = {}
    for rel in relations:
        rel_type = rel.get('type', 'Unknown')
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

    print("\nRelationship type distribution:")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type:<30} {count:>3}")

    # Phase 4: Deduplication
    print("\n" + "="*70)
    print("PHASE 4: ENTITY DEDUPLICATION")
    print("="*70)

    # Flatten entities
    flat_entities = []
    for entities in all_entities:
        flat_entities.extend(entities)

    print(f"\nDeduplicating {len(flat_entities)} entities...")
    start_time = time.time()

    conflict_agent = ConflictResolutionAgent()
    deduplicated_entities = conflict_agent.deduplicate_entities(flat_entities)

    dedup_time = time.time() - start_time
    print(f"\n✓ Entities after dedup: {len(deduplicated_entities)}")
    print(f"  Removed: {len(flat_entities) - len(deduplicated_entities)} duplicates")
    print(f"  Time: {dedup_time:.1f}s")

    # Calculate metrics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    total_time = entity_time + summary_time + relation_time + dedup_time

    print(f"\nDocuments processed: {len(documents)}")
    print(f"Entities extracted: {total_entities}")
    print(f"Entities after dedup: {len(deduplicated_entities)}")
    print(f"Relationships extracted: {len(relations)}")
    print(f"\nRelationships per node: {len(relations) / len(deduplicated_entities):.2f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Compare with baseline (from previous ingestion)
    baseline_rels_per_node = 1.17
    karma_rels_per_node = len(relations) / len(deduplicated_entities) if deduplicated_entities else 0

    print("\n" + "="*70)
    print("COMPARISON VS BASELINE")
    print("="*70)
    print(f"\nBaseline relationships/node: {baseline_rels_per_node:.2f}")
    print(f"KARMA relationships/node: {karma_rels_per_node:.2f}")
    if karma_rels_per_node > baseline_rels_per_node:
        improvement = ((karma_rels_per_node - baseline_rels_per_node) / baseline_rels_per_node) * 100
        print(f"✅ IMPROVEMENT: +{improvement:.1f}%")
    else:
        print(f"⚠️  No improvement yet")

    # Save results
    results = {
        "documents_processed": len(documents),
        "entities_extracted": total_entities,
        "entities_after_dedup": len(deduplicated_entities),
        "relationships_extracted": len(relations),
        "relationships_per_node": karma_rels_per_node,
        "relationship_types": rel_types,
        "total_time_seconds": total_time,
        "phase_times": {
            "entity_extraction": entity_time,
            "summarization": summary_time,
            "relationship_extraction": relation_time,
            "deduplication": dedup_time
        }
    }

    results_path = Path("results/karma_test_5docs.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Show sample relationships
    print("\n" + "="*70)
    print("SAMPLE RELATIONSHIPS (first 10)")
    print("="*70)
    for i, rel in enumerate(relations[:10]):
        print(f"\n{i+1}. {rel['source']} --[{rel['type']}]--> {rel['target']}")
        if 'confidence' in rel:
            print(f"   Confidence: {rel['confidence']:.2f}")
        if 'evidence' in rel:
            print(f"   Evidence: \"{rel['evidence'][:80]}...\"")

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
