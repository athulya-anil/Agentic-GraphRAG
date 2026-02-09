# Agentic GraphRAG: Mini-25 Experiment Summary

## System Overview

Agentic GraphRAG is a multi-hop question answering system that combines vector retrieval (FAISS), knowledge graph traversal (Neo4j), and LLM-based reasoning. Documents are ingested through a pipeline that extracts entities and relationships into a graph while indexing sentences as embeddings. At query time, an orchestrator agent classifies the query and routes it to one of three retrieval strategies — VECTOR (semantic search + two-hop expansion), GRAPH (entity-aware traversal), or HYBRID (both combined) — before an LLM answerer generates a response from the retrieved contexts.

## Experiments

We evaluated on 25 multi-hop HotpotQA questions under two corpus conditions:

| Setting | Docs | Supporting | Distractors |
|---|---|---|---|
| **Baseline** | 242 | 50 (2/query) | 192 (8/query) |
| **Gold-only** | 50 | 50 (2/query) | 0 |

## Results

| Strategy | Baseline EM | Gold-Only EM | Delta |
|---|---|---|---|
| VECTOR | 56% | 64% | +8% |
| GRAPH | 32% | 36% | +4% |
| HYBRID | 56% | 68% | +12% |
| ORCHESTRATOR | 52% | 60% | +8% |

## Conclusions

1. **Retrieval is not the bottleneck.** Removing all distractors only improves EM by 8-12%. Even in gold-only mode, 8/25 queries fail — and every one of them has 100% supporting document coverage. The ceiling is set by the LLM answerer, not retrieval quality.

2. **Half the failures are false negatives.** 4/8 gold-only failures are correct answers in the wrong string format (e.g., "1986 to 2013" vs "from 1986 to 2013"). Accounting for these, HYBRID's effective accuracy is ~84%.

3. **HYBRID benefits most from noise reduction.** Its +12% gain (vs +8% for VECTOR) suggests that combining graph and vector signals amplifies noise from distractors, but performs best when the corpus is clean.

4. **Graph retrieval consistently underperforms.** At 32-36% EM, it retrieves only one of two supporting docs ~50% of the time, making it unreliable for multi-hop reasoning on its own.

5. **The orchestrator hurts more than it helps.** By routing 28% of queries to the weak GRAPH strategy, it scores below pure VECTOR/HYBRID in both settings.

6. **The LLM answerer is the primary bottleneck.** The remaining true failures are LLM reasoning errors: miscomparing dates (q4), misinterpreting ambiguous question phrasing as yes/no (q24), and failing to infer implicit ratios (q19).
