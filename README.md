# Agentic GraphRAG

A hybrid multi-hop question answering system built on the HotpotQA benchmark. Combines a Neo4j knowledge graph, a FAISS dense vector index, and an LLM-based answer selector to reason across multiple documents and entities.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [System Architecture](#system-architecture)
3. [How It Works](#how-it-works)
4. [Evaluation Results](#evaluation-results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Key Design Decisions](#key-design-decisions)
9. [Known Limitations](#known-limitations)
10. [Future Work](#future-work)
11. [Citation](#citation)

---

## Problem Statement

Multi-hop question answering requires reasoning across multiple documents and entities — not just finding a single relevant passage. For example:

> *"The arena where the Lewiston Maineiacs played their home games can seat how many people?"*

Answering this requires first identifying the arena (Androscoggin Bank Colisée), then finding its seating capacity — two separate documents.

Existing approaches fall short:

| Approach | Weakness |
|---|---|
| Pure vector RAG | Retrieves passages but misses structured entity relationships |
| Pure graph RAG | Brittle — fails when entity linking is imperfect |
| Symbolic QA | No semantic reasoning (can't compare nationalities, ages, etc.) |

This system addresses all three through **adaptive routing** and **LLM-grounded answering**.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  PHASE 1 · OFFLINE INGESTION                     │
│                    (run once per corpus)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Documents                                                       │
│      │                                                           │
│      ▼                                                           │
│  SchemaAgent ──── infers entity & relationship types             │
│      │                                                           │
│      ▼                                                           │
│  EntityAgent ──── spaCy NER + Groq/Llama-3.3-70b classification │
│      │                                                           │
│      ▼                                                           │
│  RelationAgent ── LLM-based relationship extraction              │
│      │                                                           │
│      ▼                                                           │
│  ConflictResolutionAgent ── deduplication & schema validation    │
│      │                                                           │
│      ▼                                                           │
│  Sentence Tokenizer ── NLTK → spaCy → regex (exact text kept)   │
│      │                        │                                  │
│      ▼                        ▼                                  │
│  Neo4j KG             FAISS Vector Index                        │
│  (entity nodes,       (sentence-level,                          │
│   relationships,       BAAI/bge-large-en-v1.5,                  │
│   provenance)          1024-dim)                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                  PHASE 2 · ONLINE QUERY                          │
│                    (per question)                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query                                                      │
│      │                                                           │
│      ▼                                                           │
│  OrchestratorAgent                                               │
│  ├── deterministic override: "how many" → VECTOR                 │
│  └── LLM classification: SINGLE_HOP/MULTI_HOP/COMPARISON/YES_NO │
│      │                                                           │
│      ├── VECTOR ──► FAISS top-k=10 → bge-reranker-large         │
│      │              Two-hop for bridge questions:                │
│      │              Hop1: full question → extract bridge entity  │
│      │              Hop2: FAISS search on bridge entity          │
│      │                                                           │
│      └── GRAPH ───► QueryParserAgent (LLM intent parsing)       │
│                     → Neo4j Cypher 1-2 hop traversal            │
│                     → FAISS augmentation if < 3 contexts         │
│      │                                                           │
│      ▼                                                           │
│  Quality Filter (min_score=0.15, min_length=20, max=15)         │
│      │                                                           │
│      ▼                                                           │
│  LLMAnswerer (Groq/Llama-3.3-70b, temp=0.0)                    │
│  "Answer using ONLY the provided evidence. Return 'unknown'     │
│   if evidence is insufficient."                                  │
│      │                                                           │
│      ▼                                                           │
│  Answer + strategy + num_contexts + supporting_coverage          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Query Routing

Every query is classified by the `OrchestratorAgent` before retrieval:

| Query Type | Routed To | Rationale |
|---|---|---|
| `SINGLE_HOP` | VECTOR | Direct fact lookup — dense retrieval sufficient |
| `MULTI_HOP` (bridge) | VECTOR + two-hop | Graph traversal brittle; vector finds bridge entities reliably |
| `COMPARISON` | GRAPH | Requires entity attribute comparison across nodes |
| `YES_NO` | GRAPH | Requires entity co-presence and relationship verification |
| Contains "how many" | VECTOR (override) | Numeric questions consistently work better with vector |

### Ingestion Pipeline (9 stages)

1. **Schema inference** — LLM samples up to 50 documents to infer entity/relationship types
2. **Entity extraction** — spaCy provides baseline NER; LLM classifies types and enriches with summary, keywords, aliases
3. **Relationship extraction** — LLM extracts `(source, type, target, confidence)` triples
4. **Conflict resolution** — string-match deduplication; relationship conflict resolution; schema validation
5. **Sentence tokenization** — NLTK punkt with spaCy and regex fallbacks; **exact text preserved** (critical for Exact Match scoring)
6. **Neo4j graph build** — entity nodes with full metadata + provenance JSON (`[{doc_id, sent_id}]`); relationship edges
7. **Document linking** — `MENTIONED_IN` edges connect entity nodes to `Document` nodes
8. **FAISS indexing** — each sentence embedded with BAAI/bge-large-en-v1.5 (1024-dim, IndexFlatL2)
9. **Persistence** — Neo4j database + `faiss.index` / `metadata.pkl` to disk

### Graph Traversal Detail

For GRAPH-routed queries:
1. `QueryParserAgent` (LLM) extracts: anchor entity, target type, relationship type, direction (forward/reverse/bidirectional)
2. Fuzzy seed lookup: matches anchor against node `name`, `aliases`, `keywords`, `summary` fields
3. Cypher path query: 1–2 hop traversal with direction awareness
4. Context assembly: entity path string + provenance-linked sentence text
5. Fallback to FAISS if fewer than 3 contexts returned

### Answer Selection

`LLMAnswerer` (primary):
- Provider: Groq, model: `llama-3.3-70b-versatile`, temperature: 0.0
- Constrained prompt — must use only provided evidence
- Returns `"unknown"` rather than hallucinating when evidence is absent
- Appends units for numeric answers (e.g., "3,677 seated")

`AnswerSelector` (symbolic fallback):
- Deterministic, no LLM call
- YES_NO: entity + constraint token co-mention check
- COMPARISON: branch on question form
- FACTOID: KG node-name candidates ranked by priority

---

## Evaluation Results

### Benchmark: Mini-5 (5 queries, 50 documents)

> Strict EM matches only exact normalized strings. Relaxed EM (SubEM) also accepts answers where the gold string is a substring of the prediction.

| Metric | Score |
|---|---|
| Relaxed EM (SubEM) | **100%** (5/5) |
| Strict EM | **80%** (4/5) |
| Avg Supporting Coverage | 80% |
| Strategy Distribution | 3 VECTOR, 2 GRAPH |

**Per-query breakdown:**

| # | Question | Gold | Predicted | EM | Strategy |
|---|---|---|---|---|---|
| 1 | Were Scott Derrickson and Ed Wood of the same nationality? | yes | yes | ✅ | GRAPH |
| 2 | Are the Laleli Mosque and Esma Sultan Mansion in the same neighborhood? | no | no | ✅ | GRAPH |
| 3 | What government position was held by the woman who portrayed Corliss Archer? | Chief of Protocol | ...chief of protocol... | ✅ SubEM | VECTOR |
| 4 | 2014 S/S is the debut album of a group formed by who? | YG Entertainment | yg entertainment | ✅ | VECTOR |
| 5 | The arena where the Lewiston Maineiacs played can seat how many people? | 3,677 seated | 3677 seated | ✅ | VECTOR |

---

### Benchmark: Mini-25 (25 queries, 50 gold-only supporting documents, 0 distractors)

Four retrieval strategies were tested head-to-head:

| Strategy | Relaxed EM | Strict EM | Correct / 25 |
|---|---|---|---|
| VECTOR (forced) | 64% | 52% | 16 |
| GRAPH (forced) | 36% | 32% | 9 |
| **HYBRID (best)** | **68%** | **56%** | **17** |
| ORCHESTRATOR (adaptive) | 60% | 48% | 15 |

**HYBRID is the best single strategy** at 68% relaxed EM. The ORCHESTRATOR scores lower (60%) because it routes some questions to GRAPH where VECTOR would perform better — graph traversal is still brittle at this scale.

**Selected Mini-25 examples (HYBRID strategy):**

| Question | Gold | Predicted | EM |
|---|---|---|---|
| 2014 S/S debut album — group formed by who? | YG Entertainment | yg entertainment | ✅ |
| Stage name Aladin, management consultant — who? | Eenasul Fateh | eenasul fateh | ✅ |
| Lewiston Maineiacs arena seating? | 3,677 seated | 3677 seated | ✅ |
| Guns N' Roses promo year for Schwarzenegger film? | 1999 | 1999 | ✅ |
| Singer of A Rather Blustery Day — also voiced what hedgehog? | Sonic | sonic hedgehog | ✅ SubEM |
| Apple Remote alternative control device? | keyboard function keys | keyboard function keys | ✅ |
| VCU founded in what year? | 1838 | 1838 | ✅ |

---

## Installation

**Prerequisites:** Python 3.11+, Docker Desktop, [Groq API key](https://console.groq.com/keys)

```bash
# 1. Clone
git clone https://github.com/athulya-anil/Agentic-GraphRAG.git
cd Agentic-GraphRAG-private

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. Configure environment
cp .env.example .env
# Open .env and set your GROQ_API_KEY — everything else works as-is

# 4. Start Neo4j (Docker required)
./scripts/start_neo4j.sh
```

> **Note:** The FAISS index and Neo4j graph are not included in the repo — you must run ingestion before querying. See [Usage](#usage) below.

---

## Usage

### Ingest documents

```bash
# Ingest a directory of documents
python ingest.py --dir data/raw/

# Ingest specific files
python ingest.py --file doc1.txt --file doc2.pdf

# Mini-5 benchmark ingestion
python mini5/ingest_mini5.py

# Mini-25 benchmark ingestion
python mini25/ingest_mini25_gold_only.py
```

### Query

```bash
# Interactive mode
python query.py

# Single query
python query.py --query "What government position was held by the woman who portrayed Corliss Archer?"

# With options (--top-k default is 5; the pipeline internally uses 10 for vector and 15 for graph)
python query.py --query "..." --top-k 10 --verbose
```

```python
# Programmatic usage
from src.pipeline import get_retrieval_pipeline

pipeline = get_retrieval_pipeline()
result = pipeline.query(
    "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
    top_k=10
)

print(result['response'])       # "yg entertainment"
print(result['strategy'])       # "vector"
print(result['num_contexts'])   # 10
```

### Run Evaluation

```bash
# Mini-5 (5 queries)
python mini5/evaluate_mini5.py

# Mini-25 with all strategies (vector, graph, hybrid, orchestrator)
python mini25/evaluate_mini25_gold_only.py

# 100-query benchmark
python evaluate_hotpotqa_100queries.py
```

Results are saved to `data/hotpotqa/mini_5/evaluation_results.json` and `data/hotpotqa/mini_25_gold_only/evaluation_results.json`.

---

## Project Structure

```
Agentic-GraphRAG/
├── src/
│   ├── agents/
│   │   ├── schema_agent.py              # Graph schema inference (LLM)
│   │   ├── entity_agent.py              # Hybrid NER + metadata enrichment
│   │   ├── relation_agent.py            # Relationship extraction (LLM)
│   │   ├── orchestrator_agent.py        # Query routing & classification
│   │   ├── query_parser_agent.py        # Graph query intent parsing (LLM)
│   │   └── conflict_resolution_agent.py # Entity dedup & schema validation
│   ├── pipeline/
│   │   ├── ingestion.py                 # 9-stage offline ingestion pipeline
│   │   └── retrieval.py                 # Online query pipeline
│   ├── graph/
│   │   └── neo4j_manager.py             # Neo4j CRUD + retry logic
│   ├── vector/
│   │   └── faiss_index.py               # FAISS embedding + search
│   ├── answering/
│   │   ├── llm_answerer.py              # LLM-based QA (primary)
│   │   └── answer_selector.py           # Symbolic fallback
│   └── utils/
│       ├── config.py                    # Pydantic config (env-based)
│       └── llm_client.py                # Unified Groq/Ollama client
│
├── data/
│   └── hotpotqa/
│       ├── corpus.json                  # Wikipedia passage corpus
│       ├── queries_100.json             # 100-query benchmark set
│       ├── mini_5/
│       │   └── evaluation_results.json  # Mini-5 results (80% strict EM)
│       └── mini_25_gold_only/
│           └── evaluation_results.json  # Mini-25 results (68% HYBRID EM)
│
├── mini5/
│   ├── ingest_mini5.py                  # Mini-5 ingestion script
│   ├── evaluate_mini5.py                # Mini-5 evaluation script
│   └── MINI5_EVALUATION_REPORT.md       # Journey from 0% → 80% EM
│
├── mini25/
│   ├── ingest_mini25_gold_only.py       # Mini-25 ingestion script
│   └── evaluate_mini25_gold_only.py     # Mini-25 evaluation (4 strategies)
│
├── docs/
│   ├── ARCHITECTURE.md                  # Deep-dive architecture docs
│   ├── QUICKSTART.md                    # Step-by-step getting started guide
│   └── architecture_diagram.md          # Mermaid diagrams
│
├── scripts/
│   ├── setup.sh                         # One-command environment setup
│   ├── start_neo4j.sh                   # Start Neo4j via Docker
│   └── stop_neo4j.sh                    # Stop Neo4j container
│
├── ingest.py                            # CLI: ingest custom documents
├── query.py                             # CLI: interactive & single-shot queries
├── requirements.txt                     # Pinned Python dependencies
└── .env.example                         # Environment variable template
```

---

## Key Design Decisions

**Sentence-level indexing, not chunks**
Every sentence is indexed individually in FAISS with its `(doc_id, sent_id)` provenance. This preserves exact text for Exact Match evaluation and allows precise multi-hop linking.

**Hybrid routing, not one-size-fits-all**
Rather than committing to one retrieval strategy, the system classifies each query and routes it. Bridge questions go to VECTOR (where two-hop extraction works well); comparison questions go to GRAPH (where relationship traversal is superior).

**Grounded answering**
The LLM answerer is explicitly instructed to return `"unknown"` when evidence is insufficient. This eliminates hallucination at the cost of some recall — a deliberate trade-off.

**Provenance everywhere**
Every entity node, relationship edge, and FAISS vector carries `(doc_id, sent_id)` metadata. This enables the system to trace any answer back to the exact source sentence.

---

## Known Limitations

| Limitation | Impact | Details |
|---|---|---|
| Graph entity linking brittleness | High | Seed lookup fails when entity names don't substring-match nodes — root cause of 0% EM on the 100-query benchmark |
| Strict EM phrasing sensitivity | Medium | "1986 to 2013" ≠ "from 1986 to 2013"; suffix mismatches (e.g., "DSC") count as wrong |
| Numeric/comparative reasoning | Medium | LLM cannot compute "who is older" from raw birth year text without explicit reasoning |
| No test suite | Medium | Project has evaluation scripts but no pytest unit or integration tests |
| Graph retrieval at scale | High | Mini-25 GRAPH strategy achieves only 36% EM; 100-query benchmark returns empty contexts for 68/100 queries |

---

## Future Work

1. **Improved entity linking** — fuzzy matching, embedding-based lookup, or LLM-driven entity resolution to fix the graph brittleness at scale
2. **Larger evaluation** — full HotpotQA dev set (7,405 queries) with distractor documents
3. **Numeric reasoning** — chain-of-thought prompting for age/date/count comparisons
4. **Hybrid verification** — use graph to verify or rerank vector-retrieved passages
5. **Test suite** — pytest unit and integration tests for all pipeline components

---

## Performance

| Benchmark | Ingestion Time | Query Time | Memory |
|---|---|---|---|
| Mini-5 (50 docs) | ~30s | ~2–3s/query | ~2GB |
| Mini-25 (50 docs) | ~45s | ~2–3s/query | ~2GB |

---

## Key Dependencies

| Library | Version | Purpose |
|---|---|---|
| neo4j | — | Graph database |
| faiss-cpu | 1.7.4 | Vector similarity search |
| sentence-transformers | 5.2.0 | BAAI/bge embeddings |
| spaCy | 3.7.2 | NER + sentence segmentation |
| groq | 0.37.1 | LLM inference (llama-3.3-70b) |
| pydantic | 2.5.0 | Configuration management |
| tenacity | 8.2.3 | Neo4j connection retry |
| nltk | — | Sentence tokenization |

---

## Citation

```bibtex
@software{agentic_graphrag_2026,
  title   = {Agentic GraphRAG: Hybrid Multi-Hop Question Answering},
  author  = {Athulya Anil},
  year    = {2026},
  url     = {https://github.com/athulya-anil/Agentic-GraphRAG}
}
```

---

## License

MIT

## Acknowledgments

- HotpotQA dataset: Yang et al., 2018
- BGE embeddings: BAAI
- Neo4j community
