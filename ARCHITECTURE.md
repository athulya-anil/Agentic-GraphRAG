# Agentic GraphRAG - System Architecture

**A Multi-Agent System for Knowledge Graph Construction and Symbolic Answer Selection on HotpotQA**

*Last Updated: January 2026*

---

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [System Overview](#system-overview)
3. [Offline Pipeline: Ingestion](#offline-pipeline-ingestion)
4. [Online Pipeline: Query Processing](#online-pipeline-query-processing)
5. [Component Inventory](#component-inventory)
6. [Data Stores](#data-stores)
7. [Key Architectural Decisions](#key-architectural-decisions)
8. [Configuration](#configuration)
9. [File Structure](#file-structure)

---

## High-Level Architecture

```
═══════════════════════════════════════════════════════════════════════════════
                        AGENTIC GRAPHRAG ARCHITECTURE
                   HotpotQA with Symbolic Answer Selection
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 1: OFFLINE INGESTION                             │
│                    (Run once, persists to disk)                             │
└─────────────────────────────────────────────────────────────────────────────┘

    HotpotQA Documents (from queries' supporting_facts)
                    │
    ┌───────────────▼───────────────────────────────────────────────────────┐
    │  IngestionPipeline  (src/pipeline/ingestion.py)                       │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ Stage 1: SchemaAgent                                            │  │
    │  │   • Analyzes documents (sample up to 50)                        │  │
    │  │   • Infers entity_types & relation_types                        │  │
    │  │   • Output: schema dict (optional disk save)                    │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │  ┌───────────────────────────▼─────────────────────────────────────┐  │
    │  │ Stage 2: EntityAgent (Hybrid: spaCy NER + LLM)                  │  │
    │  │   • spaCy en_core_web_lg for fast NER baseline                  │  │
    │  │   • LLM (Groq/Llama-3.3-70b) for classification                 │  │
    │  │   • Entity hints from config/entity_hints.json                  │  │
    │  │   • Output: [{name, type, properties, metadata}]                │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │  ┌───────────────────────────▼─────────────────────────────────────┐  │
    │  │ Stage 3: RelationAgent (LLM)                                    │  │
    │  │   • Extracts relationships between entities                     │  │
    │  │   • Output: [{source, target, type, confidence, properties}]    │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │  ┌───────────────────────────▼─────────────────────────────────────┐  │
    │  │ Stage 4-6: ConflictResolutionAgent                              │  │
    │  │   • Deduplicate entities (string matching, no LLM)              │  │
    │  │   • Resolve relationship conflicts                              │  │
    │  │   • Validate against inferred schema                            │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │  ┌───────────────────────────▼─────────────────────────────────────┐  │
    │  │ Stage 7: Sentence Tokenization (CRITICAL for EM)                │  │
    │  │   • NLTK punkt tokenizer (fallback: spaCy, regex)               │  │
    │  │   • EXACT text preserved (no trimming)                          │  │
    │  │   • Output: [{doc_id, sent_id, sentence_text}]                  │  │
    │  │   • Provenance = (doc_id, sent_id) attached to everything       │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │          ┌───────────────────┴───────────────────┐                    │
    │          │                                       │                    │
    │  ┌───────▼──────────────────┐    ┌──────────────▼───────────────┐    │
    │  │ Stage 8: Neo4j KG Build  │    │ Stage 9: FAISS Vector Index  │    │
    │  │ • Create entity nodes    │    │ • Embed EACH SENTENCE        │    │
    │  │ • Create relationship    │    │ • Model: BAAI/bge-large-en   │    │
    │  │   edges                  │    │   -v1.5 (1024-dim)           │    │
    │  │ • Store provenance as    │    │ • Store metadata per vector: │    │
    │  │   JSON: [{doc_id,        │    │   {doc_id, sent_id,          │    │
    │  │   sent_id}, ...]         │    │    sentence_text, type}      │    │
    │  └──────────────────────────┘    └──────────────────────────────┘    │
    └───────────────────────────────────────────────────────────────────────┘
                    │                                       │
    ┌───────────────▼───────────────────────────────────────▼───────────────┐
    │                    PERSISTENT STORAGE (Disk)                          │
    ├───────────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────────────┐    ┌───────────────────────────────┐ │
    │  │ Neo4j Database              │    │ FAISS Index                   │ │
    │  │ (bolt://localhost:7687)     │    │ (data/faiss_index/)           │ │
    │  │                             │    │                               │ │
    │  │ • Nodes: entities with      │    │ • faiss.index: vectors        │ │
    │  │   - name, type, summary     │    │   (IndexFlatL2, 1024-dim)     │ │
    │  │   - keywords, aliases       │    │                               │ │
    │  │   - provenance (JSON)       │    │ • metadata.pkl: sentence      │ │
    │  │                             │    │   metadata + provenance       │ │
    │  │ • Relationships: edges      │    │                               │ │
    │  │   - type, confidence        │    │ • Sentence-level indexing     │ │
    │  │   - provenance (JSON)       │    │   (NOT arbitrary chunks)      │ │
    │  └─────────────────────────────┘    └───────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: ONLINE QUERY PIPELINE                          │
│                    (Per-query, read-only from storage)                      │
└─────────────────────────────────────────────────────────────────────────────┘

    User Question
    "Were Scott Derrickson and Ed Wood of the same nationality?"
                    │
    ┌───────────────▼───────────────────────────────────────────────────────┐
    │  RetrievalPipeline.query()  (src/pipeline/retrieval.py)               │
    │                                                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ Step 1: OrchestratorAgent - Query Classification (LLM)          │  │
    │  │   • LLM call (Groq/Llama-3.3-70b, temp=0.0)                     │  │
    │  │   • Classifies into one of 4 types:                             │  │
    │  │     - SINGLE_HOP: direct fact lookup (1 entity, 1 doc)          │  │
    │  │     - MULTI_HOP: bridge entity reasoning (2+ hops)              │  │
    │  │     - COMPARISON: compare properties of 2+ entities             │  │
    │  │     - YES_NO: boolean question                                  │  │
    │  │   • Output: QueryType enum + confidence score                   │  │
    │  │   • Only LLM call in query path (classification only)           │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │  ┌───────────────────────────▼─────────────────────────────────────┐  │
    │  │ Step 2: Routing Decision (DETERMINISTIC)                        │  │
    │  │                                                                 │  │
    │  │   Query Type          →    Strategy                             │  │
    │  │   ──────────────────────────────────────                        │  │
    │  │   SINGLE_HOP          →    VECTOR                               │  │
    │  │   MULTI_HOP           →    GRAPH                                │  │
    │  │   COMPARISON          →    GRAPH                                │  │
    │  │   YES_NO              →    GRAPH                                │  │
    │  │                                                                 │  │
    │  │   Optional: adaptive adjustment if historical perf >15% better  │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │              ┌───────────────┴───────────────┐                        │
    │              │                               │                        │
    │  ┌───────────▼────────────┐    ┌────────────▼────────────────────┐   │
    │  │ VECTOR Retrieval       │    │ GRAPH Retrieval                 │   │
    │  │                        │    │                                 │   │
    │  │ • FAISS.search()       │    │ ┌─────────────────────────────┐ │   │
    │  │ • top_k=10 (config)    │    │ │ QueryParserAgent (LLM)      │ │   │
    │  │ • Returns sentences    │    │ │ • Extract anchor_entities   │ │   │
    │  │   with provenance:     │    │ │ • Extract target_type       │ │   │
    │  │   {doc_id, sent_id,    │    │ │ • Extract relationship_type │ │   │
    │  │    text, score}        │    │ │ • Extract direction         │ │   │
    │  │                        │    │ └─────────────────────────────┘ │   │
    │  │ Fast, simple lookup    │    │              │                  │   │
    │  └────────────────────────┘    │ ┌────────────▼────────────────┐ │   │
    │              │                 │ │ Neo4j Cypher Traversal      │ │   │
    │              │                 │ │ • Find seed entities        │ │   │
    │              │                 │ │ • Multi-hop: depth 1-2      │ │   │
    │              │                 │ │ • Returns paths + context   │ │   │
    │              │                 │ │ • Augments with FAISS docs  │ │   │
    │              │                 │ │ • top_k=15 (config)         │ │   │
    │              │                 │ └─────────────────────────────┘ │   │
    │              │                 └─────────────────────────────────┘   │
    │              └───────────────────┬───────────────────────────────┘   │
    │                                  │                                   │
    │  ┌───────────────────────────────▼─────────────────────────────────┐  │
    │  │ Step 3: Reranking (VECTOR ONLY)                                 │  │
    │  │   • Model: BAAI/bge-reranker-large (CrossEncoder)              │  │
    │  │   • Applied ONLY to VECTOR retrieval results                   │  │
    │  │   • Graph order preserved (hop depth, path structure)          │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                              │                                        │
    │  ┌───────────────────────────▼─────────────────────────────────────┐  │
    │  │ Step 4: Quality Filtering                                       │  │
    │  │   • min_score threshold: 0.15                                   │  │
    │  │   • min_length: 20 chars                                        │  │
    │  │   • max_results: 15                                             │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
                    │
    Evidence Items (with provenance)
    [{doc_id, sent_id, text, score, source: "vector"|"graph", metadata}]
                    │
    ┌───────────────▼───────────────────────────────────────────────────────┐
    │                                                                       │
    │  ╔═══════════════════════════════════════════════════════════════╗   │
    │  ║  Step 5: AnswerSelector  (src/answering/answer_selector.py)   ║   │
    │  ║                                                               ║   │
    │  ║  100% SYMBOLIC / DETERMINISTIC - NO LLM GENERATION            ║   │
    │  ║                                                               ║   │
    │  ║  Dispatch by query_type:                                      ║   │
    │  ║                                                               ║   │
    │  ║  ┌─────────────────────────────────────────────────────────┐  ║   │
    │  ║  │ YES_NO:                                                 │  ║   │
    │  ║  │   • Extract query entities (via QueryParserAgent/KG)   │  ║   │
    │  ║  │   • Extract constraint tokens from query               │  ║   │
    │  ║  │   • Check: entity + constraint co-mention in evidence  │  ║   │
    │  ║  │   • ALL entities have constrained support → "yes"      │  ║   │
    │  ║  │   • Otherwise → "no"                                   │  ║   │
    │  ║  └─────────────────────────────────────────────────────────┘  ║   │
    │  ║                                                               ║   │
    │  ║  ┌─────────────────────────────────────────────────────────┐  ║   │
    │  ║  │ COMPARISON:                                             │  ║   │
    │  ║  │   • Branch on question form:                           │  ║   │
    │  ║  │     "Were/Are/Is..." → treat as YES_NO                 │  ║   │
    │  ║  │     "Which/Who/What..." → select winning entity        │  ║   │
    │  ║  │   • Date comparison: extract years, compare            │  ║   │
    │  ║  │   • Number comparison: extract values, compare         │  ║   │
    │  ║  │   • Default: entity with most evidence support         │  ║   │
    │  ║  └─────────────────────────────────────────────────────────┘  ║   │
    │  ║                                                               ║   │
    │  ║  ┌─────────────────────────────────────────────────────────┐  ║   │
    │  ║  │ SINGLE_HOP / MULTI_HOP (Factoid):                       │  ║   │
    │  ║  │   • Candidate pools:                                    │  ║   │
    │  ║  │     1. KG node names from graph retrieval metadata     │  ║   │
    │  ║  │     2. KG node names appearing in evidence text        │  ║   │
    │  ║  │     3. Value spans (numbers, genres, quoted text)      │  ║   │
    │  ║  │   • Filter out query entities (don't repeat question)  │  ║   │
    │  ║  │   • Rank by priority:                                  │  ║   │
    │  ║  │     (1) Appears in ≥2 documents                        │  ║   │
    │  ║  │     (2) Is a graph node in evidence                    │  ║   │
    │  ║  │     (3) Highest frequency                              │  ║   │
    │  ║  │   • Return top candidate                               │  ║   │
    │  ║  └─────────────────────────────────────────────────────────┘  ║   │
    │  ║                                                               ║   │
    │  ╚═══════════════════════════════════════════════════════════════╝   │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
                    │
    ┌───────────────▼───────────────────────────────────────────────────────┐
    │  Step 6: EM Normalization (HotpotQA Official)                         │
    │  Order: lower() → remove_punc() → remove_articles() → whitespace_fix()│
    │  Example: "The U.S.A." → "usa"                                        │
    └───────────────────────────────────────────────────────────────────────┘
                    │
    Predicted Answer: "yes"  (single string, EM-normalized)
```

---

## System Overview

### Two-Phase Architecture

| Phase | When | Purpose | Storage |
|-------|------|---------|---------|
| **Offline Ingestion** | Run once | Build KG and vector index from documents | Writes to Neo4j + FAISS |
| **Online Query** | Per-query | Route, retrieve, and select answer | Reads from Neo4j + FAISS |

### Critical Design Decisions

1. **Sentence-Level Indexing**: FAISS stores individual sentences (not arbitrary chunks) with `{doc_id, sent_id}` provenance
2. **Symbolic Answer Selection**: No LLM text generation at query time - only deterministic candidate ranking
3. **No Hybrid Retrieval**: System uses VECTOR *or* GRAPH strategy per query (never both)
4. **Provenance Everywhere**: Every entity, relationship, and vector stores source sentence references

---

## Offline Pipeline: Ingestion

**Entry Point**: `src/pipeline/ingestion.py` → `IngestionPipeline`

### Stage-by-Stage Flow

| Stage | Component | Input | Output | Storage |
|-------|-----------|-------|--------|---------|
| 1 | SchemaAgent | Documents (sample ≤50) | Schema dict | Memory (optional disk) |
| 2 | EntityAgent | Documents + Schema | Entities with metadata | Memory |
| 3 | RelationAgent | Documents + Entities + Schema | Relations with confidence | Memory |
| 4 | ConflictResolutionAgent | Entities | Deduplicated entities | Memory |
| 5 | ConflictResolutionAgent | Relations | Conflict-resolved relations | Memory |
| 6 | ConflictResolutionAgent | Entities + Schema | Validated entities | Memory |
| 7 | Sentence Tokenizer | Documents | `[{doc_id, sent_id, text}]` | Memory |
| 8 | Neo4j Builder | Entities + Relations + Sentences | Nodes + Edges with provenance | **Neo4j (persistent)** |
| 9 | FAISS Builder | Sentences + Entities | Vectors with metadata | **Disk (persistent)** |

### Sentence Tokenization (Stage 7)

**Critical for HotpotQA EM evaluation** - uses NLTK punkt with fallbacks:

```
Tokenizer Priority:
1. NLTK punkt (primary)
2. spaCy sentencizer (fallback)
3. Regex split on [.!?] (last resort)
```

Output format:
```python
{
    "doc_id": 0,        # Document index
    "sent_id": 3,       # Sentence index within document
    "sentence_text": "Exact text preserved."  # NO trimming
}
```

### Provenance Storage

**Neo4j nodes and relationships** store provenance as JSON strings:
```cypher
CREATE (n:Person {
    name: "John Smith",
    provenance: '[{"doc_id": 0, "sent_id": 5}, {"doc_id": 2, "sent_id": 12}]'
})
```

**FAISS metadata** stores per-vector provenance:
```python
{
    "type": "sentence",
    "doc_id": 0,
    "sent_id": 5,
    "sentence_text": "John Smith was born in 1980.",
    "text": "John Smith was born in 1980."  # backward compat
}
```

---

## Online Pipeline: Query Processing

**Entry Point**: `src/pipeline/retrieval.py` → `RetrievalPipeline.query()`

### Step 1: Query Classification (LLM)

**Component**: `OrchestratorAgent._classify_query()`

Uses LLM to classify query into one of 4 HotpotQA types:

| Query Type | Description | Example |
|------------|-------------|---------|
| `SINGLE_HOP` | Direct fact lookup (1 entity, 1 doc) | "What genre does Don Williams sing?" |
| `MULTI_HOP` | Bridge entity reasoning (2+ hops) | "What genre was sung by the artist who recorded X?" |
| `COMPARISON` | Compare properties of 2+ entities | "Were X and Y of the same nationality?" |
| `YES_NO` | Boolean question requiring aggregation | "Are both X and Y located in America?" |

### Step 2: Routing Decision (Deterministic)

**Component**: `OrchestratorAgent._select_strategy()`

```python
ROUTING_RULES = {
    QueryType.SINGLE_HOP: RetrievalStrategy.VECTOR,
    QueryType.MULTI_HOP: RetrievalStrategy.GRAPH,
    QueryType.COMPARISON: RetrievalStrategy.GRAPH,
    QueryType.YES_NO: RetrievalStrategy.GRAPH,
}
```

**Adaptive Adjustment**: If historical performance shows >15% improvement with alternate strategy, system can switch.

### Step 3: Retrieval Execution

#### VECTOR Strategy
```python
results = faiss_index.search(query, top_k=10)
# Returns: [{doc_id, sent_id, text, score, source="vector"}]
```

#### GRAPH Strategy
1. **QueryParserAgent** (LLM) extracts intent:
   - `anchor_entities`: starting points for traversal
   - `target_type`: expected answer type
   - `relationship_type`: relationship to follow
   - `direction`: forward/reverse/bidirectional

2. **Neo4j Cypher Traversal**:
```cypher
MATCH path = (start:Label)-[r*1..2]-(connected)
WHERE toLower(start.name) CONTAINS $anchor
RETURN connected, relationships(path), nodes(path)
```

3. **Augmentation**: Graph results augmented with FAISS document snippets

### Step 4: Reranking (VECTOR ONLY)

**Model**: `BAAI/bge-reranker-large` (CrossEncoder)

**CRITICAL**: Reranking is applied ONLY to VECTOR retrieval results. Graph retrieval order must be preserved (hop depth, path structure, edge confidence).

```python
# Only rerank VECTOR results
if strategy == RetrievalStrategy.VECTOR:
    reranker = CrossEncoder('BAAI/bge-reranker-large')
    scores = reranker.predict([[query, passage] for passage in passages])
```

### Step 5: Answer Selection (SYMBOLIC)

**Component**: `src/answering/answer_selector.py` → `AnswerSelector`

**CRITICAL**: This is 100% deterministic - NO LLM generation.

#### YES_NO Logic
```python
def _select_yes_no(query, evidence):
    query_entities = get_query_entities(query)
    constraint_tokens = extract_constraints(query)  # e.g., "located", "america"

    for entity in query_entities:
        # Check: entity + constraint must CO-OCCUR in same sentence
        if not any(entity in sent AND constraint in sent for sent in evidence):
            return "no"

    return "yes"  # All entities have constrained support
```

#### COMPARISON Logic
```python
def _select_comparison(query, evidence):
    if query.startswith(("Were", "Are", "Is", "Do", "Does")):
        return _select_yes_no(query, evidence)  # Boolean form
    else:
        return _select_winner(query, evidence)  # Entity selection
```

#### FACTOID Logic (SINGLE_HOP / MULTI_HOP)
```python
def _select_factoid(query, evidence):
    candidates = get_kg_node_names(evidence)
    candidates += extract_value_spans(evidence)  # numbers, genres, etc.
    candidates -= query_entities  # Don't repeat the question

    # Rank by priority
    ranked = sort_by([
        (appears_in_2plus_docs, DESC),
        (is_graph_node, DESC),
        (frequency, DESC)
    ])

    return normalize_answer(ranked[0])
```

### Step 6: EM Normalization

**HotpotQA Official Order**:
```python
def normalize_answer(s):
    s = s.lower()                    # "The U.S.A." → "the u.s.a."
    s = remove_punctuation(s)        # → "the usa"
    s = remove_articles(s)           # → "usa"
    s = whitespace_fix(s)            # → "usa"
    return s
```

---

## Component Inventory

| Layer | Component | File | Purpose | Phase | Mutates |
|-------|-----------|------|---------|-------|---------|
| **Ingestion** | SchemaAgent | `src/agents/schema_agent.py` | Infer entity/relation types | Offline | No |
| **Ingestion** | EntityAgent | `src/agents/entity_agent.py` | Hybrid spaCy+LLM extraction | Offline | No |
| **Ingestion** | RelationAgent | `src/agents/relation_agent.py` | Relationship extraction | Offline | No |
| **Ingestion** | ConflictResolutionAgent | `src/agents/conflict_resolution_agent.py` | Dedup, validate | Offline | No |
| **Ingestion** | IngestionPipeline | `src/pipeline/ingestion.py` | Orchestrate stages | Offline | **Yes** |
| **Storage** | Neo4jManager | `src/graph/neo4j_manager.py` | Graph CRUD | Both | Yes (offline) |
| **Storage** | FAISSIndex | `src/vector/faiss_index.py` | Vector operations | Both | Yes (offline) |
| **Routing** | OrchestratorAgent | `src/agents/orchestrator_agent.py` | Query classification + routing | Online | No |
| **Retrieval** | RetrievalPipeline | `src/pipeline/retrieval.py` | Execute retrieval | Online | No |
| **Retrieval** | QueryParserAgent | `src/agents/query_parser_agent.py` | Parse graph query intent | Online | No |
| **Answering** | AnswerSelector | `src/answering/answer_selector.py` | Symbolic answer extraction | Online | No |

---

## Data Stores

### Neo4j Knowledge Graph

| Aspect | Value |
|--------|-------|
| **URI** | `bolt://localhost:7687` |
| **Database** | `neo4j` |
| **Node Properties** | `name`, `type`, `summary`, `keywords`, `aliases`, `provenance` |
| **Edge Properties** | `type`, `confidence`, `provenance` |
| **Provenance Format** | JSON string: `[{"doc_id": 0, "sent_id": 5}, ...]` |

### FAISS Vector Index

| Aspect | Value |
|--------|-------|
| **Path** | `data/faiss_index/` |
| **Files** | `faiss.index`, `metadata.pkl` |
| **Index Type** | `IndexFlatL2` (exact search) |
| **Dimension** | 1024 |
| **Embedding Model** | `BAAI/bge-large-en-v1.5` |
| **Indexing Unit** | Individual sentences (NOT chunks) |
| **Metadata Per Vector** | `{type, doc_id, sent_id, sentence_text, text}` |

---

## Key Architectural Decisions

### 1. Sentence-Level Indexing (Not Chunks)

**Why**: HotpotQA EM evaluation requires exact source text. Arbitrary chunking loses sentence boundaries and provenance.

**Implementation**: Each FAISS vector = one sentence with explicit `{doc_id, sent_id}`.

### 2. Symbolic Answer Selection (No LLM Generation)

**Why**:
- Reproducible evaluation (deterministic outputs)
- Faster inference (no LLM call for answer generation)
- EM-compatible (candidates come from KG structure)

**Implementation**: `AnswerSelector` uses rule-based logic per query type.

### 3. No Hybrid Retrieval

**Why**: Removed after commit `05db459` - simpler routing, clearer failure modes.

**Implementation**: Each query uses VECTOR *xor* GRAPH, never both simultaneously.

### 4. Provenance Everywhere

**Why**: HotpotQA requires supporting fact attribution.

**Implementation**: Every node, edge, and vector stores `[{doc_id, sent_id}]` references.

### 5. LLM Usage is Minimal at Query Time

| Component | LLM Usage |
|-----------|-----------|
| OrchestratorAgent | 1 call for classification |
| QueryParserAgent | 1 call for graph queries only |
| AnswerSelector | **NONE** (symbolic) |

---

## Configuration

### Environment Variables (`.env`)

```bash
# LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j

# Embedding
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024
```

### Retrieval Parameters (`src/utils/config.py`)

```python
class RetrievalConfig:
    top_k_vector: int = 10    # Results from FAISS
    top_k_graph: int = 15     # Results from Neo4j
    # Note: No hybrid - system uses one strategy per query
```

---

## File Structure

```
agentic-graphrag/
├── src/
│   ├── agents/                         # Multi-agent system
│   │   ├── schema_agent.py             # Schema inference
│   │   ├── entity_agent.py             # Entity extraction
│   │   ├── relation_agent.py           # Relationship extraction
│   │   ├── conflict_resolution_agent.py # Dedup & validation
│   │   ├── orchestrator_agent.py       # Query routing (HotpotQA)
│   │   ├── query_parser_agent.py       # Graph query intent parsing
│   │   └── __init__.py
│   ├── answering/                      # Answer selection
│   │   ├── answer_selector.py          # SYMBOLIC answer extraction
│   │   └── __init__.py
│   ├── graph/                          # Neo4j layer
│   │   ├── neo4j_manager.py            # CRUD operations
│   │   └── __init__.py
│   ├── vector/                         # FAISS layer
│   │   ├── faiss_index.py              # Vector operations
│   │   └── __init__.py
│   ├── pipeline/                       # Orchestration
│   │   ├── ingestion.py                # Offline ingestion
│   │   ├── retrieval.py                # Online retrieval
│   │   └── __init__.py
│   └── utils/                          # Utilities
│       ├── config.py                   # Pydantic configuration
│       ├── llm_client.py               # LLM provider interface
│       └── entity_hints.py             # Entity type hints
├── data/
│   ├── faiss_index/                    # Persisted vectors
│   ├── hotpotqa/                       # Dataset + results
│   └── cache/                          # Query cache
├── config/
│   └── entity_hints.json               # Domain entity types
├── evaluate_hotpotqa_full.py           # Main evaluation script
├── ingest_hotpotqa.py                  # HotpotQA ingestion
└── query.py                            # Interactive query CLI
```

---

## Evaluation

### Metric: Exact Match (EM)

```python
def em_check(prediction, gold):
    return int(normalize_answer(prediction) == normalize_answer(gold))
```

### Evaluation Script

```bash
python evaluate_hotpotqa_full.py
```

Outputs:
- `data/hotpotqa/results_*.json` - per-query results
- `data/hotpotqa/summary_*.json` - aggregate metrics

### Metrics Tracked

- Overall EM
- EM by query type (Bridge vs Comparison)
- EM by retrieval strategy (Vector vs Graph)
- Empty context rate
- Average query latency

---

*Architecture version: 2.0 (HotpotQA + Symbolic Answer Selection)*
