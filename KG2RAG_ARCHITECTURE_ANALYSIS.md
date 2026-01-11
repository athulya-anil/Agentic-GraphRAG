# KG²RAG Architecture Analysis

**Comparison: KG²RAG (NAACL 2025) vs Agentic GraphRAG**

*Analysis Date: 2026-01-09*
*Based on: KG2RAG-main/code/util/kg_post_processor.py*

---

## Executive Summary

KG²RAG uses a **sophisticated multi-stage post-processing pipeline** to organize retrieved chunks using graph topology analysis. Their system has 3 critical components we're **completely missing**:

1. **KG Expansion** (1-hop graph traversal)
2. **Connected Components** (graph clustering)
3. **Maximum Spanning Tree** (context organization)

---

## Architectural Comparison

### Our Current Pipeline (Agentic GraphRAG)

```
Query → Orchestrator → Strategy Selection (VECTOR/GRAPH) → Retrieval → Context Assembly → LLM Generation
```

**Retrieval stage:**
- Vector: FAISS similarity search, keyword overlap filtering
- Graph: Neo4j Cypher queries, keyword matching, hop traversal

**Context assembly:** Flat concatenation of top-k chunks

---

### KG²RAG Pipeline

```
Query → Vector Retrieval → KG Expansion → Graph Construction → Connected Components → MST → Reranking → Generation
```

**Stage breakdown:**

#### Stage 1: Initial Vector Retrieval
- Semantic search with mxbai-embed-large (same as ours)
- Returns top-k seed chunks (default k=10)

#### Stage 2: KG Expansion (`KGRetrievePostProcessor`)
**Location:** `kg_post_processor.py:89-250`

```python
# Line 174: 1-hop traversal
hops = 1
for hop in range(hops):
    # For each retrieved entity
    for triplet in doc2kg[entity][idx_seq_str]:
        h, r, t = triplet
        # Add connected entities not yet retrieved
        if (h in self.ents) and (h not in retrieved_ents):
            additional_ents.add(h)
        if (t in self.ents) and (t not in retrieved_ents):
            additional_ents.add(t)
```

**What it does:**
- Takes initially retrieved entities
- Finds triplets mentioning those entities
- Adds **1-hop neighbors** (entities connected via relations)
- Propagates relevance scores to neighbors
- Expands context to include multi-hop paths

**Example:**
```
Query: "Were Scott Derrickson and Ed Wood of the same nationality?"

Initial retrieval: ["Scott Derrickson" chunk]

KG expansion via triplet: <Scott Derrickson, nationality, America>
→ Finds related entities sharing "nationality" relation
→ Discovers "Ed Wood" chunk (also has nationality: America)
→ Adds Ed Wood to context

Result: Both entities now in context for comparison
```

**We don't have this:** Our graph retrieval finds entities via keyword match, but doesn't expand via graph edges.

---

#### Stage 3: Graph Construction
**Location:** `kg_post_processor.py:276-303`

```python
# Line 276: Build MultiGraph
g = nx.MultiGraph()

# Lines 294-303: Add triplets as edges
for triplet in doc2kg[ent][idx_seq_str]:
    h, r, t = triplet
    g.add_edge(h, t, rel=r, source=node.node.id_, weight=node.score)
```

**What it does:**
- Creates NetworkX graph from retrieved chunks
- Entities = nodes
- Relations = edges with weights (retrieval scores)
- Stores source chunk ID in edge data

**We don't have this:** We store entities and relations in Neo4j, but don't construct in-memory graphs for topology analysis.

---

#### Stage 4: Connected Components Analysis
**Location:** `kg_post_processor.py:349-379`

```python
# Line 349: Find connected components
wccs = list(nx.connected_components(g))
sorted_wccs = sorted(wccs, key=len, reverse=True)

# Line 353-379: Process each component
for i in range(len(sorted_wccs)):
    wcc = sorted_wccs[i]
    if len(wcc) > 1:
        # Build MST for multi-entity clusters
        subgraph = g.subgraph(wcc)
        mst = nx.maximum_spanning_tree(subgraph, weight='weight')
```

**What it does:**
- Identifies **disconnected clusters** of entities
- Each component = a coherent topic/document
- Sorts by size (largest clusters first)

**Example:**
```
Component 1 (size=5): {Ed Wood, Tim Burton, Johnny Depp, Bela Lugosi, Martin Landau}
→ All about Ed Wood film

Component 2 (size=4): {Scott Derrickson, Doctor Strange, Marvel, Benedict Cumberbatch}
→ All about Scott Derrickson films

These are separate contexts that shouldn't be mixed!
```

**We don't have this:** We mix all retrieved chunks together without identifying topic boundaries.

---

#### Stage 5: Maximum Spanning Tree (MST)
**Location:** `kg_post_processor.py:363`

```python
# Line 363: Build MST
mst = nx.maximum_spanning_tree(subgraph, weight='weight')

# Lines 365-370: Extract chunks from MST edges
cand_ctx_list = []
for cand_edge in mst.edges(data=True):
    if cand_edge[2]['source'] != 'query':
        cand_ctx_list.append(cand_edge[2]['source'])
```

**What it does:**
- For each connected component, builds **Maximum Spanning Tree**
- Uses retrieval scores as edge weights
- MST ensures:
  - All entities in component are connected
  - Paths use highest-scoring edges (most relevant chunks)
  - No redundant connections

**Why this matters:**
Without MST, you might retrieve:
- Chunk A: "Ed Wood directed Plan 9"
- Chunk B: "Ed Wood worked with Bela Lugosi"
- Chunk C: "Johnny Depp played Ed Wood"
- Chunk D: "Tim Burton directed Ed Wood film"

MST organizes these into a coherent narrative:
```
Ed Wood (center) → connects to:
  ├─ Bela Lugosi (relation: worked_with)
  ├─ Plan 9 (relation: directed)
  └─ Johnny Depp (relation: portrayed_by) → Tim Burton (relation: directed_by)
```

Selects chunks that form connected reasoning path, not random fragments.

**We don't have this:** We return top-k chunks by score, even if they're disconnected fragments.

---

#### Stage 6: Reranking
**Location:** `kg_post_processor.py:413-431`

```python
# Line 413: Rerank with BGE
scores = self.reranker.compute_score([
    (query_bundle.query_str, cand_str)
    for cand_str in cand_strs
])

# Lines 416-420: Select top contexts
sorted_seqs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
for seq in sorted_seqs:
    if len(set(wanted_ctxs) | set(cand_ids_lists[seq])) > self.topk:
        break
    wanted_ctxs.extend(cand_ids_lists[seq])
```

**What it does:**
- Uses **BGE-reranker-large** (FlagEmbedding)
- Scores entire context combinations (not individual chunks)
- Selects top-k chunks ensuring graph connectivity

**We don't have this:** No reranker, just use initial FAISS scores.

---

## Critical Missing Components

### 1. KG Expansion (1-Hop Traversal)
**Impact:** High
**Complexity:** Medium
**Files to modify:**
- `src/pipeline/retrieval.py` - add KG expansion step after initial retrieval
- `src/graph/neo4j_manager.py` - add method to get 1-hop neighbors

**Implementation:**
```python
def expand_via_kg(initial_entities, neo4j_manager):
    """Expand entities via 1-hop graph traversal."""
    expanded = set(initial_entities)

    for entity in initial_entities:
        # Find all entities connected via relations
        neighbors = neo4j_manager.get_neighbors(entity, max_hops=1)
        expanded.update(neighbors)

    return expanded
```

---

### 2. Connected Components Analysis
**Impact:** High
**Complexity:** Low
**Files to modify:**
- `src/pipeline/retrieval.py` - add graph construction and component analysis

**Implementation:**
```python
import networkx as nx

def organize_via_components(entities, relations, chunk_scores):
    """Organize chunks into coherent components."""
    g = nx.Graph()

    # Build graph
    for entity in entities:
        g.add_node(entity)
    for h, r, t in relations:
        g.add_edge(h, t, weight=chunk_scores.get((h, t), 0.0))

    # Find components
    components = list(nx.connected_components(g))
    components.sort(key=len, reverse=True)

    return components  # [[cluster1], [cluster2], ...]
```

---

### 3. Maximum Spanning Tree
**Impact:** Very High
**Complexity:** Low (NetworkX provides it)
**Files to modify:**
- `src/pipeline/retrieval.py` - add MST construction per component

**Implementation:**
```python
def build_mst_context(component, graph, chunks_index):
    """Build MST for a component and extract chunks."""
    if len(component) <= 1:
        return list(component)

    # Get subgraph for this component
    subgraph = graph.subgraph(component)

    # Build MST (maximum spanning tree with weights)
    mst = nx.maximum_spanning_tree(subgraph, weight='weight')

    # Extract chunks from MST edges
    chunks = []
    for edge in mst.edges(data=True):
        chunk_id = edge[2].get('source')
        if chunk_id:
            chunks.append(chunk_id)

    return chunks
```

---

### 4. Reranker (BGE)
**Impact:** Medium
**Complexity:** Low
**Dependencies:** `pip install FlagEmbedding`

**Implementation:**
```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

def rerank_contexts(query, context_candidates):
    """Rerank context combinations with BGE."""
    scores = reranker.compute_score([
        (query, ctx) for ctx in context_candidates
    ])

    sorted_indices = sorted(range(len(scores)),
                           key=lambda x: scores[x],
                           reverse=True)

    return [context_candidates[i] for i in sorted_indices]
```

---

## Implementation Priority

Based on impact vs complexity:

### Priority 1: Maximum Spanning Tree (Highest Impact, Low Complexity)
- **Why:** Organizes context into coherent narratives
- **Effort:** 1-2 hours (NetworkX handles complexity)
- **Expected EM improvement:** +10-15%

### Priority 2: Connected Components (High Impact, Low Complexity)
- **Why:** Identifies topic boundaries
- **Effort:** 30 minutes
- **Expected EM improvement:** +5-10%

### Priority 3: KG Expansion (High Impact, Medium Complexity)
- **Why:** Enables multi-hop reasoning
- **Effort:** 2-3 hours (need to modify Neo4j queries)
- **Expected EM improvement:** +15-20%

### Priority 4: Reranker (Medium Impact, Low Complexity)
- **Why:** Improves context selection
- **Effort:** 1 hour
- **Expected EM improvement:** +5-10%

**Total expected improvement:** +35-55% EM (bringing us from 0% to 35-55% baseline)

With these improvements, we should achieve comparable performance to KG²RAG (they report ~60% EM).

---

## Dependencies to Add

```bash
pip install networkx FlagEmbedding
```

---

## Next Steps

1. **Wait for ingestion to complete** (~20 more minutes, currently at 221/991)
2. **Run baseline evaluation** with current architecture (expected: 5-15% EM)
3. **Implement MST + Connected Components** (easiest, highest impact)
4. **Re-evaluate** (expected: 20-30% EM)
5. **Implement KG Expansion** (enables multi-hop)
6. **Final evaluation** (expected: 35-55% EM, comparable to KG²RAG)

---

## Files to Review

To fully understand their implementation:
- `KG2RAG-main/code/kg_rag_distractor.py` - main evaluation script
- `KG2RAG-main/code/util/hotpot_evaluate_v1.py` - metrics calculation
- `KG2RAG-main/code/preprocess/hotpot_extraction.py` - triplet extraction

---

## Key Insights

1. **Graph topology matters:** MST + components turn random chunks into coherent reasoning paths
2. **Multi-hop is essential:** KG expansion finds entities not in initial retrieval
3. **Reranking helps:** BGE reranker improves final selection
4. **Implementation is straightforward:** NetworkX provides all graph algorithms

**Bottom line:** We have the data (entities, relations in Neo4j), we just need to **organize it using graph topology** like KG²RAG does.
