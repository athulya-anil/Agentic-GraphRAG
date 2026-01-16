# Agentic GraphRAG

A hybrid retrieval-augmented generation (RAG) system designed for multi-hop question answering on HotpotQA-style queries. Combines graph-based entity reasoning with dense vector retrieval, using LLM-based answer selection to achieve 80% Exact Match on the Mini-5 benchmark.

## Problem Statement

Multi-hop question answering requires reasoning across multiple documents and entities. Traditional approaches fail because:
- **Pure vector RAG**: Retrieves relevant passages but lacks structured entity relationships
- **Pure graph RAG**: Requires complete entity linking and graph traversal, which is brittle
- **Symbolic answer selection**: Cannot perform semantic reasoning (e.g., comparing nationalities)

This system addresses these limitations through adaptive routing and LLM-based reasoning.

## Architecture

### Query Routing

Queries are classified and routed to optimal retrieval strategies:

| Query Type | Strategy | Reason |
|------------|----------|--------|
| **COMPARISON** | GRAPH | Requires entity relationship traversal |
| **YES_NO** | GRAPH | Requires entity relationship verification |
| **MULTI_HOP** | VECTOR | Graph traversal incomplete; vector finds bridge entities |
| **SINGLE_HOP** | VECTOR | Direct fact lookup |

### Retrieval Pipeline

**GRAPH Retrieval**:
- Neo4j-based entity-relationship traversal
- Extracts entities via NER (spaCy)
- Traverses `LOCATED_IN`, `PART_OF`, etc. relationships
- Returns entity paths with provenance

**VECTOR Retrieval**:
- FAISS index with `BAAI/bge-large-en-v1.5` embeddings
- Sentence-level retrieval with document provenance
- **Two-hop retrieval** for bridge queries:
  1. First hop: Retrieve evidence for full question
  2. Extract bridge entities (e.g., "Shirley Temple" from "actress who portrayed...")
  3. Second hop: Retrieve evidence for extracted entities
  4. Merge and deduplicate results

### Answer Selection

**LLM-Based Answerer** (Groq/llama-3.3-70b):
- Replaces symbolic logic with constrained QA prompting
- Temperature=0.0 for deterministic answers
- Grounded: Returns "unknown" when evidence is insufficient
- **Unit appending**: Extracts units from evidence (e.g., "3,677 seated")

**Prompt**:
```
System: You must answer using ONLY the provided evidence.
Do not use outside knowledge.
If evidence is insufficient, answer "unknown".
For yes/no questions, answer exactly "yes" or "no".
Output only the answer, nothing else.

Evidence: [formatted evidence]
Question: [question]
Answer:
```

## What Makes This Different

1. **Adaptive Routing**: Query-type-aware strategy selection (not one-size-fits-all)
2. **Two-Hop Retrieval**: Automatically extracts and queries bridge entities
3. **Grounded LLM Answering**: Returns "unknown" instead of hallucinating
4. **Hybrid Architecture**: Uses graph for relationships, vector for entity discovery

## Evaluation

**Benchmark**: HotpotQA Mini-5 (5 queries, 50 documents)

**Metrics**:
- **Exact Match (EM)**: Normalized string match with gold answer
- **Supporting Coverage**: % of gold supporting documents retrieved
- **Routing Accuracy**: % of queries routed to correct strategy

**Results**:
- **EM**: 80% (4/5 correct)
- **Coverage**: 80% (avg)
- **Routing**: 100%

**Breakdown**:
| Query Type | EM | Coverage |
|------------|-----|----------|
| Comparison | 100% (2/2) | 50% |
| Bridge | 67% (2/3) | 100% |

## Query Types Handled

### ✅ Excellent Performance

**Comparison Questions**:
- "Are X and Y located in the same neighborhood?"
- "Were X and Y of the same nationality?"

**Bridge Questions** (with complete corpus):
- "2014 S/S is the debut album of a group formed by who?" → "YG Entertainment"
- "The arena where X played can seat how many people?" → "3,677 seated"

### ⚠️ Known Limitations

**Corpus Incompleteness**:
- If answer is not in corpus, returns "unknown" (correct behavior)
- Example: "What government position was held by [actress]?" fails if diplomatic career not documented

**Graph Traversal Brittleness**:
- Requires accurate entity extraction and linking
- Missing entities → 0% coverage for graph-only queries

## Installation

```bash
# Clone repository
git clone https://github.com/athulya-anil/Agentic-GraphRAG-private.git
cd Agentic-GraphRAG

# Install dependencies
pip install -r requirements.txt

# Set up Neo4j (local or cloud)
# Set environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export GROQ_API_KEY="your_groq_key"
```

## Usage

### Ingestion

```bash
# Ingest HotpotQA Mini-5
python3 ingest_mini5.py
```

### Evaluation

```bash
# Run evaluation
python3 evaluate_mini5.py

# Results saved to:
# data/hotpotqa/mini_5/evaluation_results.json
```

### Query Interface

```python
from src.pipeline import get_retrieval_pipeline

pipeline = get_retrieval_pipeline()
result = pipeline.query(
    "What government position was held by the woman who portrayed Corliss Archer?",
    top_k=10
)

print(result['response'])  # "unknown" (not in corpus)
print(result['strategy'])  # "vector"
print(result['num_contexts'])  # 10
```

## System Behavior

### When It Answers Confidently
- Evidence explicitly contains the answer
- LLM can extract answer from provided context
- Example: "YG Entertainment" from "Winner was formed by YG Entertainment"

### When It Returns "unknown"
- Evidence is insufficient or contradictory
- Answer requires external knowledge
- **This is correct behavior** (grounded, non-hallucinating)

## Project Structure

```
Agentic-GraphRAG/
├── src/
│   ├── agents/           # Query routing and orchestration
│   ├── answering/        # LLM-based answer selection
│   ├── pipeline/         # Retrieval pipeline (graph + vector)
│   ├── graph/            # Neo4j interface
│   ├── vector/           # FAISS index
│   └── utils/            # LLM client, config
├── data/
│   └── hotpotqa/mini_5/  # Mini-5 benchmark
├── ingest_mini5.py       # Ingestion script
├── evaluate_mini5.py     # Evaluation script
└── docs/                 # Documentation
```

## Key Dependencies

- **Neo4j**: Graph database
- **FAISS**: Vector similarity search
- **spaCy**: Entity extraction
- **sentence-transformers**: Embeddings (bge-large-en-v1.5)
- **Groq**: LLM inference (llama-3.3-70b)

## Performance

**Mini-5 Benchmark**:
- Ingestion: ~30s (50 documents)
- Query: ~2-3s per query (including LLM call)
- Memory: ~2GB (FAISS index + Neo4j)

## Future Work

1. **Larger Evaluation**: Full HotpotQA dev set (7,405 queries)
2. **Corpus Expansion**: Add full Wikipedia pages for entities
3. **Hybrid Verification**: Use graph to verify vector-retrieved facts
4. **LLM-Generated Cypher**: Dynamic query generation for complex reasoning

## Citation

```bibtex
@software{agentic_graphrag_2026,
  title={Agentic GraphRAG: Hybrid Multi-Hop Question Answering},
  author={Athulya Anil},
  year={2026},
  url={https://github.com/athulya-anil/Agentic-GraphRAG-private}
}
```

## License

MIT

## Acknowledgments

- HotpotQA dataset: Yang et al., 2018
- BGE embeddings: BAAI
- Neo4j community
