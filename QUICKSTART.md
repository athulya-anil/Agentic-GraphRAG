# Quick Start Guide - Agentic GraphRAG

This guide will help you get started with the Agentic GraphRAG system, optimized for HotpotQA evaluation.

## Prerequisites

Make sure you've completed the installation steps from the main README:
1. Python 3.11+ installed
2. Dependencies installed (`pip install -r requirements.txt`)
3. Neo4j running (`./scripts/start_neo4j.sh`)
4. Environment variables set in `.env` (especially `GROQ_API_KEY`)
5. spaCy model downloaded (`python -m spacy download en_core_web_lg`)

## Option A: HotpotQA Evaluation (Recommended)

### Step 1: Ingest HotpotQA Documents

```bash
# Ingest documents for the 100-query evaluation set
python ingest_hotpotqa.py
```

This will:
1. Load HotpotQA documents from supporting facts
2. Extract entities using hybrid NER + LLM
3. Build knowledge graph in Neo4j (with sentence-level provenance)
4. Build FAISS vector index (sentence-level embeddings, 1024-dim)

### Step 2: Run Evaluation

```bash
# Run full evaluation on 100 queries
python evaluate_hotpotqa_full.py
```

Output:
```
✓ FAISS: X,XXX vectors
✓ Neo4j: X,XXX nodes, X,XXX relationships

RUNNING EVALUATION (CACHE DISABLED)
Evaluating: 100%|████████████████| 100/100

RESULTS
Overall Performance:
   EM: 0.XXX (XX.X%)
   Correct: XX/100

By Question Type:
   Comparison EM: 0.XXX (21 queries)
   Bridge EM: 0.XXX (79 queries)

By Retrieval Strategy:
   Vector: 0.XXX (N queries)
   Graph: 0.XXX (N queries)

✓ Results saved to: data/hotpotqa/results_full_graphrag.json
✓ Summary saved to: data/hotpotqa/summary_full_graphrag.json
```

## Option B: Custom Documents

### Step 1: Prepare Your Documents

Place your documents in the `data/raw/` directory. Supported formats:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents
- `.docx` - Word documents

```bash
mkdir -p data/raw
cp /path/to/your/documents/*.txt data/raw/
```

### Step 2: Ingest Documents

```bash
# Ingest all documents from data/raw/
python ingest.py --dir data/raw/ --verbose
```

This will:
1. Load all supported documents
2. Run SchemaAgent (infer entity types and relationships)
3. Run EntityAgent (hybrid spaCy + LLM extraction)
4. Run RelationAgent (relationship extraction)
5. Run ConflictResolutionAgent (deduplication)
6. Store to Neo4j (with provenance) and FAISS (sentence embeddings)

### Step 3: Query Interactively

```bash
python query.py
```

Example:
```
🔍 Query: What medications treat diabetes?

💬 Response: Metformin

📊 Retrieval Stats:
   • Contexts retrieved: 10
   • Strategy used: graph
   • Query type: multi_hop
```

### Ingestion Options

```bash
# Ingest a single file
python ingest.py --file my_document.txt

# Ingest recursively
python ingest.py --dir data/raw/ --recursive

# Skip metadata enrichment (faster)
python ingest.py --dir data/raw/ --no-metadata

# Use existing schema
python ingest.py --dir data/raw/ --no-schema-inference

# Get help
python ingest.py --help
```

### Query Options

```bash
# Single query mode
python query.py --query "What treats diabetes?"

# More results
python query.py --query "Explain AI" --top-k 15

# Verbose output
python query.py --query "Tell me about diabetes" --verbose

# Get help
python query.py --help
```

## Viewing Your Knowledge Graph

Open Neo4j Browser at http://localhost:7474

Login:
- Username: `neo4j`
- Password: `password123`

Useful Cypher queries:

```cypher
-- View all node labels
CALL db.labels()

-- View all relationship types
CALL db.relationshipTypes()

-- Count nodes and relationships
MATCH (n) RETURN count(n) as nodes
MATCH ()-[r]->() RETURN count(r) as relationships

-- Find entities by name
MATCH (n) WHERE toLower(n.name) CONTAINS 'diabetes' RETURN n

-- View provenance for an entity
MATCH (n) WHERE n.name = 'Some Entity'
RETURN n.name, n.provenance

-- Graph traversal (2 hops)
MATCH path = (start)-[*1..2]-(connected)
WHERE start.name = 'Some Entity'
RETURN path LIMIT 50
```

## Understanding the Output

### Query Result Format

```python
{
    "query": "Were X and Y of the same nationality?",
    "response": "yes",                    # EM-normalized answer
    "context": [                          # Evidence with provenance
        {
            "doc_id": 0,
            "sent_id": 5,
            "text": "X is an American...",
            "score": 0.85,
            "source": "graph"
        },
        ...
    ],
    "num_contexts": 12,
    "strategy": "graph",                  # VECTOR or GRAPH
    "query_type": "comparison"            # SINGLE_HOP, MULTI_HOP, COMPARISON, YES_NO
}
```

### Evaluation Metrics

The system uses **Exact Match (EM)** as the primary metric:

```python
# HotpotQA official normalization
def normalize_answer(s):
    s = s.lower()                    # lowercase
    s = remove_punctuation(s)        # remove punctuation
    s = remove_articles(s)           # remove a, an, the
    s = whitespace_fix(s)            # normalize whitespace
    return s

em = int(normalize_answer(prediction) == normalize_answer(gold))
```

## Key Concepts

### Query Types

| Type | Description | Strategy |
|------|-------------|----------|
| SINGLE_HOP | Direct fact lookup | VECTOR |
| MULTI_HOP | Bridge entity reasoning | GRAPH |
| COMPARISON | Compare 2+ entities | GRAPH |
| YES_NO | Boolean question | GRAPH |

### Answer Selection

The **AnswerSelector** is 100% symbolic (no LLM generation):

- **YES_NO**: Entity + constraint co-mention → "yes" or "no"
- **COMPARISON**: Boolean form → YES_NO logic; entity form → winner selection
- **FACTOID**: KG node ranking by frequency and multi-doc support

### Provenance

Every result includes `{doc_id, sent_id}` tracking:
- Which document the evidence came from
- Which sentence within that document
- Enables attribution and debugging

## Troubleshooting

### "Neo4j connection failed"
```bash
# Start Neo4j
./scripts/start_neo4j.sh

# Check Docker is running
docker ps

# Verify credentials in .env
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password123
```

### "GROQ_API_KEY not found"
```bash
# Create .env from example
cp .env.example .env

# Add your key
echo "GROQ_API_KEY=your_key_here" >> .env
```

### "Empty FAISS index"
```bash
# Verify ingestion completed
python -c "from src.vector import get_faiss_index; print(f'Vectors: {get_faiss_index().get_count()}')"

# Re-run ingestion if needed
python ingest_hotpotqa.py
```

### "spaCy model not found"
```bash
python -m spacy download en_core_web_lg
```

## Next Steps

1. **Analyze results**: Check `data/hotpotqa/results_*.json` for per-query breakdown
2. **View failures**: Look at queries with `em: 0` to understand failure modes
3. **Explore graph**: Use Neo4j Browser to visualize your knowledge graph
4. **Customize**: Modify `src/answering/answer_selector.py` for different selection logic

## Files Reference

| File | Purpose |
|------|---------|
| `ingest_hotpotqa.py` | Ingest HotpotQA documents |
| `evaluate_hotpotqa_full.py` | Run evaluation |
| `query.py` | Interactive queries |
| `data/hotpotqa/results_*.json` | Per-query results |
| `data/hotpotqa/summary_*.json` | Aggregate metrics |
| `data/faiss_index/` | Vector embeddings |

## Need Help?

- Architecture details: See [ARCHITECTURE.md](ARCHITECTURE.md)
- Verify setup: `python verify_setup.py`
- Check system: `python validate_system.py`
