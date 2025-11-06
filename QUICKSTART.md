# Quick Start Guide - Agentic GraphRAG

This guide will help you get started with ingesting your own documents and querying them.

## Prerequisites

Make sure you've completed the installation steps from the main README:
1. Python 3.11+ installed
2. Dependencies installed (`pip install -r requirements.txt`)
3. Neo4j running (`./scripts/start_neo4j.sh`)
4. Environment variables set in `.env` (especially `GROQ_API_KEY`)

## Step 1: Prepare Your Documents

Place your documents in the `data/raw/` directory. Supported formats:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents (requires `pypdf`: `pip install pypdf`)
- `.docx` - Word documents (requires `python-docx`: `pip install python-docx`)

Example:
```bash
# Create the directory if it doesn't exist
mkdir -p data/raw

# Add your documents
cp /path/to/your/documents/*.txt data/raw/
```

We've included two sample documents to get you started:
- `data/raw/sample_medical.txt` - Medical/healthcare content
- `data/raw/sample_technology.txt` - Technology/AI content

## Step 2: Ingest Your Documents

Run the ingestion pipeline to build your knowledge graph:

```bash
# Ingest all documents from data/raw/
python ingest.py --dir data/raw/
```

This will:
1. Load all supported documents from the directory
2. Automatically infer the schema (entity types and relationships)
3. Extract entities using hybrid NER + LLM
4. Identify relationships between entities
5. Store everything in Neo4j (graph) and FAISS (vectors)

### Ingestion Options

```bash
# Ingest a single file
python ingest.py --file my_document.txt

# Ingest multiple specific files
python ingest.py --file doc1.txt --file doc2.pdf

# Ingest recursively (include subdirectories)
python ingest.py --dir data/raw/ --recursive

# Skip metadata enrichment for faster ingestion
python ingest.py --dir data/raw/ --no-metadata

# Use existing schema (no inference)
python ingest.py --dir data/raw/ --no-schema-inference

# Verbose output to see what's happening
python ingest.py --dir data/raw/ --verbose

# Get help
python ingest.py --help
```

## Step 3: Query Your Knowledge Graph

Once ingestion is complete, you can query your knowledge graph:

### Interactive Mode (Recommended)

```bash
python query.py
```

This opens an interactive prompt where you can:
- Ask questions naturally
- See intelligent routing (vector/graph/hybrid)
- View RAGAS evaluation metrics
- Type `help` for tips
- Type `exit` or press Ctrl+C to quit

Example queries:
```
🔍 Query: What medications treat diabetes?
🔍 Query: Explain what artificial intelligence is
🔍 Query: Who is the CEO of Tesla?
🔍 Query: Tell me about diabetes complications
```

### Single Query Mode

For one-off queries:

```bash
python query.py --query "What medications treat diabetes?"
```

### Query Options

```bash
# Retrieve more contexts
python query.py --query "Explain AI" --top-k 10

# Verbose output (show all retrieved contexts)
python query.py --query "Tell me about diabetes" --verbose

# Faster queries (skip evaluation)
python query.py --no-evaluation

# Get help
python query.py --help
```

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Make sure Neo4j is running
./scripts/start_neo4j.sh

# 2. Ingest your documents
python ingest.py --dir data/raw/ --verbose

# Output:
# ✅ Loaded 2 document(s)
# 🔄 Running ingestion pipeline...
# ✅ Ingestion complete in 45.23s
# 📊 Results:
#    • Documents processed: 2
#    • Entities extracted: 156
#    • Relations extracted: 89
#    • Neo4j nodes created: 156
#    • Neo4j edges created: 89

# 3. Query interactively
python query.py

# 🔍 Query: What medications treat type 2 diabetes?
#
# 💬 Response:
#    Metformin is the first-line medication for treating type 2 diabetes,
#    particularly in overweight individuals. Other medications include
#    Glipizide (a sulfonylurea), GLP-1 receptor agonists like Ozempic
#    and Trulicity, and SGLT2 inhibitors like Jardiance and Farxiga...
#
# 📊 Retrieval Stats:
#    • Contexts retrieved: 5
#    • Strategy used: hybrid
#
# 📈 RAGAS Metrics:
#    • Faithfulness:      0.927
#    • Answer Relevancy:  0.884
#    • Context Precision: 0.856
#    • Overall Score:     0.889
```

## Viewing Your Knowledge Graph

You can visualize your knowledge graph in Neo4j Browser:

1. Open http://localhost:7474 in your browser
2. Login with:
   - Username: `neo4j`
   - Password: `password123`
3. Run Cypher queries:

```cypher
// View all node types
CALL db.labels()

// View all relationship types
CALL db.relationshipTypes()

// Find all medications
MATCH (n:Drug) RETURN n LIMIT 25

// Find what treats diabetes
MATCH (drug:Drug)-[:TREATS]->(disease:Disease {name: "diabetes"})
RETURN drug, disease

// View the entire graph (be careful with large graphs!)
MATCH (n) RETURN n LIMIT 100
```

## Troubleshooting

### "Neo4j connection failed"
- Make sure Neo4j is running: `./scripts/start_neo4j.sh`
- Check Docker Desktop is running
- Verify Neo4j credentials in `.env` file

### "GROQ_API_KEY not found"
- Make sure you have a `.env` file (copy from `.env.example`)
- Add your Groq API key: `GROQ_API_KEY=your_key_here`
- Get a free key at: https://console.groq.com/keys

### "No documents found"
- Check your file paths are correct
- Verify file extensions are supported (.txt, .md, .pdf, .docx)
- Use `--verbose` flag to see what's happening

### "pypdf not installed" or "python-docx not installed"
- Install the missing package:
  ```bash
  pip install pypdf        # for PDF support
  pip install python-docx  # for Word document support
  ```

## Next Steps

1. **Add more documents**: Keep ingesting documents to grow your knowledge graph
   ```bash
   python ingest.py --file new_document.txt
   ```

2. **Explore the demo**: See all features with pre-loaded examples
   ```bash
   python demo.py
   ```

3. **Use the Python API**: Integrate into your own applications
   ```python
   from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
   ```

4. **Fine-tune parameters**: Adjust `top_k`, enable reranking, customize schema

5. **Deploy to production**: Use the robust error handling and logging built-in

## Need Help?

- Full documentation: See main `README.md`
- Report issues: https://github.com/anthropics/agentic-graphrag/issues
- Check configuration: `python verify_setup.py`
