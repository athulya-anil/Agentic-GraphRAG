# Agentic GraphRAG

A production-ready self-adaptive multi-agent system for autonomous knowledge graph construction and intelligent retrieval.

## 🎯 Vision

**What if you could build domain-specific knowledge graphs automatically, without manual schema design?**

Agentic GraphRAG is a novel RAG system that uses autonomous AI agents to handle the entire lifecycle of knowledge graph construction and retrieval. Unlike traditional approaches that require manual schema definition and entity extraction rules, our system **adapts to any domain automatically**.

### The Core Innovation

Instead of hard-coding schemas and extraction rules, **AI agents do all the work**:

- **SchemaAgent** analyzes your documents and infers the optimal graph structure
- **EntityAgent** extracts entities using hybrid NER (spaCy) + LLM reasoning
- **RelationAgent** identifies meaningful relationships between entities
- **OrchestratorAgent** intelligently routes queries to the best retrieval strategy
- **ReflectionAgent** continuously evaluates and improves system performance

This creates a **schema-agnostic, self-improving knowledge graph system** that works on any domain—medical literature, legal documents, technical manuals, research papers, and more.

## 🚀 Key Features

### 1. **Autonomous Schema Inference**
No need to predefine node types or relationship types. The SchemaAgent analyzes your documents and automatically discovers:
- Entity types (e.g., Disease, Drug, Symptom, Treatment)
- Relationship types (e.g., TREATS, CAUSES, DIAGNOSED_BY)
- Property schemas for each entity type

### 2. **Intelligent Multi-Strategy Retrieval**
The OrchestratorAgent dynamically chooses between:
- **Vector Search**: Semantic similarity for conceptual questions
- **Graph Traversal**: Relational queries leveraging knowledge structure
- **Hybrid Retrieval**: Combines both approaches with learned weights

### 3. **Self-Optimization Loop**
The ReflectionAgent uses RAGAS metrics to:
- Evaluate answer quality (faithfulness, relevancy)
- Assess context precision and recall
- Adjust retrieval strategies based on performance
- Suggest schema improvements

### 4. **Production-Ready Architecture**
- Robust error handling with automatic retry logic
- Batch operations for efficient large-scale processing
- Type-safe configuration with Pydantic validation
- Comprehensive logging and monitoring

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Document Ingestion                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    SchemaAgent        │
                    │  (Infer Graph Schema) │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    EntityAgent        │
                    │ (Extract Entities)    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   RelationAgent       │
                    │ (Extract Relations)   │
                    └───────────┬───────────┘
                                │
                ┌───────────────▼──────────────┐
                │                              │
        ┌───────▼────────┐          ┌────────▼─────────┐
        │   Neo4j Graph  │          │  FAISS Vector DB │
        │  (Nodes+Edges) │          │   (Embeddings)   │
        └───────┬────────┘          └────────┬─────────┘
                │                            │
                └───────────┬────────────────┘
                            │
                ┌───────────▼────────────┐
                │  OrchestratorAgent     │
                │  (Query Router)        │
                └───────────┬────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│ Vector Search  │  │ Graph Traversal│  │   Hybrid    │
└───────┬────────┘  └───────┬────────┘  └──────┬──────┘
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼────────────┐
                │   ReflectionAgent      │
                │ (Evaluate & Optimize)  │
                └────────────────────────┘
```

## 🛠️ Tech Stack

- **LLM**: Groq API (Llama 3.3-70B) - free tier with high throughput
- **Graph DB**: Neo4j (Dockerized) - property graph database
- **Vector DB**: FAISS - efficient similarity search
- **Agent Framework**: LangGraph - orchestrate multi-agent workflows
- **NER**: spaCy - fast named entity recognition
- **Evaluation**: RAGAS - RAG assessment metrics
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

## 📁 Project Structure

```
agentic-graphrag/
├── src/
│   ├── agents/           # Multi-agent implementations
│   │   ├── schema_agent.py       # Schema inference from documents
│   │   ├── entity_agent.py       # Entity extraction (NER + LLM)
│   │   ├── relation_agent.py     # Relationship extraction
│   │   ├── orchestrator_agent.py # Query routing logic
│   │   └── reflection_agent.py   # Performance evaluation
│   ├── graph/            # Neo4j graph database layer
│   │   └── neo4j_manager.py      # CRUD operations, schema management
│   ├── vector/           # Vector store implementation
│   │   └── faiss_index.py        # FAISS indexing and retrieval
│   ├── pipeline/         # Data processing pipelines
│   │   ├── ingestion.py          # Document ingestion pipeline
│   │   └── retrieval.py          # Multi-strategy retrieval
│   └── utils/            # Core utilities
│       ├── llm_client.py         # Groq API client with retry logic
│       └── config.py             # Configuration management
├── data/                 # Data directory
│   ├── raw/              # Raw documents
│   └── processed/        # Processed outputs
├── scripts/              # Utility scripts
│   ├── start_neo4j.sh    # Start Neo4j container
│   ├── stop_neo4j.sh     # Stop Neo4j container
│   └── setup.sh          # Environment setup
├── tests/                # Unit and integration tests
├── configs/              # Configuration files
└── requirements.txt      # Python dependencies
```

## 🧠 Agent Responsibilities

### **SchemaAgent**
Analyzes document corpus to infer optimal knowledge graph structure:
- Identifies entity types (node labels)
- Discovers relationship types (edge labels)
- Defines property schemas for nodes and edges
- Adapts schema as new document types are encountered

### **EntityAgent**
Extracts entities using hybrid approach:
- Fast entity recognition with spaCy NER
- LLM-powered classification for ambiguous cases
- Entity resolution and deduplication
- Property extraction for each entity

### **RelationAgent**
Identifies and extracts relationships:
- Dependency parsing for grammatical relationships
- LLM-based relation classification
- Confidence scoring for each relationship
- Temporal and conditional relationship handling

### **OrchestratorAgent**
Routes queries to optimal retrieval strategy:
- Query classification (factual vs. conceptual)
- Strategy selection (vector/graph/hybrid)
- Dynamic weight adjustment for hybrid retrieval
- Response synthesis from multiple sources

### **ReflectionAgent**
Evaluates and improves system performance:
- RAGAS metric computation (faithfulness, relevancy, precision, recall)
- Identifies failure patterns
- Suggests schema refinements
- Triggers retraining or parameter adjustment

## 🎓 Research Contributions

This project advances the state-of-the-art in several areas:

1. **Autonomous Knowledge Graph Construction**: Traditional KG construction requires domain experts to define schemas. Our approach automates this using LLM-powered agents.

2. **Adaptive Multi-Strategy Retrieval**: Most RAG systems use fixed retrieval strategies. Our OrchestratorAgent learns when to use vector search vs. graph traversal vs. hybrid approaches.

3. **Self-Improving RAG Systems**: The ReflectionAgent creates a feedback loop for continuous improvement, making the system more accurate over time without human intervention.

4. **Domain-Agnostic Design**: Unlike domain-specific solutions, this architecture works across any document corpus—from medical research to legal documents to technical manuals.

## 💡 Use Cases

- **Medical Research**: Automatically build knowledge graphs from PubMed papers linking diseases, drugs, symptoms, and treatments
- **Legal Analysis**: Extract entities and relationships from case law and regulations
- **Technical Documentation**: Create navigable knowledge graphs from API docs, manuals, and specifications
- **Academic Research**: Build citation networks and concept maps from research papers
- **Business Intelligence**: Extract insights from company reports, market research, and competitive analysis

## 📊 Evaluation Metrics

We use RAGAS (RAG Assessment) framework to evaluate:
- **Faithfulness**: Are answers grounded in retrieved context?
- **Answer Relevancy**: Do answers directly address the question?
- **Context Precision**: Is retrieved context relevant?
- **Context Recall**: Is all necessary context retrieved?

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (for Neo4j)
- Groq API key (free at https://console.groq.com)

## 📦 Installation

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/athulya-anil/agentic-graphrag.git
cd agentic-graphrag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for entity extraction)
python -m spacy download en_core_web_sm
```

### 2. Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your key from: https://console.groq.com/keys
```

Required environment variables in `.env`:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

### 3. Start Neo4j Database

```bash
# Make sure Docker Desktop is running first!

# Start Neo4j
./scripts/start_neo4j.sh
```

Access Neo4j Browser at: http://localhost:7474
- Username: `neo4j`
- Password: `password123`

### 4. Verify Installation

Test the LLM client:
```bash
python -m src.utils.llm_client
```

You should see:
```
✅ Configuration loaded successfully!
📝 Testing simple generation...
Response: Paris
📊 Testing JSON generation...
JSON Response: {...}
📈 Total tokens used: XX
✅ All tests passed!
```

## 💻 Usage

### Quick Start: Real-Time Pipeline

The simplest way to ingest your own documents and query them:

```bash
# Make sure Neo4j is running
./scripts/start_neo4j.sh

# STAGE 1: Ingest your documents
python ingest.py --dir data/raw/

# STAGE 2: Query your knowledge graph (interactive mode)
python query.py
```

#### Ingest Your Documents

```bash
# Ingest a single file
python ingest.py --file path/to/document.txt

# Ingest multiple files
python ingest.py --file doc1.txt --file doc2.pdf --file doc3.md

# Ingest all files in a directory
python ingest.py --dir data/raw/

# Ingest recursively with options
python ingest.py --dir data/raw/ --recursive --verbose

# See all options
python ingest.py --help
```

Supported formats: `.txt`, `.md`, `.pdf`, `.docx`

#### Query Your Knowledge Graph

```bash
# Interactive mode (recommended)
python query.py

# Single query from command line
python query.py --query "What medications treat diabetes?"

# Query with custom options
python query.py --query "Explain AI" --top-k 10 --verbose

# See all options
python query.py --help
```

### Running the Demo

To see the full system with pre-loaded examples:

```bash
# Make sure Neo4j is running
./scripts/start_neo4j.sh

# Run the comprehensive demo
python demo.py
```

This will demonstrate:
- Automatic schema inference from documents
- Entity extraction with metadata enrichment
- Relationship extraction and knowledge graph construction
- Intelligent query routing across multiple strategies
- Performance evaluation with RAGAS metrics

### Advanced Usage (Python API)

#### 1. Document Ingestion

```python
from src.pipeline import get_ingestion_pipeline

# Initialize pipeline
pipeline = get_ingestion_pipeline(
    schema_path="data/processed/schema.json",
    auto_refine_schema=True
)

# Ingest documents
documents = [
    "Your document text here...",
    "Another document..."
]

results = pipeline.ingest_documents(
    documents,
    infer_schema=True,
    enrich_metadata=True
)

print(f"Extracted {results['entities_extracted']} entities")
print(f"Created {results['nodes_created']} nodes")
```

#### 2. Query Retrieval

```python
from src.pipeline import get_retrieval_pipeline

# Initialize pipeline
pipeline = get_retrieval_pipeline(
    use_reranking=False,
    use_reflection=True
)

# Query with automatic routing
result = pipeline.query(
    "What medications treat diabetes?",
    top_k=5,
    evaluate=True
)

print(f"Response: {result['response']}")
print(f"Metrics: {result['metrics']}")
```

#### 3. Working with Individual Agents

```python
from src.agents import (
    get_schema_agent,
    get_entity_agent,
    get_relation_agent,
    get_orchestrator_agent
)

# Schema inference
schema_agent = get_schema_agent()
schema = schema_agent.infer_schema_from_documents(documents)

# Entity extraction
entity_agent = get_entity_agent(schema=schema)
entities = entity_agent.extract_entities_from_text(
    "Apple Inc. is headquartered in Cupertino.",
    enrich_metadata=True
)

# Relationship extraction
relation_agent = get_relation_agent(schema=schema)
relations = relation_agent.extract_relations_from_text(
    text,
    entities
)

# Query routing
orchestrator = get_orchestrator_agent()
routing = orchestrator.route_query("Your query here")
print(f"Strategy: {routing['strategy']}")
```

#### 4. Direct Database Access

```python
from src.graph import get_neo4j_manager
from src.vector import get_faiss_index

# Neo4j operations
neo4j = get_neo4j_manager()
neo4j.create_node("Person", {"name": "John", "age": 30})
schema = neo4j.get_schema()

# FAISS vector search
faiss = get_faiss_index()
faiss.add(["Text to index"], [{"metadata": "value"}])
results = faiss.search("query text", top_k=5)
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

## License

MIT License
