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
- **ConflictResolutionAgent** deduplicates entities and validates relationships
- **OrchestratorAgent** intelligently routes queries with failure-aware logic
- **GraphFailurePredictor** predicts when graph retrieval will fail
- **ReflectionAgent** continuously evaluates and improves system performance

This creates a **schema-agnostic, self-improving knowledge graph system** that works on any domain—medical literature, legal documents, technical manuals, research papers, and more.

## 🚀 Key Features

### 1. **Autonomous Schema Inference**
No need to predefine node types or relationship types. The SchemaAgent analyzes your documents and automatically discovers:
- Entity types (e.g., Disease, Drug, Symptom, Treatment)
- Relationship types (e.g., TREATS, CAUSES, DIAGNOSED_BY)
- Property schemas for each entity type

### 2. **Intelligent Query Routing with Failure Prediction**
The OrchestratorAgent dynamically chooses between:
- **Vector Search**: Semantic similarity for conceptual questions
- **Graph Traversal**: Relational queries leveraging knowledge structure

The system uses **failure-aware routing** to avoid graph retrieval for high-risk queries:
- Temporal queries (weather, "now", "current") → vector search
- Contact info (phone numbers, addresses) → vector search
- Unknown entity relationships → entity validation first

The routing decision adapts based on historical performance—if one strategy consistently outperforms another for similar queries, the system learns to prefer it.

### 3. **Performance-Aware Adaptive System**
The ReflectionAgent evaluates every response using RAGAS metrics:
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevancy**: Does it address the query?
- **Context Precision**: Was the retrieved context relevant?
- **Context Recall**: Was all necessary context retrieved?

This creates a feedback loop where:
- Performance scores are recorded per retrieval strategy
- OrchestratorAgent adjusts routing based on historical success rates
- Failure patterns are analyzed to identify systemic issues
- Schema refinements and parameter adjustments are suggested for human review

### 4. **Multi-Stage Ingestion Pipeline**
8-stage ingestion with validation and conflict resolution:
- Parse → Extract entities → Extract relations → Validate entities
- Validate relations → Resolve conflicts → Store graph → Store vectors

The ConflictResolutionAgent handles:
- Entity deduplication (same entity, different names)
- Relationship conflict resolution (contradictory facts)
- Property normalization and cleaning

### 5. **Production-Ready Architecture**
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
              ┌─────────────┴─────────────┐
              │                           │
      ┌───────▼────────┐         ┌───────▼────────┐
      │ Vector Search  │         │ Graph Traversal│
      └───────┬────────┘         └───────┬────────┘
              └─────────────┬─────────────┘
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
│   │   ├── schema_agent.py             # Schema inference from documents
│   │   ├── entity_agent.py             # Entity extraction (NER + LLM)
│   │   ├── relation_agent.py           # Relationship extraction
│   │   ├── conflict_resolution_agent.py # Entity deduplication & validation
│   │   ├── orchestrator_agent.py       # Query routing with failure-aware logic
│   │   ├── failure_predictor.py        # Graph failure risk prediction
│   │   └── reflection_agent.py         # Performance evaluation
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

### **ConflictResolutionAgent**
Ensures knowledge graph consistency and quality:
- Entity deduplication using semantic similarity + LLM reasoning
- Detects entities that are the same despite different names
- Relationship conflict resolution (handles contradictory facts)
- Property normalization and type validation
- Maintains entity merge history for transparency

### **OrchestratorAgent**
Routes queries to optimal retrieval strategy with adaptive learning:
- Query classification (factual, conceptual, relational, exploratory)
- Failure-aware routing (avoids graph for temporal/contact queries)
- Strategy selection (vector or graph) based on query type + risk
- Performance tracking per strategy (rolling window of last 100 queries)
- Automatic strategy switching when one outperforms another by >15%
- Response synthesis from retrieved context

### **GraphFailurePredictor**
Predicts when graph retrieval is likely to fail:
- Rule-based risk scoring for temporal queries (weather, "now", "current")
- Contact info detection (phone numbers, addresses)
- Relationship queries with unknown entities
- Risk levels: HIGH (avoid graph), MODERATE (validate first), LOW (use graph)

### **ReflectionAgent**
Evaluates system performance and provides actionable feedback:
- RAGAS metric computation (faithfulness, relevancy, precision, recall)
- Failure pattern analysis (which queries/metrics are failing)
- Performance trend monitoring (recent vs. historical)
- Schema refinement suggestions via LLM
- Improvement recommendations based on metric patterns

*Note: The ReflectionAgent identifies issues and suggests improvements for human review—it does not automatically retrain models or modify parameters.*

## 🎓 Research Contributions

This project advances the state-of-the-art in several areas:

1. **Autonomous Knowledge Graph Construction**: Traditional KG construction requires domain experts to define schemas. Our approach automates this using LLM-powered agents that infer structure from documents.

2. **Adaptive Query Routing**: Most RAG systems use fixed retrieval strategies. Our OrchestratorAgent learns when to use vector search vs. graph traversal based on historical performance data.

3. **RAGAS-Based Performance Feedback**: The ReflectionAgent creates a feedback loop where performance metrics inform routing decisions and surface actionable improvement suggestions.

4. **Domain-Agnostic Design**: Unlike domain-specific solutions (e.g., biomedical KG systems requiring UMLS/MeSH), this architecture works across any document corpus without predefined ontologies.

5. **End-to-End Agentic System**: Goes beyond KG construction to include intelligent retrieval and query answering—a complete pipeline from documents to answers.

6. **Failure-Aware Routing**: First system to predict graph retrieval failures before execution using empirical failure analysis and rule-based risk scoring.

## 📈 Evaluation Results

We evaluated the system on the **MS MARCO dataset** (100 real-world queries, 108 passages) using RAGAS metrics.

### Baseline Performance (Before Improvements)

```
Overall Score: 0.689 (68.9%)
Average Latency: 811ms per query
Queries Evaluated: 100

RAGAS Metrics:
├─ Faithfulness:       0.718
├─ Answer Relevancy:   0.627
├─ Context Precision:  0.738
└─ Context Recall:     0.674
```

### Strategy Performance Comparison

**Vector Search Strategy:**
- Queries: 51
- Average Score: **0.852**
- Failures: 4 (7.8% failure rate)
- Strengths: Reliable for conceptual questions

**Graph Traversal Strategy:**
- Queries: 49
- Average Score: **0.520**
- Failures: 19 (38.7% failure rate)
- Challenges: Temporal queries, missing entities, coverage gaps

### Failure Analysis

Analysis of 19 graph retrieval failures identified systematic patterns:

| Failure Type | Count | Root Cause |
|--------------|-------|------------|
| Temporal queries | 4 | Static KG cannot answer time-dependent questions |
| Contact information | 2 | Phone numbers/addresses not in knowledge graph |
| Missing entities | 9 | Queried entities not present in graph |
| Relationship gaps | 4 | Required relationships not extracted |

### Recent Improvements (Under Evaluation)

1. **GraphFailurePredictor**: Rule-based risk scoring to detect high-risk queries before execution
2. **ConflictResolutionAgent**: Entity deduplication and relationship validation
3. **Multi-stage Validation**: 8-stage ingestion pipeline with quality checks

These improvements are designed to address the identified failure patterns. Full evaluation in progress.

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
