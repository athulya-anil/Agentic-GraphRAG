# Agentic GraphRAG

A production-ready self-adaptive multi-agent system for knowledge graph construction and retrieval.

## Overview

Agentic GraphRAG is an intelligent RAG system where AI agents automatically:

1. **Infer graph schemas** from documents (SchemaAgent)
2. **Extract entities and relations** (EntityAgent, RelationAgent)
3. **Build Neo4j knowledge graphs** dynamically
4. **Route queries intelligently** between vector/graph/hybrid retrieval (OrchestratorAgent)
5. **Self-optimize through reflection** (ReflectionAgent)

## Tech Stack

- **LLM**: Groq API (Llama 3.3-70B) - free tier
- **Graph DB**: Neo4j (Docker)
- **Vector DB**: FAISS
- **Agent Framework**: LangGraph
- **NER**: spaCy
- **Evaluation**: RAGAS

## Project Structure

```
agentic-graphrag/
├── src/
│   ├── agents/           # 5+ agent implementations
│   ├── graph/            # Neo4j manager
│   ├── vector/           # FAISS index
│   ├── pipeline/         # Ingestion + retrieval
│   └── utils/            # LLM client, config
├── data/                 # Raw + processed data
├── scripts/              # Ingestion, eval, demo
├── tests/                # Unit tests
└── configs/              # YAML configs
```

## Prerequisites

- Python 3.11+
- Docker Desktop
- Groq API key (free at https://console.groq.com)

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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

## Usage

### Basic LLM Client Usage

```python
from src.utils import get_llm_client

# Initialize client
client = get_llm_client()

# Generate text
response = client.generate(
    prompt="Explain knowledge graphs in one sentence.",
    temperature=0.0
)
print(response)

# Generate structured JSON
json_response = client.generate_json(
    prompt="Create a JSON with name='John' and age=30"
)
print(json_response)
```

### Configuration Management

```python
from src.utils import get_config

# Load configuration
config = get_config()

# Access settings
print(config.llm.model)           # llama-3.3-70b-versatile
print(config.neo4j.uri)           # bolt://localhost:7687
print(config.embedding.model_name) # sentence-transformers/all-MiniLM-L6-v2
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

### Project Status

**Phase 1 (Completed)** ✅
- [x] Project structure setup
- [x] Virtual environment and dependencies
- [x] Neo4j Docker setup
- [x] Environment configuration
- [x] LLM client implementation
- [x] Configuration management

**Phase 2 (Next)**
- [ ] Neo4j manager (CRUD operations)
- [ ] FAISS vector store
- [ ] Testing suite

**Phase 3 (Future)**
- [ ] Agent implementations
- [ ] Pipeline orchestration
- [ ] Evaluation framework
- [ ] Web UI

## Troubleshooting

### Docker Not Running
```
Error: Cannot connect to Docker daemon
```
**Solution**: Start Docker Desktop and wait for it to fully initialize.

### API Key Error
```
Configuration Error: GROQ_API_KEY not set
```
**Solution**:
1. Copy `.env.example` to `.env`
2. Add your Groq API key
3. Make sure you're in the project root directory

### Neo4j Connection Failed
```
Error: Unable to connect to Neo4j
```
**Solution**:
1. Check Docker is running: `docker ps`
2. Restart Neo4j: `./scripts/start_neo4j.sh`
3. Wait 10-15 seconds for Neo4j to initialize

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

## Contact

For questions and support, please open an issue on GitHub.

---

Built with ❤️ using Claude Code
