"""
Agentic GraphRAG - Self-Adaptive Multi-Agent RAG System

A production-ready system for autonomous knowledge graph construction and intelligent retrieval.

Main Components:
- agents: Multi-agent system (Schema, Entity, Relation, Orchestrator, Reflection)
- graph: Neo4j knowledge graph management
- vector: FAISS vector store for similarity search
- pipeline: Ingestion and retrieval pipelines
- utils: Configuration and LLM client

Quick Start:
    from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline

    # Ingest documents
    ingestion = get_ingestion_pipeline()
    ingestion.ingest_documents(documents)

    # Query
    retrieval = get_retrieval_pipeline()
    result = retrieval.query("Your question here")
"""

__version__ = "0.1.0"

from . import agents
from . import graph
from . import vector
from . import pipeline
from . import utils

__all__ = ["agents", "graph", "vector", "pipeline", "utils"]
