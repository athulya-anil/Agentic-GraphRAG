"""
Pipeline module for Agentic GraphRAG

This module contains the ingestion and retrieval pipelines:
- IngestionPipeline: Document ingestion with schema inference, entity/relation extraction
- RetrievalPipeline: Intelligent query retrieval with multi-strategy routing
"""

from .ingestion import IngestionPipeline, get_ingestion_pipeline
from .retrieval import RetrievalPipeline, get_retrieval_pipeline


__all__ = [
    "IngestionPipeline",
    "get_ingestion_pipeline",
    "RetrievalPipeline",
    "get_retrieval_pipeline",
]
