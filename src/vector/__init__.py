"""
Vector Module for Agentic GraphRAG

This module provides vector storage and similarity search using FAISS.

Author: Agentic GraphRAG Team
"""

from .faiss_index import (
    FAISSIndex,
    get_faiss_index,
    reset_faiss_index,
)


__all__ = [
    "FAISSIndex",
    "get_faiss_index",
    "reset_faiss_index",
]
