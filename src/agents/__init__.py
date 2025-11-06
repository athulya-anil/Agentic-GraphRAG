"""
Agents module for Agentic GraphRAG

This module contains all autonomous agents:
- SchemaAgent: Infers graph schemas from documents
- EntityAgent: Extracts entities with metadata enrichment
- RelationAgent: Extracts relationships between entities
- OrchestratorAgent: Routes queries to optimal retrieval strategy
- ReflectionAgent: Evaluates and optimizes system performance
"""

from .schema_agent import SchemaAgent, get_schema_agent
from .entity_agent import EntityAgent, get_entity_agent
from .relation_agent import RelationAgent, get_relation_agent
from .orchestrator_agent import (
    OrchestratorAgent,
    get_orchestrator_agent,
    RetrievalStrategy,
    QueryType
)
from .reflection_agent import ReflectionAgent, get_reflection_agent


__all__ = [
    # Schema Agent
    "SchemaAgent",
    "get_schema_agent",
    # Entity Agent
    "EntityAgent",
    "get_entity_agent",
    # Relation Agent
    "RelationAgent",
    "get_relation_agent",
    # Orchestrator Agent
    "OrchestratorAgent",
    "get_orchestrator_agent",
    "RetrievalStrategy",
    "QueryType",
    # Reflection Agent
    "ReflectionAgent",
    "get_reflection_agent",
]
