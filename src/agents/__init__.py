"""
Agents module for Agentic GraphRAG

This module contains all autonomous agents:
- SchemaAgent: Infers graph schemas from documents
- EntityAgent: Extracts entities with metadata enrichment
- RelationAgent: Extracts relationships between entities
- OrchestratorAgent: Routes queries to optimal retrieval strategy
- QueryParserAgent: Parses query intent for graph traversal
- ConflictResolutionAgent: Resolves entity/relation conflicts
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
from .query_parser_agent import QueryParserAgent, get_query_parser
from .conflict_resolution_agent import ConflictResolutionAgent, get_conflict_resolution_agent


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
    # Query Parser Agent
    "QueryParserAgent",
    "get_query_parser",
    # Conflict Resolution Agent
    "ConflictResolutionAgent",
    "get_conflict_resolution_agent",
]
