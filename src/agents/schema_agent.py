"""
SchemaAgent for Agentic GraphRAG

This agent analyzes documents to automatically infer optimal knowledge graph schemas:
- Identifies entity types (node labels)
- Discovers relationship types (edge labels)
- Defines property schemas for nodes and edges
- Adapts schema as new document types are encountered

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set
from collections import Counter

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaAgent:
    """
    Agent responsible for inferring knowledge graph schemas from documents.

    The SchemaAgent analyzes document content to automatically discover:
    - Entity types (e.g., Person, Organization, Location, Disease, Drug)
    - Relationship types (e.g., WORKS_AT, TREATS, LOCATED_IN)
    - Property schemas for each entity and relationship type
    - Domain-specific terminology and concepts
    """

    def __init__(self):
        """Initialize the SchemaAgent."""
        self.llm_client = get_llm_client()
        self.config = get_config()
        self.discovered_entities: Set[str] = set()
        self.discovered_relations: Set[str] = set()
        self.entity_properties: Dict[str, Set[str]] = {}
        self.relation_properties: Dict[str, Set[str]] = {}
        logger.info("Initialized SchemaAgent")

    def infer_schema_from_documents(
        self,
        documents: List[str],
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Infer graph schema from a collection of documents.

        Args:
            documents: List of document texts
            sample_size: Optional number of documents to sample (None = use all)

        Returns:
            Dictionary containing:
                - entity_types: List of discovered entity types
                - relation_types: List of discovered relationship types
                - entity_schemas: Property schemas for each entity type
                - relation_schemas: Property schemas for each relationship type
        """
        if not documents:
            logger.warning("No documents provided for schema inference")
            return self._empty_schema()

        # Sample documents if needed
        if sample_size and sample_size < len(documents):
            import random
            documents = random.sample(documents, sample_size)
            logger.info(f"Sampling {sample_size} documents for schema inference")

        logger.info(f"Inferring schema from {len(documents)} documents...")

        # Analyze each document
        all_entities = []
        all_relations = []

        for i, doc in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")

            analysis = self._analyze_document(doc)
            all_entities.extend(analysis.get("entities", []))
            all_relations.extend(analysis.get("relations", []))

        # Aggregate and refine schema
        schema = self._aggregate_schema(all_entities, all_relations)

        logger.info(f"Discovered {len(schema['entity_types'])} entity types "
                   f"and {len(schema['relation_types'])} relation types")

        return schema

    def _analyze_document(self, document: str) -> Dict[str, Any]:
        """
        Analyze a single document to extract potential entity and relation types.

        Args:
            document: Document text

        Returns:
            Dictionary with entities and relations discovered
        """
        # Truncate very long documents
        max_length = 3000
        if len(document) > max_length:
            document = document[:max_length] + "..."

        prompt = f"""Analyze the following document and identify the types of entities and relationships present.

Document:
{document}

Please extract:
1. Entity types (e.g., Person, Organization, Location, Product, Disease, Drug, etc.)
2. Relationship types (e.g., WORKS_AT, LOCATED_IN, TREATS, MANUFACTURES, etc.)
3. Key properties for each entity type

Provide your analysis in JSON format:
{{
  "entities": [
    {{"type": "Person", "properties": ["name", "age", "occupation"]}},
    {{"type": "Organization", "properties": ["name", "industry", "founded"]}}
  ],
  "relations": [
    {{"type": "WORKS_AT", "properties": ["since", "position"]}},
    {{"type": "FOUNDED_BY", "properties": ["year"]}}
  ]
}}

Focus on extracting domain-specific entity and relationship types that are actually present in the document."""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.1)
            return response
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"entities": [], "relations": []}

    def _aggregate_schema(
        self,
        all_entities: List[Dict[str, Any]],
        all_relations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate entity and relation types across all documents.

        Args:
            all_entities: List of entity dicts from all documents
            all_relations: List of relation dicts from all documents

        Returns:
            Refined schema dictionary
        """
        # Count entity types
        entity_counter = Counter()
        entity_props: Dict[str, Set[str]] = {}

        for entity in all_entities:
            entity_type = entity.get("type", "").strip()
            if entity_type:
                entity_counter[entity_type] += 1
                if entity_type not in entity_props:
                    entity_props[entity_type] = set()
                entity_props[entity_type].update(entity.get("properties", []))

        # Count relation types
        relation_counter = Counter()
        relation_props: Dict[str, Set[str]] = {}

        for relation in all_relations:
            relation_type = relation.get("type", "").strip()
            if relation_type:
                relation_counter[relation_type] += 1
                if relation_type not in relation_props:
                    relation_props[relation_type] = set()
                relation_props[relation_type].update(relation.get("properties", []))

        # Keep only types that appear at least twice (reduce noise)
        min_frequency = 2
        entity_types = [
            entity_type for entity_type, count in entity_counter.items()
            if count >= min_frequency
        ]
        relation_types = [
            relation_type for relation_type, count in relation_counter.items()
            if count >= min_frequency
        ]

        # Build final schema
        entity_schemas = {
            entity_type: {
                "properties": list(entity_props.get(entity_type, [])),
                "frequency": entity_counter[entity_type]
            }
            for entity_type in entity_types
        }

        relation_schemas = {
            relation_type: {
                "properties": list(relation_props.get(relation_type, [])),
                "frequency": relation_counter[relation_type]
            }
            for relation_type in relation_types
        }

        # Update agent state
        self.discovered_entities.update(entity_types)
        self.discovered_relations.update(relation_types)
        self.entity_properties.update({
            k: set(v["properties"]) for k, v in entity_schemas.items()
        })
        self.relation_properties.update({
            k: set(v["properties"]) for k, v in relation_schemas.items()
        })

        return {
            "entity_types": entity_types,
            "relation_types": relation_types,
            "entity_schemas": entity_schemas,
            "relation_schemas": relation_schemas
        }

    def _empty_schema(self) -> Dict[str, Any]:
        """Return an empty schema structure."""
        return {
            "entity_types": [],
            "relation_types": [],
            "entity_schemas": {},
            "relation_schemas": {}
        }

    def refine_schema(
        self,
        existing_schema: Dict[str, Any],
        new_documents: List[str]
    ) -> Dict[str, Any]:
        """
        Refine existing schema with new documents (adaptive learning).

        Args:
            existing_schema: Previously discovered schema
            new_documents: New documents to analyze

        Returns:
            Updated schema dictionary
        """
        logger.info(f"Refining schema with {len(new_documents)} new documents")

        # Restore previous state
        self.discovered_entities.update(existing_schema.get("entity_types", []))
        self.discovered_relations.update(existing_schema.get("relation_types", []))

        # Infer from new documents
        new_schema = self.infer_schema_from_documents(new_documents)

        # Merge schemas
        merged_entity_types = list(set(
            existing_schema.get("entity_types", []) +
            new_schema.get("entity_types", [])
        ))

        merged_relation_types = list(set(
            existing_schema.get("relation_types", []) +
            new_schema.get("relation_types", [])
        ))

        # Merge property schemas
        merged_entity_schemas = existing_schema.get("entity_schemas", {}).copy()
        for entity_type, schema in new_schema.get("entity_schemas", {}).items():
            if entity_type in merged_entity_schemas:
                # Merge properties
                existing_props = set(merged_entity_schemas[entity_type]["properties"])
                new_props = set(schema["properties"])
                merged_entity_schemas[entity_type]["properties"] = list(existing_props | new_props)
            else:
                merged_entity_schemas[entity_type] = schema

        merged_relation_schemas = existing_schema.get("relation_schemas", {}).copy()
        for relation_type, schema in new_schema.get("relation_schemas", {}).items():
            if relation_type in merged_relation_schemas:
                # Merge properties
                existing_props = set(merged_relation_schemas[relation_type]["properties"])
                new_props = set(schema["properties"])
                merged_relation_schemas[relation_type]["properties"] = list(existing_props | new_props)
            else:
                merged_relation_schemas[relation_type] = schema

        logger.info(f"Refined schema: {len(merged_entity_types)} entity types, "
                   f"{len(merged_relation_types)} relation types")

        return {
            "entity_types": merged_entity_types,
            "relation_types": merged_relation_types,
            "entity_schemas": merged_entity_schemas,
            "relation_schemas": merged_relation_schemas
        }

    def get_schema_summary(self) -> str:
        """
        Get a human-readable summary of the discovered schema.

        Returns:
            Formatted schema summary string
        """
        summary = ["=" * 60]
        summary.append("DISCOVERED KNOWLEDGE GRAPH SCHEMA")
        summary.append("=" * 60)

        summary.append(f"\nEntity Types ({len(self.discovered_entities)}):")
        for entity_type in sorted(self.discovered_entities):
            props = self.entity_properties.get(entity_type, set())
            summary.append(f"  - {entity_type}")
            if props:
                summary.append(f"    Properties: {', '.join(sorted(props))}")

        summary.append(f"\nRelationship Types ({len(self.discovered_relations)}):")
        for relation_type in sorted(self.discovered_relations):
            props = self.relation_properties.get(relation_type, set())
            summary.append(f"  - {relation_type}")
            if props:
                summary.append(f"    Properties: {', '.join(sorted(props))}")

        summary.append("=" * 60)
        return "\n".join(summary)


# Singleton instance
_schema_agent: Optional[SchemaAgent] = None


def get_schema_agent() -> SchemaAgent:
    """
    Get the global SchemaAgent instance (singleton pattern).

    Returns:
        SchemaAgent: Global schema agent
    """
    global _schema_agent
    if _schema_agent is None:
        _schema_agent = SchemaAgent()
    return _schema_agent


if __name__ == "__main__":
    """Test the SchemaAgent."""
    import sys

    try:
        print("🔄 Initializing SchemaAgent...")
        agent = get_schema_agent()

        # Test documents
        test_docs = [
            """Dr. Jane Smith works at MIT in the Computer Science department.
            She researches machine learning and natural language processing.
            MIT is located in Cambridge, Massachusetts.""",

            """Tesla manufactures electric vehicles. Elon Musk is the CEO of Tesla.
            The company was founded in 2003 and is headquartered in Austin, Texas.""",

            """Aspirin is a medication used to treat pain and reduce fever.
            It is manufactured by Bayer. Aspirin treats headaches and inflammation."""
        ]

        print("\n📚 Inferring schema from sample documents...")
        schema = agent.infer_schema_from_documents(test_docs)

        print("\n" + agent.get_schema_summary())

        print("\n✅ SchemaAgent working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
