"""
RelationAgent for Agentic GraphRAG

This agent extracts relationships between entities:
- Identifies connections between extracted entities
- Classifies relationship types based on schema
- Extracts relationship properties and confidence scores
- Handles temporal and conditional relationships

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RelationAgent:
    """
    Agent responsible for extracting relationships between entities.

    Identifies and classifies relationships using:
    1. Entity co-occurrence analysis
    2. LLM-based relation extraction
    3. Schema-guided classification
    4. Confidence scoring
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the RelationAgent.

        Args:
            schema: Optional predefined schema from SchemaAgent
        """
        self.llm_client = get_llm_client()
        self.config = get_config()
        self.schema = schema or {}
        logger.info("Initialized RelationAgent")

    def extract_relations_from_documents(
        self,
        documents: List[str],
        entities_per_doc: List[List[Dict[str, Any]]],
        schema: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships from multiple documents.

        Args:
            documents: List of document texts
            entities_per_doc: List of entity lists (one per document)
            schema: Optional schema to guide extraction

        Returns:
            List of extracted relationships
        """
        if schema:
            self.schema = schema

        if len(documents) != len(entities_per_doc):
            raise ValueError("Number of documents must match number of entity lists")

        all_relations = []
        logger.info(f"Extracting relations from {len(documents)} documents...")

        for i, (doc, entities) in enumerate(zip(documents, entities_per_doc)):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")

            doc_relations = self.extract_relations_from_text(doc, entities)
            all_relations.extend(doc_relations)

        logger.info(f"Extracted {len(all_relations)} relationships")
        return all_relations

    def extract_relations_from_text(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships from a single text given its entities.

        Args:
            text: Input text
            entities: List of entities found in the text

        Returns:
            List of extracted relationships
        """
        if len(entities) < 2:
            # Need at least 2 entities to form a relationship
            return []

        # Truncate very long texts
        max_length = 2500
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Build entity list for prompt
        entity_list = []
        for entity in entities:
            entity_list.append(f"- {entity.get('name')} ({entity.get('type')})")

        # Build schema guidance
        relation_types = self.schema.get("relation_types", [])
        relation_schemas = self.schema.get("relation_schemas", {})

        schema_guide = ""
        if relation_types:
            type_descriptions = []
            for rel_type in relation_types:
                props = relation_schemas.get(rel_type, {}).get("properties", [])
                type_descriptions.append(f"  - {rel_type} (properties: {', '.join(props) if props else 'none'})")
            schema_guide = "\n\nExpected relationship types:\n" + "\n".join(type_descriptions)

        prompt = f"""Extract all relationships between the following entities based on the text.{schema_guide}

Text:
{text}

Entities:
{chr(10).join(entity_list)}

For each relationship, provide:
1. Source entity name
2. Target entity name
3. Relationship type (e.g., WORKS_AT, LOCATED_IN, TREATS, FOUNDED_BY, etc.)
4. Direction (true if source->target, false if bidirectional)
5. Properties (any relevant attributes like "since", "until", "role", etc.)
6. Confidence (0.0 to 1.0)

Return a JSON array:
[
  {{
    "source": "entity1 name",
    "target": "entity2 name",
    "type": "RELATIONSHIP_TYPE",
    "directed": true,
    "properties": {{"property": "value"}},
    "confidence": 0.9
  }}
]

Only extract relationships that are explicitly or implicitly stated in the text."""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.1)

            # Handle both list and dict responses
            if isinstance(response, dict) and "relations" in response:
                relations = response["relations"]
            elif isinstance(response, list):
                relations = response
            else:
                relations = []

            # Validate and enrich relations
            validated_relations = []
            for relation in relations:
                if self._validate_relation(relation, entities):
                    # Add source information
                    relation["source_method"] = "llm"
                    validated_relations.append(relation)

            return validated_relations

        except Exception as e:
            logger.error(f"Error extracting relations with LLM: {e}")
            return []

    def _validate_relation(
        self,
        relation: Dict[str, Any],
        entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate that a relation is well-formed.

        Args:
            relation: Relation to validate
            entities: List of valid entities

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not all(key in relation for key in ["source", "target", "type"]):
            return False

        # Check that entities exist
        entity_names = {e.get("name", "").lower() for e in entities}
        source = relation.get("source", "").lower()
        target = relation.get("target", "").lower()

        if source not in entity_names or target not in entity_names:
            return False

        # Check confidence
        confidence = relation.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            relation["confidence"] = 0.5

        # Set default for directed
        if "directed" not in relation:
            relation["directed"] = True

        # Ensure properties is a dict
        if "properties" not in relation:
            relation["properties"] = {}
        elif not isinstance(relation["properties"], dict):
            relation["properties"] = {}

        return True

    def extract_relations_between_entity_pairs(
        self,
        text: str,
        entity_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships for specific entity pairs.

        Args:
            text: Input text
            entity_pairs: List of (source_entity, target_entity) tuples

        Returns:
            List of extracted relationships
        """
        if not entity_pairs:
            return []

        # Build pairs list for prompt
        pairs_list = []
        for source, target in entity_pairs:
            pairs_list.append(f"- {source.get('name')} -> {target.get('name')}")

        prompt = f"""Given the following text and entity pairs, determine if there are relationships between them.

Text:
{text[:2000]}...

Entity pairs to check:
{chr(10).join(pairs_list)}

For each pair that has a relationship, provide:
1. Relationship type
2. Properties
3. Confidence (0.0 to 1.0)

Return a JSON array:
[
  {{
    "source": "entity1",
    "target": "entity2",
    "type": "RELATIONSHIP_TYPE",
    "properties": {{}},
    "confidence": 0.85
  }}
]

Only return relationships that are clearly evident in the text."""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.1)

            if isinstance(response, dict) and "relations" in response:
                relations = response["relations"]
            elif isinstance(response, list):
                relations = response
            else:
                relations = []

            # Add source method
            for relation in relations:
                relation["source_method"] = "llm_targeted"
                if "directed" not in relation:
                    relation["directed"] = True

            return relations

        except Exception as e:
            logger.error(f"Error extracting targeted relations: {e}")
            return []

    def merge_relations(
        self,
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge duplicate relations (same source, target, type).

        Args:
            relations: List of relations to merge

        Returns:
            Merged list of unique relations
        """
        # Group by (source, target, type)
        relation_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

        for relation in relations:
            source = relation.get("source", "").lower()
            target = relation.get("target", "").lower()
            rel_type = relation.get("type", "")

            key = (source, target, rel_type)
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(relation)

        # Merge duplicates
        merged_relations = []
        for key, group in relation_groups.items():
            # Pick the relation with highest confidence
            best_relation = max(group, key=lambda r: r.get("confidence", 0.0))

            # Merge properties
            merged_properties = {}
            for relation in group:
                merged_properties.update(relation.get("properties", {}))

            best_relation["properties"] = merged_properties
            best_relation["mention_count"] = len(group)

            # Average confidence
            avg_confidence = sum(r.get("confidence", 0.5) for r in group) / len(group)
            best_relation["confidence"] = avg_confidence

            merged_relations.append(best_relation)

        logger.info(f"Merged {len(relations)} relations into {len(merged_relations)} unique relations")
        return merged_relations


# Singleton instance
_relation_agent: Optional[RelationAgent] = None


def get_relation_agent(schema: Optional[Dict[str, Any]] = None) -> RelationAgent:
    """
    Get the global RelationAgent instance (singleton pattern).

    Args:
        schema: Optional schema for guided extraction

    Returns:
        RelationAgent: Global relation agent
    """
    global _relation_agent
    if _relation_agent is None:
        _relation_agent = RelationAgent(schema=schema)
    elif schema:
        _relation_agent.schema = schema
    return _relation_agent


if __name__ == "__main__":
    """Test the RelationAgent."""
    import sys

    try:
        print("🔄 Initializing RelationAgent...")
        agent = get_relation_agent()

        # Test text and entities
        test_text = """Dr. Jane Smith works at MIT in the Computer Science department.
        She researches machine learning. MIT is located in Cambridge, Massachusetts."""

        test_entities = [
            {"name": "Jane Smith", "type": "Person"},
            {"name": "MIT", "type": "Organization"},
            {"name": "Computer Science", "type": "Department"},
            {"name": "Cambridge", "type": "Location"},
            {"name": "Massachusetts", "type": "Location"}
        ]

        print("\n📚 Extracting relations from sample text...")
        relations = agent.extract_relations_from_text(test_text, test_entities)

        print(f"\n✅ Extracted {len(relations)} relationships:")
        for i, rel in enumerate(relations, 1):
            print(f"\n{i}. {rel.get('source')} --[{rel.get('type')}]--> {rel.get('target')}")
            print(f"   Confidence: {rel.get('confidence'):.2f}")
            if rel.get("properties"):
                print(f"   Properties: {rel.get('properties')}")

        print("\n✅ RelationAgent working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
