"""
KARMA-Inspired Enhanced Relation Agent

Key improvements over original:
1. Multi-label relationship extraction (multiple rels per entity pair)
2. Schema-guided extraction with predefined relationship types
3. Confidence scoring based on context
4. Entity-pair enumeration (catch all possible relationships)

Based on KARMA paper: "REA uses multi-label classification allowing
multiple relationships between same entity pairs"
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class KARMARelationAgent:
    """
    KARMA-inspired Relationship Extraction Agent.

    Implements multi-label relationship extraction with schema guidance.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None, schema_path: Optional[str] = None):
        """
        Initialize KARMA Relation Agent.

        Args:
            schema: Optional predefined schema
            schema_path: Path to relationship schema JSON file
        """
        self.llm_client = get_llm_client()
        self.config = get_config()
        self.schema = schema or {}

        # Load relationship schema
        if schema_path:
            self.relationship_schema = self._load_relationship_schema(schema_path)
        else:
            # Default path
            default_path = Path("data/hotpotqa/relationship_schema.json")
            if default_path.exists():
                self.relationship_schema = self._load_relationship_schema(str(default_path))
            else:
                self.relationship_schema = None

        logger.info("Initialized KARMARelationAgent (multi-label extraction enabled)")

    def _load_relationship_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load relationship schema from JSON file."""
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            logger.info(f"Loaded relationship schema from {schema_path}")
            return schema
        except Exception as e:
            logger.warning(f"Failed to load schema from {schema_path}: {e}")
            return None

    def _get_allowed_relationship_types(self) -> List[str]:
        """Get flat list of all allowed relationship types from schema."""
        if not self.relationship_schema:
            return []

        all_types = []
        for category, types in self.relationship_schema.get("relationship_types", {}).items():
            all_types.extend(types)

        return all_types

    def _generate_entity_pairs(self, entities: List[Dict[str, Any]]) -> List[Tuple[Dict, Dict]]:
        """
        Generate all possible entity pairs for relationship extraction.

        KARMA approach: Enumerate all valid entity pairs before extraction.

        Args:
            entities: List of entities

        Returns:
            List of (entity1, entity2) tuples
        """
        pairs = []

        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                pairs.append((e1, e2))

        logger.info(f"Generated {len(pairs)} entity pairs from {len(entities)} entities")
        return pairs

    def extract_relations_from_documents(
        self,
        documents: List[str],
        entities_per_doc: List[List[Dict[str, Any]]],
        summaries: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships from multiple documents using KARMA approach.

        Args:
            documents: List of document texts
            entities_per_doc: List of entity lists (one per document)
            summaries: Optional pre-computed summaries (from SummarizerAgent)
            schema: Optional schema to guide extraction

        Returns:
            List of extracted relationships with confidence scores
        """
        if schema:
            self.schema = schema

        if len(documents) != len(entities_per_doc):
            raise ValueError("Number of documents must match number of entity lists")

        all_relations = []
        logger.info(f"Extracting relations from {len(documents)} documents (KARMA multi-label mode)...")

        for i, (doc, entities) in enumerate(zip(documents, entities_per_doc)):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")

            # Use summary if available (KARMA approach)
            context = summaries[i] if summaries and i < len(summaries) else doc

            doc_relations = self.extract_relations_multi_label(context, entities)
            all_relations.extend(doc_relations)

        logger.info(f"Extracted {len(all_relations)} relationships (multi-label)")
        return all_relations

    def extract_relations_multi_label(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract multiple relationships per entity pair (KARMA multi-label approach).

        Key difference from original: Explicitly asks for ALL relationships,
        not just one per pair.

        Args:
            text: Input text (original or summary)
            entities: List of entities found in the text

        Returns:
            List of extracted relationships (potentially multiple per pair)
        """
        if len(entities) < 2:
            return []

        # Truncate if needed
        max_length = 3000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Build entity list for prompt
        entity_list = []
        for entity in entities:
            entity_list.append(f"- {entity.get('name')} ({entity.get('type', 'Unknown')})")

        # Get allowed relationship types from schema
        allowed_types = self._get_allowed_relationship_types()

        # Build schema guidance
        schema_guide = ""
        if allowed_types:
            # Group by category for better prompt
            if self.relationship_schema:
                schema_guide = "\n\nAllowed relationship types (by category):\n"
                for category, types in self.relationship_schema.get("relationship_types", {}).items():
                    schema_guide += f"\n{category}:\n"
                    for rel_type in types:
                        schema_guide += f"  - {rel_type}\n"
            else:
                schema_guide = f"\n\nAllowed relationship types: {', '.join(allowed_types[:30])}"

        # KARMA-inspired prompt: Emphasize extracting ALL relationships
        prompt = f"""Extract ALL relationships between entities in the text.
Use MULTI-LABEL extraction: if two entities have multiple relationships, extract ALL of them.

CRITICAL: Extract EVERY relationship you can find, even if entities are already connected.

Examples:
- "Scott Derrickson directed and produced Sinister" → Extract BOTH "DIRECTED" AND "PRODUCED"
- "Ed Wood was born in New York and died in Los Angeles" → Extract "BORN_IN" AND "DIED_IN"

{schema_guide}

Text:
{text}

Entities:
{chr(10).join(entity_list)}

For EACH relationship found:
1. Source entity (exact name from list above)
2. Target entity (exact name from list above)
3. Relationship type (from allowed types only)
4. Confidence (0.0-1.0): How certain are you this relationship exists?
5. Evidence (brief quote from text supporting this relationship)

Return JSON array of ALL relationships:
[
  {{
    "source": "entity1 name",
    "target": "entity2 name",
    "type": "RELATIONSHIP_TYPE",
    "confidence": 0.95,
    "evidence": "quote from text"
  }}
]

Extract as many relationships as possible. Return empty array [] if none found."""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0, max_tokens=4000)

            # Handle both list and dict responses
            if isinstance(response, dict) and "relations" in response:
                relations = response["relations"]
            elif isinstance(response, list):
                relations = response
            else:
                relations = []

            # Validate relations
            validated_relations = []
            for relation in relations:
                if self._validate_relation(relation, entities, allowed_types):
                    # Add metadata
                    relation["source_method"] = "karma_multi_label"
                    validated_relations.append(relation)

            logger.info(f"Extracted {len(validated_relations)} valid relationships from {len(entities)} entities")
            return validated_relations

        except Exception as e:
            logger.error(f"Error in multi-label relation extraction: {e}")
            return []

    def _validate_relation(
        self,
        relation: Dict[str, Any],
        entities: List[Dict[str, Any]],
        allowed_types: Optional[List[str]] = None
    ) -> bool:
        """
        Validate relationship against entities and schema.

        Args:
            relation: Relation to validate
            entities: List of valid entities
            allowed_types: Optional list of allowed relationship types

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not all(k in relation for k in ["source", "target", "type"]):
            return False

        # Check entities exist
        entity_names = {e.get('name') for e in entities}
        if relation["source"] not in entity_names or relation["target"] not in entity_names:
            logger.debug(f"Invalid relation: entities not found in list")
            return False

        # Check relationship type is in schema (if schema provided)
        if allowed_types and relation["type"] not in allowed_types:
            logger.debug(f"Invalid relation type: {relation['type']} not in schema")
            return False

        # Check confidence score (if present)
        if "confidence" in relation:
            if not (0.0 <= relation["confidence"] <= 1.0):
                return False

        return True
