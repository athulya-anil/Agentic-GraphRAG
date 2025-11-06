"""
EntityAgent for Agentic GraphRAG

This agent extracts entities from documents using a hybrid approach:
- Fast entity recognition with spaCy NER
- LLM-powered classification for ambiguous cases
- Entity resolution and deduplication
- Property extraction for each entity
- Metadata enrichment (summaries, keywords, named entities)

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, using LLM-only mode")

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityAgent:
    """
    Agent responsible for extracting entities from documents.

    Uses a hybrid approach combining:
    1. spaCy NER for fast baseline extraction
    2. LLM for context-aware classification and property extraction
    3. Entity resolution to handle duplicates and variations
    4. Metadata enrichment (summaries, keywords, entities)
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the EntityAgent.

        Args:
            schema: Optional predefined schema from SchemaAgent
        """
        self.llm_client = get_llm_client()
        self.config = get_config()
        self.schema = schema or {}

        # Load spaCy model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")

        # Entity cache for deduplication
        self.entity_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("Initialized EntityAgent")

    def extract_entities_from_documents(
        self,
        documents: List[str],
        schema: Optional[Dict[str, Any]] = None,
        enrich_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple documents.

        Args:
            documents: List of document texts
            schema: Optional schema to guide extraction
            enrich_metadata: Whether to enrich with metadata (summaries, keywords)

        Returns:
            List of extracted entities with properties and metadata
        """
        if schema:
            self.schema = schema

        all_entities = []
        logger.info(f"Extracting entities from {len(documents)} documents...")

        for i, doc in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")

            doc_entities = self.extract_entities_from_text(doc, enrich_metadata=enrich_metadata)
            all_entities.extend(doc_entities)

        # Deduplicate entities
        unique_entities = self._deduplicate_entities(all_entities)

        logger.info(f"Extracted {len(unique_entities)} unique entities "
                   f"from {len(all_entities)} total mentions")

        return unique_entities

    def extract_entities_from_text(
        self,
        text: str,
        enrich_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from a single text.

        Args:
            text: Input text
            enrich_metadata: Whether to enrich with metadata

        Returns:
            List of extracted entities
        """
        entities = []

        # Method 1: spaCy NER (if available)
        if self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)

        # Method 2: LLM-based extraction (more accurate, context-aware)
        llm_entities = self._extract_with_llm(text)
        entities.extend(llm_entities)

        # Enrich with metadata if requested (inspired by MarcoRAG)
        if enrich_metadata:
            entities = self._enrich_entities_with_metadata(entities, text)

        return entities

    def _extract_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy NER.

        Args:
            text: Input text

        Returns:
            List of entities
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity = {
                "name": ent.text,
                "type": self._map_spacy_label(ent.label_),
                "properties": {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                },
                "source": "spacy",
                "confidence": 0.7  # Base confidence for spaCy
            }
            entities.append(entity)

        return entities

    def _map_spacy_label(self, label: str) -> str:
        """
        Map spaCy labels to our schema entity types.

        Args:
            label: spaCy entity label

        Returns:
            Mapped entity type
        """
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "QUANTITY": "Quantity",
            "NORP": "Group",
            "FAC": "Facility",
            "WORK_OF_ART": "Work"
        }
        return mapping.get(label, label)

    def _extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM (context-aware, schema-guided).

        Args:
            text: Input text

        Returns:
            List of entities
        """
        # Truncate very long texts
        max_length = 2500
        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Build prompt with schema if available
        entity_types = self.schema.get("entity_types", [])
        entity_schemas = self.schema.get("entity_schemas", {})

        type_descriptions = []
        for entity_type in entity_types:
            props = entity_schemas.get(entity_type, {}).get("properties", [])
            type_descriptions.append(f"  - {entity_type} (properties: {', '.join(props)})")

        schema_guide = ""
        if type_descriptions:
            schema_guide = "\n\nExpected entity types:\n" + "\n".join(type_descriptions)

        prompt = f"""Extract all entities from the following text.{schema_guide}

Text:
{text}

For each entity, provide:
1. The entity name/text
2. The entity type
3. Relevant properties (e.g., age, occupation, location, etc.)

Return a JSON array of entities:
[
  {{
    "name": "entity name",
    "type": "EntityType",
    "properties": {{"property1": "value1", "property2": "value2"}}
  }}
]

Only extract entities that are actually present in the text."""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.1)

            # Handle both list and dict responses
            if isinstance(response, dict) and "entities" in response:
                entities_list = response["entities"]
            elif isinstance(response, list):
                entities_list = response
            else:
                entities_list = []

            # Add source and confidence
            for entity in entities_list:
                entity["source"] = "llm"
                entity["confidence"] = 0.9  # Higher confidence for LLM

            return entities_list

        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {e}")
            return []

    def _enrich_entities_with_metadata(
        self,
        entities: List[Dict[str, Any]],
        context_text: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich entities with metadata (inspired by MarcoRAG).

        Adds:
        - Summary: Brief description of the entity in context
        - Keywords: Relevant keywords associated with the entity
        - Context: Surrounding text for grounding

        Args:
            entities: List of entities to enrich
            context_text: The full text context

        Returns:
            Enriched entities
        """
        if not entities:
            return entities

        # Batch process entities for efficiency
        entity_names = [e.get("name", "") for e in entities]

        prompt = f"""Given the following text and extracted entities, provide enrichment metadata for each entity.

Text:
{context_text[:1500]}...

Entities: {', '.join(entity_names)}

For each entity, provide:
1. A brief summary (1 sentence)
2. 3-5 relevant keywords
3. A short context snippet (where it appears in text)

Return JSON:
[
  {{
    "entity": "entity name",
    "summary": "brief description",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "context": "...surrounding text..."
  }}
]"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.1)

            # Create a lookup dict
            metadata_lookup = {}
            if isinstance(response, list):
                for item in response:
                    entity_name = item.get("entity", "")
                    if entity_name:
                        metadata_lookup[entity_name.lower()] = item

            # Enrich each entity
            for entity in entities:
                entity_name = entity.get("name", "")
                metadata = metadata_lookup.get(entity_name.lower(), {})

                if metadata:
                    entity["metadata"] = {
                        "summary": metadata.get("summary", ""),
                        "keywords": metadata.get("keywords", []),
                        "context": metadata.get("context", "")
                    }

        except Exception as e:
            logger.error(f"Error enriching metadata: {e}")

        return entities

    def _deduplicate_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate entities by name and type.

        Args:
            entities: List of entities

        Returns:
            Deduplicated list of entities
        """
        # Group by (name, type) key
        entity_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        for entity in entities:
            name = entity.get("name", "").strip().lower()
            entity_type = entity.get("type", "").strip()
            if name and entity_type:
                key = (name, entity_type)
                entity_groups[key].append(entity)

        # Merge duplicates
        unique_entities = []
        for (name, entity_type), group in entity_groups.items():
            # Pick the entity with highest confidence
            best_entity = max(group, key=lambda e: e.get("confidence", 0.0))

            # Merge properties from all instances
            merged_properties = {}
            for entity in group:
                merged_properties.update(entity.get("properties", {}))

            # Merge metadata if present
            merged_metadata = {}
            for entity in group:
                if "metadata" in entity:
                    for key, value in entity["metadata"].items():
                        if key == "keywords":
                            # Merge keyword lists
                            existing = merged_metadata.get("keywords", [])
                            merged_metadata["keywords"] = list(set(existing + value))
                        else:
                            # Use longest value for other fields
                            existing = merged_metadata.get(key, "")
                            if len(str(value)) > len(str(existing)):
                                merged_metadata[key] = value

            best_entity["properties"] = merged_properties
            if merged_metadata:
                best_entity["metadata"] = merged_metadata
            best_entity["mention_count"] = len(group)

            unique_entities.append(best_entity)

        return unique_entities

    def resolve_entity(self, entity_name: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an entity from the cache.

        Args:
            entity_name: Entity name
            entity_type: Entity type

        Returns:
            Resolved entity or None
        """
        key = (entity_name.lower(), entity_type)
        return self.entity_cache.get(key)


# Singleton instance
_entity_agent: Optional[EntityAgent] = None


def get_entity_agent(schema: Optional[Dict[str, Any]] = None) -> EntityAgent:
    """
    Get the global EntityAgent instance (singleton pattern).

    Args:
        schema: Optional schema for guided extraction

    Returns:
        EntityAgent: Global entity agent
    """
    global _entity_agent
    if _entity_agent is None:
        _entity_agent = EntityAgent(schema=schema)
    elif schema:
        _entity_agent.schema = schema
    return _entity_agent


if __name__ == "__main__":
    """Test the EntityAgent."""
    import sys

    try:
        print("🔄 Initializing EntityAgent...")
        agent = get_entity_agent()

        # Test text
        test_text = """Dr. Jane Smith works at MIT in the Computer Science department.
        She researches machine learning and natural language processing.
        MIT is located in Cambridge, Massachusetts. Jane Smith published a paper
        on neural networks in 2023."""

        print("\n📚 Extracting entities from sample text...")
        entities = agent.extract_entities_from_text(test_text, enrich_metadata=True)

        print(f"\n✅ Extracted {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            print(f"\n{i}. {entity.get('name')} ({entity.get('type')})")
            print(f"   Source: {entity.get('source')}, Confidence: {entity.get('confidence')}")
            if "metadata" in entity:
                print(f"   Summary: {entity['metadata'].get('summary', 'N/A')}")
                print(f"   Keywords: {', '.join(entity['metadata'].get('keywords', []))}")

        print("\n✅ EntityAgent working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
