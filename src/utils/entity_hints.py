"""
Entity Type Hints and Validation

This module provides entity type hints and validation to improve
entity classification accuracy. It addresses common misclassification
issues (e.g., "Champ de Mars" as Organization instead of Location).

Features:
- Entity type hints from configuration
- Type validation and correction
- Integration with LLM-based classification

Author: Agentic GraphRAG Team
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityHintsManager:
    """Manager for entity type hints and validation."""

    def __init__(self, hints_path: Optional[Path] = None, llm_client=None):
        """
        Initialize the EntityHintsManager.

        Args:
            hints_path: Path to entity hints JSON file
            llm_client: Optional LLM client for validation
        """
        self.hints_path = hints_path or Path("config/entity_hints.json")
        self.entity_type_hints: Dict[str, str] = {}
        self.type_definitions: Dict[str, str] = {}
        self.validation_rules: Dict[str, Any] = {}
        self.llm_client = llm_client  # Lazy loaded if needed

        self._load_hints()

    def _load_hints(self):
        """Load entity hints from configuration file."""
        if not self.hints_path.exists():
            logger.warning(f"Entity hints file not found: {self.hints_path}")
            logger.info("Using default configuration (no hints)")
            return

        try:
            with open(self.hints_path, 'r') as f:
                config = json.load(f)

            self.entity_type_hints = config.get("entity_type_hints", {})
            self.type_definitions = config.get("entity_type_definitions", {})
            self.validation_rules = config.get("validation_rules", {})

            logger.info(f"Loaded {len(self.entity_type_hints)} entity type hints")

        except Exception as e:
            logger.error(f"Error loading entity hints: {e}")
            self.entity_type_hints = {}
            self.type_definitions = {}
            self.validation_rules = {}

    def get_type_hint(self, entity_name: str) -> Optional[str]:
        """
        Get type hint for an entity name.

        Args:
            entity_name: Entity name to look up

        Returns:
            Entity type hint or None if not found
        """
        # Exact match first
        if entity_name in self.entity_type_hints:
            return self.entity_type_hints[entity_name]

        # Case-insensitive match
        entity_lower = entity_name.lower()
        for hint_name, hint_type in self.entity_type_hints.items():
            if hint_name.lower() == entity_lower:
                return hint_type

        return None

    def apply_hints_to_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply type hints to extracted entities.

        Args:
            entities: List of entities with 'name' and 'type' fields

        Returns:
            Entities with corrected types where hints exist
        """
        corrected = 0

        for entity in entities:
            name = entity.get("name", "")
            original_type = entity.get("type", "")

            hint_type = self.get_type_hint(name)
            if hint_type and hint_type != original_type:
                logger.info(
                    f"Correcting entity type: '{name}' "
                    f"{original_type} → {hint_type}"
                )
                entity["type"] = hint_type
                entity["type_corrected"] = True
                entity["original_type"] = original_type
                corrected += 1

        if corrected > 0:
            logger.info(f"Applied hints to {corrected} entities")

        return entities

    def get_type_definitions_prompt(self) -> str:
        """
        Get entity type definitions formatted for LLM prompt.

        Returns:
            Formatted type definitions string
        """
        if not self.type_definitions:
            return ""

        lines = ["Entity Type Definitions:"]
        for entity_type, definition in self.type_definitions.items():
            lines.append(f"  - {entity_type}: {definition}")

        return "\n".join(lines)

    def should_validate_type(self, entity_type: str) -> bool:
        """
        Check if an entity type should be validated.

        Args:
            entity_type: Entity type to check

        Returns:
            True if type should be validated
        """
        must_validate = self.validation_rules.get("must_validate", [])
        return entity_type in must_validate

    def validate_entity_types(
        self,
        entities: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Validate entity types using LLM and correct obvious errors.

        Args:
            entities: List of entities to validate
            batch_size: Number of entities to validate at once

        Returns:
            Entities with validated and corrected types
        """
        if not entities or not self.llm_client:
            return entities

        # Only validate types that need validation
        must_validate = self.validation_rules.get("must_validate", [])
        if not must_validate:
            return entities

        # Filter entities that need validation
        entities_to_validate = [
            e for e in entities
            if e.get("type") in must_validate
        ]

        if not entities_to_validate:
            return entities

        logger.info(f"Validating {len(entities_to_validate)} entities...")

        # Process in batches
        validated = []
        for i in range(0, len(entities_to_validate), batch_size):
            batch = entities_to_validate[i:i + batch_size]

            # Build validation prompt
            entity_list = []
            for j, ent in enumerate(batch):
                entity_list.append(
                    f"{j+1}. \"{ent.get('name')}\" classified as {ent.get('type')}"
                )

            entity_list_str = "\n".join(entity_list)

            type_defs = self.get_type_definitions_prompt()

            prompt = f"""Review these entity classifications and correct any obvious errors.

{type_defs}

Entities to validate:
{entity_list_str}

For each entity, determine if the classification is correct. If not, provide the correct type.
Return JSON array with validation results:
[
  {{"index": 1, "correct": true}},
  {{"index": 2, "correct": false, "corrected_type": "Location", "reason": "It's a geographic place"}}
]"""

            try:
                response = self.llm_client.generate_json(prompt, temperature=0.0)

                # Handle both list and dict responses
                if isinstance(response, dict) and "validations" in response:
                    validations = response["validations"]
                elif isinstance(response, list):
                    validations = response
                else:
                    logger.warning("Unexpected validation response format")
                    validated.extend(batch)
                    continue

                # Apply corrections
                for validation in validations:
                    idx = validation.get("index", 0) - 1
                    if 0 <= idx < len(batch):
                        entity = batch[idx]

                        if not validation.get("correct", True):
                            corrected_type = validation.get("corrected_type")
                            reason = validation.get("reason", "LLM validation")

                            if corrected_type:
                                logger.info(
                                    f"Validation correction: '{entity['name']}' "
                                    f"{entity['type']} → {corrected_type} ({reason})"
                                )
                                entity["original_type"] = entity["type"]
                                entity["type"] = corrected_type
                                entity["validation_corrected"] = True
                                entity["validation_reason"] = reason

                validated.extend(batch)

            except Exception as e:
                logger.error(f"Error validating entity types: {e}")
                validated.extend(batch)

        # Merge validated entities back
        validated_map = {id(e): e for e in validated}
        result = []
        for entity in entities:
            if id(entity) in validated_map:
                result.append(validated_map[id(entity)])
            else:
                result.append(entity)

        return result

    def get_common_errors(self) -> Dict[str, str]:
        """
        Get common entity classification errors.

        Returns:
            Dictionary of error patterns
        """
        return self.validation_rules.get("common_errors", {})


# Singleton instance
_hints_manager: Optional[EntityHintsManager] = None


def get_entity_hints_manager(
    hints_path: Optional[Path] = None,
    llm_client=None
) -> EntityHintsManager:
    """
    Get the global EntityHintsManager instance (singleton).

    Args:
        hints_path: Optional path to hints file
        llm_client: Optional LLM client for validation

    Returns:
        EntityHintsManager: Global hints manager
    """
    global _hints_manager

    if _hints_manager is None:
        _hints_manager = EntityHintsManager(
            hints_path=hints_path,
            llm_client=llm_client
        )
    elif llm_client is not None and _hints_manager.llm_client is None:
        # Set LLM client if not already set
        _hints_manager.llm_client = llm_client

    return _hints_manager


if __name__ == "__main__":
    """Test the entity hints manager."""
    import sys

    try:
        print("🔄 Initializing EntityHintsManager...")
        manager = get_entity_hints_manager()

        # Test entity type hints
        test_entities = [
            {"name": "Champ de Mars", "type": "Organization"},
            {"name": "Eiffel Tower", "type": "Location"},
            {"name": "Paris", "type": "Location"},
            {"name": "Unknown Entity", "type": "Thing"}
        ]

        print("\n📋 Original entities:")
        for entity in test_entities:
            print(f"  - {entity['name']}: {entity['type']}")

        # Apply hints
        corrected = manager.apply_hints_to_entities(test_entities)

        print("\n✅ After applying hints:")
        for entity in corrected:
            if entity.get("type_corrected"):
                print(f"  - {entity['name']}: {entity['original_type']} → {entity['type']}")
            else:
                print(f"  - {entity['name']}: {entity['type']} (unchanged)")

        # Test type definitions
        print(f"\n📖 Type Definitions:")
        print(manager.get_type_definitions_prompt())

        print("\n✅ EntityHintsManager working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
