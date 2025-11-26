"""
ConflictResolutionAgent for Agentic GraphRAG

This agent resolves conflicts and deduplicates entities/relationships:
- Detects duplicate entities with different names
- Merges entity properties and metadata
- Resolves contradictory relationships
- Validates against schema constraints
- Assigns confidence scores to resolutions

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from difflib import SequenceMatcher

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConflictResolutionAgent:
    """
    Agent responsible for resolving conflicts in extracted knowledge.

    Handles:
    1. Entity deduplication (same entity, different names)
    2. Property conflict resolution (contradictory attributes)
    3. Relationship conflict resolution (contradictory edges)
    4. Schema validation and alignment
    5. Confidence scoring for merged entities
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConflictResolutionAgent.

        Args:
            schema: Optional schema to validate against
        """
        self.llm_client = get_llm_client()
        self.config = get_config()
        self.schema = schema or {}

        # Thresholds for conflict resolution
        self.entity_similarity_threshold = 0.85  # For string matching
        self.embedding_similarity_threshold = 0.90  # For semantic similarity

        logger.info("Initialized ConflictResolutionAgent")

    def deduplicate_entities(
        self,
        entities: List[Dict[str, Any]],
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate entities that refer to the same real-world object.

        Args:
            entities: List of extracted entities
            use_llm: Whether to use LLM for disambiguation

        Returns:
            List of deduplicated entities
        """
        if not entities:
            return []

        logger.info(f"Deduplicating {len(entities)} entities...")

        # Group entities by type for efficiency
        entities_by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get('type', 'Unknown')
            entities_by_type[entity_type].append(entity)

        deduplicated = []

        for entity_type, type_entities in entities_by_type.items():
            # Find duplicate clusters
            clusters = self._find_duplicate_clusters(type_entities, use_llm)

            # Merge each cluster
            for cluster in clusters:
                merged_entity = self._merge_entity_cluster(cluster)
                deduplicated.append(merged_entity)

        logger.info(f"Deduplicated to {len(deduplicated)} unique entities")
        return deduplicated

    def _find_duplicate_clusters(
        self,
        entities: List[Dict[str, Any]],
        use_llm: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Find clusters of duplicate entities.

        Uses multiple signals:
        1. String similarity (name matching)
        2. Property overlap
        3. LLM verification for ambiguous cases
        """
        clusters = []
        processed = set()

        for i, entity in enumerate(entities):
            if i in processed:
                continue

            # Start new cluster
            cluster = [entity]
            processed.add(i)

            # Find similar entities
            for j, other_entity in enumerate(entities[i+1:], start=i+1):
                if j in processed:
                    continue

                if self._are_duplicates(entity, other_entity, use_llm):
                    cluster.append(other_entity)
                    processed.add(j)

            clusters.append(cluster)

        return clusters

    def _are_duplicates(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        use_llm: bool = True
    ) -> bool:
        """
        Determine if two entities are duplicates.

        Uses multiple signals:
        1. Exact name match
        2. String similarity
        3. Property overlap
        4. LLM verification (if enabled)
        """
        name1 = entity1.get('name', '').lower().strip()
        name2 = entity2.get('name', '').lower().strip()

        # Exact match
        if name1 == name2:
            return True

        # String similarity
        similarity = SequenceMatcher(None, name1, name2).ratio()
        if similarity >= self.entity_similarity_threshold:
            return True

        # Check for obvious variations (e.g., "USA" vs "United States")
        if self._are_name_variations(name1, name2):
            return True

        # Property overlap check
        if self._have_high_property_overlap(entity1, entity2):
            if use_llm:
                # Use LLM for final verification
                return self._llm_verify_duplicates(entity1, entity2)
            else:
                return True

        return False

    def _are_name_variations(self, name1: str, name2: str) -> bool:
        """Check if names are common variations."""
        # Common abbreviations and variations
        variations = {
            ('usa', 'united states', 'united states of america'),
            ('uk', 'united kingdom', 'great britain'),
            ('ny', 'new york'),
            ('ca', 'california'),
            ('dr', 'doctor'),
            ('prof', 'professor'),
        }

        for variation_set in variations:
            if name1 in variation_set and name2 in variation_set:
                return True

        return False

    def _have_high_property_overlap(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> bool:
        """Check if entities have significant property overlap."""
        props1 = entity1.get('properties', {})
        props2 = entity2.get('properties', {})

        if not props1 or not props2:
            return False

        # Count matching properties
        matching = 0
        total = 0

        for key in set(props1.keys()) | set(props2.keys()):
            total += 1
            if key in props1 and key in props2:
                if props1[key] == props2[key]:
                    matching += 1

        # Require >70% overlap
        return (matching / total) > 0.7 if total > 0 else False

    def _llm_verify_duplicates(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> bool:
        """Use LLM to verify if entities are duplicates."""
        prompt = f"""Are these two entities referring to the same real-world object?

Entity 1:
- Name: {entity1.get('name')}
- Type: {entity1.get('type')}
- Properties: {json.dumps(entity1.get('properties', {}), indent=2)}

Entity 2:
- Name: {entity2.get('name')}
- Type: {entity2.get('type')}
- Properties: {json.dumps(entity2.get('properties', {}), indent=2)}

Answer with JSON:
{{
  "are_duplicates": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0)
            return response.get('are_duplicates', False)
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}, defaulting to conservative (False)")
            return False

    def _merge_entity_cluster(
        self,
        cluster: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge a cluster of duplicate entities into a single canonical entity.

        Strategy:
        1. Use most common name
        2. Merge all properties (keep highest confidence)
        3. Combine metadata
        4. Average confidence scores
        """
        if len(cluster) == 1:
            return cluster[0]

        # Choose canonical name (most common or longest)
        names = [e.get('name', '') for e in cluster]
        canonical_name = max(set(names), key=names.count) if names else cluster[0].get('name')

        # Merge properties
        merged_properties = {}
        for entity in cluster:
            props = entity.get('properties', {})
            for key, value in props.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                # Keep value from entity with higher confidence
                elif entity.get('confidence', 0) > cluster[0].get('confidence', 0):
                    merged_properties[key] = value

        # Average confidence
        confidences = [e.get('confidence', 0.5) for e in cluster]
        avg_confidence = sum(confidences) / len(confidences)

        # Merge metadata
        all_sources = set()
        for entity in cluster:
            source = entity.get('source')
            if source:
                all_sources.add(source)

        merged_entity = {
            'name': canonical_name,
            'type': cluster[0].get('type'),
            'properties': merged_properties,
            'confidence': avg_confidence,
            'source': list(all_sources),
            'merged_from': len(cluster),  # Track that this was merged
            'alternative_names': list(set(names) - {canonical_name})
        }

        logger.debug(f"Merged {len(cluster)} entities into: {canonical_name}")
        return merged_entity

    def resolve_relationship_conflicts(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Resolve contradictory relationships.

        Handles cases like:
        - Same source/target, different relationship types
        - Contradictory properties on relationships

        Args:
            relationships: List of extracted relationships

        Returns:
            List of resolved relationships
        """
        if not relationships:
            return []

        logger.info(f"Resolving conflicts in {len(relationships)} relationships...")

        # Group relationships by (source, target) pair
        rel_groups = defaultdict(list)
        for rel in relationships:
            key = (rel.get('source'), rel.get('target'))
            rel_groups[key].append(rel)

        resolved = []

        for (source, target), rels in rel_groups.items():
            if len(rels) == 1:
                # No conflict
                resolved.append(rels[0])
            else:
                # Multiple relationships between same entities
                merged_rel = self._resolve_relationship_group(rels)
                resolved.append(merged_rel)

        logger.info(f"Resolved to {len(resolved)} unique relationships")
        return resolved

    def _resolve_relationship_group(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolve a group of relationships between the same entities.

        Strategy:
        1. If all have same type → merge
        2. If different types → keep highest confidence
        3. Merge properties
        """
        if len(relationships) == 1:
            return relationships[0]

        # Check if all same type
        types = [r.get('type') for r in relationships]
        if len(set(types)) == 1:
            # Same type, merge properties
            return self._merge_relationships(relationships)
        else:
            # Different types, keep highest confidence
            best_rel = max(relationships, key=lambda r: r.get('confidence', 0.0))
            logger.warning(
                f"Conflicting relationship types between "
                f"{relationships[0].get('source')} and {relationships[0].get('target')}: "
                f"{types}. Keeping: {best_rel.get('type')}"
            )
            return best_rel

    def _merge_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge relationships of the same type."""
        # Use first as base
        merged = relationships[0].copy()

        # Merge properties
        merged_properties = {}
        for rel in relationships:
            props = rel.get('properties', {})
            merged_properties.update(props)

        # Average confidence
        confidences = [r.get('confidence', 0.5) for r in relationships]
        avg_confidence = sum(confidences) / len(confidences)

        merged['properties'] = merged_properties
        merged['confidence'] = avg_confidence
        merged['merged_from'] = len(relationships)

        return merged

    def validate_against_schema(
        self,
        entities: List[Dict[str, Any]],
        strict: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate entities against schema.

        Args:
            entities: List of entities to validate
            strict: If True, reject invalid entities. If False, flag warnings.

        Returns:
            Tuple of (valid_entities, invalid_entities)
        """
        if not self.schema or 'entity_types' not in self.schema:
            logger.warning("No schema provided, skipping validation")
            return entities, []

        valid = []
        invalid = []

        allowed_types = set(self.schema.get('entity_types', []))

        for entity in entities:
            entity_type = entity.get('type')

            if entity_type in allowed_types:
                valid.append(entity)
            else:
                if strict:
                    logger.warning(f"Entity type '{entity_type}' not in schema, rejecting")
                    invalid.append(entity)
                else:
                    logger.warning(f"Entity type '{entity_type}' not in schema, but allowing")
                    valid.append(entity)

        return valid, invalid


# Singleton instance
_conflict_resolution_agent: Optional[ConflictResolutionAgent] = None


def get_conflict_resolution_agent(
    schema: Optional[Dict[str, Any]] = None
) -> ConflictResolutionAgent:
    """
    Get the global ConflictResolutionAgent instance (singleton pattern).

    Args:
        schema: Optional schema to validate against

    Returns:
        ConflictResolutionAgent: Global conflict resolution agent
    """
    global _conflict_resolution_agent
    if _conflict_resolution_agent is None:
        _conflict_resolution_agent = ConflictResolutionAgent(schema=schema)
    return _conflict_resolution_agent


if __name__ == "__main__":
    """Test the ConflictResolutionAgent."""
    import sys

    try:
        print("🔄 Testing ConflictResolutionAgent...")
        agent = get_conflict_resolution_agent()

        # Test entity deduplication
        test_entities = [
            {
                'name': 'Apple Inc.',
                'type': 'Organization',
                'properties': {'founded': '1976'},
                'confidence': 0.9
            },
            {
                'name': 'Apple',
                'type': 'Organization',
                'properties': {'founded': '1976', 'ceo': 'Tim Cook'},
                'confidence': 0.8
            },
            {
                'name': 'Microsoft',
                'type': 'Organization',
                'properties': {'founded': '1975'},
                'confidence': 0.95
            }
        ]

        deduplicated = agent.deduplicate_entities(test_entities, use_llm=False)
        print(f"\n✅ Deduplicated {len(test_entities)} → {len(deduplicated)} entities")
        for entity in deduplicated:
            print(f"  - {entity['name']} (merged from {entity.get('merged_from', 1)})")

        # Test relationship conflict resolution
        test_relationships = [
            {
                'source': 'Apple',
                'target': 'iPhone',
                'type': 'PRODUCES',
                'confidence': 0.9
            },
            {
                'source': 'Apple',
                'target': 'iPhone',
                'type': 'PRODUCES',
                'confidence': 0.85,
                'properties': {'year': '2007'}
            }
        ]

        resolved = agent.resolve_relationship_conflicts(test_relationships)
        print(f"\n✅ Resolved {len(test_relationships)} → {len(resolved)} relationships")

        print("\n✅ ConflictResolutionAgent working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
