"""
Schema Aligner

Ensures extracted relationships conform to predefined schema.
Uses rule-based validation: exact match, synonym mapping, fuzzy matching.

NOTE: This is NOT an agent (no LLM calls). It performs purely deterministic
schema validation using string matching and predefined mappings.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class SchemaAligner:
    """
    Schema Aligner (rule-based, no LLM).

    Validates and aligns relationships to a predefined schema/ontology
    using exact matching, synonym mapping, and fuzzy string matching.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize Schema Aligner.

        Args:
            schema: Optional schema definition
        """
        self.schema = schema or self._get_default_schema()

        # Build lookup structures
        self.allowed_types = set(self.schema['relationship_types'])
        self.type_synonyms = self.schema.get('synonyms', {})
        self.entity_types = set(self.schema.get('entity_types', []))

        logger.info(f"Initialized SchemaAligner with {len(self.allowed_types)} relationship types")

    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default schema for general-purpose knowledge graphs."""
        return {
            'relationship_types': [
                # People & Organizations
                'WORKS_FOR', 'EMPLOYED_BY', 'CEO_OF', 'FOUNDER_OF',
                'MEMBER_OF', 'PART_OF', 'SUBSIDIARY_OF',

                # Creative works
                'DIRECTED', 'PRODUCED', 'ACTED_IN', 'STARS', 'APPEARED_IN',
                'WROTE', 'AUTHORED', 'COMPOSED',

                # Locations
                'LOCATED_IN', 'BASED_IN', 'HEADQUARTERS_IN',
                'BORN_IN', 'DIED_IN', 'LIVED_IN',

                # Events & Time
                'OCCURRED_IN', 'FOUNDED_IN', 'RELEASED_ON',
                'BORN_ON', 'DIED_ON',

                # Relationships
                'MARRIED_TO', 'PARENT_OF', 'CHILD_OF', 'SIBLING_OF',
                'FRIEND_OF', 'COLLEAGUE_OF',

                # Business
                'ACQUIRED', 'MERGED_WITH', 'INVESTED_IN', 'FUNDED',
                'COMPETITOR_OF', 'PARTNER_WITH',

                # Education & Research
                'STUDIED_AT', 'GRADUATED_FROM', 'TEACHES_AT',
                'RESEARCHED', 'DISCOVERED', 'INVENTED',

                # Generic (use sparingly)
                'RELATED_TO', 'ASSOCIATED_WITH', 'HAS_PROPERTY'
            ],
            'synonyms': {
                'WORKS_AT': 'WORKS_FOR',
                'EMPLOYEE_OF': 'EMPLOYED_BY',
                'CHIEF_OF': 'CEO_OF',
                'CREATED': 'FOUNDED_IN',
                'ESTABLISHED': 'FOUNDED_IN',
                'ACTED': 'ACTED_IN',
                'STARRED': 'STARS',
                'FEATURED_IN': 'APPEARED_IN',
                'PENNED': 'WROTE',
                'WROTE_BY': 'WROTE',
                'LOCATED': 'LOCATED_IN',
                'IN': 'LOCATED_IN',
                'HQ_IN': 'HEADQUARTERS_IN',
                'MARRIED': 'MARRIED_TO',
                'SPOUSE': 'MARRIED_TO',
                'ACQUIRED_BY': 'ACQUIRED',
                'BOUGHT': 'ACQUIRED',
                'PURCHASED': 'ACQUIRED',
            },
            'entity_types': [
                'PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT',
                'DATE', 'PRODUCT', 'WORK', 'CONCEPT'
            ]
        }

    def fuzzy_match_type(self, rel_type: str, threshold: float = 0.8) -> Optional[str]:
        """
        Fuzzy match relationship type to schema.

        Args:
            rel_type: Relationship type to match
            threshold: Similarity threshold (0-1)

        Returns:
            Matched schema type or None
        """
        best_match = None
        best_score = 0.0

        for schema_type in self.allowed_types:
            score = SequenceMatcher(None, rel_type.upper(), schema_type).ratio()
            if score > best_score:
                best_score = score
                best_match = schema_type

        return best_match if best_score >= threshold else None

    def align_relationship_type(self, rel_type: str) -> Optional[str]:
        """
        Align relationship type to schema.

        Steps:
        1. Normalize and check exact match
        2. Check synonym mapping
        3. Try fuzzy matching
        4. Fallback to LLM reasoning (for complex cases)

        Args:
            rel_type: Original relationship type

        Returns:
            Aligned relationship type or None if invalid
        """
        # 1. Exact match
        normalized = rel_type.upper().strip()
        if normalized in self.allowed_types:
            return normalized

        # 2. Synonym mapping
        if normalized in self.type_synonyms:
            mapped = self.type_synonyms[normalized]
            logger.debug(f"Mapped {rel_type} -> {mapped} (synonym)")
            return mapped

        # 3. Fuzzy matching
        fuzzy_match = self.fuzzy_match_type(normalized)
        if fuzzy_match:
            logger.debug(f"Mapped {rel_type} -> {fuzzy_match} (fuzzy)")
            return fuzzy_match

        # 4. Could not align
        logger.warning(f"Could not align relationship type: {rel_type}")
        return None

    def validate_relationship(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and align a single relationship.

        Args:
            relationship: Relationship to validate

        Returns:
            Validated relationship with alignment metadata
        """
        original_type = relationship.get('type', '')

        # Align type
        aligned_type = self.align_relationship_type(original_type)

        if not aligned_type:
            relationship['validation'] = {
                'is_valid': False,
                'reason': 'type_not_in_schema',
                'original_type': original_type
            }
            return relationship

        # Update type if changed
        if aligned_type != original_type:
            relationship['type'] = aligned_type
            relationship['original_type'] = original_type

        # Check entity validity
        source = relationship.get('source', '')
        target = relationship.get('target', '')

        if not source or not target:
            relationship['validation'] = {
                'is_valid': False,
                'reason': 'missing_entities'
            }
            return relationship

        # Mark as valid
        relationship['validation'] = {
            'is_valid': True,
            'aligned': aligned_type != original_type
        }

        return relationship

    def align_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Align all relationships to schema.

        Args:
            relationships: List of relationships

        Returns:
            Tuple of (valid_relationships, invalid_relationships)
        """
        valid = []
        invalid = []

        alignment_stats = {
            'total': len(relationships),
            'exact_match': 0,
            'aligned': 0,
            'invalid': 0
        }

        for rel in relationships:
            validated_rel = self.validate_relationship(rel)

            if validated_rel.get('validation', {}).get('is_valid', False):
                valid.append(validated_rel)

                if validated_rel.get('validation', {}).get('aligned', False):
                    alignment_stats['aligned'] += 1
                else:
                    alignment_stats['exact_match'] += 1
            else:
                invalid.append(validated_rel)
                alignment_stats['invalid'] += 1

        logger.info(
            f"Schema alignment: {alignment_stats['exact_match']} exact, "
            f"{alignment_stats['aligned']} aligned, "
            f"{alignment_stats['invalid']} invalid "
            f"(out of {alignment_stats['total']})"
        )

        return valid, invalid

    def suggest_schema_extensions(
        self,
        invalid_relationships: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Suggest new relationship types based on invalid relationships.

        Args:
            invalid_relationships: List of invalid relationships

        Returns:
            List of suggested relationship types
        """
        type_counts = {}

        for rel in invalid_relationships:
            original_type = rel.get('original_type') or rel.get('type', '')
            if original_type:
                type_counts[original_type] = type_counts.get(original_type, 0) + 1

        # Suggest types that appear frequently
        suggestions = [
            type_name for type_name, count in sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            ) if count >= 3  # Appears at least 3 times
        ]

        if suggestions:
            logger.info(f"Schema extension suggestions: {suggestions}")

        return suggestions
