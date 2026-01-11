"""
Evaluator Agent - KARMA Implementation

Aggregates multiple verification signals into confidence scores.
Based on KARMA: C(t) = σ(Σαᵢvᵢ(t)), Cl(t) = σ(Σβⱼcⱼ(t)), R(t) = σ(Σγₖrₖ(t))
Final decision: integrate if (C+Cl+R)/3 ≥ Θ
"""

import logging
import math
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    KARMA Evaluator Agent.

    Aggregates Confidence, Clarity, and Relevance scores to make
    final integration decisions.
    """

    def __init__(self, integration_threshold: float = 0.7):
        """
        Initialize Evaluator Agent.

        Args:
            integration_threshold: Minimum avg score to integrate (0.0-1.0)
        """
        self.integration_threshold = integration_threshold
        logger.info(f"Initialized EvaluatorAgent (threshold={integration_threshold})")

    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-x))

    def calculate_confidence(self, relationship: Dict[str, Any]) -> float:
        """
        Calculate Confidence score C(t).

        KARMA approach: C(t) = σ(Σαᵢvᵢ(t))

        Signals:
        - LLM confidence score
        - Evidence quality
        - Source method reliability
        """
        signals = []

        # Signal 1: LLM confidence
        llm_conf = relationship.get('confidence', 0.5)
        signals.append(('llm_conf', llm_conf, 2.0))  # weight=2.0

        # Signal 2: Evidence quality (has evidence quote)
        has_evidence = 1.0 if relationship.get('evidence') else 0.3
        signals.append(('has_evidence', has_evidence, 1.0))

        # Signal 3: Source method reliability
        source = relationship.get('source_method', 'unknown')
        source_reliability = {
            'karma_multi_label': 0.9,
            'llm': 0.7,
            'rule_based': 0.5,
            'unknown': 0.3
        }.get(source, 0.5)
        signals.append(('source_method', source_reliability, 1.5))

        # Weighted sum
        weighted_sum = sum(score * weight for _, score, weight in signals)
        total_weight = sum(weight for _, _, weight in signals)

        # Normalize and apply sigmoid
        normalized = weighted_sum / total_weight
        confidence = self.sigmoid((normalized - 0.5) * 4)  # Scale to make sigmoid meaningful

        return confidence

    def calculate_clarity(self, relationship: Dict[str, Any]) -> float:
        """
        Calculate Clarity score Cl(t).

        KARMA approach: Cl(t) = σ(Σβⱼcⱼ(t))

        Signals:
        - Entity name clarity (not ambiguous)
        - Relationship type specificity
        - Property completeness
        """
        signals = []

        # Signal 1: Entity names are clear (not empty, not placeholders)
        source_clear = 1.0 if (relationship.get('source') and
                              len(relationship['source']) > 2 and
                              relationship['source'].lower() not in ['unknown', 'n/a']) else 0.2
        target_clear = 1.0 if (relationship.get('target') and
                              len(relationship['target']) > 2 and
                              relationship['target'].lower() not in ['unknown', 'n/a']) else 0.2

        signals.append(('entity_clarity', (source_clear + target_clear) / 2, 2.0))

        # Signal 2: Relationship type is specific (not generic)
        rel_type = relationship.get('type', '')
        is_specific = 0.9 if rel_type not in ['RELATED_TO', 'ASSOCIATED_WITH'] else 0.4
        signals.append(('type_specificity', is_specific, 1.5))

        # Signal 3: Has supporting properties
        has_properties = 1.0 if relationship.get('properties') else 0.5
        signals.append(('has_properties', has_properties, 0.5))

        # Weighted sum and sigmoid
        weighted_sum = sum(score * weight for _, score, weight in signals)
        total_weight = sum(weight for _, _, weight in signals)

        normalized = weighted_sum / total_weight
        clarity = self.sigmoid((normalized - 0.5) * 4)

        return clarity

    def calculate_relevance(self, relationship: Dict[str, Any]) -> float:
        """
        Calculate Relevance score R(t).

        KARMA approach: R(t) = σ(Σγₖrₖ(t))

        Signals:
        - Relationship type is in schema
        - Entities are important (not trivial)
        - Relationship is unique (not duplicate)
        """
        signals = []

        # Signal 1: Type is in allowed schema (validated earlier)
        # If it passed validation, it's in schema
        signals.append(('in_schema', 1.0, 2.0))

        # Signal 2: Relationship type frequency (rarer = more relevant)
        # This would require corpus statistics, default to 0.7
        signals.append(('uniqueness', 0.7, 1.0))

        # Signal 3: Has conflict marker (contradicts existing) - lower relevance
        has_conflict = relationship.get('has_conflict', False)
        conflict_penalty = 0.3 if has_conflict else 1.0
        signals.append(('no_conflict', conflict_penalty, 1.5))

        # Weighted sum and sigmoid
        weighted_sum = sum(score * weight for _, score, weight in signals)
        total_weight = sum(weight for _, _, weight in signals)

        normalized = weighted_sum / total_weight
        relevance = self.sigmoid((normalized - 0.5) * 4)

        return relevance

    def evaluate_relationship(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single relationship with full KARMA scoring.

        Args:
            relationship: Relationship to evaluate

        Returns:
            Relationship with added evaluation scores
        """
        confidence = self.calculate_confidence(relationship)
        clarity = self.calculate_clarity(relationship)
        relevance = self.calculate_relevance(relationship)

        # KARMA final score
        final_score = (confidence + clarity + relevance) / 3

        # Add evaluation metadata
        relationship['evaluation'] = {
            'confidence': confidence,
            'clarity': clarity,
            'relevance': relevance,
            'final_score': final_score,
            'should_integrate': final_score >= self.integration_threshold
        }

        return relationship

    def evaluate_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluate all relationships and filter by integration threshold.

        Args:
            relationships: List of relationships

        Returns:
            Tuple of (accepted_relationships, rejected_relationships)
        """
        accepted = []
        rejected = []

        for rel in relationships:
            evaluated_rel = self.evaluate_relationship(rel)

            if evaluated_rel['evaluation']['should_integrate']:
                accepted.append(evaluated_rel)
            else:
                rejected.append(evaluated_rel)

        logger.info(f"Evaluated {len(relationships)} relationships: {len(accepted)} accepted, {len(rejected)} rejected")

        # Log score distribution
        if accepted:
            avg_score = sum(r['evaluation']['final_score'] for r in accepted) / len(accepted)
            logger.info(f"Average final score (accepted): {avg_score:.3f}")

        return accepted, rejected
