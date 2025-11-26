"""
Failure Predictor for Graph Retrieval

Predicts when graph retrieval is likely to fail and suggests alternative strategies.
Based on analysis of graph failures, this module identifies high-risk queries.

Author: Agentic GraphRAG Team
"""

import logging
import re
from typing import Dict, Any, Tuple, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphFailurePredictor:
    """
    Predicts when graph retrieval is likely to fail.

    Based on analysis of 19 graph failures from MS MARCO evaluation:
    - 21% temporal queries (weather, "now", "current")
    - 11% contact info (phone numbers, addresses)
    - 21% relationship queries with missing entities
    - 47% other (general missing entity/coverage issues)
    """

    def __init__(self):
        """Initialize the failure predictor."""

        # Rule-based patterns for high-risk queries
        self.temporal_patterns = [
            r'\bweather\b',
            r'\bcurrent\b',
            r'\btoday\b',
            r'\bnow\b',
            r'\blatest\b',
            r'\brecent\b',
            r'\bright now\b'
        ]

        self.contact_patterns = [
            r'\bphone number\b',
            r'\bphone\s+number\b',
            r'\bcontact\b.*\bnumber\b',
            r'\baddress\b',
            r'\btoll.?free\b'
        ]

        self.relationship_patterns = [
            r'^what (does|is|treats|causes|prevents|belongs)',
            r'^who (is|was|authored|founded)',
            r'^which ',
            r'what.*(treat|cause|prevent|produce)',
            r'who.*(author|found|create)'
        ]

        # Statistics from analysis
        self.failure_stats = {
            'temporal': 0.21,  # 21% of failures were temporal
            'contact': 0.11,   # 11% were contact info
            'relationship_missing_entity': 0.21  # 21% were relationship queries with missing entities
        }

        logger.info("Initialized GraphFailurePredictor")

    def predict_failure_risk(
        self,
        query: str,
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predict the risk that graph retrieval will fail for this query.

        Args:
            query: User query
            query_analysis: Optional query analysis from OrchestratorAgent

        Returns:
            Tuple of (risk_score, reasoning)
            - risk_score: 0.0 to 1.0 (higher = more likely to fail)
            - reasoning: Dict with details about the prediction
        """
        query_lower = query.lower().strip()

        risk_score = 0.0
        risk_factors = []
        suggestions = []

        # Check for temporal patterns
        is_temporal = self._is_temporal_query(query_lower)
        if is_temporal:
            risk_score += 0.9  # Very high risk
            risk_factors.append('temporal_query')
            suggestions.append('Use vector search - temporal data not in static KG')

        # Check for contact info patterns
        is_contact = self._is_contact_query(query_lower)
        if is_contact:
            risk_score += 0.85  # High risk
            risk_factors.append('contact_info')
            suggestions.append('Use vector search - contact info rarely in KG')

        # Check for relationship queries (moderate risk if entities unknown)
        is_relationship = self._is_relationship_query(query_lower)
        if is_relationship and not is_temporal:
            # Lower risk if we know entities exist, higher if uncertain
            if query_analysis and query_analysis.get('needs_entities'):
                risk_score += 0.4  # Moderate risk
                risk_factors.append('relationship_unknown_entities')
                suggestions.append('Check if entities exist in KG before using graph')
            else:
                # Relationship query but might be general concept
                risk_score += 0.2  # Low-moderate risk

        # Cap at 1.0
        risk_score = min(risk_score, 1.0)

        reasoning = {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
            'suggestions': suggestions,
            'is_temporal': is_temporal,
            'is_contact': is_contact,
            'is_relationship': is_relationship
        }

        return risk_score, reasoning

    def _is_temporal_query(self, query: str) -> bool:
        """Check if query asks for temporal/real-time data."""
        for pattern in self.temporal_patterns:
            if re.search(pattern, query):
                return True
        return False

    def _is_contact_query(self, query: str) -> bool:
        """Check if query asks for contact information."""
        for pattern in self.contact_patterns:
            if re.search(pattern, query):
                return True
        return False

    def _is_relationship_query(self, query: str) -> bool:
        """Check if query asks about entity relationships."""
        for pattern in self.relationship_patterns:
            if re.search(pattern, query):
                return True
        return False

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level."""
        if risk_score >= 0.7:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MODERATE'
        elif risk_score >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'

    def should_avoid_graph(
        self,
        query: str,
        threshold: float = 0.6
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if graph retrieval should be avoided for this query.

        Args:
            query: User query
            threshold: Risk threshold above which to avoid graph (default: 0.6)

        Returns:
            Tuple of (should_avoid, reasoning)
        """
        risk_score, reasoning = self.predict_failure_risk(query)

        should_avoid = risk_score >= threshold

        reasoning['threshold'] = threshold
        reasoning['should_avoid'] = should_avoid
        reasoning['alternative'] = 'vector' if should_avoid else None

        return should_avoid, reasoning


# Singleton instance
_failure_predictor: Optional[GraphFailurePredictor] = None


def get_failure_predictor() -> GraphFailurePredictor:
    """
    Get the global GraphFailurePredictor instance (singleton pattern).

    Returns:
        GraphFailurePredictor: Global failure predictor
    """
    global _failure_predictor
    if _failure_predictor is None:
        _failure_predictor = GraphFailurePredictor()
    return _failure_predictor


if __name__ == "__main__":
    """Test the failure predictor."""
    import sys

    try:
        print("🔄 Testing GraphFailurePredictor...\n")
        predictor = get_failure_predictor()

        # Test queries from actual failures
        test_queries = [
            # Temporal (should be HIGH risk)
            "weather in greek isles may",
            "what is the current stock price?",

            # Contact info (should be HIGH risk)
            "arcadis phone number",
            "phone number to cancel sirius xm",

            # Relationship with unknown entities (should be MODERATE risk)
            "what drugs treat uti in dogs",
            "who authored desperation",

            # General queries (should be LOW risk)
            "what is photosynthesis?",
            "explain machine learning",

            # Good graph queries (should be MINIMAL risk)
            "what is aripiprazole used for?",
        ]

        for query in test_queries:
            risk_score, reasoning = predictor.predict_failure_risk(query)
            should_avoid, decision = predictor.should_avoid_graph(query, threshold=0.6)

            print(f"Query: \"{query}\"")
            print(f"  Risk Score: {risk_score:.2f} ({reasoning['risk_level']})")
            if reasoning['risk_factors']:
                print(f"  Risk Factors: {', '.join(reasoning['risk_factors'])}")
            print(f"  Should Avoid Graph: {'YES' if should_avoid else 'NO'}")
            if should_avoid and reasoning['suggestions']:
                print(f"  → {reasoning['suggestions'][0]}")
            print()

        print("✅ GraphFailurePredictor working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
