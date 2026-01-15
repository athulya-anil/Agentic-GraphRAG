"""
OrchestratorAgent for Agentic GraphRAG (HotpotQA-optimized)

This agent intelligently routes queries to the optimal retrieval strategy:
- Classifies HotpotQA query types (single-hop, multi-hop, comparison, yes/no)
- Routes multi-hop/comparison/yes-no → GRAPH
- Routes single-hop → VECTOR
- Synthesizes responses from retrieved context

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config
from ..answering import get_answer_selector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enumeration of retrieval strategies."""
    VECTOR = "vector"       # Semantic similarity search
    GRAPH = "graph"         # Graph traversal/relational queries


class QueryType(Enum):
    """HotpotQA query type taxonomy."""
    SINGLE_HOP = "single_hop"      # Direct fact lookup (1 entity, 1 document)
    MULTI_HOP = "multi_hop"        # Bridge entity reasoning (2+ hops)
    COMPARISON = "comparison"      # Compare properties of 2+ entities
    YES_NO = "yes_no"             # Boolean questions requiring aggregation


class OrchestratorAgent:
    """
    Agent responsible for query routing and response synthesis (HotpotQA).

    Routing logic (simple, no ambiguity):
    - MULTI_HOP → GRAPH (bridge entities require traversal)
    - COMPARISON → GRAPH (must retrieve multiple entities)
    - YES_NO → GRAPH (requires evidence aggregation)
    - SINGLE_HOP → VECTOR (semantic similarity sufficient)
    """

    def __init__(self):
        """Initialize the OrchestratorAgent."""
        self.llm_client = get_llm_client()
        self.config = get_config()
        self.answer_selector = get_answer_selector()

        # Performance tracking for adaptive learning
        self.strategy_performance: Dict[str, List[float]] = {
            "vector": [],
            "graph": []
        }

        logger.info("Initialized OrchestratorAgent (HotpotQA-optimized)")

    def route_query(
        self,
        query: str,
        available_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the optimal retrieval strategy.

        Args:
            query: User query
            available_strategies: List of available strategies (None = all)

        Returns:
            Dictionary containing:
                - strategy: RetrievalStrategy
                - query_type: QueryType
                - confidence: float
                - params: Dict with strategy-specific parameters
        """
        if available_strategies is None:
            available_strategies = ["vector", "graph"]

        # Classify query
        query_type, query_analysis = self._classify_query(query)

        # Select strategy (simple routing)
        strategy = self._select_strategy(
            query_type=query_type,
            available_strategies=available_strategies
        )

        # Get strategy-specific parameters
        params = self._get_strategy_params(strategy, query_analysis)

        logger.info(f"Routed query to {strategy.value} strategy (type: {query_type.value})")

        return {
            "strategy": strategy,
            "query_type": query_type,
            "confidence": query_analysis.get("confidence", 0.7),
            "params": params,
            "analysis": query_analysis
        }

    def _classify_query(self, query: str) -> tuple[QueryType, Dict[str, Any]]:
        """
        Classify a HotpotQA query by type.

        Args:
            query: User query

        Returns:
            Tuple of (QueryType, analysis_dict)
        """
        prompt = f"""Analyze this query for a multi-hop QA system (HotpotQA).

Query: "{query}"

Classify into ONE of these types:

1. SINGLE_HOP: Direct fact lookup requiring one entity/document
   - Example: "What genre of music does Don Williams sing?"
   - Characteristics: One entity, one relationship or property

2. MULTI_HOP: Bridge entity reasoning requiring 2+ hops
   - Example: "What genre was sung by the artist who first recorded 'Come from the Heart'?"
   - Characteristics: Must find intermediate entity (bridge) to reach answer
   - Pattern: "X of Y of Z" or "Who/What that did Y?"

3. COMPARISON: Compare properties of 2+ entities
   - Example: "Were Scott Derrickson and Ed Wood of the same nationality?"
   - Characteristics: Two or more entities mentioned, asking about commonality/difference
   - Keywords: "both", "same", "versus", "and", "which ... or"

4. YES_NO: Boolean question requiring evidence aggregation
   - Example: "Are both Rutgers and Carnegie Mellon located in America?"
   - Characteristics: Answer is yes/no, often overlaps with comparison
   - Keywords: "is", "are", "does", "do", "was", "were" + expecting boolean

IMPORTANT:
- Multi-hop: Look for nested structures ("director of film starring X")
- Comparison: Count entities - if 2+, it's comparison
- Yes/no: Check if answer format is boolean (not factoid)
- If multiple types apply, prefer: COMPARISON > YES_NO > MULTI_HOP > SINGLE_HOP

Return JSON:
{{
  "type": "MULTI_HOP",
  "num_hops": 2,
  "num_entities": 1,
  "is_comparison": false,
  "is_yes_no": false,
  "confidence": 0.9,
  "reasoning": "Query requires finding intermediate entity"
}}"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0)

            # Map type string to enum
            type_str = response.get("type", "SINGLE_HOP").upper()
            type_str = type_str.replace("-", "_")  # Handle "SINGLE-HOP" → "SINGLE_HOP"

            query_type = QueryType[type_str] if type_str in QueryType.__members__ else QueryType.SINGLE_HOP

            return query_type, response

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to multi-hop (safer for HotpotQA)
            return QueryType.MULTI_HOP, {
                "type": "MULTI_HOP",
                "confidence": 0.5,
                "reasoning": "Classification failed, defaulting to multi-hop"
            }

    def _select_strategy(
        self,
        query_type: QueryType,
        available_strategies: List[str]
    ) -> RetrievalStrategy:
        """
        Select retrieval strategy based on query type.

        Simple routing logic (no ambiguity):
        - MULTI_HOP → GRAPH
        - COMPARISON → GRAPH
        - YES_NO → GRAPH
        - SINGLE_HOP → VECTOR

        Args:
            query_type: Classified query type
            available_strategies: Available strategies

        Returns:
            Selected RetrievalStrategy
        """
        # HotpotQA routing: multi-hop/comparison/yes-no → graph
        if query_type in [QueryType.MULTI_HOP, QueryType.COMPARISON, QueryType.YES_NO]:
            preferred = RetrievalStrategy.GRAPH
        else:  # SINGLE_HOP
            preferred = RetrievalStrategy.VECTOR

        # Ensure strategy is available
        if preferred.value not in available_strategies:
            logger.warning(f"Preferred strategy {preferred.value} not available, falling back")
            if "vector" in available_strategies:
                return RetrievalStrategy.VECTOR
            elif "graph" in available_strategies:
                return RetrievalStrategy.GRAPH

        # Consider historical performance (adaptive learning)
        if self._has_performance_data():
            adjusted_strategy = self._adjust_based_on_performance(preferred, available_strategies)
            if adjusted_strategy:
                logger.info(f"Adjusted strategy based on performance: {preferred.value} → {adjusted_strategy.value}")
                return adjusted_strategy

        return preferred

    def _get_strategy_params(
        self,
        strategy: RetrievalStrategy,
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get strategy-specific parameters.

        Args:
            strategy: Selected retrieval strategy
            query_analysis: Query analysis results

        Returns:
            Dictionary of parameters
        """
        params = {}

        if strategy == RetrievalStrategy.VECTOR:
            params["top_k"] = self.config.retrieval.top_k_vector
            params["method"] = "similarity"

        elif strategy == RetrievalStrategy.GRAPH:
            params["top_k"] = self.config.retrieval.top_k_graph
            params["max_depth"] = 2
            params["method"] = "traversal"

        return params

    def _has_performance_data(self) -> bool:
        """Check if we have enough performance data for adaptation."""
        min_samples = 5
        return any(len(scores) >= min_samples for scores in self.strategy_performance.values())

    def _adjust_based_on_performance(
        self,
        preferred: RetrievalStrategy,
        available_strategies: List[str]
    ) -> Optional[RetrievalStrategy]:
        """
        Adjust strategy based on historical performance.

        Args:
            preferred: Initially preferred strategy
            available_strategies: Available strategies

        Returns:
            Adjusted strategy or None to keep preferred
        """
        # Calculate average performance for each strategy
        avg_performance = {}
        for strategy, scores in self.strategy_performance.items():
            if scores and strategy in available_strategies:
                avg_performance[strategy] = sum(scores) / len(scores)

        if not avg_performance:
            return None

        # If preferred strategy is significantly underperforming, switch
        preferred_perf = avg_performance.get(preferred.value, 0.0)
        best_strategy = max(avg_performance, key=avg_performance.get)
        best_perf = avg_performance[best_strategy]

        # Switch if best is significantly better (>15% improvement)
        if best_perf > preferred_perf * 1.15:
            logger.info(f"Performance-based switch: {preferred.value} ({preferred_perf:.2f}) → {best_strategy} ({best_perf:.2f})")
            return RetrievalStrategy[best_strategy.upper()]

        return None

    def record_performance(
        self,
        strategy: RetrievalStrategy,
        score: float
    ) -> None:
        """
        Record performance of a strategy for adaptive learning.

        Args:
            strategy: Strategy that was used
            score: Performance score (0.0 to 1.0)
        """
        if 0.0 <= score <= 1.0:
            self.strategy_performance[strategy.value].append(score)
            # Keep only last 100 scores
            if len(self.strategy_performance[strategy.value]) > 100:
                self.strategy_performance[strategy.value] = self.strategy_performance[strategy.value][-100:]

    def synthesize_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        strategy: RetrievalStrategy,
        query_type: Optional[QueryType] = None
    ) -> str:
        """
        Synthesize answer using symbolic AnswerSelector (HotpotQA EM-correct).

        CRITICAL: This uses DETERMINISTIC answer selection (NOT text generation).
        Output = single answer string for EM evaluation.

        Args:
            query: User query
            retrieved_context: List of retrieved context items
            strategy: Strategy that was used
            query_type: Query type (YES_NO, COMPARISON, SINGLE_HOP, MULTI_HOP)

        Returns:
            Single answer string (EM-valid)
        """
        if not retrieved_context:
            return ""

        # Default to SINGLE_HOP if query_type not provided
        if query_type is None:
            query_type = QueryType.SINGLE_HOP

        # Use AnswerSelector for deterministic answer extraction
        answer = self.answer_selector.select_answer(
            query=query,
            query_type=query_type.value,
            evidence=retrieved_context
        )

        # Fallback if no answer selected
        if not answer or answer.strip() == "":
            logger.warning(f"AnswerSelector returned empty answer for query: {query[:50]}...")
            return ""

        return answer


# Singleton instance
_orchestrator_agent: Optional[OrchestratorAgent] = None


def get_orchestrator_agent() -> OrchestratorAgent:
    """
    Get the global OrchestratorAgent instance (singleton pattern).

    Returns:
        OrchestratorAgent: Global orchestrator agent
    """
    global _orchestrator_agent
    if _orchestrator_agent is None:
        _orchestrator_agent = OrchestratorAgent()
    return _orchestrator_agent


if __name__ == "__main__":
    """Test the OrchestratorAgent with HotpotQA queries."""
    import sys

    try:
        print("🔄 Initializing OrchestratorAgent (HotpotQA)...")
        agent = get_orchestrator_agent()

        # Test queries (HotpotQA-style)
        test_queries = [
            "What genre of music does Don Williams sing?",  # SINGLE_HOP
            "What genre was sung by the artist who first recorded 'Come from the Heart'?",  # MULTI_HOP
            "Were Scott Derrickson and Ed Wood of the same nationality?",  # COMPARISON
            "Are both Rutgers and Carnegie Mellon located in America?",  # YES_NO
        ]

        print("\n📚 Testing query routing (HotpotQA)...")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            routing = agent.route_query(query)
            print(f"  Strategy: {routing['strategy'].value}")
            print(f"  Type: {routing['query_type'].value}")
            print(f"  Confidence: {routing['confidence']:.2f}")
            print(f"  Params: {routing['params']}")

        print("\n✅ OrchestratorAgent working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
