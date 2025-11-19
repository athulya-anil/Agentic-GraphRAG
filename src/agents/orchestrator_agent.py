"""
OrchestratorAgent for Agentic GraphRAG

This agent intelligently routes queries to the optimal retrieval strategy:
- Classifies queries (factual vs. conceptual, relational vs. semantic)
- Selects retrieval strategy (vector/graph/hybrid)
- Dynamically adjusts weights for hybrid retrieval
- Synthesizes responses from multiple sources

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional, Literal
from enum import Enum

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


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
    HYBRID = "hybrid"       # Combination of both


class QueryType(Enum):
    """Enumeration of query types."""
    FACTUAL = "factual"           # Specific facts (e.g., "Who is the CEO?")
    CONCEPTUAL = "conceptual"     # Broad concepts (e.g., "Explain RAG")
    RELATIONAL = "relational"     # Entity relationships (e.g., "What does X treat?")
    EXPLORATORY = "exploratory"   # Open-ended (e.g., "Tell me about...")


class OrchestratorAgent:
    """
    Agent responsible for query routing and response synthesis.

    Analyzes queries to determine:
    1. Query type (factual, conceptual, relational, exploratory)
    2. Optimal retrieval strategy (vector, graph, hybrid)
    3. Hybrid weights (if hybrid strategy chosen)
    4. Response synthesis approach
    """

    def __init__(self):
        """Initialize the OrchestratorAgent."""
        self.llm_client = get_llm_client()
        self.config = get_config()

        # Performance tracking for adaptive learning
        self.strategy_performance: Dict[str, List[float]] = {
            "vector": [],
            "graph": [],
            "hybrid": []
        }

        logger.info("Initialized OrchestratorAgent")

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
            available_strategies = ["vector", "graph", "hybrid"]

        # Classify query
        query_type, query_analysis = self._classify_query(query)

        # Select strategy
        strategy = self._select_strategy(
            query=query,
            query_type=query_type,
            query_analysis=query_analysis,
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
        Classify a query to understand its nature.

        Args:
            query: User query

        Returns:
            Tuple of (QueryType, analysis_dict)
        """
        prompt = f"""Analyze the following user query and classify its type for a knowledge graph retrieval system.

Query: "{query}"

Classify the query into one of these types:
1. FACTUAL: Asks for specific facts or data points (e.g., "What is the population of Paris?")
2. CONCEPTUAL: Asks for explanations or broad concepts (e.g., "Explain how neural networks work")
3. RELATIONAL: Asks about relationships between entities (e.g., "What drugs treat diabetes?")
4. EXPLORATORY: Open-ended exploration (e.g., "Tell me about machine learning")

IMPORTANT: Determine if the query needs relationships between entities.
Questions that need relationships include:
- "Who/What/Where/When" questions involving entities (e.g., "Who founded Apple?" needs PERSON-FOUNDED->COMPANY relationship)
- Questions about entity attributes that are other entities (e.g., "Where is X located?" needs ENTITY-LOCATED_IN->LOCATION)
- Questions asking what entities do/cause/treat/produce (e.g., "What treats diabetes?" needs DRUG-TREATS->DISEASE)
- Multi-hop queries (e.g., "Who founded the company that makes iPhone?" needs two relationships)

Questions that DON'T need relationships:
- Asking for explanations of concepts (e.g., "What is machine learning?")
- Asking for definitions (e.g., "Define photosynthesis")
- Asking how things work conceptually (e.g., "How does gravity work?")

Also determine:
- needs_relationships: Does answering this require traversing entity relationships in a graph? (true/false)
- needs_semantic: Does it need semantic/conceptual understanding beyond facts? (true/false)
- needs_entities: Is it looking for specific entities? (true/false)
- suggested_strategy: Best approach (vector=semantic search, graph=relationship traversal, hybrid=both)

Return JSON:
{{
  "type": "FACTUAL",
  "needs_relationships": true,
  "needs_semantic": false,
  "needs_entities": true,
  "suggested_strategy": "graph",
  "confidence": 0.9,
  "reasoning": "Brief explanation"
}}"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0)

            # Map type string to enum
            type_str = response.get("type", "EXPLORATORY").upper()
            query_type = QueryType[type_str] if type_str in QueryType.__members__ else QueryType.EXPLORATORY

            return query_type, response

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to exploratory + hybrid
            return QueryType.EXPLORATORY, {
                "needs_relationships": True,
                "needs_semantic": True,
                "suggested_strategy": "hybrid",
                "confidence": 0.5
            }

    def _select_strategy(
        self,
        query: str,
        query_type: QueryType,
        query_analysis: Dict[str, Any],
        available_strategies: List[str]
    ) -> RetrievalStrategy:
        """
        Select the optimal retrieval strategy (GRAPH or VECTOR only).

        Args:
            query: User query
            query_type: Classified query type
            query_analysis: Query analysis results
            available_strategies: Available strategies

        Returns:
            Selected RetrievalStrategy (GRAPH or VECTOR)
        """
        # Strategy selection logic based on query type and analysis
        # ONLY GRAPH or VECTOR - no hybrid

        if query_type == QueryType.RELATIONAL:
            # Relational queries always use graph traversal
            preferred = RetrievalStrategy.GRAPH
        elif query_type == QueryType.FACTUAL:
            # Factual queries: check if relationships needed
            if query_analysis.get("needs_relationships", False):
                preferred = RetrievalStrategy.GRAPH
            else:
                preferred = RetrievalStrategy.VECTOR
        elif query_type == QueryType.CONCEPTUAL:
            # Conceptual queries use semantic search
            preferred = RetrievalStrategy.VECTOR
        else:  # EXPLORATORY
            # Exploratory: if needs relationships → graph, else → vector
            if query_analysis.get("needs_relationships", False):
                preferred = RetrievalStrategy.GRAPH
            else:
                preferred = RetrievalStrategy.VECTOR

        # Ensure strategy is available, fallback to vector if needed
        if preferred.value not in available_strategies:
            if "vector" in available_strategies:
                preferred = RetrievalStrategy.VECTOR
            elif "graph" in available_strategies:
                preferred = RetrievalStrategy.GRAPH

        # Consider historical performance (adaptive learning)
        if self._has_performance_data():
            adjusted_strategy = self._adjust_based_on_performance(preferred, available_strategies)
            if adjusted_strategy:
                preferred = adjusted_strategy

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

        elif strategy == RetrievalStrategy.HYBRID:
            params["top_k_vector"] = self.config.retrieval.top_k_vector
            params["top_k_graph"] = self.config.retrieval.top_k_graph

            # Adjust alpha based on query needs
            if query_analysis.get("needs_relationships", False):
                # More weight on graph
                params["alpha"] = 0.3  # 0 = all graph, 1 = all vector
            elif query_analysis.get("needs_semantic", True):
                # More weight on vector
                params["alpha"] = 0.7
            else:
                # Balanced
                params["alpha"] = self.config.retrieval.hybrid_alpha

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
            logger.info(f"Switching from {preferred.value} to {best_strategy} based on performance")
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
        strategy: RetrievalStrategy
    ) -> str:
        """
        Synthesize a response from retrieved context.

        Args:
            query: User query
            retrieved_context: List of retrieved context items
            strategy: Strategy that was used

        Returns:
            Synthesized response
        """
        if not retrieved_context:
            return "I don't have enough information to answer this query."

        # Build context string
        context_parts = []
        for i, item in enumerate(retrieved_context[:10], 1):  # Limit to top 10
            text = item.get("text", item.get("content", ""))
            source = item.get("source", "unknown")
            context_parts.append(f"[{i}] {text} (source: {source})")

        context_str = "\n\n".join(context_parts)

        prompt = f"""Based on the following context, answer the user's query.

Context:
{context_str}

Query: {query}

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so.
Cite sources using [1], [2], etc. when appropriate."""

        try:
            response = self.llm_client.generate(prompt, temperature=0.1)
            return response
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            return "Sorry, I encountered an error generating the response."


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
    """Test the OrchestratorAgent."""
    import sys

    try:
        print("🔄 Initializing OrchestratorAgent...")
        agent = get_orchestrator_agent()

        # Test queries
        test_queries = [
            "Who is the CEO of Tesla?",
            "Explain how neural networks work",
            "What drugs treat diabetes?",
            "Tell me about machine learning applications"
        ]

        print("\n📚 Testing query routing...")
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
