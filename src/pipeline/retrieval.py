"""
Retrieval Pipeline for Agentic GraphRAG

This module handles intelligent query retrieval:
1. Query analysis and routing (OrchestratorAgent)
2. Multi-strategy retrieval (vector/graph/hybrid)
3. Context reranking (optional cross-encoder)
4. Response generation
5. Performance tracking (ReflectionAgent)

Author: Agentic GraphRAG Team
"""

import logging
from typing import List, Dict, Any, Optional

from ..agents import (
    get_orchestrator_agent,
    get_reflection_agent,
    RetrievalStrategy
)
from ..graph import get_neo4j_manager
from ..vector import get_faiss_index
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    End-to-end query retrieval pipeline with intelligent routing.

    Features:
    - Automatic query routing to optimal strategy
    - Vector similarity search
    - Graph traversal queries
    - Hybrid retrieval with dynamic weighting
    - Optional cross-encoder reranking
    - Response generation and evaluation
    """

    def __init__(
        self,
        use_reranking: bool = False,
        use_reflection: bool = True
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            use_reranking: Whether to use cross-encoder reranking
            use_reflection: Whether to track performance with ReflectionAgent
        """
        self.config = get_config()
        self.use_reranking = use_reranking
        self.use_reflection = use_reflection

        # Initialize agents
        self.orchestrator = get_orchestrator_agent()
        self.reflection_agent = get_reflection_agent() if use_reflection else None

        # Initialize storage
        self.neo4j_manager = get_neo4j_manager()
        self.faiss_index = get_faiss_index()

        # Cross-encoder for reranking (lazy load)
        self.reranker = None
        if use_reranking:
            self._initialize_reranker()

        logger.info("Initialized RetrievalPipeline")

    def _initialize_reranker(self):
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder reranker")
        except ImportError:
            logger.warning("sentence-transformers not available, reranking disabled")
            self.use_reranking = False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query
            top_k: Number of results to return (uses config default if None)
            strategy: Force a specific strategy (None = auto-route)
            return_metadata: Whether to include metadata in results

        Returns:
            List of retrieved context items
        """
        # Route query if strategy not specified
        if strategy is None:
            routing = self.orchestrator.route_query(query)
            strategy = routing["strategy"]
            params = routing["params"]
            logger.info(f"Query routed to {strategy.value} strategy")
        else:
            # Use provided strategy with default params
            params = self._get_default_params(strategy)

        # Override top_k if provided
        if top_k is not None:
            if "top_k" in params:
                params["top_k"] = top_k
            if "top_k_vector" in params:
                params["top_k_vector"] = top_k
            if "top_k_graph" in params:
                params["top_k_graph"] = top_k

        # Retrieve based on strategy
        if strategy == RetrievalStrategy.VECTOR:
            results = self._retrieve_vector(query, params, return_metadata)
        elif strategy == RetrievalStrategy.GRAPH:
            results = self._retrieve_graph(query, params, return_metadata)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self._retrieve_hybrid(query, params, return_metadata)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Rerank if enabled
        if self.use_reranking and results:
            results = self._rerank_results(query, results)

        logger.info(f"Retrieved {len(results)} results for query")
        return results

    def _retrieve_vector(
        self,
        query: str,
        params: Dict[str, Any],
        return_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using vector similarity search.

        Args:
            query: User query
            params: Retrieval parameters
            return_metadata: Whether to include metadata

        Returns:
            List of results
        """
        top_k = params.get("top_k", self.config.retrieval.top_k_vector)
        results = self.faiss_index.search(
            query=query,
            top_k=top_k,
            return_metadata=return_metadata
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.get("metadata", {}).get("text", ""),
                "score": result.get("score", 0.0),
                "source": "vector",
                "metadata": result.get("metadata", {}) if return_metadata else {}
            })

        return formatted_results

    def _retrieve_graph(
        self,
        query: str,
        params: Dict[str, Any],
        return_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using graph traversal.

        Args:
            query: User query
            params: Retrieval parameters
            return_metadata: Whether to include metadata

        Returns:
            List of results
        """
        top_k = params.get("top_k", self.config.retrieval.top_k_graph)
        max_depth = params.get("max_depth", 2)

        # Extract entity names from query (simple approach)
        # In production, would use NER here
        query_lower = query.lower()

        # Search for entities in Neo4j that match query terms
        query_words = set(query_lower.split())
        entity_results = []

        try:
            # Get all nodes (in production, would filter more efficiently)
            all_labels = self.neo4j_manager.get_schema()["labels"]

            for label in all_labels[:10]:  # Limit labels checked
                # Search for nodes with names containing query words
                for word in list(query_words)[:5]:  # Limit words checked
                    if len(word) > 3:  # Skip short words
                        cypher = f"""
                        MATCH (n:{label})
                        WHERE toLower(n.name) CONTAINS $word
                        RETURN n, id(n) as node_id
                        LIMIT {top_k}
                        """
                        results = self.neo4j_manager.execute_query(
                            cypher, {"word": word}
                        )

                        for result in results:
                            node = dict(result["n"])
                            node_id = result["node_id"]

                            # Get relationships
                            relationships = self.neo4j_manager.get_relationships(
                                label,
                                {"name": node.get("name")},
                                direction="both"
                            )

                            # Build context text
                            context_parts = [f"{label}: {node.get('name')}"]

                            if "summary" in node and node["summary"]:
                                context_parts.append(f"Summary: {node['summary']}")

                            for rel in relationships[:3]:  # Limit relationships
                                rel_type = rel["relationship_type"]
                                connected = rel["connected_node"].get("name", "unknown")
                                context_parts.append(f"{rel_type} {connected}")

                            context_text = ". ".join(context_parts)

                            entity_results.append({
                                "text": context_text,
                                "score": 0.8,  # Fixed score for graph results
                                "source": "graph",
                                "metadata": {
                                    "node": node,
                                    "node_id": node_id,
                                    "relationships": relationships
                                } if return_metadata else {}
                            })

        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")

        # Deduplicate and limit
        seen_texts = set()
        unique_results = []
        for result in entity_results:
            text = result["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        return unique_results

    def _retrieve_hybrid(
        self,
        query: str,
        params: Dict[str, Any],
        return_metadata: bool
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid approach (vector + graph).

        Args:
            query: User query
            params: Retrieval parameters
            return_metadata: Whether to include metadata

        Returns:
            List of combined results
        """
        top_k_vector = params.get("top_k_vector", self.config.retrieval.top_k_vector)
        top_k_graph = params.get("top_k_graph", self.config.retrieval.top_k_graph)
        alpha = params.get("alpha", self.config.retrieval.hybrid_alpha)

        # Get results from both sources
        vector_results = self._retrieve_vector(
            query, {"top_k": top_k_vector}, return_metadata
        )
        graph_results = self._retrieve_graph(
            query, {"top_k": top_k_graph}, return_metadata
        )

        # Combine and reweight scores
        # alpha = 0: all graph, alpha = 1: all vector
        combined = []

        for result in vector_results:
            result["score"] = result["score"] * alpha
            combined.append(result)

        for result in graph_results:
            result["score"] = result["score"] * (1 - alpha)
            combined.append(result)

        # Sort by combined score
        combined.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate by text
        seen_texts = set()
        unique_combined = []
        for result in combined:
            text = result["text"]
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_combined.append(result)

        # Return top results
        top_k_total = max(top_k_vector, top_k_graph)
        return unique_combined[:top_k_total]

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.

        Args:
            query: User query
            results: Initial results

        Returns:
            Reranked results
        """
        if not self.reranker or not results:
            return results

        try:
            # Prepare pairs for reranking
            pairs = [[query, result["text"]] for result in results if result["text"]]

            if not pairs:
                return results

            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)

            # Update scores
            for i, score in enumerate(rerank_scores):
                if i < len(results):
                    results[i]["rerank_score"] = float(score)
                    results[i]["original_score"] = results[i]["score"]
                    results[i]["score"] = float(score)

            # Re-sort by new scores
            results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
            logger.info("Reranked results using cross-encoder")

        except Exception as e:
            logger.error(f"Error in reranking: {e}")

        return results

    def _get_default_params(self, strategy: RetrievalStrategy) -> Dict[str, Any]:
        """Get default parameters for a strategy."""
        if strategy == RetrievalStrategy.VECTOR:
            return {"top_k": self.config.retrieval.top_k_vector}
        elif strategy == RetrievalStrategy.GRAPH:
            return {"top_k": self.config.retrieval.top_k_graph, "max_depth": 2}
        elif strategy == RetrievalStrategy.HYBRID:
            return {
                "top_k_vector": self.config.retrieval.top_k_vector,
                "top_k_graph": self.config.retrieval.top_k_graph,
                "alpha": self.config.retrieval.hybrid_alpha
            }
        return {}

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
        evaluate: bool = False,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete query pipeline: retrieve + generate + optionally evaluate.

        Args:
            query: User query
            top_k: Number of results
            strategy: Force specific strategy
            evaluate: Whether to evaluate with ReflectionAgent
            ground_truth: Optional ground truth for evaluation

        Returns:
            Dictionary with response, context, and optional metrics
        """
        # Retrieve context
        context = self.retrieve(query, top_k=top_k, strategy=strategy)

        # Generate response
        response = self.orchestrator.synthesize_response(
            query, context, strategy or RetrievalStrategy.HYBRID
        )

        result = {
            "query": query,
            "response": response,
            "context": context,
            "num_contexts": len(context)
        }

        # Evaluate if requested
        if evaluate and self.reflection_agent:
            metrics = self.reflection_agent.evaluate_response(
                query, response, context, ground_truth
            )
            result["metrics"] = metrics

            # Record performance for orchestrator
            if strategy:
                self.orchestrator.record_performance(strategy, metrics.get("overall", 0.5))

        return result


# Singleton instance
_retrieval_pipeline: Optional[RetrievalPipeline] = None


def get_retrieval_pipeline(
    use_reranking: bool = False,
    use_reflection: bool = True
) -> RetrievalPipeline:
    """
    Get the global RetrievalPipeline instance (singleton pattern).

    Args:
        use_reranking: Whether to use cross-encoder reranking
        use_reflection: Whether to use reflection agent

    Returns:
        RetrievalPipeline: Global retrieval pipeline
    """
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline(
            use_reranking=use_reranking,
            use_reflection=use_reflection
        )
    return _retrieval_pipeline


if __name__ == "__main__":
    """Test the retrieval pipeline."""
    import sys

    try:
        print("🔄 Initializing retrieval pipeline...")
        pipeline = get_retrieval_pipeline(use_reflection=False)

        # Test query
        test_query = "What is machine learning?"

        print(f"\n🔍 Testing retrieval for: '{test_query}'")
        result = pipeline.query(test_query, top_k=5, evaluate=False)

        print(f"\n✅ Query complete!")
        print(f"  Response: {result['response'][:200]}...")
        print(f"  Contexts retrieved: {result['num_contexts']}")

        if result['context']:
            print(f"\n📄 Top context:")
            top_context = result['context'][0]
            print(f"  Text: {top_context['text'][:150]}...")
            print(f"  Score: {top_context['score']:.3f}")
            print(f"  Source: {top_context['source']}")

        print("\n✅ Retrieval pipeline working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
