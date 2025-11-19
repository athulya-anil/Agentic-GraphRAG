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
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..agents import (
    get_orchestrator_agent,
    get_reflection_agent,
    RetrievalStrategy
)
from ..agents.query_parser_agent import get_query_parser
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
        use_reflection: bool = True,
        enable_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            use_reranking: Whether to use cross-encoder reranking
            use_reflection: Whether to track performance with ReflectionAgent
            enable_cache: Whether to enable query result caching
            cache_dir: Directory for cache storage
        """
        self.config = get_config()
        self.use_reranking = use_reranking
        self.use_reflection = use_reflection
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or Path("data/cache/queries")

        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def _get_cache_key(self, query: str, top_k: Optional[int], strategy: Optional[RetrievalStrategy]) -> str:
        """Generate cache key from query parameters."""
        cache_input = json.dumps({
            "query": query.lower().strip(),
            "top_k": top_k or self.config.retrieval.top_k_vector,
            "strategy": strategy.value if strategy else "auto"
        }, sort_keys=True)
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve query result from cache."""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    logger.debug(f"Cache hit for query: {cache_key}")
                    return cached
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save query result to cache."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Remove non-serializable fields
            cacheable_result = {
                "query": result.get("query"),
                "response": result.get("response"),
                "num_contexts": result.get("num_contexts"),
                "strategy": result.get("strategy")
            }

            with open(cache_file, 'w') as f:
                json.dump(cacheable_result, f)
            logger.debug(f"Cached query result: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

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

        # Filter low-quality results
        if results:
            results = self._filter_low_quality_results(query, results)

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
        Retrieve using graph traversal with multi-hop support and intelligent query parsing.

        Args:
            query: User query
            params: Retrieval parameters
            return_metadata: Whether to include metadata

        Returns:
            List of results
        """
        top_k = params.get("top_k", self.config.retrieval.top_k_graph)
        max_depth = params.get("max_depth", 2)
        entity_results = []

        try:
            # Step 1: Parse query intent using LLM (smart approach!)
            schema = self.neo4j_manager.get_schema()
            parser = get_query_parser(schema={
                "entity_types": schema.get("labels", []),
                "relationship_types": schema.get("relationship_types", [])
            })

            intent = parser.parse_query(query)
            logger.info(
                f"Query intent: target={intent.get('target_type')}, "
                f"anchor={intent.get('anchor_entity')}, "
                f"direction={intent.get('direction')}"
            )

            # Step 2: Use intent to guide retrieval
            anchor_entity = intent.get("anchor_entity", "")
            anchor_type = intent.get("anchor_type", "")
            target_type = intent.get("target_type", "")
            relationship = intent.get("relationship_type", "")
            direction = intent.get("direction", "bidirectional")

            # Step 3: Find anchor (seed) entities using enhanced search
            seed_entities = []

            if anchor_entity and anchor_type:
                # Use parsed anchor for precise matching
                search_labels = [anchor_type] if anchor_type != "Unknown" else schema.get("labels", [])[:10]
            else:
                # Fallback to keyword-based search
                search_labels = schema.get("labels", [])[:10]

            for label in search_labels:
                # Search for anchor entity in multiple fields
                search_term = anchor_entity.lower() if anchor_entity else ""
                if not search_term:
                    # Fallback: extract from query
                    query_words = [w for w in query.lower().split() if len(w) > 3]
                    search_term = query_words[0] if query_words else ""

                if search_term:
                    cypher = f"""
                    MATCH (n:{label})
                    WHERE toLower(n.name) CONTAINS $word
                       OR toLower(coalesce(n.aliases, '')) CONTAINS $word
                       OR toLower(coalesce(n.keywords, '')) CONTAINS $word
                       OR toLower(coalesce(n.summary, '')) CONTAINS $word
                    RETURN n, id(n) as node_id, $word as matched_word
                    LIMIT {min(top_k, 5)}
                    """
                    results = self.neo4j_manager.execute_query(
                        cypher, {"word": search_term}
                    )

                    for result in results:
                        seed_entities.append({
                            "node": dict(result["n"]),
                            "node_id": result["node_id"],
                            "label": label,
                            "matched_word": result["matched_word"]
                        })

            # Step 4: Intelligent traversal based on query intent
            for seed in seed_entities[:top_k]:
                node = seed["node"]
                label = seed["label"]
                node_id = seed["node_id"]

                # Build relationship pattern based on parsed direction
                if relationship and direction == "forward":
                    # anchor -[REL]-> target
                    rel_pattern = f"-[r:{relationship}*1..{max_depth}]->"
                elif relationship and direction == "reverse":
                    # anchor <-[REL]- target
                    rel_pattern = f"<-[r:{relationship}*1..{max_depth}]-"
                else:
                    # Bidirectional or unknown - search both ways
                    rel_pattern = f"-[r*1..{max_depth}]-"

                # Optional target type filter
                target_filter = f":{target_type}" if target_type and target_type != "Unknown" else ""

                # Get multi-hop paths using intent-aware Cypher
                cypher = f"""
                MATCH path = (start:{label}){rel_pattern}(connected{target_filter})
                WHERE id(start) = $node_id
                WITH path, connected, start, length(path) as depth
                ORDER BY depth
                LIMIT {min(top_k * 2, 20)}
                RETURN
                    start,
                    connected,
                    [r in relationships(path) | type(r)] as rel_types,
                    [n in nodes(path) | coalesce(n.name, labels(n)[0])] as path_names,
                    depth
                """

                try:
                    path_results = self.neo4j_manager.execute_query(
                        cypher, {"node_id": node_id}
                    )

                    # Build comprehensive context from paths
                    context_parts = [f"{label}: {node.get('name')}"]

                    if "summary" in node and node["summary"]:
                        context_parts.append(f"Summary: {node['summary']}")

                    # Add path information for multi-hop reasoning
                    paths_seen = set()
                    for path_result in path_results:
                        rel_types = path_result["rel_types"]
                        path_names = path_result["path_names"]
                        depth = path_result["depth"]

                        # Create path description
                        path_desc = " → ".join([
                            f"{path_names[i]} -{rel_types[i]}-> {path_names[i+1]}"
                            for i in range(len(rel_types))
                        ])

                        if path_desc and path_desc not in paths_seen:
                            paths_seen.add(path_desc)
                            context_parts.append(f"Path (depth {depth}): {path_desc}")

                            # Limit path descriptions to avoid bloat
                            if len(paths_seen) >= 5:
                                break

                    context_text = ". ".join(context_parts)

                    # Score based on relevance and path depth
                    # Shorter paths get higher scores
                    avg_depth = sum(r["depth"] for r in path_results) / max(len(path_results), 1)
                    score = max(0.5, 1.0 - (avg_depth / max_depth) * 0.3)

                    entity_results.append({
                        "text": context_text,
                        "score": score,
                        "source": "graph",
                        "metadata": {
                            "node": node,
                            "node_id": node_id,
                            "paths_found": len(paths_seen),
                            "avg_path_depth": avg_depth
                        } if return_metadata else {}
                    })

                except Exception as e:
                    logger.error(f"Error in multi-hop traversal: {e}")
                    # Fallback to simple retrieval
                    context_text = f"{label}: {node.get('name')}"
                    entity_results.append({
                        "text": context_text,
                        "score": 0.7,
                        "source": "graph",
                        "metadata": {"node": node, "node_id": node_id} if return_metadata else {}
                    })

        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")

        # Deduplicate and rank by score
        seen_texts = set()
        unique_results = []
        # Sort by score descending
        entity_results.sort(key=lambda x: x["score"], reverse=True)

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

    def _filter_low_quality_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        min_score: float = 0.25,  # Balanced threshold
        max_results: int = 10    # Keep reasonable number of results
    ) -> List[Dict[str, Any]]:
        """
        Filter out low-quality and irrelevant results to improve context precision.

        Args:
            query: User query
            results: Retrieved results
            min_score: Minimum score threshold (default: 0.3)
            max_results: Maximum number of results to keep (default: 8)

        Returns:
            Filtered results
        """
        if not results:
            return results

        filtered = []
        query_lower = query.lower()

        # Extract meaningful query words (remove stop words)
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'are',
                      'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having',
                      'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
                      'be', 'can', 'could', 'would', 'should', 'may', 'might', 'must',
                      'shall', 'will', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'to', 'and', 'or', 'but', 'if', 'then', 'so', 'than',
                      'too', 'very', 'just', 'about', 'into', 'through', 'during',
                      'before', 'after', 'above', 'below', 'between', 'under', 'again',
                      'further', 'once', 'here', 'there', 'when', 'where', 'why',
                      'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'any', 'both'}

        query_words = set(w for w in query_lower.split() if w not in stop_words and len(w) > 2)

        for result in results:
            text = result.get("text", "").lower()
            score = result.get("score", 0.0)

            # Filter 1: Score threshold
            if score < min_score:
                continue

            # Filter 2: Minimum length (avoid too short/empty contexts)
            if len(text) < 20:
                continue

            # Filter 3: Semantic relevance check - require keyword overlap
            # This is key for context precision
            text_words = set(w for w in text.split() if len(w) > 2)
            overlap = len(query_words & text_words)

            # Calculate overlap ratio
            if query_words:
                overlap_ratio = overlap / len(query_words)
            else:
                overlap_ratio = 0.5  # Default for empty query words

            # Filter based on score and overlap combination
            # Only filter truly irrelevant results - balance precision with recall
            if score < 0.4 and overlap_ratio < 0.15:
                continue
            elif score < 0.5 and overlap == 0:
                continue

            # Add relevance boost to score based on overlap
            result["adjusted_score"] = score * (1 + overlap_ratio * 0.3)
            filtered.append(result)

        # Sort by adjusted score and limit
        filtered.sort(key=lambda x: x.get("adjusted_score", x["score"]), reverse=True)
        filtered = filtered[:max_results]

        if len(filtered) < len(results):
            logger.info(f"Filtered {len(results) - len(filtered)} low-quality results (kept {len(filtered)})")

        return filtered

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
        ground_truth: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Complete query pipeline: retrieve + generate + optionally evaluate.

        Args:
            query: User query
            top_k: Number of results
            strategy: Force specific strategy
            evaluate: Whether to evaluate with ReflectionAgent
            ground_truth: Optional ground truth for evaluation
            use_cache: Whether to use cached results (default: True)

        Returns:
            Dictionary with response, context, and optional metrics
        """
        # Check cache first (if not evaluating, since evaluation needs fresh metrics)
        if use_cache and not evaluate:
            cache_key = self._get_cache_key(query, top_k, strategy)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached query result")
                return cached_result

        # Route query if strategy not specified
        if strategy is None:
            routing = self.orchestrator.route_query(query)
            strategy = routing["strategy"]
            logger.info(f"Query routed to {strategy.value} strategy")

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
            "num_contexts": len(context),
            "strategy": strategy.value  # Track which strategy was used
        }

        # Evaluate if requested
        if evaluate and self.reflection_agent:
            metrics = self.reflection_agent.evaluate_response(
                query, response, context, ground_truth
            )
            result["metrics"] = metrics

            # Record performance for orchestrator
            self.orchestrator.record_performance(strategy, metrics.get("overall", 0.5))

        # Cache result if not evaluating
        if use_cache and not evaluate:
            cache_key = self._get_cache_key(query, top_k, strategy)
            self._save_to_cache(cache_key, result)

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
