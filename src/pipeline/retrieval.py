"""
Retrieval Pipeline for Agentic GraphRAG (HotpotQA-optimized)

This module handles intelligent query retrieval:
1. Query analysis and routing (OrchestratorAgent)
2. Multi-strategy retrieval (vector or graph)
3. Result filtering
4. Response generation (extractive concatenation)

Author: Agentic GraphRAG Team
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..agents import (
    get_orchestrator_agent,
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
    End-to-end query retrieval pipeline with intelligent routing (HotpotQA).

    Features:
    - Automatic query routing to optimal strategy (VECTOR vs GRAPH)
    - Vector similarity search (FAISS)
    - Graph traversal queries (Neo4j multi-hop)
    - BAAI/bge-reranker-large for result reranking
    - Result filtering
    - Extractive response generation
    """

    def __init__(
        self,
        use_reranking: bool = True,
        enable_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            use_reranking: Whether to use BAAI/bge-reranker-large
            enable_cache: Whether to enable query result caching
            cache_dir: Directory for cache storage
        """
        self.config = get_config()
        self.use_reranking = use_reranking
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or Path("data/cache/queries")

        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize agents
        self.orchestrator = get_orchestrator_agent()

        # Initialize storage
        self.neo4j_manager = get_neo4j_manager()
        self.faiss_index = get_faiss_index()

        # Initialize reranker (BAAI/bge-reranker-large)
        self.reranker = None
        if use_reranking:
            self._initialize_reranker()

        logger.info("Initialized RetrievalPipeline (HotpotQA-optimized)")

    def _initialize_reranker(self):
        """Initialize BAAI/bge-reranker-large for HotpotQA."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('BAAI/bge-reranker-large')
            logger.info("Loaded BAAI/bge-reranker-large")
        except ImportError:
            logger.warning("sentence-transformers not available, reranking disabled")
            self.use_reranking = False
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
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
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # TWO-HOP RETRIEVAL for bridge questions
        # If this is a MULTI_HOP query with VECTOR strategy, perform second hop
        if strategy == RetrievalStrategy.VECTOR and results:
            # Check if query type is MULTI_HOP (bridge question)
            routing = self.orchestrator.route_query(query)
            query_type = routing.get("query_type")
            
            if query_type and query_type.value == "multi_hop":
                logger.info("MULTI_HOP query detected - attempting two-hop retrieval")
                bridge_entities = self._extract_bridge_entities(query, results)
                
                if bridge_entities:
                    logger.info(f"Extracted bridge entities: {bridge_entities}")
                    # Perform second-hop retrieval for each entity
                    second_hop_results = []
                    for entity in bridge_entities[:2]:  # Max 2 entities
                        logger.info(f"Second-hop retrieval for: {entity}")
                        second_hop = self.faiss_index.search(
                            query=entity,
                            top_k=5,
                            return_metadata=True
                        )
                        second_hop_results.extend(second_hop)
                    
                    # Merge and deduplicate
                    if second_hop_results:
                        results = self._merge_and_deduplicate(results, second_hop_results)
                        logger.info(f"After two-hop: {len(results)} total results")

        # Rerank if enabled (BAAI/bge-reranker-large) - VECTOR ONLY
        # Graph retrieval order must be preserved (hop depth, path structure, edge confidence)
        if self.use_reranking and results and strategy == RetrievalStrategy.VECTOR:
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
        Retrieve using vector similarity search with sentence-level provenance.

        Returns evidence items with:
        - doc_id: document index
        - sent_id: sentence index within document
        - text: sentence text (exact)
        - source: "vector"
        - score: similarity score

        Args:
            query: User query
            params: Retrieval parameters
            return_metadata: Whether to include metadata

        Returns:
            List of results with sentence-level provenance
        """
        top_k = params.get("top_k", self.config.retrieval.top_k_vector)
        results = self.faiss_index.search(
            query=query,
            top_k=top_k,
            return_metadata=True  # Always need metadata for provenance
        )

        # Format results with sentence-level provenance
        formatted_results = []
        for result in results:
            meta = result.get("metadata", {})

            # Extract provenance from metadata
            doc_id = meta.get("doc_id", meta.get("doc_index", 0))
            sent_id = meta.get("sent_id", meta.get("chunk_index", 0))

            # Use sentence_text if available, fallback to text
            text = meta.get("sentence_text", meta.get("text", ""))

            formatted_results.append({
                "doc_id": doc_id,
                "sent_id": sent_id,
                "text": text,
                "score": result.get("score", 0.0),
                "source": "vector",
                "metadata": meta if return_metadata else {}
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
            # Support both single and multiple anchor entities (for comparison questions)
            anchor_entities_list = intent.get("anchor_entities", [])
            if not anchor_entities_list and intent.get("anchor_entity"):
                # Backward compatibility: single anchor_entity
                anchor_entities_list = [intent.get("anchor_entity")]

            anchor_type = intent.get("anchor_type", "")
            target_type = intent.get("target_type", "")
            relationship = intent.get("relationship_type", "")
            direction = intent.get("direction", "bidirectional")

            # Step 3: Find anchor (seed) entities using enhanced search
            # For comparison questions, we search for ALL anchor entities
            seed_entities = []

            if anchor_entities_list and anchor_type:
                # Use parsed anchors for precise matching
                search_labels = [anchor_type] if anchor_type != "Unknown" else schema.get("labels", [])[:10]
            else:
                # Fallback to keyword-based search
                search_labels = schema.get("labels", [])[:10]

            # Search for each anchor entity (handles both single and multi-entity queries)
            for anchor_entity in anchor_entities_list:
                for label in search_labels:
                    # Search for this anchor entity in multiple fields
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

            # FALLBACK: If no entities found with parsed intent, try keyword-based search
            if not seed_entities:
                logger.info(f"No entities found with parsed intent, falling back to keyword search")
                # Extract meaningful keywords from query
                stop_words = {'what', 'which', 'when', 'where', 'who', 'how', 'does', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'was', 'were', 'been', 'provide', 'provides', 'providing'}
                query_words = [w for w in query.lower().split() if len(w) > 3 and w not in stop_words]

                # Search all entity types with each keyword
                all_labels = schema.get("labels", [])[:15]
                for keyword in query_words[:3]:  # Use top 3 keywords
                    for label in all_labels:
                        cypher = f"""
                        MATCH (n:{label})
                        WHERE toLower(n.name) CONTAINS $word
                           OR toLower(coalesce(n.aliases, '')) CONTAINS $word
                           OR toLower(coalesce(n.keywords, '')) CONTAINS $word
                           OR toLower(coalesce(n.summary, '')) CONTAINS $word
                        RETURN n, id(n) as node_id, $word as matched_word
                        LIMIT {min(top_k, 3)}
                        """
                        fallback_results = self.neo4j_manager.execute_query(
                            cypher, {"word": keyword}
                        )

                        for result in fallback_results:
                            seed_entities.append({
                                "node": dict(result["n"]),
                                "node_id": result["node_id"],
                                "label": label,
                                "matched_word": result["matched_word"]
                            })

                        if len(seed_entities) >= top_k:
                            break
                    if len(seed_entities) >= top_k:
                        break

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
                # CRITICAL FIX: Also fetch Document nodes via MENTIONED_IN
                cypher = f"""
                MATCH path = (start:{label}){rel_pattern}(connected{target_filter})
                WHERE id(start) = $node_id
                OPTIONAL MATCH (start)-[:MENTIONED_IN]->(doc:Document)
                WITH path, connected, start, doc, length(path) as depth
                ORDER BY depth
                LIMIT {min(top_k * 2, 20)}
                RETURN
                    start,
                    connected,
                    doc,
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

                    # CRITICAL FIX: Extract Document info from query results
                    doc_title = None
                    doc_id_from_node = None
                    if path_results and "doc" in path_results[0]:
                        doc_node = path_results[0]["doc"]
                        if doc_node:
                            doc_title = doc_node.get("title")
                            doc_id_from_node = doc_node.get("doc_id")

                    # IMPORTANT: Also retrieve source document text from FAISS
                    # This ensures we get the rich original content, not just entity names
                    entity_name = node.get('name', '')
                    if entity_name:
                        # Search FAISS for documents mentioning this entity
                        doc_results = self.faiss_index.search(
                            query=entity_name,
                            top_k=2,
                            return_metadata=True
                        )
                        for doc_result in doc_results:
                            doc_text = doc_result.get("metadata", {}).get("text", "")
                            if doc_text and len(doc_text) > 50:
                                # Only include if it mentions the entity
                                if entity_name.lower() in doc_text.lower():
                                    context_parts.append(f"Source document: {doc_text}")

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

                    # Extract provenance from node if available
                    node_provenance = node.get("provenance", "[]")
                    try:
                        import json
                        prov_list = json.loads(node_provenance) if isinstance(node_provenance, str) else node_provenance
                    except (json.JSONDecodeError, TypeError):
                        prov_list = []

                    # Use first provenance entry for doc_id/sent_id, or defaults
                    doc_id = prov_list[0]["doc_id"] if prov_list else (doc_id_from_node or 0)
                    sent_id = prov_list[0]["sent_id"] if prov_list else 0

                    entity_results.append({
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "doc_title": doc_title,  # CRITICAL: Add doc_title for coverage
                        "text": context_text,
                        "score": score,
                        "source": "graph",
                        "metadata": {
                            "node": node,
                            "node_id": node_id,
                            "paths_found": len(paths_seen),
                            "avg_path_depth": avg_depth,
                            "provenance": prov_list  # Full provenance list
                        } if return_metadata else {}
                    })

                except Exception as e:
                    logger.error(f"Error in multi-hop traversal: {e}")
                    # Fallback to simple retrieval with provenance
                    context_text = f"{label}: {node.get('name')}"

                    # Try to extract provenance from node
                    node_provenance = node.get("provenance", "[]")
                    try:
                        import json
                        prov_list = json.loads(node_provenance) if isinstance(node_provenance, str) else node_provenance
                    except (json.JSONDecodeError, TypeError):
                        prov_list = []

                    doc_id = prov_list[0]["doc_id"] if prov_list else 0
                    sent_id = prov_list[0]["sent_id"] if prov_list else 0

                    entity_results.append({
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "doc_title": None,  # Fallback case - no doc title available
                        "text": context_text,
                        "score": 0.7,
                        "source": "graph",
                        "metadata": {"node": node, "node_id": node_id, "provenance": prov_list} if return_metadata else {}
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

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using BAAI/bge-reranker-large.

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
            pairs = [[query, result["text"]] for result in results if result.get("text")]

            if not pairs:
                return results

            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)

            # Update scores
            for i, score in enumerate(rerank_scores):
                if i < len(results):
                    results[i]["rerank_score"] = float(score)
                    results[i]["original_score"] = results[i].get("score", 0.0)
                    results[i]["score"] = float(score)

            # Re-sort by new scores
            results.sort(key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)
            logger.info(f"Reranked {len(results)} results using BAAI/bge-reranker-large")

        except Exception as e:
            logger.error(f"Error in reranking: {e}")

        return results

    def _filter_low_quality_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        min_score: float = 0.15,  # Lower for HotpotQA bridge entities
        max_results: int = 15     # More contexts for multi-hop
    ) -> List[Dict[str, Any]]:
        """
        Filter out low-quality results (HotpotQA-optimized).

        Simplified filtering - no lexical overlap checks to avoid
        filtering out bridge entities with low keyword overlap.

        Args:
            query: User query
            results: Retrieved results
            min_score: Minimum score threshold
            max_results: Maximum number of results to keep

        Returns:
            Filtered results
        """
        if not results:
            return results

        filtered = []

        for result in results:
            text = result.get("text", "")
            score = result.get("score", 0.0)
            source = result.get("source", "vector")

            # Filter 1: Score threshold
            if score < min_score:
                continue

            # Filter 2: Minimum length (avoid too short/empty contexts)
            if len(text) < 20:
                continue

            # No lexical overlap filtering - trust embedding similarity
            # and graph traversal for HotpotQA

            filtered.append(result)

        # Sort by score and limit
        filtered.sort(key=lambda x: x["score"], reverse=True)
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
        return {}

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Complete query pipeline: retrieve + generate (HotpotQA).

        Args:
            query: User query
            top_k: Number of results
            strategy: Force specific strategy
            use_cache: Whether to use cached results (default: True)

        Returns:
            Dictionary with response, context, and metadata
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(query, top_k, strategy)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached query result")
                return cached_result

        # Route query if strategy not specified
        query_type = None
        if strategy is None:
            routing = self.orchestrator.route_query(query)
            strategy = routing["strategy"]
            query_type = routing["query_type"]
            logger.info(f"Query routed to {strategy.value} strategy (type: {query_type.value})")

        # Retrieve context
        context = self.retrieve(query, top_k=top_k, strategy=strategy)

        # Generate response using AnswerSelector (EM-correct)
        response = self.orchestrator.synthesize_response(
            query=query,
            retrieved_context=context,
            strategy=strategy or RetrievalStrategy.VECTOR,
            query_type=query_type
        )

        result = {
            "query": query,
            "response": response,
            "context": context,
            "num_contexts": len(context),
            "strategy": strategy.value,
            "query_type": query_type.value if query_type else "unknown"
        }

        # Cache result
        if use_cache:
            cache_key = self._get_cache_key(query, top_k, strategy)
            self._save_to_cache(cache_key, result)

        return result
    
    def _extract_bridge_entities(
        self,
        query: str,
        evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract candidate bridge entities from first-hop evidence.
        
        Strategy:
        1. Look for proper nouns (capitalized phrases)
        2. Prefer multi-word entities (full names)
        3. Exclude query terms and generic words
        
        Args:
            query: Original query
            evidence: First-hop evidence
        
        Returns:
            List of candidate bridge entities
        """
        import re
        from collections import Counter
        
        # Extract all capitalized phrases (2-4 words for full names)
        candidates = []
        for item in evidence:
            text = item.get("metadata", {}).get("text", "") or item.get("text", "")
            if not text:
                continue
            
            # Match capitalized words (proper nouns)
            # Prefer longer phrases (full names like "Shirley Temple")
            matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
            candidates.extend(matches)
        
        if not candidates:
            return []
        
        # Count frequency
        entity_counts = Counter(candidates)
        
        # Filter out query terms and generic words
        query_words_lower = set(word.lower() for word in query.split())
        generic_words = {'american', 'person', 'organization', 'location', 'facility', 'creativework'}
        
        filtered = []
        for entity, count in entity_counts.most_common(10):
            entity_words = entity.split()
            entity_words_lower = set(word.lower() for word in entity_words)
            
            # Skip if all words are in query
            if entity_words_lower.issubset(query_words_lower):
                continue
            
            # Skip generic entity types
            if any(word.lower() in generic_words for word in entity_words):
                continue
            
            # Prefer entities with 2+ words (full names)
            if len(entity_words) >= 2 and count >= 2:
                filtered.append(entity)
            elif len(entity_words) >= 2 and count >= 1:
                # Allow single mention if it's a full name
                filtered.append(entity)
        
        logger.debug(f"Bridge entity candidates: {filtered[:5]}")
        return filtered[:2]  # Return top 2
    
    def _merge_and_deduplicate(
        self,
        first_hop: List[Dict[str, Any]],
        second_hop: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge first-hop and second-hop results, removing duplicates.
        
        Args:
            first_hop: Results from first retrieval
            second_hop: Results from second retrieval
        
        Returns:
            Merged and deduplicated results
        """
        seen_texts = set()
        merged = []
        
        # Add first-hop results (higher priority)
        for item in first_hop:
            text = item.get("metadata", {}).get("text", "") or item.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                merged.append(item)
        
        # Add second-hop results (if not duplicate)
        for item in second_hop:
            text = item.get("metadata", {}).get("text", "") or item.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                merged.append(item)
        
        # Cap total results to avoid overwhelming LLM
        max_results = 15
        if len(merged) > max_results:
            logger.info(f"Capping results from {len(merged)} to {max_results}")
            merged = merged[:max_results]
        
        return merged


# Singleton instance
_retrieval_pipeline: Optional[RetrievalPipeline] = None


def get_retrieval_pipeline(use_reranking: bool = True) -> RetrievalPipeline:
    """
    Get the global RetrievalPipeline instance (singleton pattern).

    Args:
        use_reranking: Whether to use BAAI/bge-reranker-large (default: True)

    Returns:
        RetrievalPipeline: Global retrieval pipeline (HotpotQA-optimized)
    """
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline(use_reranking=use_reranking)
    return _retrieval_pipeline


if __name__ == "__main__":
    """Test the retrieval pipeline (HotpotQA)."""
    import sys

    try:
        print("🔄 Initializing retrieval pipeline (HotpotQA-optimized)...")
        pipeline = get_retrieval_pipeline()

        # Test query (HotpotQA-style)
        test_query = "What genre was sung by the artist who first recorded 'Come from the Heart'?"

        print(f"\n🔍 Testing retrieval for: '{test_query}'")
        result = pipeline.query(test_query, top_k=10)

        print(f"\n✅ Query complete!")
        print(f"  Strategy: {result['strategy']}")
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
