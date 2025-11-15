"""
Query Parser Agent

Uses LLM to understand query intent and structure for better graph traversal.

Extracts:
- What we're looking for (subject/target)
- What we know (object/anchor)
- Relationship type and direction
- Query pattern (forward, reverse, bidirectional)

Author: Agentic GraphRAG Team
"""

import logging
from typing import Dict, Any, Optional, List
from ..utils.llm_client import get_llm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryParserAgent:
    """
    Parses natural language queries to extract graph query patterns.

    Examples:
    - "What does Aspirin treat?" → {target: "Disease", anchor: "Aspirin", relationship: "TREATS", direction: "forward"}
    - "Which drugs treat diabetes?" → {target: "Drug", anchor: "diabetes", relationship: "TREATS", direction: "reverse"}
    - "Who founded Apple?" → {target: "Person", anchor: "Apple", relationship: "FOUNDED", direction: "reverse"}
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize Query Parser Agent.

        Args:
            schema: Optional graph schema with entity and relationship types
        """
        self.llm_client = get_llm_client()
        self.schema = schema or {}
        logger.info("Initialized QueryParserAgent")

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query to extract graph query pattern.

        Args:
            query: Natural language query

        Returns:
            Dictionary with query pattern:
            {
                "target_type": "Drug",  # What we're looking for
                "anchor_entity": "diabetes",  # What we know
                "relationship_type": "TREATS",  # Connection type
                "direction": "reverse",  # forward/reverse/bidirectional
                "anchor_type": "Disease",  # Type of anchor entity
                "confidence": 0.9
            }
        """
        # Build schema context for LLM
        schema_context = ""
        if self.schema:
            if "entity_types" in self.schema:
                schema_context += f"\nAvailable entity types: {', '.join(self.schema['entity_types'])}"
            if "relationship_types" in self.schema:
                schema_context += f"\nAvailable relationships: {', '.join(self.schema['relationship_types'])}"

        prompt = f"""Parse this query to extract the graph query pattern.{schema_context}

Query: "{query}"

Analyze:
1. What is the user looking for? (target entity type)
2. What do they already know? (anchor entity/value)
3. What's the relationship connecting them?
4. Is this a forward or reverse query?

Examples:
- "What does Aspirin treat?"
  → Looking for: Disease, Know: "Aspirin", Relationship: TREATS, Direction: forward

- "Which drugs treat diabetes?"
  → Looking for: Drug, Know: "diabetes", Relationship: TREATS, Direction: reverse

- "Who manufactures Aspirin?"
  → Looking for: Organization, Know: "Aspirin", Relationship: MANUFACTURES, Direction: reverse

- "Where is Tesla headquartered?"
  → Looking for: Location, Know: "Tesla", Relationship: HEADQUARTERED_IN, Direction: forward

Return JSON:
{{
  "target_type": "EntityType of what we're looking for",
  "anchor_entity": "the specific entity/value we know",
  "anchor_type": "EntityType of the anchor",
  "relationship_type": "RELATIONSHIP_TYPE",
  "direction": "forward or reverse",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0)

            # Validate response
            if not isinstance(response, dict):
                logger.warning(f"Invalid response format from LLM: {response}")
                return self._fallback_parse(query)

            # Add query for reference
            response["original_query"] = query

            logger.info(
                f"Parsed query: target={response.get('target_type')}, "
                f"anchor={response.get('anchor_entity')}, "
                f"relationship={response.get('relationship_type')}, "
                f"direction={response.get('direction')}"
            )

            return response

        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return self._fallback_parse(query)

    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """
        Fallback parser when LLM fails.

        Simple heuristics:
        - "which/what" at start → likely reverse query
        - Entity names → extract as anchors
        """
        query_lower = query.lower()

        return {
            "target_type": "Unknown",
            "anchor_entity": None,
            "anchor_type": "Unknown",
            "relationship_type": None,
            "direction": "bidirectional",  # Search both ways when uncertain
            "confidence": 0.3,
            "reasoning": "Fallback parser - LLM failed",
            "original_query": query
        }

    def construct_cypher_query(
        self,
        parse_result: Dict[str, Any],
        max_results: int = 5
    ) -> tuple[str, Dict[str, Any]]:
        """
        Construct a Cypher query based on parsed query pattern.

        Args:
            parse_result: Result from parse_query()
            max_results: Maximum results to return

        Returns:
            Tuple of (cypher_query, parameters)
        """
        target_type = parse_result.get("target_type", "")
        anchor_entity = parse_result.get("anchor_entity", "")
        anchor_type = parse_result.get("anchor_type", "")
        relationship = parse_result.get("relationship_type", "")
        direction = parse_result.get("direction", "bidirectional")

        params = {
            "anchor": anchor_entity.lower() if anchor_entity else "",
            "limit": max_results
        }

        # Build relationship pattern based on direction
        if direction == "forward" and relationship:
            # anchor -[REL]-> target
            rel_pattern = f"-[r:{relationship}]->"
        elif direction == "reverse" and relationship:
            # anchor <-[REL]- target
            rel_pattern = f"<-[r:{relationship}]-"
        else:
            # bidirectional
            rel_pattern = "-[r]-" if relationship else "-[r]-"

        # Build MATCH pattern
        if anchor_type and target_type:
            match_pattern = f"MATCH (anchor:{anchor_type}){rel_pattern}(target:{target_type})"
        elif anchor_type:
            match_pattern = f"MATCH (anchor:{anchor_type}){rel_pattern}(target)"
        else:
            match_pattern = f"MATCH (anchor){rel_pattern}(target)"

        # Build WHERE clause for anchor matching
        where_clauses = [
            "toLower(anchor.name) CONTAINS $anchor",
            "OR toLower(coalesce(anchor.aliases, '')) CONTAINS $anchor",
            "OR toLower(coalesce(anchor.keywords, '')) CONTAINS $anchor"
        ]
        where_clause = f"WHERE {' '.join(where_clauses)}"

        # Complete query
        cypher = f"""
        {match_pattern}
        {where_clause}
        RETURN target, anchor, type(r) as relationship
        LIMIT $limit
        """

        return cypher, params


# Singleton instance
_query_parser: Optional[QueryParserAgent] = None


def get_query_parser(schema: Optional[Dict[str, Any]] = None) -> QueryParserAgent:
    """
    Get the global QueryParserAgent instance (singleton).

    Args:
        schema: Optional graph schema

    Returns:
        QueryParserAgent: Global query parser
    """
    global _query_parser

    if _query_parser is None or schema is not None:
        _query_parser = QueryParserAgent(schema=schema)

    return _query_parser


if __name__ == "__main__":
    """Test the query parser."""
    print("🔄 Testing QueryParserAgent...\n")

    parser = QueryParserAgent(schema={
        "entity_types": ["Drug", "Disease", "Organization", "Person", "Location"],
        "relationship_types": ["TREATS", "MANUFACTURES", "FOUNDED", "HEADQUARTERED_IN"]
    })

    test_queries = [
        "What does Aspirin treat?",
        "Which drugs treat diabetes?",
        "Who manufactures Aspirin?",
        "Which diseases are treated by Metformin?",
        "Where is Tesla headquartered?",
        "Who founded Apple?"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        result = parser.parse_query(query)
        print(f"  Target: {result.get('target_type')}")
        print(f"  Anchor: {result.get('anchor_entity')} ({result.get('anchor_type')})")
        print(f"  Relationship: {result.get('relationship_type')}")
        print(f"  Direction: {result.get('direction')}")
        print(f"  Confidence: {result.get('confidence')}")

        cypher, params = parser.construct_cypher_query(result)
        print(f"  Cypher: {cypher.strip()[:100]}...")
        print()
