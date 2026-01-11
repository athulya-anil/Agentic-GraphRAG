"""
Reader Agent - KARMA Implementation

Scores document segments based on likelihood of containing novel relationships.
Based on KARMA: "Reader agents parse normalized text into logical segments
and score relevance (0-1)"
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ReaderAgent:
    """
    KARMA Reader Agent.

    Scores document segments for relevance to relationship extraction.
    Filters out low-value segments to focus on relationship-rich content.
    """

    def __init__(self, llm_client, relevance_threshold: float = 0.5):
        """
        Initialize Reader Agent.

        Args:
            llm_client: UnifiedLLMClient instance
            relevance_threshold: Minimum score to keep a segment (0.0-1.0)
        """
        self.llm_client = llm_client
        self.relevance_threshold = relevance_threshold
        logger.info(f"Initialized ReaderAgent (threshold={relevance_threshold})")

    def score_document_relevance(
        self,
        document: str,
        existing_entities: List[Dict[str, Any]]
    ) -> float:
        """
        Score entire document for relationship extraction relevance.

        KARMA approach: R(doc) = LLM_reader(doc, existing_KG)

        Args:
            document: Full document text
            existing_entities: Entities already in knowledge graph

        Returns:
            Relevance score 0.0-1.0
        """
        # Build context about existing entities
        entity_names = [e.get('name') for e in existing_entities[:10]]
        context = f"Known entities: {', '.join(entity_names)}" if entity_names else "No existing entities"

        prompt = f"""Score this document's relevance for extracting NEW relationships on a scale of 0.0 to 1.0.

Consider:
- Does it contain explicit relationships (verbs like "directed", "founded", "created")?
- Does it mention multiple entities that could be connected?
- Does it contain factual statements vs just descriptions?
- Are the relationships likely to be novel and useful?

Context: {context}

Document:
{document[:1000]}

Return ONLY a JSON object:
{{
  "relevance_score": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0, max_tokens=200)

            score = float(response.get("relevance_score", 0.5))
            reasoning = response.get("reasoning", "")

            logger.info(f"Document relevance: {score:.2f} - {reasoning[:50]}")
            return score

        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}")
            return 0.5  # Default to medium relevance

    def filter_documents_by_relevance(
        self,
        documents: List[str],
        entities_per_doc: List[List[Dict[str, Any]]]
    ) -> Tuple[List[str], List[List[Dict[str, Any]]], List[float]]:
        """
        Filter documents based on relevance scores.

        Args:
            documents: List of documents
            entities_per_doc: Entities for each document

        Returns:
            Tuple of (filtered_docs, filtered_entities, scores)
        """
        filtered_docs = []
        filtered_entities = []
        scores = []

        for i, (doc, entities) in enumerate(zip(documents, entities_per_doc)):
            score = self.score_document_relevance(doc, entities)
            scores.append(score)

            if score >= self.relevance_threshold:
                filtered_docs.append(doc)
                filtered_entities.append(entities)
                logger.info(f"Document {i+1}: KEPT (score={score:.2f})")
            else:
                logger.info(f"Document {i+1}: FILTERED (score={score:.2f})")

        logger.info(f"Filtered {len(filtered_docs)}/{len(documents)} documents above threshold")
        return filtered_docs, filtered_entities, scores
