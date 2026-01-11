"""
Summarizer Agent - KARMA-inspired

Condenses documents while preserving:
- Entity names
- Action verbs (relationship indicators)
- Temporal markers
- Causal phrases

Based on KARMA paper: "Summarizer agents condense high-relevance segments
into concise summaries while preserving technical details"
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """
    KARMA-inspired Summarizer Agent.

    Preserves relationship-indicating context while condensing text.
    Goal: Keep summaries under 100 words while maintaining entity relationships.
    """

    def __init__(self, llm_client):
        """
        Initialize Summarizer Agent.

        Args:
            llm_client: UnifiedLLMClient instance
        """
        self.llm_client = llm_client
        logger.info("Initialized SummarizerAgent (KARMA-inspired)")

    def summarize_for_relationships(self, document: str, entities: List[Dict[str, Any]]) -> str:
        """
        Summarize document preserving relationship context.

        KARMA approach: Preserve domain-specific terms that indicate relationships.

        Args:
            document: Full document text
            entities: List of entities extracted from document

        Returns:
            Condensed summary (<100 words) preserving relationship signals
        """
        # Extract entity names for the prompt
        entity_names = [e.get('name', '') for e in entities if e.get('name')]

        prompt = f"""Summarize this text in UNDER 100 words while preserving:

1. ALL entity names exactly as written: {', '.join(entity_names[:20])}
2. Action verbs indicating relationships: directed, created, founded, invented, born, died, worked, studied, produced, released
3. Temporal markers: dates, years, time periods (e.g., "in 2012", "during", "after")
4. Causal phrases: led to, resulted in, because of, due to
5. Quantitative data: numbers, statistics, measurements

DO NOT:
- Remove entity names
- Paraphrase relationships
- Add interpretations

Text to summarize:
{document[:4000]}

Summary (<100 words):"""

        try:
            summary = self.llm_client.generate(
                prompt,
                temperature=0.0,
                max_tokens=200  # ~100 words
            )

            logger.info(f"Summarized document: {len(document)} chars → {len(summary)} chars")
            return summary.strip()

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: return truncated original
            return document[:500] + "..."

    def batch_summarize(self, documents: List[str], entities_list: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Summarize multiple documents in batch.

        Args:
            documents: List of document texts
            entities_list: List of entity lists (one per document)

        Returns:
            List of summaries
        """
        summaries = []

        for i, (doc, entities) in enumerate(zip(documents, entities_list)):
            logger.info(f"Summarizing document {i+1}/{len(documents)}")
            summary = self.summarize_for_relationships(doc, entities)
            summaries.append(summary)

        return summaries
