"""
LLM-Based Answerer for HotpotQA

Replaces symbolic answer selection with LLM-based reading.
Uses existing UnifiedLLMClient infrastructure.

Author: Agentic GraphRAG Team
"""

import logging
from typing import List, Dict, Any, Optional
from ..utils.llm_client import get_llm_client
from .answer_selector import normalize_answer

logger = logging.getLogger(__name__)


class LLMAnswerer:
    """
    LLM-based answer selection for HotpotQA.
    
    Replaces symbolic logic with constrained QA prompting.
    """
    
    def __init__(self):
        """Initialize LLM answerer with existing LLM client."""
        self.llm_client = get_llm_client()
        logger.info("Initialized LLMAnswerer")
    
    def answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        required_type: Optional[str] = None
    ) -> str:
        """
        Generate answer from question and evidence using LLM.
        
        Args:
            question: User query
            evidence: Retrieved context items with metadata
            required_type: Expected answer type (ignored - LLM handles this)
        
        Returns:
            Normalized answer string (EM-compatible)
        """
        if not evidence:
            logger.warning(f"No evidence provided for question: {question[:50]}...")
            return ""
        
        # Format evidence into structured prompt
        formatted_evidence = self._format_evidence(evidence)
        
        # Build constrained QA prompt
        user_prompt = self._build_prompt(question, formatted_evidence)
        system_prompt = self._get_system_prompt()
        
        # Call LLM
        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,  # Deterministic for QA
                max_tokens=50  # Short answers only
            )
            
            # Post-process numeric answers to include units from evidence
            processed_answer = self._append_unit_if_numeric(response.strip(), evidence)
            
            # Normalize for EM evaluation
            normalized = normalize_answer(processed_answer)
            
            logger.info(f"LLM answer: '{normalized}' (raw: '{response[:50]}...')")
            return normalized
            
        except Exception as e:
            logger.error(f"LLM answering failed: {e}")
            return ""
    
    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """
        Format evidence into structured text.
        
        Format:
        [Document: {doc_title}]
        {sentence_text}
        
        [Document: {doc_title}]
        {sentence_text}
        """
        formatted_parts = []
        
        for item in evidence:
            doc_title = item.get("doc_title", "Unknown")
            text = item.get("text", "")
            
            if text:
                formatted_parts.append(f"[Document: {doc_title}]\n{text}")
        
        return "\n\n".join(formatted_parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for constrained QA."""
        return (
            "You must answer using ONLY the provided evidence. "
            "Do not use outside knowledge. "
            "If the evidence is insufficient, answer 'unknown'. "
            "For yes/no questions, answer exactly 'yes' or 'no'. "
            "Output only the answer, nothing else."
        )
    
    def _build_prompt(self, question: str, formatted_evidence: str) -> str:
        """Build user prompt with evidence and question."""
        return (
            f"Evidence:\n{formatted_evidence}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    
    def _append_unit_if_numeric(self, answer: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Append unit to numeric answer if found in evidence.
        
        For "how many" questions, LLM often returns just the number.
        This method searches evidence for units adjacent to the number
        and appends them to match gold answer format.
        
        Args:
            answer: LLM's raw answer
            evidence: Retrieved evidence items
        
        Returns:
            Answer with unit appended if found in evidence
        """
        import re
        
        # Only process if answer looks numeric
        answer_stripped = answer.strip()
        if not re.search(r'\d', answer_stripped):
            return answer
        
        # Common unit words for HotpotQA
        unit_words = [
            'seated', 'people', 'members', 'episodes', 'seasons',
            'years', 'months', 'days', 'hours', 'minutes',
            'meters', 'feet', 'miles', 'kilometers',
            'pounds', 'kilograms', 'tons',
            'dollars', 'euros', 'pounds'
        ]
        
        # Extract just the digits from answer for matching
        answer_digits = re.sub(r'[^\d]', '', answer_stripped)
        if not answer_digits:
            return answer
        
        # Search evidence for number + unit pattern
        for item in evidence:
            text = item.get("metadata", {}).get("text", "") or item.get("text", "")
            if not text:
                continue
            
            # Look for the number (with or without commas) followed by a unit
            # Create a flexible pattern: match digits with optional commas
            # Example: answer="3677" should match "3,677 seated" in text
            
            # Build pattern that allows commas between digits
            # For "3677", match "3,?6,?7,?7" (commas optional)
            flexible_digits = ',?'.join(answer_digits)
            
            # Try multiple pattern variations
            patterns = [
                rf'\b{flexible_digits}\s+(\w+)\b',  # "3,677 seated"
                rf'\({flexible_digits}\s+(\w+)\)',  # "(3,677 seated)"
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    potential_unit = match.group(1).lower()
                    if potential_unit in unit_words:
                        # Found a unit! Append it
                        logger.info(f"Appending unit '{potential_unit}' to answer '{answer_stripped}'")
                        return f"{answer_stripped} {potential_unit}"
        
        # No unit found, return original
        return answer


# Singleton instance
_llm_answerer: Optional[LLMAnswerer] = None


def get_llm_answerer() -> LLMAnswerer:
    """Get or create singleton LLMAnswerer instance."""
    global _llm_answerer
    if _llm_answerer is None:
        _llm_answerer = LLMAnswerer()
    return _llm_answerer
