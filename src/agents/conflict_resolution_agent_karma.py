"""
KARMA Conflict Resolution Agent with LLM Debate

Based on KARMA: "CRA resolves contradictions through LLM-based debate mechanisms"
Reduces contradictory edges by 18.6%
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class KARMAConflictResolutionAgent:
    """
    KARMA-inspired Conflict Resolution with LLM debate.

    Identifies and resolves contradicting relationships through debate.
    """

    def __init__(self, llm_client):
        """
        Initialize KARMA Conflict Resolution Agent.

        Args:
            llm_client: UnifiedLLMClient instance
        """
        self.llm_client = llm_client
        self.contradictory_pairs = self._load_contradictory_pairs()
        logger.info("Initialized KARMAConflictResolutionAgent (LLM debate enabled)")

    def _load_contradictory_pairs(self) -> Dict[str, List[str]]:
        """
        Load known contradictory relationship pairs.

        KARMA approach: Domain knowledge of contradictions
        (e.g., "inhibits" vs "activates" typically contradict)
        """
        return {
            "DIRECTED": ["NOT_DIRECTED"],
            "CREATED": ["NOT_CREATED", "DESTROYED"],
            "FOUNDED": ["DISSOLVED", "DESTROYED"],
            "BORN_IN": ["NOT_BORN_IN"],
            "DIED_IN": ["NOT_DIED_IN"],
            "LOCATED_IN": ["NOT_LOCATED_IN"],
            "MEMBER_OF": ["NOT_MEMBER_OF", "LEFT"],
            "OWNS": ["DOES_NOT_OWN", "SOLD"],
            "SUPPORTS": ["OPPOSES"],
            "INCREASES": ["DECREASES"],
            "ACTIVATES": ["INHIBITS", "DEACTIVATES"],
        }

    def are_contradictory(self, rel1: Dict[str, Any], rel2: Dict[str, Any]) -> bool:
        """
        Check if two relationships are potentially contradictory.

        Args:
            rel1: First relationship
            rel2: Second relationship

        Returns:
            True if they might contradict
        """
        # Must involve same entities
        if (rel1['source'] != rel2['source'] or
            rel1['target'] != rel2['target']):
            return False

        # Check if relationship types contradict
        type1 = rel1['type']
        type2 = rel2['type']

        # Same type with different directions could contradict
        if type1 == type2:
            return False  # Same relationship type = agreement, not contradiction

        # Check known contradictory pairs
        if type1 in self.contradictory_pairs:
            if type2 in self.contradictory_pairs[type1]:
                return True

        if type2 in self.contradictory_pairs:
            if type1 in self.contradictory_pairs[type2]:
                return True

        return False

    def resolve_contradiction_with_debate(
        self,
        rel1: Dict[str, Any],
        rel2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve contradiction using LLM debate.

        KARMA approach: LLMCRA(t, t') → {Agree, Contradict, Ambiguous}

        Args:
            rel1: First relationship
            rel2: Second relationship

        Returns:
            Resolution decision: {"action": "keep_rel1" | "keep_rel2" | "keep_both" | "discard_both",
                                  "reasoning": "..."}
        """
        prompt = f"""Two relationships contradict. Resolve this contradiction.

Relationship 1:
  {rel1['source']} --[{rel1['type']}]--> {rel1['target']}
  Confidence: {rel1.get('confidence', 'N/A')}
  Evidence: {rel1.get('evidence', 'N/A')}

Relationship 2:
  {rel2['source']} --[{rel2['type']}]--> {rel2['target']}
  Confidence: {rel2.get('confidence', 'N/A')}
  Evidence: {rel2.get('evidence', 'N/A')}

Decision options:
1. keep_rel1: Relationship 1 is correct, discard 2
2. keep_rel2: Relationship 2 is correct, discard 1
3. keep_both: Both are correct (not actually contradictory, e.g., different time periods)
4. discard_both: Both are incorrect or too uncertain

Consider:
- Evidence quality
- Confidence scores
- Temporal context (same time vs different times)
- Specificity (specific beats general)

Return JSON:
{{
  "action": "keep_rel1" | "keep_rel2" | "keep_both" | "discard_both",
  "reasoning": "explanation"
}}
"""

        try:
            response = self.llm_client.generate_json(prompt, temperature=0.0, max_tokens=500)

            action = response.get("action", "keep_both")  # Default to keeping both
            reasoning = response.get("reasoning", "")

            logger.info(f"Debate resolved: {action} - {reasoning[:50]}")

            return {
                "action": action,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error(f"Debate resolution failed: {e}")
            return {"action": "keep_both", "reasoning": "Debate failed, keeping both"}

    def resolve_conflicts(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Resolve all contradictions in relationship list using LLM debate.

        Args:
            relationships: List of relationships

        Returns:
            Tuple of (filtered_relationships, conflicts_resolved_count)
        """
        # Group relationships by entity pair
        pair_groups = defaultdict(list)
        for rel in relationships:
            key = (rel['source'], rel['target'])
            pair_groups[key].append(rel)

        resolved_relationships = []
        conflicts_resolved = 0

        # Check each pair for contradictions
        for (src, tgt), rels in pair_groups.items():
            if len(rels) == 1:
                # No conflict possible
                resolved_relationships.append(rels[0])
                continue

            # Check for contradictions
            kept_rels = []
            discarded = set()

            for i, rel1 in enumerate(rels):
                if i in discarded:
                    continue

                has_contradiction = False

                for j, rel2 in enumerate(rels[i+1:], start=i+1):
                    if j in discarded:
                        continue

                    if self.are_contradictory(rel1, rel2):
                        has_contradiction = True
                        conflicts_resolved += 1

                        logger.info(f"Contradiction detected: {rel1['type']} vs {rel2['type']}")

                        # Resolve with LLM debate
                        resolution = self.resolve_contradiction_with_debate(rel1, rel2)

                        if resolution['action'] == 'keep_rel1':
                            discarded.add(j)
                        elif resolution['action'] == 'keep_rel2':
                            discarded.add(i)
                            break  # rel1 is discarded, move to next
                        elif resolution['action'] == 'discard_both':
                            discarded.add(i)
                            discarded.add(j)
                            break
                        # else keep_both: do nothing

                if i not in discarded:
                    kept_rels.append(rel1)

            resolved_relationships.extend(kept_rels)

        logger.info(f"Resolved {conflicts_resolved} conflicts, kept {len(resolved_relationships)}/{len(relationships)} relationships")

        return resolved_relationships, conflicts_resolved
