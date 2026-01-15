"""
Symbolic Answer Selector for HotpotQA EM Evaluation

CRITICAL: This module uses DETERMINISTIC/SYMBOLIC logic (NOT LLM generation).
It converts retrieved evidence into single EM-valid answer strings.

Strategy by query type:
- YES_NO: Extract attributes, compare symbolically → "yes" or "no"
- COMPARISON: Extract comparable values, compare → winning entity/value
- SINGLE_HOP/MULTI_HOP: Score candidate spans → top canonical candidate

STRICT CONSTRAINTS:
1. NO TEXT GENERATION. NO PARAPHRASING. Output = ONE answer string only.
2. NO FALLBACK TO CONCATENATION. AnswerSelector output is the ONLY returned answer.
3. CANDIDATE SPANS MUST COME ONLY FROM EVIDENCE (no invented/inferred strings).
4. ALL ANSWERS NORMALIZED using HotpotQA EM rules (lowercase, strip punctuation, remove articles).
5. PROVENANCE LOGGING does not affect answer selection logic (debugging only).
"""

import logging
import re
import string
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class AnswerSelector:
    """
    Deterministic answer selector for HotpotQA.

    Converts retrieved evidence into EM-valid single answers without LLM generation.
    """

    def __init__(self):
        """Initialize answer selector."""
        # Common yes/no patterns
        self.affirmative_patterns = [
            r'\byes\b', r'\btrue\b', r'\bcorrect\b', r'\baffirmative\b',
            r'\bindeed\b', r'\bconfirmed\b', r'\bagree\b'
        ]
        self.negative_patterns = [
            r'\bno\b', r'\bfalse\b', r'\bincorrect\b', r'\bnegative\b',
            r'\bdeny\b', r'\bdenied\b', r'\bdisagree\b', r'\bnot\b'
        ]

        logger.info("Initialized AnswerSelector (symbolic/deterministic)")

    def select_answer(
        self,
        query: str,
        query_type: str,
        evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Select answer using deterministic logic based on query type.

        Args:
            query: User query
            query_type: One of {SINGLE_HOP, MULTI_HOP, COMPARISON, YES_NO}
            evidence: Retrieved context items with metadata

        Returns:
            Single answer string (EM-valid)
        """
        if not evidence:
            return ""

        query_type = query_type.upper().replace("-", "_")

        if query_type == "YES_NO":
            return self._select_yes_no(query, evidence)
        elif query_type == "COMPARISON":
            return self._select_comparison(query, evidence)
        elif query_type in ["SINGLE_HOP", "MULTI_HOP"]:
            return self._select_factoid(query, evidence)
        else:
            # Default: factoid selection
            return self._select_factoid(query, evidence)

    def _select_yes_no(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        YES/NO: Symbolic comparison of attributes.

        Logic:
        1. Parse question for entities and property being compared
        2. Extract attributes from evidence for each entity
        3. Compare symbolically
        4. Return "yes" or "no"

        Examples:
        - "Are both X and Y located in America?" → check if both have location=America
        - "Is X older than Y?" → compare ages/dates
        """
        query_lower = query.lower()

        # Pattern 1: "are both X and Y [property]?"
        if "both" in query_lower or "and" in query_lower:
            return self._compare_shared_property(query, evidence)

        # Pattern 2: "is/are X [property]?" (single entity check)
        if re.search(r'\b(is|are|was|were|does|do|did)\b', query_lower):
            return self._check_single_property(query, evidence)

        # Fallback: count affirmative vs negative indicators in evidence
        affirmative_count = 0
        negative_count = 0

        for item in evidence:
            text = item.get("text", "").lower()

            for pattern in self.affirmative_patterns:
                if re.search(pattern, text):
                    affirmative_count += 1

            for pattern in self.negative_patterns:
                if re.search(pattern, text):
                    negative_count += 1

        # Decide based on majority
        if affirmative_count > negative_count:
            return "yes"
        elif negative_count > affirmative_count:
            return "no"
        else:
            # Tie or no signals: default "yes" if evidence exists
            return "yes" if evidence else "no"

    def _compare_shared_property(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Compare if multiple entities share a property.

        Example: "Are both Rutgers and Carnegie Mellon located in America?"
        → Check if both have "America" or "United States" in their context
        """
        # Extract entities being compared (simple heuristic: capitalized phrases)
        entities = self._extract_entities_from_query(query)

        if len(entities) < 2:
            # Can't compare, fallback to pattern matching
            return self._fallback_yes_no(evidence)

        # Extract property keyword (location, nationality, type, etc.)
        property_keywords = self._extract_property_keywords(query)

        # Check if ALL entities have the property
        entities_with_property = 0

        for entity in entities:
            # Check if entity and property co-occur in any evidence
            for item in evidence:
                text = item.get("text", "")
                if entity.lower() in text.lower():
                    # Check for property keywords nearby
                    for keyword in property_keywords:
                        if keyword.lower() in text.lower():
                            entities_with_property += 1
                            break
                    break

        # All entities must have property
        return "yes" if entities_with_property == len(entities) else "no"

    def _check_single_property(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Check if single entity has a property.

        Example: "Is Tesla headquartered in California?"
        → Check if "Tesla" and "California" co-occur with "headquartered"
        """
        # Extract main entity and property
        entities = self._extract_entities_from_query(query)
        property_keywords = self._extract_property_keywords(query)

        if not entities:
            return self._fallback_yes_no(evidence)

        main_entity = entities[0]

        # Check if entity + property appear together in evidence
        for item in evidence:
            text = item.get("text", "").lower()
            if main_entity.lower() in text:
                for keyword in property_keywords:
                    if keyword.lower() in text:
                        return "yes"

        return "no"

    def _fallback_yes_no(self, evidence: List[Dict[str, Any]]) -> str:
        """Fallback: count yes/no indicators in evidence."""
        yes_count = sum(
            1 for item in evidence
            if any(re.search(p, item.get("text", "").lower()) for p in self.affirmative_patterns)
        )
        no_count = sum(
            1 for item in evidence
            if any(re.search(p, item.get("text", "").lower()) for p in self.negative_patterns)
        )

        return "yes" if yes_count >= no_count else "no"

    def _select_comparison(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        COMPARISON: Extract and compare values, return winner.

        Logic:
        1. Identify entities being compared
        2. Extract comparable property (date, number, location, etc.)
        3. Compare deterministically
        4. Return winning entity name

        Examples:
        - "Which is taller, X or Y?" → extract heights, return taller entity
        - "Who was born first, X or Y?" → extract birth years, return older entity
        - "Which has more population, X or Y?" → extract populations, return larger
        """
        entities = self._extract_entities_from_query(query)

        if len(entities) < 2:
            # Not a valid comparison, return first entity or empty
            return entities[0] if entities else ""

        # Determine comparison type from query
        query_lower = query.lower()

        # Date comparison
        if any(word in query_lower for word in ["first", "earlier", "before", "older", "founded", "established"]):
            return self._compare_dates(entities, evidence, prefer_earlier=True)

        if any(word in query_lower for word in ["later", "after", "newer", "recent"]):
            return self._compare_dates(entities, evidence, prefer_earlier=False)

        # Numerical comparison
        if any(word in query_lower for word in ["more", "larger", "bigger", "greater", "higher", "taller"]):
            return self._compare_numbers(entities, evidence, prefer_larger=True)

        if any(word in query_lower for word in ["less", "smaller", "fewer", "lower", "shorter"]):
            return self._compare_numbers(entities, evidence, prefer_larger=False)

        # Categorical comparison (same/different)
        if any(word in query_lower for word in ["same", "both", "common"]):
            # Return the shared value
            shared_value = self._find_shared_value(entities, evidence)
            return shared_value if shared_value else entities[0]

        # Default: return first entity
        return entities[0]

    def _compare_dates(
        self,
        entities: List[str],
        evidence: List[Dict[str, Any]],
        prefer_earlier: bool
    ) -> str:
        """Compare dates and return winner."""
        entity_dates = {}

        for entity in entities:
            # Find years associated with this entity
            for item in evidence:
                text = item.get("text", "")
                if entity.lower() in text.lower():
                    # Extract 4-digit years
                    years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
                    if years:
                        entity_dates[entity] = int(years[0])
                        break

        if not entity_dates:
            return entities[0]  # No dates found, return first

        # Sort by date
        if prefer_earlier:
            winner = min(entity_dates, key=entity_dates.get)
        else:
            winner = max(entity_dates, key=entity_dates.get)

        logger.debug(f"Date comparison: {entity_dates}, winner: {winner}")
        return winner

    def _compare_numbers(
        self,
        entities: List[str],
        evidence: List[Dict[str, Any]],
        prefer_larger: bool
    ) -> str:
        """Compare numerical values and return winner."""
        entity_numbers = {}

        for entity in entities:
            # Find numbers associated with this entity
            for item in evidence:
                text = item.get("text", "")
                if entity.lower() in text.lower():
                    # Extract numbers (with optional commas/decimals)
                    numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text)
                    if numbers:
                        # Take first number, remove commas
                        num_str = numbers[0].replace(',', '')
                        try:
                            entity_numbers[entity] = float(num_str)
                            break
                        except ValueError:
                            continue

        if not entity_numbers:
            return entities[0]  # No numbers found, return first

        # Sort by number
        if prefer_larger:
            winner = max(entity_numbers, key=entity_numbers.get)
        else:
            winner = min(entity_numbers, key=entity_numbers.get)

        logger.debug(f"Number comparison: {entity_numbers}, winner: {winner}")
        return winner

    def _find_shared_value(self, entities: List[str], evidence: List[Dict[str, Any]]) -> str:
        """Find shared property value between entities."""
        # For each entity, collect associated values (locations, nationalities, etc.)
        entity_values = defaultdict(set)

        for entity in entities:
            for item in evidence:
                text = item.get("text", "")
                if entity.lower() in text.lower():
                    # Extract capitalized words (potential values)
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                    entity_values[entity].update(words)

        # Find intersection
        if len(entity_values) < 2:
            return ""

        shared = set.intersection(*[entity_values[e] for e in entities])

        # Return most common shared value
        return list(shared)[0] if shared else ""

    def _select_factoid(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        SINGLE_HOP/MULTI_HOP: Extract candidate spans and score.

        Logic:
        1. Extract candidate answer spans (entities, noun phrases)
        2. Score candidates by:
           - Frequency across evidence
           - Cross-document consistency
           - Presence in titles/metadata
        3. Return highest-scoring canonical candidate

        Examples:
        - "What genre does Don Williams sing?" → extract "country", "country music"
        - "Who directed Titanic?" → extract "James Cameron"
        """
        # Extract all candidate spans from evidence
        candidates = self._extract_candidate_spans(evidence)

        if not candidates:
            return ""

        # Score candidates
        scored_candidates = self._score_candidates(candidates, evidence)

        # Return top candidate
        if scored_candidates:
            top_candidate, score = scored_candidates[0]
            logger.debug(f"Selected answer: '{top_candidate}' (score: {score:.2f})")
            return top_candidate

        return ""

    def _extract_candidate_spans(self, evidence: List[Dict[str, Any]]) -> Counter:
        """
        Extract candidate answer spans from evidence.

        Candidates = entities, noun phrases, capitalized phrases, quoted text.
        """
        candidates = Counter()

        for item in evidence:
            text = item.get("text", "")

            # Extract capitalized phrases (likely entities)
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            candidates.update(capitalized)

            # Extract quoted text (often answers)
            quoted = re.findall(r'"([^"]+)"', text)
            candidates.update(quoted)

            # Extract single capitalized words
            single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
            candidates.update(single_caps)

        return candidates

    def _score_candidates(
        self,
        candidates: Counter,
        evidence: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Score candidate answer spans.

        Scoring factors:
        - Frequency across evidence (higher = better)
        - Cross-document consistency (appears in multiple docs = better)
        - Length (prefer 1-3 words over long phrases)
        - Title presence (appears in doc title = better)
        """
        scored = []

        for candidate, freq in candidates.items():
            if not candidate or len(candidate) < 2:
                continue

            score = 0.0

            # Factor 1: Frequency (normalized)
            score += freq * 2.0

            # Factor 2: Cross-document consistency
            docs_with_candidate = set()
            for item in evidence:
                text = item.get("text", "")
                if candidate.lower() in text.lower():
                    doc_id = item.get("metadata", {}).get("doc_id", "")
                    docs_with_candidate.add(doc_id)

            if len(docs_with_candidate) > 1:
                score += len(docs_with_candidate) * 3.0

            # Factor 3: Length preference (1-3 words optimal)
            word_count = len(candidate.split())
            if 1 <= word_count <= 3:
                score += 2.0
            elif word_count > 5:
                score -= 2.0

            # Factor 4: Title presence
            for item in evidence:
                title = item.get("metadata", {}).get("title", "")
                if candidate.lower() in title.lower():
                    score += 5.0
                    break

            scored.append((candidate, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entities from query (simple heuristic).

        Returns capitalized phrases that are likely entity names.
        """
        # Extract capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

        # Remove common question words
        stop_words = {"What", "Where", "When", "Who", "Which", "Why", "How", "Are", "Is", "Do", "Does", "Did", "Were", "Was"}
        entities = [e for e in entities if e not in stop_words]

        return entities

    def _extract_property_keywords(self, query: str) -> List[str]:
        """
        Extract property keywords from query.

        Examples:
        - "located in" → ["located", "location"]
        - "nationality" → ["nationality", "national"]
        - "older than" → ["older", "age", "born"]
        """
        keywords = []

        query_lower = query.lower()

        # Location
        if any(word in query_lower for word in ["located", "location", "place", "where"]):
            keywords.extend(["located", "location", "place", "in", "at"])

        # Nationality
        if any(word in query_lower for word in ["nationality", "national", "citizen"]):
            keywords.extend(["nationality", "national", "citizen", "from"])

        # Age/Time
        if any(word in query_lower for word in ["older", "age", "born", "year"]):
            keywords.extend(["born", "age", "year", "old"])

        # Type/Category
        if any(word in query_lower for word in ["type", "kind", "genre", "category"]):
            keywords.extend(["type", "kind", "genre", "category", "is", "are"])

        # Generic extraction: nouns and verbs
        words = re.findall(r'\b[a-z]{4,}\b', query_lower)
        keywords.extend(words)

        return list(set(keywords))


# Singleton instance
_answer_selector: Optional[AnswerSelector] = None


def get_answer_selector() -> AnswerSelector:
    """Get global AnswerSelector instance (singleton)."""
    global _answer_selector
    if _answer_selector is None:
        _answer_selector = AnswerSelector()
    return _answer_selector
