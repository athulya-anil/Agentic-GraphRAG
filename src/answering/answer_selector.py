"""
Symbolic Answer Selector for HotpotQA EM Evaluation

CRITICAL: This module uses DETERMINISTIC/SYMBOLIC logic (NOT LLM generation).
It converts retrieved evidence into single EM-valid answer strings.

Strategy by query type:
- YES_NO: Entity + constraint token co-mention → "yes" or "no"
- COMPARISON: Branch on question form → YES_NO or entity selection
- SINGLE_HOP/MULTI_HOP: KG node-name candidates → priority ranking

STRICT CONSTRAINTS:
1. NO TEXT GENERATION. NO PARAPHRASING. Output = ONE answer string only.
2. NO FALLBACK TO CONCATENATION.
3. CANDIDATES COME FROM KG NODE NAMES (structural, not regex heuristics).
4. ALL ANSWERS NORMALIZED using HotpotQA EM rules.
5. YES/NO requires entity + constraint co-mention (not just entity presence).

Author: Agentic GraphRAG Team
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
    Uses structural entity sources (KG nodes) instead of regex heuristics.
    """

    def __init__(self):
        """Initialize answer selector."""
        # Cache for KG node names (loaded on first use)
        self._kg_node_names: Optional[Set[str]] = None

        logger.info("Initialized AnswerSelector (symbolic/deterministic)")

    # =========================================================================
    # MAIN DISPATCH
    # =========================================================================

    def select_answer(
        self,
        query: str,
        query_type: str,
        evidence: List[Dict[str, Any]],
        required_type: str = "unknown"
    ) -> str:
        """
        Select answer using deterministic logic based on query type.

        Args:
            query: User query
            query_type: One of {SINGLE_HOP, MULTI_HOP, COMPARISON, YES_NO}
            evidence: Retrieved context items with provenance
            required_type: Expected answer type (e.g. "number", "person")

        Returns:
            Single answer string (EM-valid, normalized)
        """
        if not evidence:
            logger.warning("No evidence provided to AnswerSelector")
            return ""

        query_type = query_type.upper().replace("-", "_")

        # Log provenance for debugging (does not affect answer)
        self._log_provenance(query, evidence)

        if query_type == "YES_NO":
            answer = self._select_yes_no(query, evidence)
        elif query_type == "COMPARISON":
            answer = self._select_comparison(query, evidence)
        elif query_type in ["SINGLE_HOP", "MULTI_HOP"]:
            answer = self._select_factoid(query, evidence, required_type=required_type)
        else:
            # Default: factoid selection
            answer = self._select_factoid(query, evidence, required_type=required_type)

        logger.info(f"Selected answer: '{answer}' for query_type: {query_type}")
        return answer

    # =========================================================================
    # YES/NO SELECTION (FIX 1: Entity + Constraint Co-mention)
    # =========================================================================

    def _select_yes_no(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        YES/NO = symbolic set satisfaction with CONSTRAINT CO-MENTION.

        Rule:
        - YES if: ALL query entities have ≥1 sentence where
          (entity appears) AND (≥1 constraint token from query appears)
        - NO otherwise

        This prevents false positives from mere entity mention.

        Args:
            query: User query
            evidence: Retrieved evidence

        Returns:
            "yes" or "no"
        """
        # Get entities from query (structural, via QueryParserAgent)
        query_entities = self._get_query_entities_structural(query)
        constraint_tokens = self._extract_constraint_tokens(query)

        logger.debug(f"YES/NO - entities: {query_entities}, constraints: {constraint_tokens}")

        # WARN if no constraint tokens found - helps debug false "no"
        if not constraint_tokens:
            logger.warning(
                f"YES/NO: No constraint tokens extracted from query: '{query[:60]}...' "
                "This may cause false 'no' answers."
            )

        if len(query_entities) == 0:
            # No entities detected - check if any evidence exists
            logger.warning(f"YES/NO: No entities found in query: '{query[:60]}...'")
            return "yes" if evidence else "no"

        # Check: does each entity have co-mention with constraint in SAME sentence?
        entities_with_constrained_support = set()

        for entity in query_entities:
            entity_lower = entity.lower()
            for item in evidence:
                sent_text = item.get("text", "").lower()

                # Entity must appear in sentence
                if entity_lower not in sent_text:
                    continue

                # At least one constraint token must co-appear in SAME sentence
                has_constraint = any(
                    token in sent_text for token in constraint_tokens
                )

                if has_constraint:
                    entities_with_constrained_support.add(entity)
                    logger.debug(
                        f"Entity '{entity}' has constrained support in: "
                        f"doc_id={item.get('doc_id')}, sent_id={item.get('sent_id')}"
                    )
                    break

        # Symbolic satisfaction: ALL entities must have constrained support
        if entities_with_constrained_support == query_entities:
            return "yes"
        else:
            return "no"

    def _extract_constraint_tokens(self, query: str) -> Set[str]:
        """
        Extract constraint tokens from query.

        Constraint tokens are content words that must co-occur with entities
        for a YES answer. Keep list MINIMAL to avoid false negatives.

        Examples:
        - "Are both X and Y located in America?" → {"located", "america"}
        - "Were X and Y born in the same country?" → {"born", "country"}

        Args:
            query: User query

        Returns:
            Set of constraint tokens (lowercase)
        """
        query_lower = query.lower()
        constraint_tokens = set()

        # MINIMAL constraint set - only HotpotQA-relevant relation words
        # Keep small to avoid false negatives
        minimal_constraints = {
            'located', 'born', 'directed', 'written', 'capital',
            'country', 'nationality', 'founded', 'produced', 'american',
            'genre', 'type', 'member', 'played', 'starred'
        }

        # Only add constraints that appear in the query
        for constraint in minimal_constraints:
            if constraint in query_lower:
                constraint_tokens.add(constraint)

        # Also extract location/nationality values from query (e.g., "America", "British")
        # These are often the property values being checked
        location_patterns = re.findall(r'\b(america|usa|united states|britain|british|'
                                       r'france|french|germany|german|japan|japanese|'
                                       r'canada|canadian|australia|australian)\b', query_lower)
        constraint_tokens.update(location_patterns)

        return constraint_tokens

    # =========================================================================
    # COMPARISON SELECTION (FIX 3: Branch on Question Form)
    # =========================================================================

    def _select_comparison(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        COMPARISON: Branch based on question form.

        - "Were/Are/Is/Do/Does..." → treat as YES_NO (answer: "yes"/"no")
        - "Which/Who/What..." → select winning entity

        Args:
            query: User query
            evidence: Retrieved evidence

        Returns:
            "yes"/"no" OR entity name (normalized)
        """
        query_lower = query.lower().strip()

        # Detect question form by first word
        yes_no_starters = ('were', 'are', 'is', 'was', 'do', 'does', 'did', 'has', 'have', 'had')
        selection_starters = ('which', 'who', 'what', 'whose', 'whom')

        words = query_lower.split()
        first_word = words[0] if words else ""

        # Branch 1: Yes/No form comparison ("Were X and Y the same nationality?")
        if first_word in yes_no_starters:
            logger.debug("COMPARISON detected as YES/NO form")
            return self._select_yes_no(query, evidence)

        # Branch 2: Selection form comparison ("Which is taller, X or Y?")
        if first_word in selection_starters:
            logger.debug("COMPARISON detected as selection form")
            return self._select_comparison_winner(query, evidence)

        # Default: try selection
        return self._select_comparison_winner(query, evidence)

    def _select_comparison_winner(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Select winning entity for "Which/Who/What" comparison queries.

        Args:
            query: User query
            evidence: Retrieved evidence

        Returns:
            Winning entity name (normalized)
        """
        entities = list(self._get_query_entities_structural(query))

        if len(entities) == 0:
            # Fallback: try to get candidates from evidence
            candidates = self._get_candidates_from_evidence(evidence, query)
            if candidates:
                # Return most frequent
                freq = Counter()
                for item in evidence:
                    text = item.get("text", "").lower()
                    for c in candidates:
                        if c.lower() in text:
                            freq[c] += 1
                if freq:
                    return self._normalize_answer(freq.most_common(1)[0][0])
            return ""
        if len(entities) == 1:
            return self._normalize_answer(entities[0])

        query_lower = query.lower()

        # Date comparison (earlier/first vs later/recent)
        if any(w in query_lower for w in ["first", "earlier", "older", "before"]):
            return self._compare_dates(entities, evidence, prefer_earlier=True)
        if any(w in query_lower for w in ["later", "after", "newer", "recent"]):
            return self._compare_dates(entities, evidence, prefer_earlier=False)

        # Numeric comparison (more/larger vs less/smaller)
        if any(w in query_lower for w in ["more", "larger", "greater", "higher", "taller", "longer", "bigger"]):
            return self._compare_numbers(entities, evidence, prefer_larger=True)
        if any(w in query_lower for w in ["less", "smaller", "fewer", "lower", "shorter"]):
            return self._compare_numbers(entities, evidence, prefer_larger=False)

        # Default: return entity with most evidence support
        entity_support = Counter()
        for entity in entities:
            entity_lower = entity.lower()
            for item in evidence:
                if entity_lower in item.get("text", "").lower():
                    entity_support[entity] += 1

        if entity_support:
            winner = entity_support.most_common(1)[0][0]
            return self._normalize_answer(winner)

        return self._normalize_answer(entities[0])

    def _compare_dates(
        self,
        entities: List[str],
        evidence: List[Dict[str, Any]],
        prefer_earlier: bool
    ) -> str:
        """Compare dates and return winner."""
        entity_dates = {}

        for entity in entities:
            entity_lower = entity.lower()
            for item in evidence:
                text = item.get("text", "")
                if entity_lower in text.lower():
                    # Extract 4-digit years
                    years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
                    if years:
                        entity_dates[entity] = int(years[0])
                        break

        if not entity_dates:
            return self._normalize_answer(entities[0])

        if prefer_earlier:
            winner = min(entity_dates, key=entity_dates.get)
        else:
            winner = max(entity_dates, key=entity_dates.get)

        logger.debug(f"Date comparison: {entity_dates}, winner: {winner}")
        return self._normalize_answer(winner)

    def _compare_numbers(
        self,
        entities: List[str],
        evidence: List[Dict[str, Any]],
        prefer_larger: bool
    ) -> str:
        """Compare numerical values and return winner."""
        entity_numbers = {}

        for entity in entities:
            entity_lower = entity.lower()
            for item in evidence:
                text = item.get("text", "")
                if entity_lower in text.lower():
                    # Extract numbers (with optional commas/decimals)
                    numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text)
                    if numbers:
                        num_str = numbers[0].replace(',', '')
                        try:
                            entity_numbers[entity] = float(num_str)
                            break
                        except ValueError:
                            continue

        if not entity_numbers:
            return self._normalize_answer(entities[0])

        if prefer_larger:
            winner = max(entity_numbers, key=entity_numbers.get)
        else:
            winner = min(entity_numbers, key=entity_numbers.get)

        logger.debug(f"Number comparison: {entity_numbers}, winner: {winner}")
        return self._normalize_answer(winner)

    # =========================================================================
    # FACTOID SELECTION (FIX 2: Use KG Node-Name Set)
    # =========================================================================

    def _select_factoid(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        required_type: str = "unknown"
    ) -> str:
        """
        SINGLE_HOP/MULTI_HOP: Select answer from evidence.

        Candidate pools:
        1. Entity candidates (KG node names)
        2. Value candidates (numbers, genres, quoted text)

        Priority order (simple, debuggable):
        1. Appears in ≥2 documents
        2. Appears as graph node in evidence
        3. Highest frequency in evidence

        Args:
            query: User query
            evidence: Retrieved evidence

        Returns:
            Top candidate answer (normalized)
        """
        # Get candidates from BOTH entity and value pools
        candidates = self._get_candidates_from_evidence(evidence, query)

        # Fix 3: Filter by required type
        candidates = self._filter_by_type(candidates, required_type)

        if not candidates:
            logger.warning("No candidates found for factoid selection")
            return ""

        # Track metrics per candidate
        candidate_docs = defaultdict(set)
        candidate_freq = Counter()
        graph_nodes_in_evidence = set()

        for item in evidence:
            text = item.get("text", "").lower()
            doc_id = item.get("doc_id", 0)
            source = item.get("source", "")
            meta = item.get("metadata", {})

            # Track which candidates appear in this sentence
            for candidate in candidates:
                if candidate.lower() in text:
                    candidate_docs[candidate].add(doc_id)
                    candidate_freq[candidate] += 1

            # Track graph nodes from evidence
            if source == "graph" and "node" in meta:
                node_name = meta["node"].get("name", "")
                if node_name:
                    graph_nodes_in_evidence.add(node_name)

        # Filter out query entities from candidates (answer shouldn't repeat question)
        query_entities = self._get_query_entities_structural(query)
        query_entity_lower = {e.lower() for e in query_entities}
        candidates = {c for c in candidates if c.lower() not in query_entity_lower}

        if not candidates:
            logger.warning("All candidates filtered as query entities")
            return ""

        # Priority ranking (simple, no tuned weights)
        def rank_candidate(c):
            multi_doc = len(candidate_docs.get(c, set())) >= 2
            is_graph_node = c in graph_nodes_in_evidence
            freq = candidate_freq.get(c, 0)
            # Priority tuple: (multi_doc, is_graph, freq)
            return (multi_doc, is_graph_node, freq)

        ranked = sorted(candidates, key=rank_candidate, reverse=True)

        if ranked:
            top_candidate = ranked[0]
            logger.debug(
                f"Factoid selection: top='{top_candidate}', "
                f"multi_doc={len(candidate_docs.get(top_candidate, set())) >= 2}, "
                f"is_graph={top_candidate in graph_nodes_in_evidence}, "
                f"freq={candidate_freq.get(top_candidate, 0)}"
            )
            return self._normalize_answer(top_candidate)

        return ""

    # =========================================================================
    # ENTITY EXTRACTION (STRUCTURAL, NOT REGEX)
    # =========================================================================

    def _get_kg_node_names(self) -> Set[str]:
        """
        Get all entity names from knowledge graph (cached).

        Used for candidate extraction when metadata is incomplete.
        This is the primary source of candidates (structural, not heuristic).

        Returns:
            Set of all KG node names
        """
        if self._kg_node_names is not None:
            return self._kg_node_names

        try:
            from ..graph import get_neo4j_manager
            neo4j = get_neo4j_manager()

            # Query all node names
            result = neo4j.execute_query(
                "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name"
            )
            self._kg_node_names = {r["name"] for r in result if r["name"]}
            logger.info(f"Loaded {len(self._kg_node_names)} KG node names")
        except Exception as e:
            logger.warning(f"Failed to load KG node names: {e}")
            self._kg_node_names = set()

        return self._kg_node_names

    def _get_candidates_from_evidence(
        self,
        evidence: List[Dict[str, Any]],
        query: str = ""
    ) -> Set[str]:
        """
        Extract candidate answers from evidence.

        Two candidate pools (both symbolic, no LLM):
        1. ENTITY candidates: KG node names (high precision)
        2. VALUE candidates: spans from evidence (numbers, quoted text, descriptors)

        Args:
            evidence: Retrieved evidence
            query: Original query (for intent-based extraction)

        Returns:
            Set of candidate answer strings
        """
        candidates = set()

        # =====================================================================
        # POOL 1: Entity candidates (KG-backed)
        # =====================================================================

        # Source 1a: Graph nodes from retrieval metadata
        for item in evidence:
            meta = item.get("metadata", {})
            if "node" in meta:
                node_name = meta["node"].get("name", "")
                if node_name:
                    candidates.add(node_name)

        # Source 1b: KG node names that appear in evidence text
        kg_nodes = self._get_kg_node_names()
        for item in evidence:
            text = item.get("text", "").lower()
            for node_name in kg_nodes:
                if node_name.lower() in text:
                    candidates.add(node_name)

        # =====================================================================
        # POOL 2: Value candidates (span extraction, still symbolic)
        # =====================================================================

        value_candidates = self._extract_value_spans(evidence, query)
        candidates.update(value_candidates)

        return candidates

    def _extract_value_spans(self, evidence: List[Dict[str, Any]], query: str) -> Set[str]:
        """
        Extract value/span candidates from evidence text.

        This is SYMBOLIC extraction (pattern-based), NOT generation.
        Handles HotpotQA answers that are values, not entities:
        - Numbers ("10", "1958")
        - Genres/types ("country", "music", "rock")
        - Quoted strings
        - Short descriptors

        Args:
            evidence: Retrieved evidence
            query: Original query for intent detection

        Returns:
            Set of value candidate strings
        """
        values = set()
        query_lower = query.lower()

        for item in evidence:
            text = item.get("text", "")
            text_lower = text.lower()

            # -----------------------------------------------------------------
            # Numbers (for "how many", "how much", "what year" questions)
            # -----------------------------------------------------------------
            if any(w in query_lower for w in ["how many", "how much", "number", "year", "age", "award"]):
                # Extract standalone numbers
                numbers = re.findall(r'\b(\d+)\b', text)
                values.update(numbers)

            # -----------------------------------------------------------------
            # Quoted strings (often exact answers)
            # -----------------------------------------------------------------
            quoted = re.findall(r'"([^"]+)"', text)
            for q in quoted:
                if len(q) < 100:  # Avoid long passages
                    values.add(q)

            # -----------------------------------------------------------------
            # Genre/type/kind patterns (for "what genre/type" questions)
            # -----------------------------------------------------------------
            if any(w in query_lower for w in ["genre", "type", "kind", "style", "form"]):
                # Common genre words
                genre_words = ["rock", "pop", "country", "jazz", "classical", "hip hop",
                               "metal", "folk", "blues", "soul", "r&b", "electronic",
                               "punk", "alternative", "indie", "reggae"]
                for genre in genre_words:
                    if genre in text_lower:
                        values.add(genre)

                # Pattern: "X is a/an [TYPE] Y" or "[TYPE] music/film/artist"
                type_patterns = [
                    r'\b([\w-]+)\s+(?:music|film|movie|band|artist|singer|genre)\b',
                    r'\b(?:is a|is an|was a|was an)\s+([\w-]+)\b',
                ]
                for pattern in type_patterns:
                    matches = re.findall(pattern, text_lower)
                    values.update(m for m in matches if len(m) > 2)

            # -----------------------------------------------------------------
            # Art/medium types (for "what type of art" questions)
            # -----------------------------------------------------------------
            if any(w in query_lower for w in ["art", "medium", "form"]):
                art_words = ["music", "opera", "ballet", "painting", "sculpture",
                             "film", "cinema", "theater", "theatre", "dance"]
                for art in art_words:
                    if art in text_lower:
                        values.add(art)

            # -----------------------------------------------------------------
            # Title/profession (for "what title/profession" questions)
            # -----------------------------------------------------------------
            if any(w in query_lower for w in ["title", "profession", "occupation", "job"]):
                title_words = ["doctor", "professor", "director", "president", "ceo",
                               "actor", "actress", "singer", "writer", "author",
                               "scientist", "engineer", "lawyer", "physician"]
                for title in title_words:
                    if title in text_lower:
                        values.add(title)

        return values

    def _get_query_entities_structural(self, query: str) -> Set[str]:
        """
        Get entities from query using QueryParserAgent (structural).

        Falls back to KG node intersection if QueryParserAgent unavailable.
        NO regex capitalized-phrase heuristics.

        Args:
            query: User query

        Returns:
            Set of entity names found in query
        """
        entities = set()

        # Try QueryParserAgent first
        try:
            from ..agents.query_parser_agent import get_query_parser
            parser = get_query_parser()
            intent = parser.parse_query(query)

            # Extract from parsed anchor entities
            if intent.get("anchor_entities"):
                entities.update(intent["anchor_entities"])
            if intent.get("anchor_entity"):
                entities.add(intent["anchor_entity"])

            logger.debug(f"QueryParserAgent found entities: {entities}")
        except Exception as e:
            logger.debug(f"QueryParserAgent unavailable: {e}")

        # Fallback: find KG nodes that appear in query
        if not entities:
            query_lower = query.lower()
            kg_nodes = self._get_kg_node_names()
            for node_name in kg_nodes:
                if node_name.lower() in query_lower:
                    entities.add(node_name)
            logger.debug(f"KG intersection found entities: {entities}")

        return {e for e in entities if e}

    # =========================================================================
    # NORMALIZATION (EM-compatible)
    # =========================================================================

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for HotpotQA EM evaluation.

        MUST MATCH Search-R1 / Official HotpotQA exactly:
        Order: lower → remove_punc → remove_articles → white_space_fix

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer string
        """
        if not answer:
            return ""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        # EXACT order from Search-R1 / HotpotQA official
        return white_space_fix(remove_articles(remove_punc(lower(answer))))

    # =========================================================================
    # PROVENANCE LOGGING (Debugging only, does NOT affect answer)
    # =========================================================================

    def _log_provenance(self, query: str, evidence: List[Dict[str, Any]]) -> None:
        """
        Log provenance information for debugging.

        This does NOT affect answer selection - purely for debugging.

        Args:
            query: User query
            evidence: Retrieved evidence
        """
        logger.debug(f"AnswerSelector processing query: '{query[:50]}...'")
        logger.debug(f"Evidence count: {len(evidence)}")

        for i, item in enumerate(evidence[:5]):  # Log first 5
            doc_id = item.get("doc_id", "?")
            sent_id = item.get("sent_id", "?")
            source = item.get("source", "?")
            text_preview = item.get("text", "")[:50]
            logger.debug(f"  [{i}] doc={doc_id}, sent={sent_id}, src={source}: '{text_preview}...'")

    def _filter_by_type(self, candidates: Set[str], required_type: str) -> Set[str]:
        """
        Filter candidates by expected answer type (Symbolic).
        """
        if required_type == "unknown" or not candidates:
            return candidates
            
        filtered = set()
        logger.debug(f"Filtering {len(candidates)} candidates by type '{required_type}'")
        
        for cand in candidates:
            is_match = False
            c_lower = cand.lower()
            
            # Number check
            is_digit = any(char.isdigit() for char in cand)
            
            if required_type == "number":
                # Must contain digit or number word
                if is_digit or c_lower in ["one", "two", "three", "ten", "zero"]:
                    is_match = True
            
            elif required_type == "date":
                # Must look like year or contain month
                if re.search(r'\b\d{4}\b', cand):
                    is_match = True
            
            elif required_type == "person":
                # Heuristic: Uppercase first letter (often lost in norm but still check)
                # Eliminate obvious non-persons (digits)
                if not is_digit:
                    is_match = True
                    
            elif required_type == "organization":
                if not is_digit:
                    is_match = True
                    
            elif required_type == "location":
                if not is_digit:
                    is_match = True
            
            else:
                is_match = True
                
            if is_match:
                filtered.add(cand)
                
        logger.debug(f"Filtered {len(candidates)} -> {len(filtered)} candidates")
        return filtered if filtered else candidates  # Fallback to all if overly aggressive


# Singleton instance
_answer_selector: Optional[AnswerSelector] = None


def get_answer_selector() -> AnswerSelector:
    """Get global AnswerSelector instance (singleton)."""
    global _answer_selector
    if _answer_selector is None:
        _answer_selector = AnswerSelector()
    return _answer_selector


# =============================================================================
# STANDALONE EM FUNCTIONS (Search-R1 / HotpotQA compatible)
# =============================================================================

def normalize_answer(s: str) -> str:
    """
    Normalize answer string for EM comparison.

    EXACT match to Search-R1 / Official HotpotQA evaluation.
    Order: lower → remove_punc → remove_articles → white_space_fix
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers) -> int:
    """
    Check exact match between prediction and golden answers.

    EXACT match to Search-R1 implementation.

    Args:
        prediction: Predicted answer string
        golden_answers: Single answer string or list of acceptable answers

    Returns:
        1 if match, 0 otherwise
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)

    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1

    return 0


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Compute exact match score (HotpotQA official format).

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        True if exact match, False otherwise
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)
