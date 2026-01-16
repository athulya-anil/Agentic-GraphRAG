"""
Answer Selection Module

Symbolic/deterministic answer selection for HotpotQA EM evaluation.
Includes Search-R1 / HotpotQA compatible EM functions.
"""

from .answer_selector import (
    AnswerSelector,
    get_answer_selector,
    normalize_answer,
    exact_match_score
)
from .llm_answerer import (
    LLMAnswerer,
    get_llm_answerer
)

__all__ = [
    "AnswerSelector",
    "get_answer_selector",
    "LLMAnswerer",
    "get_llm_answerer",
    "normalize_answer",
    "exact_match_score"
]
