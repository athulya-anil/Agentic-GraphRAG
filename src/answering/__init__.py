"""
Answer Selection Module

Symbolic/deterministic answer selection for HotpotQA EM evaluation.
"""

from .answer_selector import AnswerSelector, get_answer_selector

__all__ = ['AnswerSelector', 'get_answer_selector']
