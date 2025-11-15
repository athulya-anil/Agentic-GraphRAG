#!/usr/bin/env python3
"""
Quick Evaluation Script - Generates Results Without RAGAS

This faster version evaluates the system without full RAGAS metrics to generate
results quickly for your professor. It measures:
- Response quality (via LLM-as-judge)
- Retrieval latency
- Success rate
- Query performance across types

Run this first to get quick results, then run full evaluation.py later for complete metrics.
