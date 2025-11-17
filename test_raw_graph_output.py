#!/usr/bin/env python3
"""
Test script to show RAW graph retrieval output before LLM synthesis
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_retrieval_pipeline
from src.agents import RetrievalStrategy
import json


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def main():
    """Show raw graph output."""
    print_section("RAW GRAPH RETRIEVAL OUTPUT TEST")

    # Initialize pipeline
    pipeline = get_retrieval_pipeline(use_reflection=False)

    # Test query
    query = "What medications treat diabetes?"

    print(f"Query: \"{query}\"\n")
    print("Forcing GRAPH strategy (no LLM synthesis yet)...\n")

    # Get raw retrieval result (before synthesis)
    result = pipeline.query(
        query,
        top_k=5,
        strategy=RetrievalStrategy.GRAPH,
        evaluate=False  # Skip evaluation to see just retrieval
    )

    print_section("RAW CONTEXTS FROM GRAPH TRAVERSAL")

    contexts = result.get('context', [])

    if not contexts:
        print("❌ No contexts retrieved from graph")
        return

    print(f"Total contexts retrieved: {len(contexts)}\n")

    for i, ctx in enumerate(contexts, 1):
        print(f"{'─' * 70}")
        print(f"Context #{i}:")
        print(f"{'─' * 70}")

        # Show raw fields
        print(f"Source: {ctx.get('source', 'unknown')}")
        print(f"Score: {ctx.get('score', 0.0):.3f}")

        # This is the RAW graph data before LLM processing
        print(f"\nRAW TEXT:\n{ctx.get('text', 'N/A')}\n")

        # Show metadata
        if 'metadata' in ctx:
            print(f"Metadata: {json.dumps(ctx['metadata'], indent=2)}\n")

    print(f"{'=' * 70}\n")

    # Now show what LLM does with it
    print_section("AFTER LLM SYNTHESIS")
    print(f"Response:\n{result['response']}\n")

    print(f"{'=' * 70}\n")
    print("✅ Comparison complete!")
    print("\nKey points:")
    print("- RAW output = Structured data from graph traversal")
    print("- LLM synthesis = Natural language answer from raw data")
    print("- Graph provides facts, LLM provides readable response")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
