#!/usr/bin/env python3
"""
Agentic GraphRAG - Query CLI

This script allows you to query your knowledge graph interactively or from command line.

Usage:
    # Interactive mode
    python query.py

    # Single query from command line
    python query.py --query "What medications treat diabetes?"

    # Query with custom options
    python query.py --query "Explain machine learning" --top-k 10 --no-evaluation

Author: Agentic GraphRAG Team
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_retrieval_pipeline


def print_result(result: dict, verbose: bool = False):
    """Pretty print query results."""
    print(f"\n{'─' * 70}")
    print(f"💬 Response:")
    print(f"{'─' * 70}")
    print(f"{result['response']}\n")

    # Print retrieval stats
    print(f"📊 Retrieval Stats:")
    print(f"   • Contexts retrieved: {result['num_contexts']}")
    print(f"   • Strategy used: {result.get('strategy', 'unknown')}")

    # Print metrics if available
    if result.get('metrics'):
        metrics = result['metrics']
        print(f"\n📈 RAGAS Metrics:")
        print(f"   • Faithfulness:      {metrics.get('faithfulness', 0):.3f}")
        print(f"   • Answer Relevancy:  {metrics.get('answer_relevancy', 0):.3f}")
        print(f"   • Context Precision: {metrics.get('context_precision', 0):.3f}")
        print(f"   • Overall Score:     {metrics.get('overall', 0):.3f}")

    # Print top contexts if verbose
    if verbose and result.get('context'):
        print(f"\n📄 Retrieved Contexts:")
        for i, ctx in enumerate(result['context'][:3], 1):
            print(f"\n   Context {i}:")
            print(f"   • Source: {ctx.get('source', 'unknown')}")
            print(f"   • Score: {ctx.get('score', 0):.3f}")
            print(f"   • Text: {ctx['text'][:200]}...")


def interactive_mode(pipeline, top_k: int, evaluate: bool, verbose: bool):
    """Run in interactive query mode."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - INTERACTIVE QUERY MODE")
    print("=" * 70)
    print("\n💡 Tips:")
    print("   • Type your questions naturally")
    print("   • Use 'exit', 'quit', or Ctrl+C to exit")
    print("   • Type 'help' for more information")
    print("\n" + "─" * 70)

    while True:
        try:
            # Get user query
            query = input("\n🔍 Query: ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'help':
                print("\n📖 Help:")
                print("   • Ask any question about your ingested documents")
                print("   • The system will automatically choose the best retrieval strategy")
                print("   • Relational queries → Graph traversal")
                print("   • Conceptual queries → Vector search")
                print("   • Complex queries → Hybrid approach")
                print("\n   Examples:")
                print("   - What medications treat diabetes?")
                print("   - Explain what machine learning is")
                print("   - Who is the CEO of Tesla?")
                print("   - Tell me about MIT and its research")
                continue

            # Run query
            result = pipeline.query(
                query,
                top_k=top_k,
                evaluate=evaluate
            )

            # Display results
            print_result(result, verbose)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


def single_query_mode(pipeline, query: str, top_k: int, evaluate: bool, verbose: bool):
    """Run a single query and exit."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - QUERY")
    print("=" * 70)
    print(f"\n🔍 Query: {query}")

    try:
        result = pipeline.query(
            query,
            top_k=top_k,
            evaluate=evaluate
        )
        print_result(result, verbose)
        print("\n" + "=" * 70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Run query interface from command line."""
    parser = argparse.ArgumentParser(
        description='Query the Agentic GraphRAG knowledge graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python query.py

  # Single query
  python query.py --query "What medications treat diabetes?"

  # Query with custom options
  python query.py --query "Explain AI" --top-k 10 --verbose

  # Disable evaluation for faster queries
  python query.py --no-evaluation
        """
    )

    # Query options
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Query string (if not provided, enters interactive mode)'
    )

    # Retrieval options
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of contexts to retrieve (default: 5)'
    )
    parser.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable cross-encoder reranking (faster but less accurate)'
    )
    parser.add_argument(
        '--no-reflection',
        action='store_true',
        help='Disable reflection agent (no self-optimization)'
    )
    parser.add_argument(
        '--no-evaluation',
        action='store_true',
        help='Skip RAGAS evaluation (faster queries)'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output (show all contexts)'
    )

    args = parser.parse_args()

    # Initialize retrieval pipeline
    print("🔧 Initializing retrieval pipeline...")
    try:
        pipeline = get_retrieval_pipeline(
            use_reranking=not args.no_reranking,
            use_reflection=not args.no_reflection
        )
        print("✅ Pipeline ready!\n")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("\n💡 Make sure:")
        print("   • Neo4j is running: ./scripts/start_neo4j.sh")
        print("   • Documents are ingested: python ingest.py --help")
        print("   • Environment variables are set in .env")
        sys.exit(1)

    # Run in appropriate mode
    if args.query:
        # Single query mode
        single_query_mode(
            pipeline,
            args.query,
            args.top_k,
            not args.no_evaluation,
            args.verbose
        )
    else:
        # Interactive mode
        interactive_mode(
            pipeline,
            args.top_k,
            not args.no_evaluation,
            args.verbose
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
