#!/usr/bin/env python3
"""
Agentic GraphRAG - End-to-End Demo

This script demonstrates the complete Agentic GraphRAG system:
1. Document ingestion with automatic schema inference
2. Entity and relationship extraction with metadata enrichment
3. Knowledge graph construction
4. Intelligent query routing and retrieval
5. Performance evaluation and self-optimization

"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
from src.agents import get_schema_agent, get_reflection_agent


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a formatted section title."""
    print(f"\n{'─' * 70}")
    print(f"📌 {title}")
    print(f"{'─' * 70}")


def main():
    """Run the complete demo."""
    print_header("AGENTIC GRAPHRAG - END-TO-END DEMO")
    print("\nA self-adaptive multi-agent system for autonomous knowledge graph")
    print("construction and intelligent retrieval.\n")

    # ========================================================================
    # STAGE 1: DOCUMENT INGESTION
    # ========================================================================
    print_section("Stage 1: Document Ingestion")

    # Sample documents from different domains
    documents = [
        # Medical/Healthcare domain
        """Aspirin is a medication used to reduce pain, fever, and inflammation.
        It is manufactured by Bayer and other pharmaceutical companies.
        Aspirin treats headaches, muscle pain, and can help prevent heart attacks.
        Common side effects include stomach irritation and bleeding.""",

        """Diabetes is a chronic disease that affects how the body processes blood sugar.
        Type 2 diabetes is the most common form. Metformin is a medication commonly
        prescribed to treat type 2 diabetes. It helps control blood sugar levels.
        The American Diabetes Association provides guidelines for diabetes management.""",

        # Technology/Business domain
        """Tesla is an American electric vehicle and clean energy company founded by
        Elon Musk. The company is headquartered in Austin, Texas. Tesla manufactures
        electric cars, battery energy storage, and solar panels. Elon Musk serves as
        CEO and has been instrumental in the company's growth.""",

        """Apple Inc. is a technology company that designs and sells consumer electronics.
        Tim Cook is the CEO of Apple. The company was founded by Steve Jobs, Steve Wozniak,
        and Ronald Wayne in 1976. Apple is headquartered in Cupertino, California and
        produces the iPhone, iPad, and Mac computers.""",

        # Academic/Research domain
        """Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        Deep learning is a type of machine learning that uses neural networks with
        multiple layers. These techniques are used in natural language processing,
        computer vision, and speech recognition.""",

        """MIT (Massachusetts Institute of Technology) is a private research university
        located in Cambridge, Massachusetts. The university has a strong Computer Science
        department that conducts research in artificial intelligence, robotics, and
        distributed systems. Many notable researchers work at MIT's CSAIL laboratory."""
    ]

    print(f"\n📚 Ingesting {len(documents)} documents from multiple domains...")
    print("   - Medical/Healthcare (2 docs)")
    print("   - Technology/Business (2 docs)")
    print("   - Academic/Research (2 docs)")

    # Initialize ingestion pipeline
    schema_path = Path("data/processed/schema.json")
    ingestion_pipeline = get_ingestion_pipeline(
        schema_path=schema_path,
        auto_refine_schema=True
    )

    # Ingest documents
    print("\n🔄 Running ingestion pipeline...")
    print("   • Inferring graph schema")
    print("   • Extracting entities with metadata")
    print("   • Identifying relationships")
    print("   • Building knowledge graph")
    print("   • Creating vector embeddings")

    results = ingestion_pipeline.ingest_documents(
        documents,
        infer_schema=True,
        enrich_metadata=True
    )

    print(f"\n✅ Ingestion complete in {results['duration_seconds']:.2f}s")
    print(f"\n   Statistics:")
    print(f"   • Documents processed: {results['documents_processed']}")
    print(f"   • Entities extracted: {results['entities_extracted']}")
    print(f"   • Relations extracted: {results['relations_extracted']}")
    print(f"   • Neo4j nodes created: {results['nodes_created']}")
    print(f"   • Neo4j edges created: {results['edges_created']}")

    # Display discovered schema
    print_section("Discovered Schema")
    schema_agent = get_schema_agent()
    print(schema_agent.get_schema_summary())

    # ========================================================================
    # STAGE 2: INTELLIGENT QUERY RETRIEVAL
    # ========================================================================
    print_section("Stage 2: Intelligent Query Retrieval")

    # Initialize retrieval pipeline
    retrieval_pipeline = get_retrieval_pipeline(
        use_reranking=False,  # Set to True if cross-encoder is available
        use_reflection=True
    )

    # Test queries of different types
    test_queries = [
        {
            "query": "What medications treat diabetes?",
            "type": "Relational (graph-focused)",
            "description": "Expects graph traversal to find Drug->TREATS->Disease relationships"
        },
        {
            "query": "Explain what machine learning is",
            "type": "Conceptual (vector-focused)",
            "description": "Expects semantic search for conceptual explanation"
        },
        {
            "query": "Who is the CEO of Tesla?",
            "type": "Factual (hybrid)",
            "description": "Specific fact that might benefit from both strategies"
        },
        {
            "query": "Tell me about MIT and its research",
            "type": "Exploratory (hybrid)",
            "description": "Open-ended query requiring comprehensive context"
        }
    ]

    print(f"\n🔍 Testing {len(test_queries)} queries with intelligent routing...\n")

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        query_type = test_case["type"]

        print(f"\n{'─' * 70}")
        print(f"Query {i}: \"{query}\"")
        print(f"Expected Type: {query_type}")
        print(f"{'─' * 70}")

        # Run query
        result = retrieval_pipeline.query(
            query,
            top_k=5,
            evaluate=True  # Enable evaluation
        )

        # Display results
        print(f"\n💬 Response:")
        print(f"   {result['response'][:300]}...")

        print(f"\n📊 Retrieval Stats:")
        print(f"   • Contexts retrieved: {result['num_contexts']}")

        if result.get('metrics'):
            metrics = result['metrics']
            print(f"\n📈 RAGAS Metrics:")
            print(f"   • Faithfulness:      {metrics.get('faithfulness', 0):.3f}")
            print(f"   • Answer Relevancy:  {metrics.get('answer_relevancy', 0):.3f}")
            print(f"   • Context Precision: {metrics.get('context_precision', 0):.3f}")
            print(f"   • Overall Score:     {metrics.get('overall', 0):.3f}")

        if result['context']:
            print(f"\n📄 Top Context Source:")
            top_ctx = result['context'][0]
            print(f"   Source: {top_ctx['source']}")
            print(f"   Score: {top_ctx['score']:.3f}")
            print(f"   Text: {top_ctx['text'][:150]}...")

    # ========================================================================
    # STAGE 3: PERFORMANCE ANALYSIS
    # ========================================================================
    print_section("Stage 3: Performance Analysis & Self-Optimization")

    reflection_agent = get_reflection_agent()

    # Show performance summary
    print(reflection_agent.get_performance_summary())

    # Analyze failures
    print(f"\n🔍 Failure Analysis:")
    failure_analysis = reflection_agent.analyze_failures(threshold=0.7)
    print(f"   • Total evaluations: {failure_analysis['total_evaluations']}")
    print(f"   • Failures: {failure_analysis['failures']}")
    print(f"   • Failure rate: {failure_analysis['failure_rate']*100:.1f}%")

    if failure_analysis.get('failing_metrics'):
        print(f"\n   Failing Metrics:")
        for metric, count in failure_analysis['failing_metrics'].items():
            print(f"     • {metric}: {count} failures")

    # Get improvement suggestions
    print(f"\n💡 Improvement Suggestions:")
    suggestions = reflection_agent.suggest_improvements(failure_analysis)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")

    # ========================================================================
    # SYSTEM STATISTICS
    # ========================================================================
    print_section("System Statistics")

    stats = ingestion_pipeline.get_statistics()
    print(f"\n📊 Knowledge Graph:")
    print(f"   • Entity types: {stats['schema_summary']['entity_types']}")
    print(f"   • Relation types: {stats['schema_summary']['relation_types']}")
    print(f"   • Total nodes: {stats['neo4j_stats']['total_nodes']}")
    print(f"   • Total relationships: {stats['neo4j_stats']['total_relationships']}")

    print(f"\n🔢 Vector Store:")
    print(f"   • Total vectors: {stats['faiss_stats']['total_vectors']}")

    print(f"\n📈 Processing Stats:")
    print(f"   • Documents ingested: {stats['documents_processed']}")
    print(f"   • Entities extracted: {stats['entities_extracted']}")
    print(f"   • Relations extracted: {stats['relations_extracted']}")

    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print_header("DEMO COMPLETE")

    print("\n✅ Successfully demonstrated:")
    print("   • Autonomous schema inference from multi-domain documents")
    print("   • Entity extraction with metadata enrichment")
    print("   • Relationship extraction and knowledge graph construction")
    print("   • Intelligent query routing (vector/graph/hybrid)")
    print("   • Performance evaluation with RAGAS metrics")
    print("   • Self-optimization recommendations")

    print("\n🎯 Key Features:")
    print("   • Schema-agnostic: Works on any domain automatically")
    print("   • Self-adaptive: Learns optimal retrieval strategies")
    print("   • Production-ready: Robust error handling and logging")
    print("   • Metadata-enriched: Summaries, keywords, and context")

    print("\n📚 Next Steps:")
    print("   • Ingest your own documents")
    print("   • Fine-tune retrieval parameters")
    print("   • Enable cross-encoder reranking")
    print("   • Deploy to production")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
