#!/usr/bin/env python3
"""
Verification script for Agentic GraphRAG setup.

This script tests each component of the system to ensure everything is configured correctly.
Run this after setting up your .env file and starting Neo4j.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def print_result(test_name, passed, message=""):
    """Print test result with formatting."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"     {message}")
    return passed

def test_environment():
    """Test if .env file exists and has required variables."""
    print_header("1. Environment Configuration")

    env_file = Path(".env")
    if not env_file.exists():
        print_result("Environment file", False, ".env file not found")
        print("\n💡 Setup instructions:")
        print("   1. Copy the example file: cp .env.example .env")
        print("   2. Edit .env and add your Groq API key")
        print("   3. Get a free key at: https://console.groq.com/keys")
        return False

    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    # Check required variables
    required_vars = [
        ("GROQ_API_KEY", "Groq API key"),
        ("NEO4J_URI", "Neo4j URI"),
        ("NEO4J_USER", "Neo4j username"),
        ("NEO4J_PASSWORD", "Neo4j password"),
    ]

    all_present = True
    for var_name, description in required_vars:
        value = os.getenv(var_name)
        if not value or value == f"your_{var_name.lower()}_here":
            print_result(description, False, f"{var_name} not set")
            all_present = False
        else:
            print_result(description, True, f"{var_name} is configured")

    return all_present

def test_dependencies():
    """Test if required Python packages are installed."""
    print_header("2. Python Dependencies")

    required_packages = [
        ("groq", "Groq API client"),
        ("neo4j", "Neo4j driver"),
        ("faiss", "FAISS vector search"),
        ("sentence_transformers", "Sentence transformers"),
        ("dotenv", "Python-dotenv"),
    ]

    all_installed = True
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_result(description, True, f"{package} installed")
        except ImportError:
            print_result(description, False, f"{package} not installed")
            all_installed = False

    if not all_installed:
        print("\n💡 Install missing packages:")
        print("   pip install -r requirements.txt")

    return all_installed

def test_llm_client():
    """Test LLM client connection to Groq."""
    print_header("3. LLM Client (Groq API)")

    try:
        from src.utils.llm_client import get_llm_client

        client = get_llm_client()
        print_result("Client initialization", True, "LLM client created successfully")

        # Test simple generation
        response = client.generate(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.0,
        )

        if response and len(response) > 0:
            print_result("API connection", True, f"Response received: '{response.strip()}'")
            print_result("Token tracking", True, f"Tokens used: {client.get_token_usage()}")
            return True
        else:
            print_result("API connection", False, "Empty response received")
            return False

    except ValueError as e:
        print_result("Configuration", False, str(e))
        return False
    except Exception as e:
        print_result("LLM client", False, str(e))
        return False

def test_neo4j():
    """Test Neo4j connection and operations."""
    print_header("4. Neo4j Graph Database")

    try:
        from src.graph.neo4j_manager import get_neo4j_manager

        manager = get_neo4j_manager()
        print_result("Connection", True, "Connected to Neo4j")

        # Test basic operations
        schema = manager.get_schema()
        print_result("Schema query", True, f"Found {len(schema['labels'])} labels, {len(schema['relationship_types'])} relationship types")

        node_count = manager.get_node_count()
        rel_count = manager.get_relationship_count()
        print_result("Database stats", True, f"{node_count} nodes, {rel_count} relationships")

        # Test CRUD operations
        test_node = manager.create_node(
            label="TestNode",
            properties={"name": "verification_test", "timestamp": "test"},
            merge=True
        )
        print_result("Node creation", True, f"Created test node with ID {test_node.get('_id')}")

        # Clean up
        manager.delete_node(
            label="TestNode",
            properties={"name": "verification_test"}
        )
        print_result("Node deletion", True, "Cleaned up test node")

        manager.close()
        return True

    except Exception as e:
        print_result("Neo4j connection", False, str(e))
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Docker Desktop is running")
        print("   2. Start Neo4j: ./scripts/start_neo4j.sh")
        print("   3. Check if Neo4j is accessible at: http://localhost:7474")
        print("   4. Verify credentials in .env file match Neo4j settings")
        return False

def test_faiss():
    """Test FAISS vector store."""
    print_header("5. FAISS Vector Store")

    try:
        from src.vector.faiss_index import FAISSIndex

        # Initialize index
        index = FAISSIndex()
        print_result("Index initialization", True, f"Created FAISS index with dimension {index.dimension}")
        print_result("Embedding model", True, f"Loaded model: {index.embedding_config.model_name}")

        # Test embedding
        test_text = "This is a test sentence for embedding."
        embedding = index.embed_text(test_text)
        print_result("Text embedding", True, f"Generated embedding with shape {embedding.shape}")

        # Test adding and searching
        test_docs = [
            "Neo4j is a graph database.",
            "FAISS enables similarity search.",
            "Python is a programming language."
        ]

        ids = index.add(test_docs)
        print_result("Add documents", True, f"Added {len(ids)} documents")

        # Test search
        results = index.search("What is Neo4j?", top_k=2)
        print_result("Similarity search", True, f"Found {len(results)} results")

        if results:
            best_match = results[0]
            print(f"     Best match: '{best_match['metadata']['text']}' (score: {best_match['score']:.4f})")

        # Clean up
        index.clear()
        print_result("Cleanup", True, "Cleared test data")

        # Proper cleanup to avoid segmentation faults
        index.cleanup()

        return True

    except Exception as e:
        print_result("FAISS vector store", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure cleanup even on error
        try:
            if 'index' in locals():
                index.cleanup()
        except:
            pass

def main():
    """Run all verification tests."""
    print("\n🔍 Agentic GraphRAG - System Verification")
    print("=" * 60)

    results = {
        "Environment": test_environment(),
        "Dependencies": test_dependencies(),
    }

    # Only test services if environment is configured
    if results["Environment"] and results["Dependencies"]:
        results["LLM Client"] = test_llm_client()
        results["Neo4j"] = test_neo4j()
        results["FAISS"] = test_faiss()
    else:
        print("\n⚠️  Skipping service tests due to configuration issues")
        results["LLM Client"] = False
        results["Neo4j"] = False
        results["FAISS"] = False

    # Summary
    print_header("Verification Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} components verified successfully")
    print(f"{'='*60}\n")

    if passed == total:
        print("🎉 All systems operational! Your setup is ready to use.")
        return 0
    else:
        print("⚠️  Some components need attention. See messages above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
