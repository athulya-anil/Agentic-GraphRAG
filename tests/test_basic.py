"""
Basic tests for Agentic GraphRAG components

Run with: pytest tests/test_basic.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_config, get_llm_client
from src.agents import (
    get_schema_agent,
    get_entity_agent,
    get_relation_agent,
    get_orchestrator_agent,
    get_reflection_agent
)


class TestConfiguration:
    """Test configuration management."""

    def test_config_loading(self):
        """Test that configuration loads successfully."""
        config = get_config()
        assert config is not None
        assert config.llm.model is not None
        assert config.neo4j.uri is not None

    def test_llm_client(self):
        """Test LLM client initialization."""
        client = get_llm_client()
        assert client is not None


class TestAgents:
    """Test agent initialization."""

    def test_schema_agent(self):
        """Test SchemaAgent initialization."""
        agent = get_schema_agent()
        assert agent is not None

    def test_entity_agent(self):
        """Test EntityAgent initialization."""
        agent = get_entity_agent()
        assert agent is not None

    def test_relation_agent(self):
        """Test RelationAgent initialization."""
        agent = get_relation_agent()
        assert agent is not None

    def test_orchestrator_agent(self):
        """Test OrchestratorAgent initialization."""
        agent = get_orchestrator_agent()
        assert agent is not None

    def test_reflection_agent(self):
        """Test ReflectionAgent initialization."""
        agent = get_reflection_agent()
        assert agent is not None


class TestSchemaInference:
    """Test schema inference functionality."""

    def test_schema_inference_basic(self):
        """Test basic schema inference."""
        agent = get_schema_agent()
        test_docs = [
            "John works at Google. Google is located in Mountain View.",
            "Alice is a doctor at Stanford Hospital."
        ]

        schema = agent.infer_schema_from_documents(test_docs)

        assert "entity_types" in schema
        assert "relation_types" in schema
        assert len(schema["entity_types"]) > 0


class TestEntityExtraction:
    """Test entity extraction functionality."""

    def test_entity_extraction_basic(self):
        """Test basic entity extraction."""
        agent = get_entity_agent()
        test_text = "Apple Inc. is headquartered in Cupertino, California."

        entities = agent.extract_entities_from_text(test_text, enrich_metadata=False)

        assert isinstance(entities, list)
        # Check that at least some entities were extracted
        assert len(entities) >= 0


class TestQueryRouting:
    """Test query routing functionality."""

    def test_query_routing(self):
        """Test query classification and routing."""
        agent = get_orchestrator_agent()

        test_queries = [
            "Who is the CEO of Tesla?",
            "Explain machine learning",
            "What treats diabetes?"
        ]

        for query in test_queries:
            routing = agent.route_query(query)

            assert "strategy" in routing
            assert "query_type" in routing
            assert "confidence" in routing
            assert "params" in routing


class TestReflection:
    """Test reflection and evaluation functionality."""

    def test_evaluation(self):
        """Test response evaluation."""
        agent = get_reflection_agent()

        query = "What is AI?"
        response = "AI is artificial intelligence, the simulation of human intelligence by machines."
        context = [{"text": "Artificial intelligence is intelligence demonstrated by machines."}]

        metrics = agent.evaluate_response(query, response, context)

        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "overall" in metrics
        assert 0.0 <= metrics["overall"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
