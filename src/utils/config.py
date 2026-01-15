"""
Configuration Management for Agentic GraphRAG

This module handles loading and validating configuration from environment variables
and YAML files using Pydantic for type safety and validation.

Author: Agentic GraphRAG Team
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import yaml


# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM settings (Groq, Ollama, etc)."""

    provider: str = Field(default="groq", description="LLM provider (groq, ollama, gemini)")
    api_key: str = Field(default="", description="API key (not needed for Ollama)")
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model name"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=16384)

    @validator("api_key")
    def validate_api_key(cls, v: str, values: dict) -> str:
        """Validate API key is not empty for providers that need it."""
        provider = values.get("provider", "groq")
        # Ollama runs locally and doesn't need an API key
        if provider == "ollama":
            return v
        # Other providers need an API key
        if not v or v == "your_groq_api_key_here":
            raise ValueError(
                f"{provider.upper()}_API_KEY not set. Please set it in .env file or environment."
            )
        return v


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j graph database."""

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="password123", description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")

    @validator("password")
    def validate_password(cls, v: str) -> str:
        """Validate password is not empty."""
        if not v:
            raise ValueError("NEO4J_PASSWORD cannot be empty")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model and text processing."""

    model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace model name for embeddings"
    )
    dimension: int = Field(default=1024, ge=1, description="Embedding dimension")
    # TODO: Optimize chunking after span-level provenance implemented
    # HotpotQA requires sentence-level evidence tracking
    # Current: whole-document storage (no chunk provenance)
    chunk_size: int = Field(default=512, ge=1, description="Text chunk size")
    chunk_overlap: int = Field(default=50, ge=0, description="Chunk overlap size")

    @validator("chunk_overlap")
    def validate_overlap(cls, v: int, values: dict) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = values.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v


class VectorStoreConfig(BaseModel):
    """Configuration for vector store (FAISS)."""

    index_path: Path = Field(
        default=Path("data/faiss_index"),
        description="Path to FAISS index"
    )
    store_type: Literal["faiss"] = Field(default="faiss", description="Vector store type")


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""

    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry attempts")
    timeout_seconds: int = Field(default=30, ge=5, le=300, description="Timeout in seconds")
    verbose: bool = Field(default=True, description="Enable verbose logging")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval parameters (HotpotQA-optimized)."""

    top_k_vector: int = Field(default=10, ge=1, le=30, description="Top K for vector search")
    top_k_graph: int = Field(default=15, ge=1, le=50, description="Top K for graph search")
    # System uses either VECTOR or GRAPH (not hybrid)
    # HotpotQA requires more contexts for multi-hop reasoning


class ApplicationConfig(BaseModel):
    """Main application configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    environment: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Application environment"
    )


class Config(BaseModel):
    """
    Main configuration class that combines all sub-configurations.

    This class loads configuration from environment variables and provides
    type-safe access to all application settings.
    """

    llm: LLMConfig
    neo4j: Neo4jConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    agent: AgentConfig
    retrieval: RetrievalConfig
    app: ApplicationConfig

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.

        Returns:
            Config: Initialized configuration object

        Raises:
            ValueError: If required environment variables are missing
        """
        # Determine LLM provider and load appropriate settings
        llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()

        if llm_provider == "ollama":
            llm_config = LLMConfig(
                provider="ollama",
                api_key="",  # Ollama doesn't need an API key
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "16000")),
            )
        elif llm_provider == "groq":
            llm_config = LLMConfig(
                provider="groq",
                api_key=os.getenv("GROQ_API_KEY", ""),
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=float(os.getenv("GROQ_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "2048")),
            )
        elif llm_provider == "gemini":
            llm_config = LLMConfig(
                provider="gemini",
                api_key=os.getenv("GEMINI_API_KEY", ""),
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048")),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        return cls(
            llm=llm_config,
            neo4j=Neo4jConfig(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password123"),
                database=os.getenv("NEO4J_DATABASE", "neo4j"),
            ),
            embedding=EmbeddingConfig(
                model_name=os.getenv(
                    "EMBEDDING_MODEL",
                    "BAAI/bge-large-en-v1.5"
                ),
                dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
                chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            ),
            vector_store=VectorStoreConfig(
                index_path=Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index")),
                store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),  # type: ignore
            ),
            agent=AgentConfig(
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30")),
                verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
            ),
            retrieval=RetrievalConfig(
                top_k_vector=int(os.getenv("TOP_K_VECTOR", "10")),
                top_k_graph=int(os.getenv("TOP_K_GRAPH", "15")),
            ),
            app=ApplicationConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO"),  # type: ignore
                environment=os.getenv("ENV", "development"),  # type: ignore
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config: Initialized configuration object

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Config: Global configuration object
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("✅ Configuration loaded successfully!")
        print(f"\nLLM Model: {config.llm.model}")
        print(f"Neo4j URI: {config.neo4j.uri}")
        print(f"Embedding Model: {config.embedding.model_name}")
        print(f"Log Level: {config.app.log_level}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
