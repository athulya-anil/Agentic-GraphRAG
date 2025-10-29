"""
FAISS Vector Store for Agentic GraphRAG

This module provides vector storage and similarity search using FAISS with:
- Sentence embedding using HuggingFace models
- FAISS index management (flat and IVF)
- Vector similarity search
- Persistence (save/load)
- Metadata management

Author: Agentic GraphRAG Team
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..utils.config import get_config, EmbeddingConfig, VectorStoreConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS-based vector store with embedding capabilities.

    Provides functionality for:
    - Embedding text using sentence transformers
    - Creating and managing FAISS indexes
    - Adding vectors with metadata
    - Similarity search
    - Persistence

    Attributes:
        embedding_config: Embedding configuration
        vector_config: Vector store configuration
        model: Sentence transformer model for embeddings
        index: FAISS index
        metadata: List of metadata for each vector
    """

    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
        vector_config: Optional[VectorStoreConfig] = None,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS vector store.

        Args:
            embedding_config: Optional embedding configuration
            vector_config: Optional vector store configuration
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)

        Raises:
            ValueError: If configuration is invalid
        """
        config = get_config()
        self.embedding_config = embedding_config or config.embedding
        self.vector_config = vector_config or config.vector_store
        self.use_gpu = use_gpu

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_config.model_name}")
        self.model = SentenceTransformer(self.embedding_config.model_name)
        self.dimension = self.embedding_config.dimension

        # Initialize FAISS index
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._create_index()

        logger.info(f"Initialized FAISS index with dimension {self.dimension}")

    def _create_index(self, index_type: str = "Flat") -> None:
        """
        Create a new FAISS index.

        Args:
            index_type: Type of index ("Flat" for exact search, "IVF" for approximate)
        """
        if index_type == "Flat":
            # Exact search - slower but accurate
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            # Approximate search - faster but may miss some results
            quantizer = faiss.IndexFlatL2(self.dimension)
            n_cells = 100  # Number of cells for IVF
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_cells)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving FAISS index to GPU")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

        logger.debug(f"Created {index_type} FAISS index")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings with shape (n_texts, dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension).astype('float32')

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings.astype('float32')

    def add(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False
    ) -> List[int]:
        """
        Add texts to the vector store.

        Args:
            texts: List of texts to add
            metadata: Optional list of metadata dicts (one per text)
            show_progress: Whether to show embedding progress

        Returns:
            List of assigned IDs for the added vectors
        """
        if not texts:
            return []

        # Generate embeddings
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.embed_texts(texts, show_progress=show_progress)

        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        end_id = self.index.ntotal
        ids = list(range(start_id, end_id))

        # Store metadata
        if metadata is None:
            metadata = [{"text": text} for text in texts]
        else:
            # Add text to metadata if not present
            for i, meta in enumerate(metadata):
                if "text" not in meta:
                    meta["text"] = texts[i]

        self.metadata.extend(metadata)

        logger.info(f"Added {len(texts)} vectors to index (total: {self.index.ntotal})")
        return ids

    def add_single(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a single text to the vector store.

        Args:
            text: Text to add
            metadata: Optional metadata dict

        Returns:
            Assigned ID for the vector
        """
        ids = self.add([text], [metadata] if metadata else None)
        return ids[0] if ids else -1

    def search(
        self,
        query: str,
        top_k: int = 5,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using a text query.

        Args:
            query: Query text
            top_k: Number of results to return
            return_metadata: Whether to include metadata in results

        Returns:
            List of search results with distances and optional metadata
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []

        # Generate query embedding
        query_embedding = self.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            result = {
                "id": int(idx),
                "distance": float(distance),
                "score": 1.0 / (1.0 + float(distance))  # Convert distance to similarity score
            }

            if return_metadata and idx < len(self.metadata):
                result["metadata"] = self.metadata[idx]

            results.append(result)

        logger.debug(f"Search returned {len(results)} results")
        return results

    def search_by_vector(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using a vector query.

        Args:
            vector: Query vector
            top_k: Number of results to return
            return_metadata: Whether to include metadata in results

        Returns:
            List of search results with distances and optional metadata
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []

        # Ensure vector is the right shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(vector, min(top_k, self.index.ntotal))

        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            result = {
                "id": int(idx),
                "distance": float(distance),
                "score": 1.0 / (1.0 + float(distance))
            }

            if return_metadata and idx < len(self.metadata):
                result["metadata"] = self.metadata[idx]

            results.append(result)

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query

        Returns:
            List of result lists (one per query)
        """
        if not queries:
            return []

        # Generate query embeddings
        query_embeddings = self.embed_texts(queries)

        # Search
        distances, indices = self.index.search(query_embeddings, min(top_k, self.index.ntotal))

        # Format results for each query
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for distance, idx in zip(query_distances, query_indices):
                if idx == -1:
                    continue

                result = {
                    "id": int(idx),
                    "distance": float(distance),
                    "score": 1.0 / (1.0 + float(distance)),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                }
                results.append(result)

            all_results.append(results)

        return all_results

    def delete(self, ids: List[int]) -> bool:
        """
        Delete vectors by IDs.

        Note: FAISS doesn't support efficient deletion, so we recreate the index.

        Args:
            ids: List of IDs to delete

        Returns:
            True if deletion was successful
        """
        if not ids:
            return True

        ids_set = set(ids)
        logger.info(f"Deleting {len(ids)} vectors from index...")

        # Get all vectors except the ones to delete
        all_vectors = []
        new_metadata = []

        for i in range(self.index.ntotal):
            if i not in ids_set:
                vector = self.index.reconstruct(int(i))
                all_vectors.append(vector)
                if i < len(self.metadata):
                    new_metadata.append(self.metadata[i])

        # Recreate index
        self._create_index()
        if all_vectors:
            vectors_array = np.array(all_vectors).astype('float32')
            self.index.add(vectors_array)

        self.metadata = new_metadata
        logger.info(f"Index recreated with {self.index.ntotal} vectors")
        return True

    def clear(self) -> None:
        """Clear all vectors from the index."""
        logger.warning("Clearing all vectors from index")
        self._create_index()
        self.metadata = []

    def get_count(self) -> int:
        """
        Get number of vectors in the index.

        Returns:
            Number of vectors
        """
        return self.index.ntotal

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save index and metadata to disk.

        Args:
            path: Optional custom save path (defaults to config path)
        """
        save_path = path or self.vector_config.index_path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = save_path / "faiss.index"
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))

        # Save metadata
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved index to {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load index and metadata from disk.

        Args:
            path: Optional custom load path (defaults to config path)

        Returns:
            True if loading was successful
        """
        load_path = path or self.vector_config.index_path
        load_path = Path(load_path)

        index_file = load_path / "faiss.index"
        metadata_file = load_path / "metadata.pkl"

        if not index_file.exists():
            logger.warning(f"Index file not found: {index_file}")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))

            if self.use_gpu and faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    self.metadata = pickle.load(f)
            else:
                self.metadata = []

            logger.info(f"Loaded index from {load_path} ({self.index.ntotal} vectors)")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


# Singleton instance
_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    """
    Get the global FAISS index instance (singleton pattern).

    Returns:
        FAISSIndex: Global FAISS index
    """
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex()
    return _faiss_index


def reset_faiss_index() -> None:
    """Reset the global FAISS index (useful for testing)."""
    global _faiss_index
    _faiss_index = None


if __name__ == "__main__":
    """Test the FAISS index."""
    import sys

    try:
        print("🔄 Initializing FAISS index...")
        index = get_faiss_index()

        print(f"\n📊 Index info:")
        print(f"  Dimension: {index.dimension}")
        print(f"  Model: {index.embedding_config.model_name}")
        print(f"  Current vectors: {index.get_count()}")

        # Test embedding
        print("\n🧪 Testing embedding...")
        sample_text = "Knowledge graphs represent information as nodes and relationships."
        embedding = index.embed_text(sample_text)
        print(f"  Text: '{sample_text}'")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding sample: {embedding[:5]}...")

        # Test adding and searching
        print("\n📝 Testing add and search...")
        test_texts = [
            "Neo4j is a graph database management system.",
            "FAISS enables efficient similarity search.",
            "Machine learning models can generate embeddings."
        ]
        ids = index.add(test_texts)
        print(f"  Added {len(ids)} texts with IDs: {ids}")

        # Search
        query = "What is Neo4j?"
        results = index.search(query, top_k=2)
        print(f"\n🔍 Search results for '{query}':")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Text: {result['metadata']['text']}")

        print(f"\n📈 Total vectors: {index.get_count()}")
        print("\n✅ FAISS index working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
