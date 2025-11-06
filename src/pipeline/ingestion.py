"""
Ingestion Pipeline for Agentic GraphRAG

This module orchestrates the document ingestion process:
1. Schema inference from documents
2. Entity extraction with metadata enrichment
3. Relationship extraction
4. Knowledge graph construction (Neo4j)
5. Vector index creation (FAISS)

Author: Agentic GraphRAG Team
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..agents import (
    get_schema_agent,
    get_entity_agent,
    get_relation_agent
)
from ..graph import get_neo4j_manager
from ..vector import get_faiss_index
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Stages:
    1. Schema Inference: Analyze documents to discover graph structure
    2. Entity Extraction: Extract entities with metadata enrichment
    3. Relationship Extraction: Identify connections between entities
    4. Graph Construction: Build Neo4j knowledge graph
    5. Vector Indexing: Create FAISS embeddings for retrieval
    """

    def __init__(
        self,
        schema_path: Optional[Path] = None,
        auto_refine_schema: bool = True
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            schema_path: Optional path to load/save schema
            auto_refine_schema: Whether to automatically refine schema with new documents
        """
        self.config = get_config()
        self.schema_path = schema_path
        self.auto_refine_schema = auto_refine_schema

        # Initialize agents
        self.schema_agent = get_schema_agent()
        self.entity_agent = None  # Created after schema inference
        self.relation_agent = None  # Created after schema inference

        # Initialize storage
        self.neo4j_manager = get_neo4j_manager()
        self.faiss_index = get_faiss_index()

        # State
        self.schema: Optional[Dict[str, Any]] = None
        self.ingestion_stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "nodes_created": 0,
            "edges_created": 0
        }

        # Load existing schema if available
        if schema_path and schema_path.exists():
            self._load_schema()

        logger.info("Initialized IngestionPipeline")

    def ingest_documents(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        infer_schema: bool = True,
        enrich_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest documents into the system.

        Args:
            documents: List of document texts
            document_metadata: Optional metadata for each document
            infer_schema: Whether to infer/refine schema
            enrich_metadata: Whether to enrich entities with metadata

        Returns:
            Ingestion statistics and results
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return self.ingestion_stats

        logger.info(f"Starting ingestion of {len(documents)} documents...")
        start_time = datetime.now()

        # Stage 1: Schema Inference
        if infer_schema:
            self.schema = self._infer_or_refine_schema(documents)
            logger.info("✓ Schema inference complete")
        elif self.schema is None:
            raise ValueError("No schema available. Set infer_schema=True or load a schema.")

        # Initialize agents with schema
        self.entity_agent = get_entity_agent(schema=self.schema)
        self.relation_agent = get_relation_agent(schema=self.schema)

        # Stage 2: Entity Extraction
        logger.info("Extracting entities...")
        all_entities = []
        entities_per_doc = []

        for i, doc in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")

            entities = self.entity_agent.extract_entities_from_text(
                doc, enrich_metadata=enrich_metadata
            )
            all_entities.extend(entities)
            entities_per_doc.append(entities)

        logger.info(f"✓ Extracted {len(all_entities)} entities")

        # Stage 3: Relationship Extraction
        logger.info("Extracting relationships...")
        all_relations = self.relation_agent.extract_relations_from_documents(
            documents, entities_per_doc, schema=self.schema
        )
        logger.info(f"✓ Extracted {len(all_relations)} relationships")

        # Stage 4: Graph Construction
        logger.info("Building knowledge graph...")
        nodes_created, edges_created = self._build_knowledge_graph(all_entities, all_relations)
        logger.info(f"✓ Created {nodes_created} nodes and {edges_created} edges")

        # Stage 5: Vector Indexing
        logger.info("Building vector index...")
        vectors_created = self._build_vector_index(documents, all_entities, document_metadata)
        logger.info(f"✓ Indexed {vectors_created} vectors")

        # Update stats
        self.ingestion_stats["documents_processed"] += len(documents)
        self.ingestion_stats["entities_extracted"] += len(all_entities)
        self.ingestion_stats["relations_extracted"] += len(all_relations)
        self.ingestion_stats["nodes_created"] += nodes_created
        self.ingestion_stats["edges_created"] += edges_created

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Ingestion complete in {duration:.2f}s")

        # Save schema if path provided
        if self.schema_path:
            self._save_schema()

        return {
            **self.ingestion_stats,
            "duration_seconds": duration,
            "schema": self.schema
        }

    def _infer_or_refine_schema(self, documents: List[str]) -> Dict[str, Any]:
        """
        Infer new schema or refine existing one.

        Args:
            documents: Documents to analyze

        Returns:
            Schema dictionary
        """
        if self.schema is None:
            # First time: infer schema
            logger.info("Inferring schema from documents...")
            schema = self.schema_agent.infer_schema_from_documents(
                documents, sample_size=min(50, len(documents))
            )
        elif self.auto_refine_schema:
            # Refine existing schema
            logger.info("Refining existing schema...")
            schema = self.schema_agent.refine_schema(
                self.schema, documents[:20]  # Sample for refinement
            )
        else:
            # Use existing schema
            schema = self.schema

        return schema

    def _build_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Build knowledge graph in Neo4j.

        Args:
            entities: Extracted entities
            relations: Extracted relationships

        Returns:
            Tuple of (nodes_created, edges_created)
        """
        nodes_created = 0
        edges_created = 0

        # Create entity nodes
        entity_lookup = {}  # Map (name, type) -> node properties

        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "Entity")
            properties = entity.get("properties", {})

            # Add core properties
            node_props = {
                "name": name,
                **properties
            }

            # Add metadata if present
            if "metadata" in entity:
                metadata = entity["metadata"]
                node_props["summary"] = metadata.get("summary", "")
                node_props["keywords"] = ",".join(metadata.get("keywords", []))

            # Create or merge node
            try:
                node = self.neo4j_manager.create_node(
                    label=entity_type,
                    properties=node_props,
                    merge=True  # Avoid duplicates
                )
                entity_lookup[(name.lower(), entity_type)] = node
                nodes_created += 1
            except Exception as e:
                logger.error(f"Error creating node for {name}: {e}")

        # Create relationship edges
        for relation in relations:
            source_name = relation.get("source", "").lower()
            target_name = relation.get("target", "").lower()
            rel_type = relation.get("type", "RELATED_TO")
            rel_props = relation.get("properties", {})

            # Add confidence
            rel_props["confidence"] = relation.get("confidence", 0.5)

            # Find source and target nodes
            source_node = None
            target_node = None

            # Try to find in entity lookup
            for (name, _), node in entity_lookup.items():
                if name == source_name:
                    source_node = node
                if name == target_name:
                    target_node = node

            if not source_node or not target_node:
                continue

            # Create relationship
            try:
                self.neo4j_manager.create_relationship(
                    from_label=source_node.get("type", "Entity"),
                    from_properties={"name": source_name},
                    to_label=target_node.get("type", "Entity"),
                    to_properties={"name": target_name},
                    relationship_type=rel_type,
                    relationship_properties=rel_props,
                    merge=True
                )
                edges_created += 1
            except Exception as e:
                logger.error(f"Error creating relationship {source_name}->{target_name}: {e}")

        return nodes_created, edges_created

    def _build_vector_index(
        self,
        documents: List[str],
        entities: List[Dict[str, Any]],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Build vector index for documents and entities.

        Args:
            documents: Document texts
            entities: Extracted entities
            document_metadata: Optional document metadata

        Returns:
            Number of vectors created
        """
        vectors_created = 0

        # Index documents
        doc_metadata = document_metadata or [{}] * len(documents)
        for i, (doc, meta) in enumerate(zip(documents, doc_metadata)):
            # Chunk document
            chunks = self._chunk_document(doc)

            for j, chunk in enumerate(chunks):
                chunk_meta = {
                    **meta,
                    "type": "document_chunk",
                    "doc_index": i,
                    "chunk_index": j,
                    "text": chunk
                }

                # Add to index
                self.faiss_index.add_single(chunk, metadata=chunk_meta)
                vectors_created += 1

        # Index entities (with their enriched metadata)
        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "")

            # Create searchable text for entity
            entity_text = f"{name} ({entity_type})"

            if "metadata" in entity:
                summary = entity["metadata"].get("summary", "")
                keywords = ", ".join(entity["metadata"].get("keywords", []))
                entity_text += f". {summary}. Keywords: {keywords}"

            entity_meta = {
                "type": "entity",
                "entity_name": name,
                "entity_type": entity_type,
                "text": entity_text,
                **entity.get("metadata", {})
            }

            self.faiss_index.add_single(entity_text, metadata=entity_meta)
            vectors_created += 1

        return vectors_created

    def _chunk_document(
        self,
        document: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document text
            chunk_size: Optional chunk size (uses config default)
            chunk_overlap: Optional overlap size (uses config default)

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.embedding.chunk_size
        chunk_overlap = chunk_overlap or self.config.embedding.chunk_overlap

        if len(document) <= chunk_size:
            return [document]

        chunks = []
        start = 0

        while start < len(document):
            end = start + chunk_size
            chunk = document[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap

        return chunks

    def _load_schema(self) -> None:
        """Load schema from file."""
        try:
            with open(self.schema_path, "r") as f:
                self.schema = json.load(f)
            logger.info(f"Loaded schema from {self.schema_path}")
        except Exception as e:
            logger.error(f"Error loading schema: {e}")

    def _save_schema(self) -> None:
        """Save schema to file."""
        if not self.schema or not self.schema_path:
            return

        try:
            self.schema_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.schema_path, "w") as f:
                json.dump(self.schema, f, indent=2)
            logger.info(f"Saved schema to {self.schema_path}")
        except Exception as e:
            logger.error(f"Error saving schema: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.ingestion_stats,
            "schema_summary": {
                "entity_types": len(self.schema.get("entity_types", [])) if self.schema else 0,
                "relation_types": len(self.schema.get("relation_types", [])) if self.schema else 0
            },
            "neo4j_stats": {
                "total_nodes": self.neo4j_manager.get_node_count(),
                "total_relationships": self.neo4j_manager.get_relationship_count()
            },
            "faiss_stats": {
                "total_vectors": self.faiss_index.get_count()
            }
        }


# Singleton instance
_ingestion_pipeline: Optional[IngestionPipeline] = None


def get_ingestion_pipeline(
    schema_path: Optional[Path] = None,
    auto_refine_schema: bool = True
) -> IngestionPipeline:
    """
    Get the global IngestionPipeline instance (singleton pattern).

    Args:
        schema_path: Optional path to schema file
        auto_refine_schema: Whether to auto-refine schema

    Returns:
        IngestionPipeline: Global ingestion pipeline
    """
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(
            schema_path=schema_path,
            auto_refine_schema=auto_refine_schema
        )
    return _ingestion_pipeline


if __name__ == "__main__":
    """Test the ingestion pipeline."""
    import sys

    try:
        print("🔄 Initializing ingestion pipeline...")
        pipeline = get_ingestion_pipeline()

        # Test documents
        test_docs = [
            """Dr. Jane Smith works at MIT in the Computer Science department.
            She researches machine learning and natural language processing.
            MIT is located in Cambridge, Massachusetts.""",

            """Tesla manufactures electric vehicles. Elon Musk is the CEO of Tesla.
            The company was founded in 2003 and is headquartered in Austin, Texas.""",
        ]

        print("\n📚 Ingesting sample documents...")
        results = pipeline.ingest_documents(test_docs, infer_schema=True, enrich_metadata=True)

        print(f"\n✅ Ingestion complete!")
        print(f"  Documents processed: {results['documents_processed']}")
        print(f"  Entities extracted: {results['entities_extracted']}")
        print(f"  Relations extracted: {results['relations_extracted']}")
        print(f"  Nodes created: {results['nodes_created']}")
        print(f"  Edges created: {results['edges_created']}")
        print(f"  Duration: {results['duration_seconds']:.2f}s")

        print("\n📊 System statistics:")
        stats = pipeline.get_statistics()
        print(f"  Entity types: {stats['schema_summary']['entity_types']}")
        print(f"  Relation types: {stats['schema_summary']['relation_types']}")
        print(f"  Total Neo4j nodes: {stats['neo4j_stats']['total_nodes']}")
        print(f"  Total Neo4j relationships: {stats['neo4j_stats']['total_relationships']}")
        print(f"  Total FAISS vectors: {stats['faiss_stats']['total_vectors']}")

        print("\n✅ Ingestion pipeline working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
