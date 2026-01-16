"""
Ingestion Pipeline for Agentic GraphRAG

This module orchestrates the document ingestion process:
1. Schema inference from documents
2. Entity extraction with metadata enrichment
3. Relationship extraction
4. Knowledge graph construction (Neo4j) with sentence-level provenance
5. Vector index creation (FAISS) at sentence level

Author: Agentic GraphRAG Team
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..agents import (
    get_schema_agent,
    get_entity_agent,
    get_relation_agent
)
from ..agents.conflict_resolution_agent import get_conflict_resolution_agent
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
    End-to-end document ingestion pipeline with multi-stage validation.

    Stages:
    1. Schema Inference: Analyze documents to discover graph structure
    2. Entity Extraction: Extract entities with metadata enrichment
    3. Relationship Extraction: Identify connections between entities
    4. Conflict Resolution: Deduplicate entities and resolve contradictions
    5. Schema Validation: Validate against inferred schema
    6. Graph Construction: Build Neo4j knowledge graph
    7. Vector Indexing: Create FAISS embeddings for retrieval
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
        self.conflict_resolver = None  # Created after schema inference

        # Initialize storage
        self.neo4j_manager = get_neo4j_manager()
        self.faiss_index = get_faiss_index()

        # State
        self.schema: Optional[Dict[str, Any]] = None
        self.ingestion_stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "entities_after_dedup": 0,
            "relations_extracted": 0,
            "relations_after_resolution": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "conflicts_resolved": 0
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
        self.conflict_resolver = get_conflict_resolution_agent(schema=self.schema)

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

        # Stage 4: Conflict Resolution (Entity Deduplication)
        logger.info("Deduplicating entities...")
        original_entity_count = len(all_entities)
        all_entities = self.conflict_resolver.deduplicate_entities(
            all_entities, use_llm=False  # Use string matching only for speed
        )
        duplicates_merged = original_entity_count - len(all_entities)
        logger.info(f"✓ Deduplicated {original_entity_count} → {len(all_entities)} entities ({duplicates_merged} merged)")

        # Stage 5: Conflict Resolution (Relationship Conflicts)
        logger.info("Resolving relationship conflicts...")
        original_rel_count = len(all_relations)
        all_relations = self.conflict_resolver.resolve_relationship_conflicts(all_relations)
        conflicts_resolved = original_rel_count - len(all_relations)
        logger.info(f"✓ Resolved {original_rel_count} → {len(all_relations)} relationships ({conflicts_resolved} conflicts)")

        # Stage 6: Schema Validation
        logger.info("Validating against schema...")
        all_entities, invalid_entities = self.conflict_resolver.validate_against_schema(
            all_entities, strict=False  # Allow entities not in schema with warning
        )
        if invalid_entities:
            logger.warning(f"Found {len(invalid_entities)} entities not matching schema")

        # Update conflict resolution stats
        self.ingestion_stats["conflicts_resolved"] = duplicates_merged + conflicts_resolved

        # Stage 7: Sentence Tokenization (for provenance)
        logger.info("Tokenizing documents into sentences...")
        all_sentences = []
        for doc_id, doc in enumerate(documents):
            sentences = self._sentence_tokenize(doc, doc_id)
            all_sentences.extend(sentences)
        logger.info(f"✓ Tokenized {len(documents)} documents into {len(all_sentences)} sentences")

        # Stage 8: Graph Construction (with sentence-level provenance)
        logger.info("Building knowledge graph with provenance...")
        nodes_created, edges_created = self._build_knowledge_graph(
            all_entities, all_relations, all_sentences, documents, document_metadata
        )
        logger.info(f"✓ Created {nodes_created} nodes and {edges_created} edges")

        # Stage 9: Vector Indexing (at sentence level)
        logger.info("Building sentence-level vector index...")
        vectors_created, _ = self._build_vector_index(
            documents, all_entities, document_metadata, all_sentences
        )
        logger.info(f"✓ Indexed {vectors_created} vectors (sentences + entities)")

        # Update stats
        self.ingestion_stats["documents_processed"] += len(documents)
        self.ingestion_stats["entities_extracted"] += original_entity_count
        self.ingestion_stats["entities_after_dedup"] += len(all_entities)
        self.ingestion_stats["relations_extracted"] += original_rel_count
        self.ingestion_stats["relations_after_resolution"] += len(all_relations)
        self.ingestion_stats["nodes_created"] += nodes_created
        self.ingestion_stats["edges_created"] += edges_created

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Ingestion complete in {duration:.2f}s")

        # Save schema if path provided
        if self.schema_path:
            self._save_schema()

        # Save FAISS index to disk
        logger.info("Saving FAISS index to disk...")
        self.faiss_index.save()
        logger.info("✓ FAISS index saved")

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
        relations: List[Dict[str, Any]],
        all_sentences: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[int, int]:
        """
        Build knowledge graph in Neo4j with sentence-level provenance.

        Each node and relationship stores provenance as a LIST of {doc_id, sent_id}
        since entities/relations may appear in multiple sentences.

        Args:
            entities: Extracted entities
            relations: Extracted relationships
            all_sentences: Tokenized sentences for provenance tracking

        Returns:
            Tuple of (nodes_created, edges_created)
        """
        nodes_created = 0
        edges_created = 0
        all_sentences = all_sentences or []

        # Create Document nodes
        if documents:
            logger.info(f"Creating {len(documents)} Document nodes...")
            for i, doc_text in enumerate(documents):
                doc_id = i
                meta = document_metadata[i] if document_metadata and i < len(document_metadata) else {}
                title = meta.get("title") or meta.get("doc_title") or f"Document {i}"

                props = {
                    "doc_id": doc_id,
                    "title": title,
                    "length": len(doc_text)
                }
                
                try:
                    self.neo4j_manager.create_node(
                        label="Document",
                        properties=props,
                        merge=True
                    )
                    nodes_created += 1
                except Exception as e:
                    logger.error(f"Error creating Document node {doc_id}: {e}")

        # Create entity nodes
        entity_lookup = {}  # Map (name, type) -> (node properties, entity_type)

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
                node_props["aliases"] = ",".join(metadata.get("aliases", []))

            # Find provenance for this entity (LIST of {doc_id, sent_id})
            if all_sentences:
                provenance = self._find_sentence_for_entity(name, all_sentences)
                # Store as JSON string for Neo4j compatibility
                node_props["provenance"] = json.dumps(provenance) if provenance else "[]"
            else:
                node_props["provenance"] = "[]"

            # Create or merge node
            try:
                node = self.neo4j_manager.create_node(
                    label=entity_type,
                    properties=node_props,
                    merge=True  # Avoid duplicates
                )
                # Store both node and entity_type
                entity_lookup[(name.lower(), entity_type)] = (node, entity_type)
                nodes_created += 1
            except Exception as e:
                logger.error(f"Error creating node for {name}: {e}")

            # Link to Document (MENTIONED_IN)
            if "provenance" in node_props and node_props["provenance"] != "[]":
                try:
                    prov_list = json.loads(node_props["provenance"])
                    # Get unique doc_ids from provenance
                    doc_ids = set(p["doc_id"] for p in prov_list)
                    
                    for doc_id in doc_ids:
                        self.neo4j_manager.create_relationship(
                            from_label=entity_type,
                            from_properties={"name": name},
                            to_label="Document",
                            to_properties={"doc_id": doc_id},
                            relationship_type="MENTIONED_IN",
                            merge=True
                        )
                        edges_created += 1
                except Exception as e:
                    logger.warning(f"Error linking entity {name} to document: {e}")

        # Create relationship edges
        for relation in relations:
            source_name = relation.get("source", "").lower()
            target_name = relation.get("target", "").lower()
            rel_type = relation.get("type", "RELATED_TO")
            rel_props = relation.get("properties", {})

            # Add confidence
            rel_props["confidence"] = relation.get("confidence", 0.5)

            # Find provenance for this relationship (sentences mentioning both entities)
            if all_sentences:
                rel_provenance = self._find_sentence_for_relationship(
                    relation.get("source", ""),
                    relation.get("target", ""),
                    all_sentences
                )
                rel_props["provenance"] = json.dumps(rel_provenance) if rel_provenance else "[]"
            else:
                rel_props["provenance"] = "[]"

            # Find source and target nodes with their types
            source_node = None
            source_type = None
            target_node = None
            target_type = None

            # Try to find in entity lookup
            for (name, etype), (node, node_type) in entity_lookup.items():
                if name == source_name and source_node is None:
                    source_node = node
                    source_type = node_type
                if name == target_name and target_node is None:
                    target_node = node
                    target_type = node_type

            if not source_node or not target_node:
                logger.debug(f"Skipping relationship {source_name}->{target_name}: nodes not found")
                continue

            # Create relationship
            try:
                self.neo4j_manager.create_relationship(
                    from_label=source_type,
                    from_properties={"name": source_node.get("name")},
                    to_label=target_type,
                    to_properties={"name": target_node.get("name")},
                    relationship_type=rel_type,
                    relationship_properties=rel_props,
                    merge=True
                )
                edges_created += 1
            except Exception as e:
                logger.error(f"Error creating relationship {source_name}->{target_name}: {e}")

        return nodes_created, edges_created

    def _find_sentence_for_relationship(
        self,
        source_name: str,
        target_name: str,
        sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, int]]:
        """
        Find sentences where both source and target entities co-occur.

        Args:
            source_name: Source entity name
            target_name: Target entity name
            sentences: List of sentence dicts from _sentence_tokenize

        Returns:
            List of {doc_id, sent_id} where both entities appear
        """
        provenance = []
        source_lower = source_name.lower()
        target_lower = target_name.lower()

        for sent in sentences:
            text_lower = sent["sentence_text"].lower()
            if source_lower in text_lower and target_lower in text_lower:
                provenance.append({
                    "doc_id": sent["doc_id"],
                    "sent_id": sent["sent_id"]
                })

        return provenance

    def _build_vector_index(
        self,
        documents: List[str],
        entities: List[Dict[str, Any]],
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        all_sentences: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Build vector index at SENTENCE level for HotpotQA provenance.

        Each sentence is indexed with:
        - doc_id: document index
        - sent_id: sentence index within document
        - sentence_text: exact sentence text (no trimming)

        Args:
            documents: Document texts
            entities: Extracted entities
            document_metadata: Optional document metadata
            all_sentences: Pre-tokenized sentences (if None, will tokenize)

        Returns:
            Tuple of (vectors_created, all_sentences for reuse in graph building)
        """
        # Collect all texts and metadata for batch insertion
        all_texts = []
        all_metadata = []

        # Tokenize documents into sentences if not provided
        if all_sentences is None:
            all_sentences = []
            for doc_id, doc in enumerate(documents):
                sentences = self._sentence_tokenize(doc, doc_id)
                all_sentences.extend(sentences)
            logger.info(f"Tokenized {len(documents)} documents into {len(all_sentences)} sentences")

        # Index sentences (instead of arbitrary chunks)
        doc_metadata_list = document_metadata or [{}] * len(documents)
        for sent in all_sentences:
            doc_id = sent["doc_id"]
            sent_id = sent["sent_id"]
            sentence_text = sent["sentence_text"]

            # Skip empty sentences
            if not sentence_text.strip():
                continue

            # Get document-level metadata if available
            doc_meta = doc_metadata_list[doc_id] if doc_id < len(doc_metadata_list) else {}

            sent_meta = {
                **doc_meta,
                "type": "sentence",
                "doc_id": doc_id,
                "sent_id": sent_id,
                "sentence_text": sentence_text,  # EXACT text
                "text": sentence_text  # For backward compatibility
            }
            all_texts.append(sentence_text)
            all_metadata.append(sent_meta)

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

            # Find provenance for this entity
            provenance = self._find_sentence_for_entity(name, all_sentences)

            entity_meta = {
                "type": "entity",
                "entity_name": name,
                "entity_type": entity_type,
                "text": entity_text,
                "provenance": provenance,  # List of {doc_id, sent_id}
                **entity.get("metadata", {})
            }
            all_texts.append(entity_text)
            all_metadata.append(entity_meta)

        # Add all vectors in batches to avoid memory fragmentation
        batch_size = 50  # Add in batches of 50 to prevent segfault
        vectors_created = 0
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_metadata = all_metadata[i:i+batch_size]
            self.faiss_index.add(batch_texts, metadata=batch_metadata)
            vectors_created += len(batch_texts)
            logger.info(f"Added batch {i//batch_size + 1} ({vectors_created}/{len(all_texts)} vectors)")

        return vectors_created, all_sentences

    def _chunk_document(
        self,
        document: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Chunk a document into smaller pieces.
        DEPRECATED: Use _sentence_tokenize for HotpotQA sentence-level provenance.

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

    def _sentence_tokenize(self, document: str, doc_id: int) -> List[Dict[str, Any]]:
        """
        Tokenize document into sentences with stable IDs for HotpotQA provenance.

        Fallback chain:
        1. NLTK punkt (with auto-download guard)
        2. spaCy sentencizer (if available)
        3. Regex fallback (last resort)

        Stores EXACT text (no trimming) - critical for EM evaluation.

        Args:
            document: Document text
            doc_id: Document index for provenance

        Returns:
            List of dicts with doc_id, sent_id, sentence_text
        """
        sentences = []

        # Try NLTK first (with download guard)
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
                try:
                    nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
                except Exception:
                    pass

            sentences = nltk.sent_tokenize(document)
            logger.debug(f"NLTK tokenized doc {doc_id} into {len(sentences)} sentences")
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}")
            sentences = []

        # Fallback to spaCy if NLTK failed
        if not sentences:
            try:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
                except OSError:
                    nlp = spacy.blank("en")

                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")

                doc = nlp(document)
                sentences = [sent.text for sent in doc.sents]
                logger.info(f"Used spaCy sentencizer as fallback for doc {doc_id}")
            except Exception as e:
                logger.warning(f"spaCy sentence tokenization failed: {e}")
                sentences = []

        # Last resort: regex fallback
        if not sentences:
            # Split on .!? followed by whitespace and capital letter (or end)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', document)
            if not sentences or (len(sentences) == 1 and not sentences[0].strip()):
                # If no splits or empty, treat whole doc as one sentence
                sentences = [document] if document.strip() else []
            logger.warning(f"Used regex fallback for sentence tokenization on doc {doc_id}")

        # Build output with EXACT text (no modification)
        return [
            {
                "doc_id": doc_id,
                "sent_id": sent_idx,
                "sentence_text": sent  # EXACT, no strip/modification
            }
            for sent_idx, sent in enumerate(sentences)
        ]

    def _find_sentence_for_entity(
        self,
        entity_name: str,
        sentences: List[Dict[str, Any]]
    ) -> List[Dict[str, int]]:
        """
        Find which sentences contain an entity (for provenance tracking).

        Args:
            entity_name: Entity name to search for
            sentences: List of sentence dicts from _sentence_tokenize

        Returns:
            List of {doc_id, sent_id} where entity appears
        """
        provenance = []
        entity_lower = entity_name.lower()

        for sent in sentences:
            if entity_lower in sent["sentence_text"].lower():
                provenance.append({
                    "doc_id": sent["doc_id"],
                    "sent_id": sent["sent_id"]
                })

        return provenance

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
