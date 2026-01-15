"""
KARMA-Enabled Ingestion Pipeline

Enhanced ingestion pipeline using KARMA multi-agent system for improved:
- Entity extraction accuracy
- Relationship extraction recall (target: 2-3x improvement)
- Conflict resolution (18.6% reduction in contradictions)
- Schema compliance (31.2% improvement)

Based on KARMA: "Knowledge graph Augmentation from multi-agent RAG with MCP"
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.agents.karma_controller import KARMAController
from src.utils.llm_client import get_llm_client
from src.graph import get_neo4j_manager
from src.vector import get_faiss_index
from src.utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KARMAIngestionPipeline:
    """
    KARMA-enhanced document ingestion pipeline.

    Uses multi-agent system with:
    - Reader Agent (relevance scoring)
    - Entity Agent (entity extraction)
    - Summarizer (contextual summaries)
    - Relation Agent (multi-label relationship extraction)
    - Schema Alignment (type validation)
    - Conflict Resolution (LLM debate)
    - Evaluator (confidence aggregation)
    """

    def __init__(
        self,
        relevance_threshold: float = 0.5,
        integration_threshold: float = 0.7,
        enable_conflict_resolution: bool = True,
        enable_schema_validation: bool = True
    ):
        """
        Initialize KARMA ingestion pipeline.

        Args:
            relevance_threshold: Minimum relevance score for processing segments
            integration_threshold: Minimum confidence score for integration
            enable_conflict_resolution: Enable LLM-based conflict resolution
            enable_schema_validation: Enable schema alignment
        """
        self.config = get_config()

        # Initialize LLM client
        self.llm_client = get_llm_client()

        # Initialize KARMA controller
        self.karma_controller = KARMAController(
            llm_client=self.llm_client,
            relevance_threshold=relevance_threshold,
            integration_threshold=integration_threshold,
            enable_conflict_resolution=enable_conflict_resolution,
            enable_schema_validation=enable_schema_validation
        )

        # Initialize storage
        self.neo4j_manager = get_neo4j_manager()
        self.faiss_index = get_faiss_index()

        # State
        self.ingestion_stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "relationships_accepted": 0,
            "relationships_rejected": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "avg_relevance": 0,
            "acceptance_rate": 0
        }

        logger.info("Initialized KARMA Ingestion Pipeline")

    def ingest_documents(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Ingest documents using KARMA multi-agent system.

        Args:
            documents: List of document texts
            document_metadata: Optional metadata for each document
            batch_size: Process documents in batches for progress tracking

        Returns:
            Ingestion statistics and results
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return self.ingestion_stats

        logger.info(f"Starting KARMA ingestion of {len(documents)} documents...")
        logger.info(f"Configuration:")
        logger.info(f"  - Relevance threshold: {self.karma_controller.reader.relevance_threshold}")
        logger.info(f"  - Integration threshold: {self.karma_controller.evaluator.integration_threshold}")
        logger.info(f"  - Conflict resolution: {self.karma_controller.enable_conflict_resolution}")
        logger.info(f"  - Schema validation: {self.karma_controller.enable_schema_validation}")

        start_time = datetime.now()

        # Prepare document objects
        doc_objects = []
        for i, doc_text in enumerate(documents):
            doc_obj = {
                'id': f'doc_{i}',
                'content': doc_text
            }
            if document_metadata and i < len(document_metadata):
                doc_obj.update(document_metadata[i])
            doc_objects.append(doc_obj)

        # Process through KARMA
        logger.info("="*70)
        logger.info("Starting KARMA Multi-Agent Processing")
        logger.info("="*70)

        # Process in batches for progress tracking
        all_entities = []
        all_relationships = []
        all_summaries = []

        for batch_start in range(0, len(doc_objects), batch_size):
            batch_end = min(batch_start + batch_size, len(doc_objects))
            batch_docs = doc_objects[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1} (documents {batch_start+1}-{batch_end})")

            result = self.karma_controller.process_documents(batch_docs)

            all_entities.extend(result['entities'])
            all_relationships.extend(result['relationships'])
            all_summaries.extend(result['summaries'])

        logger.info("="*70)
        logger.info("KARMA Processing Complete")
        logger.info("="*70)

        # Get final metrics from KARMA controller
        karma_metrics = self.karma_controller._aggregate_metrics([
            {'stages': self.karma_controller.global_context.get('metadata', {})}
        ])

        # Build knowledge graph
        logger.info("Building knowledge graph in Neo4j...")
        nodes_created, edges_created = self._build_knowledge_graph(
            all_entities,
            all_relationships
        )
        logger.info(f"✓ Created {nodes_created} nodes and {edges_created} edges")

        # Build vector index
        logger.info("Building vector index...")
        vectors_created = self._build_vector_index(
            documents,
            all_entities,
            document_metadata
        )
        logger.info(f"✓ Indexed {vectors_created} vectors")

        # Calculate final stats
        elapsed = (datetime.now() - start_time).total_seconds()

        self.ingestion_stats.update({
            "documents_processed": len(documents),
            "entities_extracted": len(all_entities),
            "relationships_accepted": len(all_relationships),
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "elapsed_seconds": elapsed,
            "docs_per_second": len(documents) / elapsed if elapsed > 0 else 0
        })

        # Log final summary
        logger.info("="*70)
        logger.info("INGESTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Documents processed: {self.ingestion_stats['documents_processed']}")
        logger.info(f"Entities extracted: {self.ingestion_stats['entities_extracted']}")
        logger.info(f"Relationships extracted: {self.ingestion_stats['relationships_accepted']}")
        logger.info(f"Graph nodes created: {self.ingestion_stats['nodes_created']}")
        logger.info(f"Graph edges created: {self.ingestion_stats['edges_created']}")
        logger.info(f"Processing time: {elapsed:.2f}s ({self.ingestion_stats['docs_per_second']:.2f} docs/sec)")
        logger.info(f"Relationships per document: {len(all_relationships) / len(documents):.2f}")
        logger.info(f"Entities per document: {len(all_entities) / len(documents):.2f}")
        logger.info("="*70)

        return {
            'stats': self.ingestion_stats,
            'entities': all_entities,
            'relationships': all_relationships,
            'summaries': all_summaries
        }

    def _build_knowledge_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Build knowledge graph in Neo4j.

        Args:
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            Tuple of (nodes_created, edges_created)
        """
        # Create entity nodes
        nodes_created = 0
        for entity in entities:
            properties = {
                'name': entity['name'],
                'type': entity.get('type', 'ENTITY')
            }

            # Add metadata
            if 'metadata' in entity:
                properties.update(entity['metadata'])

            self.neo4j_manager.create_node(
                node_type=entity.get('type', 'ENTITY'),
                properties=properties
            )
            nodes_created += 1

        # Create relationships
        edges_created = 0
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            rel_type = rel.get('type', 'RELATED_TO')

            if not source or not target:
                continue

            properties = {}

            # Add confidence scores
            if 'evaluation' in rel:
                properties['confidence'] = rel['evaluation']['final_score']
                properties['clarity'] = rel['evaluation']['clarity']
                properties['relevance'] = rel['evaluation']['relevance']

            # Add evidence
            if 'evidence' in rel:
                properties['evidence'] = rel['evidence']

            # Add properties
            if 'properties' in rel and isinstance(rel['properties'], dict):
                properties.update(rel['properties'])

            self.neo4j_manager.create_relationship(
                source_node_name=source,
                target_node_name=target,
                relationship_type=rel_type,
                properties=properties
            )
            edges_created += 1

        return nodes_created, edges_created

    def _build_vector_index(
        self,
        documents: List[str],
        entities: List[Dict[str, Any]],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Build FAISS vector index for retrieval.

        Args:
            documents: Original documents
            entities: Extracted entities
            document_metadata: Optional metadata

        Returns:
            Number of vectors indexed
        """
        # Create vectors for documents
        for i, doc in enumerate(documents):
            metadata = {
                'doc_id': f'doc_{i}',
                'type': 'document'
            }

            if document_metadata and i < len(document_metadata):
                metadata.update(document_metadata[i])

            self.faiss_index.add_document(
                text=doc,
                metadata=metadata
            )

        # Create vectors for entities (for entity-centric retrieval)
        for entity in entities:
            entity_text = f"{entity['name']}: {entity.get('type', 'ENTITY')}"

            # Add context if available
            if 'metadata' in entity and 'description' in entity['metadata']:
                entity_text += f". {entity['metadata']['description']}"

            self.faiss_index.add_document(
                text=entity_text,
                metadata={
                    'entity_name': entity['name'],
                    'entity_type': entity.get('type', 'ENTITY'),
                    'type': 'entity'
                }
            )

        return len(documents) + len(entities)

    def save_results(self, output_path: Path, results: Dict[str, Any]):
        """
        Save ingestion results to JSON.

        Args:
            output_path: Path to save results
            results: Results dictionary
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make results JSON-serializable
        serializable_results = {
            'stats': results['stats'],
            'entities': results['entities'],
            'relationships': results['relationships'],
            'summaries': [s for s in results['summaries'] if s]  # Filter None
        }

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved results to: {output_path}")


def create_karma_pipeline(**kwargs) -> KARMAIngestionPipeline:
    """Factory function to create KARMA pipeline."""
    return KARMAIngestionPipeline(**kwargs)
