"""
Pipeline Orchestrator

Orchestrates agents in the proper sequence for knowledge graph construction:
1. Reader Agent (relevance scoring)
2. Entity Agent (entity extraction)
3. Summarizer Agent (contextual summaries)
4. Relation Agent (multi-label relationship extraction)
5. Schema Aligner (type validation) - rule-based, no LLM
6. Conflict Resolution Agent (LLM debate)
7. Confidence Evaluator (confidence aggregation) - symbolic, no LLM

NOTE: This is NOT an agent (no direct LLM calls). It orchestrates other
components and manages the pipeline workflow.
"""

import logging
from typing import List, Dict, Any, Optional
from src.utils.llm_client import LLMClient
from src.agents.reader_agent import ReaderAgent
from src.agents.entity_agent import EntityAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.multi_label_relation_agent import MultiLabelRelationAgent
from src.agents.schema_aligner import SchemaAligner
from src.agents.debate_conflict_resolver import DebateConflictResolver
from src.agents.confidence_evaluator import ConfidenceEvaluator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Pipeline Orchestrator (coordinator, no direct LLM calls).

    Orchestrates the full pipeline for knowledge graph construction.
    Delegates LLM calls to individual agents.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        relevance_threshold: float = 0.5,
        integration_threshold: float = 0.7,
        enable_conflict_resolution: bool = True,
        enable_schema_validation: bool = True
    ):
        """
        Initialize Pipeline Orchestrator.

        Args:
            llm_client: LLM client for agents
            relevance_threshold: Minimum relevance score for processing
            integration_threshold: Minimum score for integration
            enable_conflict_resolution: Enable conflict resolution
            enable_schema_validation: Enable schema alignment
        """
        self.llm_client = llm_client

        # Initialize agents (LLM-powered)
        self.reader = ReaderAgent(llm_client, relevance_threshold)
        self.entity_agent = EntityAgent(llm_client)
        self.summarizer = SummarizerAgent(llm_client)
        self.relation_agent = MultiLabelRelationAgent()
        self.conflict_agent = DebateConflictResolver(llm_client) if enable_conflict_resolution else None

        # Initialize non-agent components (no LLM)
        self.schema_aligner = SchemaAligner() if enable_schema_validation else None
        self.evaluator = ConfidenceEvaluator(integration_threshold)

        self.enable_conflict_resolution = enable_conflict_resolution
        self.enable_schema_validation = enable_schema_validation

        # Context management
        self.global_context = {
            'entities': {},
            'relationships': [],
            'summaries': [],
            'metadata': {}
        }

        logger.info("Initialized PipelineOrchestrator with all components")

    def process_document(
        self,
        document: Dict[str, Any],
        existing_entities: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a single document through the full KARMA pipeline.

        Args:
            document: Document to process (must have 'content' key)
            existing_entities: Existing entities for context (optional)

        Returns:
            Processed results with entities and relationships
        """
        doc_id = document.get('id', 'unknown')
        content = document.get('content', '')

        logger.info(f"Processing document {doc_id} through KARMA pipeline")

        # Track pipeline metrics
        metrics = {
            'doc_id': doc_id,
            'stages': {}
        }

        # Stage 1: Reader Agent - Relevance scoring
        logger.debug(f"[{doc_id}] Stage 1: Reader Agent")
        segments = self.reader.split_into_segments(content)
        scored_segments = self.reader.score_segments(segments)

        relevant_segments = [
            seg for seg in scored_segments
            if seg['relevance_score'] >= self.reader.relevance_threshold
        ]

        metrics['stages']['reader'] = {
            'total_segments': len(segments),
            'relevant_segments': len(relevant_segments),
            'avg_relevance': sum(s['relevance_score'] for s in scored_segments) / len(scored_segments) if scored_segments else 0
        }

        logger.info(f"[{doc_id}] Reader: {len(relevant_segments)}/{len(segments)} segments relevant")

        # If no relevant segments, skip
        if not relevant_segments:
            logger.info(f"[{doc_id}] No relevant segments, skipping document")
            return {
                'doc_id': doc_id,
                'entities': [],
                'relationships': [],
                'summary': None,
                'metrics': metrics
            }

        # Stage 2: Entity Agent - Entity extraction
        logger.debug(f"[{doc_id}] Stage 2: Entity Agent")
        entities = self.entity_agent.extract_entities(content)

        metrics['stages']['entities'] = {
            'count': len(entities)
        }

        logger.info(f"[{doc_id}] Extracted {len(entities)} entities")

        # Stage 3: Summarizer - Generate contextual summary
        logger.debug(f"[{doc_id}] Stage 3: Summarizer")
        summary = self.summarizer.summarize(content, entities)

        # Stage 4: Relation Agent - Multi-label relationship extraction
        logger.debug(f"[{doc_id}] Stage 4: KARMA Relation Agent (multi-label)")
        relationships = self.relation_agent.extract_relationships(
            content,
            entities,
            summary_context=summary
        )

        metrics['stages']['relationships_raw'] = {
            'count': len(relationships)
        }

        logger.info(f"[{doc_id}] Extracted {len(relationships)} relationships")

        # Stage 5: Schema Alignment (optional)
        if self.enable_schema_validation and self.schema_aligner:
            logger.debug(f"[{doc_id}] Stage 5: Schema Alignment")
            relationships, invalid_rels = self.schema_aligner.align_relationships(relationships)

            metrics['stages']['schema'] = {
                'valid': len(relationships),
                'invalid': len(invalid_rels)
            }

            logger.info(f"[{doc_id}] Schema validation: {len(relationships)} valid, {len(invalid_rels)} invalid")

        # Stage 6: Conflict Resolution (optional)
        if self.enable_conflict_resolution and self.conflict_agent and existing_entities:
            logger.debug(f"[{doc_id}] Stage 6: Conflict Resolution")

            # Build existing knowledge base
            existing_kb = {
                'entities': existing_entities,
                'relationships': self.global_context.get('relationships', [])
            }

            relationships = self.conflict_agent.resolve_conflicts(
                relationships,
                existing_kb
            )

            metrics['stages']['conflict_resolution'] = {
                'relationships_after': len(relationships)
            }

            logger.info(f"[{doc_id}] After conflict resolution: {len(relationships)} relationships")

        # Stage 7: Evaluator - Final scoring and filtering
        logger.debug(f"[{doc_id}] Stage 7: Evaluator")
        accepted_rels, rejected_rels = self.evaluator.evaluate_relationships(relationships)

        metrics['stages']['evaluator'] = {
            'accepted': len(accepted_rels),
            'rejected': len(rejected_rels),
            'acceptance_rate': len(accepted_rels) / len(relationships) if relationships else 0
        }

        logger.info(
            f"[{doc_id}] Evaluator: {len(accepted_rels)} accepted, "
            f"{len(rejected_rels)} rejected"
        )

        # Update global context
        self.global_context['entities'].update({e['name']: e for e in entities})
        self.global_context['relationships'].extend(accepted_rels)
        self.global_context['summaries'].append(summary)

        return {
            'doc_id': doc_id,
            'entities': entities,
            'relationships': accepted_rels,
            'rejected_relationships': rejected_rels,
            'summary': summary,
            'metrics': metrics
        }

    def process_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process multiple documents through KARMA pipeline.

        Args:
            documents: List of documents to process

        Returns:
            Aggregated results
        """
        all_entities = []
        all_relationships = []
        all_summaries = []
        document_metrics = []

        logger.info(f"Processing {len(documents)} documents through KARMA")

        for i, doc in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}")

            # Pass existing entities for conflict resolution
            result = self.process_document(doc, existing_entities=all_entities)

            all_entities.extend(result['entities'])
            all_relationships.extend(result['relationships'])
            all_summaries.append(result['summary'])
            document_metrics.append(result['metrics'])

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(document_metrics)

        logger.info(
            f"KARMA processing complete: {len(all_entities)} entities, "
            f"{len(all_relationships)} relationships"
        )

        return {
            'entities': all_entities,
            'relationships': all_relationships,
            'summaries': all_summaries,
            'metrics': aggregate_metrics
        }

    def _aggregate_metrics(self, document_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all documents."""
        aggregate = {
            'total_documents': len(document_metrics),
            'total_entities': 0,
            'total_relationships_accepted': 0,
            'total_relationships_rejected': 0,
            'avg_relevance': 0,
            'schema_validation': {'valid': 0, 'invalid': 0},
            'acceptance_rate': 0
        }

        for metrics in document_metrics:
            stages = metrics.get('stages', {})

            # Entities
            aggregate['total_entities'] += stages.get('entities', {}).get('count', 0)

            # Relationships
            evaluator = stages.get('evaluator', {})
            aggregate['total_relationships_accepted'] += evaluator.get('accepted', 0)
            aggregate['total_relationships_rejected'] += evaluator.get('rejected', 0)

            # Relevance
            reader = stages.get('reader', {})
            aggregate['avg_relevance'] += reader.get('avg_relevance', 0)

            # Schema
            schema = stages.get('schema', {})
            aggregate['schema_validation']['valid'] += schema.get('valid', 0)
            aggregate['schema_validation']['invalid'] += schema.get('invalid', 0)

        # Calculate averages
        if document_metrics:
            aggregate['avg_relevance'] /= len(document_metrics)

        total_rels = aggregate['total_relationships_accepted'] + aggregate['total_relationships_rejected']
        if total_rels > 0:
            aggregate['acceptance_rate'] = aggregate['total_relationships_accepted'] / total_rels

        aggregate['relationships_per_document'] = (
            aggregate['total_relationships_accepted'] / aggregate['total_documents']
            if aggregate['total_documents'] > 0 else 0
        )

        return aggregate
