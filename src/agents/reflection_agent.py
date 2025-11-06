"""
ReflectionAgent for Agentic GraphRAG

This agent evaluates and optimizes system performance:
- Computes RAGAS metrics (faithfulness, relevancy, precision, recall)
- Identifies failure patterns and bottlenecks
- Suggests schema refinements
- Triggers parameter adjustments and retraining

Author: Agentic GraphRAG Team
"""

import logging
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from ..utils.llm_client import get_llm_client
from ..utils.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReflectionAgent:
    """
    Agent responsible for system evaluation and self-optimization.

    Performs:
    1. RAGAS metric computation (faithfulness, relevancy, precision, recall)
    2. Failure pattern analysis
    3. Schema refinement suggestions
    4. Parameter optimization recommendations
    """

    def __init__(self):
        """Initialize the ReflectionAgent."""
        self.llm_client = get_llm_client()
        self.config = get_config()

        # Performance history
        self.evaluation_history: List[Dict[str, Any]] = []
        self.failure_patterns: Dict[str, int] = defaultdict(int)

        logger.info("Initialized ReflectionAgent")

    def evaluate_response(
        self,
        query: str,
        response: str,
        retrieved_context: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single response using RAGAS metrics.

        Args:
            query: User query
            response: Generated response
            retrieved_context: Retrieved context items
            ground_truth: Optional ground truth answer for comparison

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # 1. Faithfulness: Is the response grounded in the context?
        metrics["faithfulness"] = self._compute_faithfulness(response, retrieved_context)

        # 2. Answer Relevancy: Does the answer address the query?
        metrics["answer_relevancy"] = self._compute_answer_relevancy(query, response)

        # 3. Context Precision: Is the retrieved context relevant?
        metrics["context_precision"] = self._compute_context_precision(
            query, retrieved_context, response
        )

        # 4. Context Recall: Is all necessary context retrieved?
        if ground_truth:
            metrics["context_recall"] = self._compute_context_recall(
                ground_truth, retrieved_context
            )

        # Compute overall score
        metrics["overall"] = sum(metrics.values()) / len(metrics)

        # Record in history
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "metrics": metrics
        })

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def _compute_faithfulness(
        self,
        response: str,
        retrieved_context: List[Dict[str, Any]]
    ) -> float:
        """
        Compute faithfulness: Are claims in the response supported by context?

        Args:
            response: Generated response
            retrieved_context: Retrieved context

        Returns:
            Faithfulness score (0.0 to 1.0)
        """
        if not response or not retrieved_context:
            return 0.0

        # Build context string
        context_texts = [
            item.get("text", item.get("content", ""))
            for item in retrieved_context
        ]
        context_str = "\n\n".join(context_texts)

        prompt = f"""Evaluate if the following response is faithful to the given context.
Check if all claims in the response are supported by the context.

Context:
{context_str[:2000]}...

Response:
{response}

Rate faithfulness from 0.0 to 1.0:
- 1.0: All claims are supported by context
- 0.5: Some claims are supported
- 0.0: Claims are not supported or contradict context

Return JSON: {{"faithfulness": 0.85, "reasoning": "brief explanation"}}"""

        try:
            result = self.llm_client.generate_json(prompt, temperature=0.0)
            return float(result.get("faithfulness", 0.5))
        except Exception as e:
            logger.error(f"Error computing faithfulness: {e}")
            return 0.5

    def _compute_answer_relevancy(self, query: str, response: str) -> float:
        """
        Compute answer relevancy: Does the response address the query?

        Args:
            query: User query
            response: Generated response

        Returns:
            Relevancy score (0.0 to 1.0)
        """
        if not response:
            return 0.0

        prompt = f"""Evaluate if the following response is relevant to the query.

Query: {query}

Response: {response}

Rate relevancy from 0.0 to 1.0:
- 1.0: Directly answers the query
- 0.5: Partially relevant
- 0.0: Not relevant or off-topic

Return JSON: {{"relevancy": 0.9, "reasoning": "brief explanation"}}"""

        try:
            result = self.llm_client.generate_json(prompt, temperature=0.0)
            return float(result.get("relevancy", 0.5))
        except Exception as e:
            logger.error(f"Error computing relevancy: {e}")
            return 0.5

    def _compute_context_precision(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        response: str
    ) -> float:
        """
        Compute context precision: Is the retrieved context relevant?

        Args:
            query: User query
            retrieved_context: Retrieved context
            response: Generated response

        Returns:
            Precision score (0.0 to 1.0)
        """
        if not retrieved_context:
            return 0.0

        # Sample top contexts
        sample_size = min(5, len(retrieved_context))
        sample_contexts = retrieved_context[:sample_size]

        relevant_count = 0
        for context in sample_contexts:
            text = context.get("text", context.get("content", ""))

            prompt = f"""Is the following context relevant for answering the query?

Query: {query}

Context: {text[:500]}

Answer with JSON: {{"relevant": true/false, "reasoning": "brief explanation"}}"""

            try:
                result = self.llm_client.generate_json(prompt, temperature=0.0)
                if result.get("relevant", False):
                    relevant_count += 1
            except Exception as e:
                logger.error(f"Error checking context relevance: {e}")

        precision = relevant_count / sample_size if sample_size > 0 else 0.0
        return precision

    def _compute_context_recall(
        self,
        ground_truth: str,
        retrieved_context: List[Dict[str, Any]]
    ) -> float:
        """
        Compute context recall: Is all necessary information retrieved?

        Args:
            ground_truth: Ground truth answer
            retrieved_context: Retrieved context

        Returns:
            Recall score (0.0 to 1.0)
        """
        if not ground_truth or not retrieved_context:
            return 0.0

        context_texts = [
            item.get("text", item.get("content", ""))
            for item in retrieved_context
        ]
        context_str = "\n\n".join(context_texts[:5])

        prompt = f"""Does the retrieved context contain all information needed to answer with the ground truth?

Ground Truth: {ground_truth}

Retrieved Context:
{context_str[:2000]}...

Rate recall from 0.0 to 1.0:
- 1.0: All necessary information is present
- 0.5: Some information is present
- 0.0: Missing critical information

Return JSON: {{"recall": 0.8, "reasoning": "brief explanation"}}"""

        try:
            result = self.llm_client.generate_json(prompt, temperature=0.0)
            return float(result.get("recall", 0.5))
        except Exception as e:
            logger.error(f"Error computing recall: {e}")
            return 0.5

    def analyze_failures(
        self,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Analyze failure patterns from evaluation history.

        Args:
            threshold: Score threshold below which to consider a failure

        Returns:
            Analysis report
        """
        if not self.evaluation_history:
            return {"message": "No evaluation history available"}

        failures = []
        metric_failures = defaultdict(list)

        for eval_item in self.evaluation_history:
            metrics = eval_item.get("metrics", {})
            overall = metrics.get("overall", 0.0)

            if overall < threshold:
                failures.append(eval_item)

                # Track which metrics failed
                for metric, score in metrics.items():
                    if score < threshold and metric != "overall":
                        metric_failures[metric].append(eval_item["query"])

        failure_rate = len(failures) / len(self.evaluation_history)

        analysis = {
            "total_evaluations": len(self.evaluation_history),
            "failures": len(failures),
            "failure_rate": failure_rate,
            "failing_metrics": {
                metric: len(queries) for metric, queries in metric_failures.items()
            },
            "sample_failures": failures[:5]  # Sample of failures
        }

        logger.info(f"Failure analysis: {failure_rate*100:.1f}% failure rate")
        return analysis

    def suggest_improvements(
        self,
        failure_analysis: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Suggest improvements based on performance analysis.

        Args:
            failure_analysis: Optional failure analysis results

        Returns:
            List of improvement suggestions
        """
        if failure_analysis is None:
            failure_analysis = self.analyze_failures()

        suggestions = []

        # Check failure rate
        failure_rate = failure_analysis.get("failure_rate", 0.0)
        if failure_rate > 0.3:
            suggestions.append("High failure rate detected. Consider retraining or schema refinement.")

        # Check specific metrics
        failing_metrics = failure_analysis.get("failing_metrics", {})

        if failing_metrics.get("faithfulness", 0) > 5:
            suggestions.append(
                "Low faithfulness scores detected. Improve context retrieval quality."
            )

        if failing_metrics.get("answer_relevancy", 0) > 5:
            suggestions.append(
                "Low answer relevancy. Review query routing and response generation."
            )

        if failing_metrics.get("context_precision", 0) > 5:
            suggestions.append(
                "Low context precision. Refine retrieval strategy or add reranking."
            )

        if failing_metrics.get("context_recall", 0) > 5:
            suggestions.append(
                "Low context recall. Increase top_k or improve chunking strategy."
            )

        # Check recent trend
        if len(self.evaluation_history) >= 10:
            recent_scores = [
                e["metrics"]["overall"]
                for e in self.evaluation_history[-10:]
            ]
            avg_recent = sum(recent_scores) / len(recent_scores)

            if avg_recent < 0.6:
                suggestions.append(
                    f"Recent performance degradation (avg: {avg_recent:.2f}). "
                    "Consider parameter adjustment or schema update."
                )

        if not suggestions:
            suggestions.append("System performing well. No immediate improvements needed.")

        return suggestions

    def suggest_schema_refinements(
        self,
        current_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest schema refinements based on performance.

        Args:
            current_schema: Current knowledge graph schema

        Returns:
            Schema refinement suggestions
        """
        # Analyze which entity/relation types appear in failures
        failure_entities = defaultdict(int)
        failure_relations = defaultdict(int)

        for eval_item in self.evaluation_history:
            if eval_item.get("metrics", {}).get("overall", 1.0) < 0.6:
                query = eval_item.get("query", "")
                # Simple heuristic: extract potential entities/relations from query
                # In practice, would use NER here
                words = query.lower().split()
                for word in words:
                    if word in ["who", "where", "when"]:
                        failure_entities["Person"] += 1

        suggestions = {
            "add_entity_types": [],
            "add_relation_types": [],
            "modify_properties": {},
            "reasoning": ""
        }

        # Generate suggestions using LLM
        entity_types = current_schema.get("entity_types", [])
        relation_types = current_schema.get("relation_types", [])

        prompt = f"""Based on the current schema, suggest refinements to improve performance.

Current Entity Types: {', '.join(entity_types)}
Current Relation Types: {', '.join(relation_types)}

Consider:
1. Missing entity types that might be useful
2. Missing relationship types
3. Additional properties that should be tracked

Return JSON:
{{
  "add_entity_types": ["NewType1", "NewType2"],
  "add_relation_types": ["NEW_RELATION"],
  "reasoning": "Explanation of suggestions"
}}"""

        try:
            result = self.llm_client.generate_json(prompt, temperature=0.2)
            suggestions.update(result)
        except Exception as e:
            logger.error(f"Error generating schema suggestions: {e}")

        return suggestions

    def get_performance_summary(self) -> str:
        """
        Get a human-readable performance summary.

        Returns:
            Formatted performance summary
        """
        if not self.evaluation_history:
            return "No evaluation data available."

        # Calculate averages
        all_metrics = defaultdict(list)
        for eval_item in self.evaluation_history:
            for metric, score in eval_item.get("metrics", {}).items():
                all_metrics[metric].append(score)

        avg_metrics = {
            metric: sum(scores) / len(scores)
            for metric, scores in all_metrics.items()
        }

        # Build summary
        summary = ["=" * 60]
        summary.append("PERFORMANCE SUMMARY")
        summary.append("=" * 60)
        summary.append(f"\nTotal Evaluations: {len(self.evaluation_history)}")

        summary.append("\nAverage Metrics:")
        for metric, avg in sorted(avg_metrics.items()):
            bar = "█" * int(avg * 20)
            summary.append(f"  {metric:20s}: {avg:.3f} {bar}")

        # Recent trend
        if len(self.evaluation_history) >= 5:
            recent = [e["metrics"]["overall"] for e in self.evaluation_history[-5:]]
            avg_recent = sum(recent) / len(recent)
            summary.append(f"\nRecent Performance (last 5): {avg_recent:.3f}")

        summary.append("=" * 60)
        return "\n".join(summary)


# Singleton instance
_reflection_agent: Optional[ReflectionAgent] = None


def get_reflection_agent() -> ReflectionAgent:
    """
    Get the global ReflectionAgent instance (singleton pattern).

    Returns:
        ReflectionAgent: Global reflection agent
    """
    global _reflection_agent
    if _reflection_agent is None:
        _reflection_agent = ReflectionAgent()
    return _reflection_agent


if __name__ == "__main__":
    """Test the ReflectionAgent."""
    import sys

    try:
        print("🔄 Initializing ReflectionAgent...")
        agent = get_reflection_agent()

        # Test evaluation
        test_query = "What is machine learning?"
        test_response = "Machine learning is a subset of AI that enables systems to learn from data."
        test_context = [
            {"text": "Machine learning is a method of data analysis that automates analytical model building."},
            {"text": "It is a branch of artificial intelligence based on the idea that systems can learn from data."}
        ]

        print("\n📊 Evaluating sample response...")
        metrics = agent.evaluate_response(test_query, test_response, test_context)

        print(f"\nMetrics:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.3f}")

        print("\n" + agent.get_performance_summary())

        print("\n✅ ReflectionAgent working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
