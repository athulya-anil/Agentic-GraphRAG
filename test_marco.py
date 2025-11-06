#!/usr/bin/env python3
"""
Test Agentic GraphRAG with MS MARCO-style data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline, get_retrieval_pipeline
from src.agents import get_reflection_agent

# Sample passages (MS MARCO style - information-rich passages)
documents = [
    """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading 
    artists and intellectuals for its design, but it has become a global cultural icon of France 
    and one of the most recognizable structures in the world. The tower is 330 metres tall and 
    was the tallest man-made structure in the world until the completion of the Chrysler Building 
    in New York in 1930.""",
    
    """Amazon Web Services (AWS) is a subsidiary of Amazon that provides on-demand cloud computing 
    platforms and APIs to individuals, companies, and governments. Founded in 2006, AWS has become 
    the world's most comprehensive and broadly adopted cloud platform. It offers over 200 fully 
    featured services from data centers globally. Andy Jassy was the CEO of AWS from its launch 
    until 2021, when he became CEO of Amazon. The headquarters are located in Seattle, Washington.""",
    
    """Photosynthesis is a process used by plants and other organisms to convert light energy into 
    chemical energy that can later be released to fuel the organism's activities. This chemical 
    energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon 
    dioxide and water. Oxygen is produced as a waste product. Most plants, algae, and cyanobacteria 
    perform photosynthesis. The process is vital for life on Earth as it produces the oxygen we 
    breathe and forms the base of the food chain.""",
    
    """Tesla, Inc. is an American electric vehicle and clean energy company based in Austin, Texas. 
    Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined as chairman 
    of the board in 2004 and became CEO in 2008. The company's name is a tribute to inventor and 
    electrical engineer Nikola Tesla. Tesla produces electric cars (Model S, Model 3, Model X, 
    Model Y), battery energy storage systems, and solar panels. As of 2023, Tesla is the world's 
    most valuable automaker.""",
    
    """COVID-19 is a contagious disease caused by the SARS-CoV-2 virus. The first known case was 
    identified in Wuhan, China, in December 2019. The World Health Organization declared the 
    outbreak a Public Health Emergency of International Concern in January 2020 and a pandemic 
    in March 2020. Symptoms range from mild to severe and can include fever, cough, and difficulty 
    breathing. Vaccines were developed in record time, with Pfizer-BioNTech, Moderna, and others 
    receiving emergency authorization in late 2020."""
]

# Test queries with ground truth
test_queries = [
    {
        "query": "Who designed the Eiffel Tower?",
        "ground_truth": "Gustave Eiffel's company designed and built the Eiffel Tower",
        "expected_entities": ["Eiffel Tower", "Gustave Eiffel"]
    },
    {
        "query": "When was AWS founded and who was its first CEO?",
        "ground_truth": "AWS was founded in 2006 and Andy Jassy was CEO from launch until 2021",
        "expected_entities": ["AWS", "Andy Jassy", "Amazon"]
    },
    {
        "query": "What does photosynthesis produce?",
        "ground_truth": "Photosynthesis produces oxygen and converts light energy into chemical energy stored in carbohydrates",
        "expected_entities": ["photosynthesis", "oxygen", "carbon dioxide"]
    },
    {
        "query": "Who founded Tesla and when?",
        "ground_truth": "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning, with Elon Musk joining in 2004",
        "expected_entities": ["Tesla", "Martin Eberhard", "Marc Tarpenning", "Elon Musk"]
    },
    {
        "query": "When was COVID-19 first identified?",
        "ground_truth": "The first known case of COVID-19 was identified in Wuhan, China, in December 2019",
        "expected_entities": ["COVID-19", "Wuhan", "China"]
    }
]

print('=' * 80)
print('  AGENTIC GRAPHRAG - MS MARCO STYLE TEST')
print('=' * 80)

# Stage 1: Ingestion
print('\n📚 STAGE 1: Document Ingestion')
print('─' * 80)
print(f'Ingesting {len(documents)} passages...')

ingestion = get_ingestion_pipeline(
    schema_path=Path("data/processed/schema_marco.json"),
    auto_refine_schema=True
)

results = ingestion.ingest_documents(documents, infer_schema=True, enrich_metadata=True)

print(f'\n✅ Ingestion Complete:')
print(f'   • Documents: {results["documents_processed"]}')
print(f'   • Entities: {results["entities_extracted"]}')
print(f'   • Relations: {results["relations_extracted"]}')
print(f'   • Neo4j Nodes: {results["nodes_created"]}')
print(f'   • Neo4j Edges: {results["edges_created"]}')
print(f'   • Duration: {results["duration_seconds"]:.2f}s')

# Stage 2: Query & Evaluation
print('\n\n📚 STAGE 2: Query Testing with Ground Truth Evaluation')
print('─' * 80)

retrieval = get_retrieval_pipeline(use_reranking=False, use_reflection=True)
reflection = get_reflection_agent()

all_metrics = []

for i, test_case in enumerate(test_queries, 1):
    query = test_case["query"]
    ground_truth = test_case["ground_truth"]
    
    print(f'\n[Query {i}] {query}')
    print(f'   Ground Truth: {ground_truth[:80]}...')
    
    # Query the system
    result = retrieval.query(
        query,
        top_k=3,
        evaluate=True,
        ground_truth=ground_truth
    )
    
    # Show results
    print(f'\n   Response: {result["response"][:150]}...')
    
    if result.get('metrics'):
        metrics = result['metrics']
        all_metrics.append(metrics)
        print(f'\n   📊 RAGAS Metrics:')
        print(f'      Faithfulness:      {metrics.get("faithfulness", 0):.3f}')
        print(f'      Answer Relevancy:  {metrics.get("answer_relevancy", 0):.3f}')
        print(f'      Context Precision: {metrics.get("context_precision", 0):.3f}')
        print(f'      Context Recall:    {metrics.get("context_recall", 0):.3f}')
        print(f'      Overall:           {metrics.get("overall", 0):.3f}')
    
    print(f'\n   📄 Top Retrieved Context:')
    if result['context']:
        top = result['context'][0]
        print(f'      Source: {top["source"]}')
        print(f'      Score: {top["score"]:.3f}')
        print(f'      Text: {top["text"][:100]}...')

# Stage 3: Overall Performance
print('\n\n📚 STAGE 3: Overall Performance Analysis')
print('─' * 80)

if all_metrics:
    avg_metrics = {
        'faithfulness': sum(m.get('faithfulness', 0) for m in all_metrics) / len(all_metrics),
        'answer_relevancy': sum(m.get('answer_relevancy', 0) for m in all_metrics) / len(all_metrics),
        'context_precision': sum(m.get('context_precision', 0) for m in all_metrics) / len(all_metrics),
        'context_recall': sum(m.get('context_recall', 0) for m in all_metrics) / len(all_metrics),
        'overall': sum(m.get('overall', 0) for m in all_metrics) / len(all_metrics)
    }
    
    print('\n📊 Average Performance:')
    for metric, value in avg_metrics.items():
        bar = '█' * int(value * 20)
        print(f'   {metric:20s}: {value:.3f} {bar}')

print('\n' + '=' * 80)
print('  ✅ MS MARCO STYLE TEST COMPLETE')
print('=' * 80)
print('\nView the knowledge graph in Neo4j Browser: http://localhost:7474')
print('Run query: MATCH (n) RETURN n LIMIT 50')
