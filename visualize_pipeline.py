#!/usr/bin/env python3
"""
Pipeline Visualization for Agentic GraphRAG

Creates visual diagrams of the system architecture and data flow.
Generates both static images and interactive flowcharts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'ingestion': '#3498db',    # Blue
    'agent': '#e74c3c',        # Red
    'storage': '#2ecc71',      # Green
    'retrieval': '#f39c12',    # Orange
    'evaluation': '#9b59b6',   # Purple
    'data': '#34495e'          # Dark gray
}

def create_architecture_diagram():
    """Create high-level architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Agentic GraphRAG Architecture',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Layer 1: Document Input
    doc_box = FancyBboxPatch((5, 8.5), 4, 0.6,
                             boxstyle="round,pad=0.1",
                             edgecolor=colors['data'],
                             facecolor=colors['data'],
                             alpha=0.3, linewidth=2)
    ax.add_patch(doc_box)
    ax.text(7, 8.8, 'Raw Documents\n(.txt, .pdf, .md)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(7, 7.8), xytext=(7, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Layer 2: Schema Agent
    schema_box = FancyBboxPatch((4.5, 7), 5, 0.7,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['agent'],
                                facecolor=colors['agent'],
                                alpha=0.3, linewidth=2)
    ax.add_patch(schema_box)
    ax.text(7, 7.35, 'SchemaAgent\nAutomatic Schema Inference',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(7, 6.3), xytext=(7, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Layer 3: Entity & Relation Agents (parallel)
    entity_box = FancyBboxPatch((1, 5.3), 3, 0.9,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['agent'],
                                facecolor=colors['agent'],
                                alpha=0.3, linewidth=2)
    ax.add_patch(entity_box)
    ax.text(2.5, 5.75, 'EntityAgent\nHybrid NER + LLM\nMetadata Enrichment',
            ha='center', va='center', fontsize=9, fontweight='bold')

    relation_box = FancyBboxPatch((10, 5.3), 3, 0.9,
                                  boxstyle="round,pad=0.1",
                                  edgecolor=colors['agent'],
                                  facecolor=colors['agent'],
                                  alpha=0.3, linewidth=2)
    ax.add_patch(relation_box)
    ax.text(11.5, 5.75, 'RelationAgent\nLLM-based\nRelationship Extraction',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows from schema to agents
    ax.annotate('', xy=(2.5, 6.2), xytext=(6, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(11.5, 6.2), xytext=(8, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Layer 4: Storage (parallel)
    neo4j_box = FancyBboxPatch((1, 3.8), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor=colors['storage'],
                               facecolor=colors['storage'],
                               alpha=0.3, linewidth=2)
    ax.add_patch(neo4j_box)
    ax.text(2.5, 4.2, 'Neo4j Graph DB\nNodes & Edges\nRelationships',
            ha='center', va='center', fontsize=9, fontweight='bold')

    faiss_box = FancyBboxPatch((10, 3.8), 3, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor=colors['storage'],
                               facecolor=colors['storage'],
                               alpha=0.3, linewidth=2)
    ax.add_patch(faiss_box)
    ax.text(11.5, 4.2, 'FAISS Vector Store\nEmbeddings\nSemantic Search',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows to storage
    ax.annotate('', xy=(2.5, 4.6), xytext=(2.5, 5.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(11.5, 4.6), xytext=(11.5, 5.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # User Query
    query_box = FancyBboxPatch((5, 3.0), 4, 0.5,
                               boxstyle="round,pad=0.1",
                               edgecolor=colors['data'],
                               facecolor=colors['data'],
                               alpha=0.3, linewidth=2)
    ax.add_patch(query_box)
    ax.text(7, 3.25, 'User Query',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow to orchestrator
    ax.annotate('', xy=(7, 2.7), xytext=(7, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Layer 5: Orchestrator
    orch_box = FancyBboxPatch((4.5, 1.9), 5, 0.7,
                              boxstyle="round,pad=0.1",
                              edgecolor=colors['retrieval'],
                              facecolor=colors['retrieval'],
                              alpha=0.3, linewidth=2)
    ax.add_patch(orch_box)
    ax.text(7, 2.25, 'OrchestratorAgent\nIntelligent Query Routing',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Bidirectional arrows to storage
    ax.annotate('', xy=(2.5, 3.8), xytext=(5.5, 2.3),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=colors['retrieval']))
    ax.annotate('', xy=(11.5, 3.8), xytext=(8.5, 2.3),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=colors['retrieval']))

    # Layer 6: Retrieval Strategies
    vector_box = FancyBboxPatch((0.5, 0.8), 2.5, 0.5,
                                boxstyle="round,pad=0.05",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.2, linewidth=1.5)
    ax.add_patch(vector_box)
    ax.text(1.75, 1.05, 'Vector Search',
            ha='center', va='center', fontsize=9)

    graph_box = FancyBboxPatch((5.75, 0.8), 2.5, 0.5,
                               boxstyle="round,pad=0.05",
                               edgecolor=colors['retrieval'],
                               facecolor=colors['retrieval'],
                               alpha=0.2, linewidth=1.5)
    ax.add_patch(graph_box)
    ax.text(7, 1.05, 'Graph Traversal',
            ha='center', va='center', fontsize=9)

    hybrid_box = FancyBboxPatch((11, 0.8), 2.5, 0.5,
                                boxstyle="round,pad=0.05",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.2, linewidth=1.5)
    ax.add_patch(hybrid_box)
    ax.text(12.25, 1.05, 'Hybrid (Both)',
            ha='center', va='center', fontsize=9)

    # Arrows to strategies
    ax.annotate('', xy=(1.75, 1.3), xytext=(6, 1.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7, 1.3), xytext=(7, 1.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(12.25, 1.3), xytext=(8, 1.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Response
    response_box = FancyBboxPatch((5, 0.1), 4, 0.4,
                                  boxstyle="round,pad=0.1",
                                  edgecolor=colors['data'],
                                  facecolor=colors['data'],
                                  alpha=0.3, linewidth=2)
    ax.add_patch(response_box)
    ax.text(7, 0.3, 'Generated Response',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to response
    ax.annotate('', xy=(6.5, 0.5), xytext=(1.75, 0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7, 0.5), xytext=(7, 0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7.5, 0.5), xytext=(12.25, 0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Reflection Agent (side loop)
    reflect_box = FancyBboxPatch((0.3, 2.2), 2, 0.5,
                                 boxstyle="round,pad=0.05",
                                 edgecolor=colors['evaluation'],
                                 facecolor=colors['evaluation'],
                                 alpha=0.3, linewidth=1.5)
    ax.add_patch(reflect_box)
    ax.text(1.3, 2.45, 'ReflectionAgent\nRAGAS Evaluation',
            ha='center', va='center', fontsize=8, fontweight='bold')

    # Feedback loop
    ax.annotate('', xy=(2.3, 2.4), xytext=(4.5, 2.25),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                               color=colors['evaluation'], linestyle='dashed'))

    plt.tight_layout()
    return fig


def create_ingestion_flow():
    """Create detailed ingestion pipeline flow."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6, 7.5, 'Ingestion Pipeline Flow',
            ha='center', va='top', fontsize=16, fontweight='bold')

    y = 6.5
    x_left = 1
    x_right = 7

    # Step 1: Input
    ax.add_patch(FancyBboxPatch((x_left, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['data'],
                                facecolor=colors['data'],
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 2, y + 0.3, '1. Load Documents',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.add_patch(FancyBboxPatch((x_right, y), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='gray',
                                facecolor='white',
                                alpha=0.5, linewidth=1))
    ax.text(x_right + 2, y + 0.3, 'Files: .txt, .pdf, .md, .docx',
            ha='center', va='center', fontsize=9, style='italic')

    # Step 2: Schema Inference
    y -= 1.2
    ax.annotate('', xy=(x_left + 2, y + 0.6), xytext=(x_left + 2, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((x_left, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['agent'],
                                facecolor=colors['agent'],
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 2, y + 0.3, '2. Infer Schema',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.add_patch(FancyBboxPatch((x_right, y), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='gray',
                                facecolor='white',
                                alpha=0.5, linewidth=1))
    ax.text(x_right + 2, y + 0.3, 'LLM analyzes → Entity & Relation types',
            ha='center', va='center', fontsize=9, style='italic')

    # Step 3: Entity Extraction
    y -= 1.2
    ax.annotate('', xy=(x_left + 2, y + 0.6), xytext=(x_left + 2, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((x_left, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['agent'],
                                facecolor=colors['agent'],
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 2, y + 0.3, '3. Extract Entities',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.add_patch(FancyBboxPatch((x_right, y), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='gray',
                                facecolor='white',
                                alpha=0.5, linewidth=1))
    ax.text(x_right + 2, y + 0.3, 'Hybrid: spaCy NER + LLM enrichment',
            ha='center', va='center', fontsize=9, style='italic')

    # Step 4: Relation Extraction
    y -= 1.2
    ax.annotate('', xy=(x_left + 2, y + 0.6), xytext=(x_left + 2, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((x_left, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['agent'],
                                facecolor=colors['agent'],
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 2, y + 0.3, '4. Extract Relations',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.add_patch(FancyBboxPatch((x_right, y), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='gray',
                                facecolor='white',
                                alpha=0.5, linewidth=1))
    ax.text(x_right + 2, y + 0.3, 'LLM identifies relationships between entities',
            ha='center', va='center', fontsize=9, style='italic')

    # Step 5: Store in Neo4j
    y -= 1.2
    ax.annotate('', xy=(x_left + 1, y + 0.6), xytext=(x_left + 2, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((x_left - 1, y), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['storage'],
                                facecolor=colors['storage'],
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 0.5, y + 0.3, '5. Neo4j Graph',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Step 6: Store in FAISS
    ax.annotate('', xy=(x_left + 5, y + 0.6), xytext=(x_left + 3, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((x_left + 3.5, y), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['storage'],
                                facecolor=colors['storage'],
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 5, y + 0.3, '6. FAISS Vectors',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Output
    y -= 1.2
    ax.annotate('', xy=(x_left + 2, y + 0.6), xytext=(x_left + 0.5, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(x_left + 2, y + 0.6), xytext=(x_left + 5, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((x_left, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='green',
                                facecolor='green',
                                alpha=0.3, linewidth=2))
    ax.text(x_left + 2, y + 0.3, '✓ Ready for Queries',
            ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def create_retrieval_flow():
    """Create detailed retrieval pipeline flow."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6, 7.5, 'Retrieval Pipeline Flow',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Query input
    y = 6.5
    ax.add_patch(FancyBboxPatch((4, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['data'],
                                facecolor=colors['data'],
                                alpha=0.3, linewidth=2))
    ax.text(6, y + 0.3, 'User Query',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Query analysis
    y -= 1.2
    ax.annotate('', xy=(6, y + 0.6), xytext=(6, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((3.5, y), 5, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.3, linewidth=2))
    ax.text(6, y + 0.3, 'Orchestrator: Analyze Query Type',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Three strategy branches
    y -= 1.5

    # Vector search
    ax.annotate('', xy=(1.5, y + 0.6), xytext=(5, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['retrieval']))
    ax.add_patch(FancyBboxPatch((0.5, y), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.2, linewidth=2))
    ax.text(1.5, y + 0.3, 'Vector Search',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.5, y - 0.2, '(Conceptual)',
            ha='center', va='center', fontsize=7, style='italic')

    # Graph traversal
    ax.annotate('', xy=(6, y + 0.6), xytext=(6, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['retrieval']))
    ax.add_patch(FancyBboxPatch((5, y), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.2, linewidth=2))
    ax.text(6, y + 0.3, 'Graph Traversal',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6, y - 0.2, '(Relational)',
            ha='center', va='center', fontsize=7, style='italic')

    # Hybrid
    ax.annotate('', xy=(10.5, y + 0.6), xytext=(7, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['retrieval']))
    ax.add_patch(FancyBboxPatch((9.5, y), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.2, linewidth=2))
    ax.text(10.5, y + 0.3, 'Hybrid',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(10.5, y - 0.2, '(Multi-hop)',
            ha='center', va='center', fontsize=7, style='italic')

    # Context retrieval
    y -= 1.5
    ax.annotate('', xy=(6, y + 0.6), xytext=(1.5, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(6, y + 0.6), xytext=(6, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(6, y + 0.6), xytext=(10.5, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    ax.add_patch(FancyBboxPatch((4, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['retrieval'],
                                facecolor=colors['retrieval'],
                                alpha=0.3, linewidth=2))
    ax.text(6, y + 0.3, 'Retrieved Context',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # LLM generation
    y -= 1.2
    ax.annotate('', xy=(6, y + 0.6), xytext=(6, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((3.5, y), 5, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor=colors['agent'],
                                facecolor=colors['agent'],
                                alpha=0.3, linewidth=2))
    ax.text(6, y + 0.3, 'LLM: Generate Answer from Context',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Response
    y -= 1.2
    ax.annotate('', xy=(6, y + 0.6), xytext=(6, y + 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((4, y), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='green',
                                facecolor='green',
                                alpha=0.3, linewidth=2))
    ax.text(6, y + 0.3, 'Final Response',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Reflection feedback loop
    ax.add_patch(FancyBboxPatch((9, y + 1.2), 2.5, 0.5,
                                boxstyle="round,pad=0.05",
                                edgecolor=colors['evaluation'],
                                facecolor=colors['evaluation'],
                                alpha=0.3, linewidth=1.5))
    ax.text(10.25, y + 1.45, 'ReflectionAgent\nEvaluate Quality',
            ha='center', va='center', fontsize=8, fontweight='bold')

    ax.annotate('', xy=(9, y + 1.4), xytext=(8, y + 0.3),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                               color=colors['evaluation'], linestyle='dashed'))

    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - PIPELINE VISUALIZATION")
    print("=" * 70)

    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Generating visualizations...")

    # 1. Architecture diagram
    print("\n1. Creating architecture diagram...")
    fig1 = create_architecture_diagram()
    fig1.savefig(output_dir / "architecture.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/architecture.png")

    # 2. Ingestion flow
    print("\n2. Creating ingestion pipeline flow...")
    fig2 = create_ingestion_flow()
    fig2.savefig(output_dir / "ingestion_pipeline.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/ingestion_pipeline.png")

    # 3. Retrieval flow
    print("\n3. Creating retrieval pipeline flow...")
    fig3 = create_retrieval_flow()
    fig3.savefig(output_dir / "retrieval_pipeline.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir}/retrieval_pipeline.png")

    plt.close('all')

    print("\n" + "=" * 70)
    print("  ✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll diagrams saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  • architecture.png - High-level system architecture")
    print("  • ingestion_pipeline.png - Document ingestion flow")
    print("  • retrieval_pipeline.png - Query retrieval flow")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
