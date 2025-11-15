#!/usr/bin/env python3
"""
Visualization Script for Agentic GraphRAG Evaluation Results

Generates publication-quality plots from evaluation data:
1. Performance comparison across query types
2. RAGAS metrics breakdown
3. Latency analysis
4. Success rate visualization
5. Heatmaps and correlation plots

Requirements: matplotlib, seaborn, pandas
Install with: pip install matplotlib seaborn pandas
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\n💡 Install visualization dependencies:")
    print("   pip install matplotlib seaborn pandas")
    sys.exit(1)


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    json_file = results_dir / "results.json"

    if not json_file.exists():
        print(f"❌ Results file not found: {json_file}")
        print("\n💡 Run evaluation first:")
        print("   python evaluation.py")
        sys.exit(1)

    with open(json_file, 'r') as f:
        return json.load(f)


def create_performance_comparison(df: pd.DataFrame, output_dir: Path):
    """Create bar chart comparing performance across query types."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by query type and calculate mean overall score
    grouped = df.groupby('query_type')['overall_score'].mean().sort_values(ascending=False)

    colors = sns.color_palette("husl", len(grouped))
    bars = ax.bar(grouped.index, grouped.values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Overall Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison Across Query Types', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / "performance_by_query_type.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_metrics_breakdown(df: pd.DataFrame, output_dir: Path):
    """Create grouped bar chart for RAGAS metrics breakdown."""
    fig, ax = plt.subplots(figsize=(14, 7))

    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    query_types = df['query_type'].unique()

    # Calculate means for each metric and query type
    data = []
    for qtype in query_types:
        qtype_data = df[df['query_type'] == qtype]
        data.append([qtype_data[metric].mean() for metric in metrics])

    # Create grouped bar chart
    x = np.arange(len(metrics))
    width = 0.15
    multiplier = 0

    for i, qtype in enumerate(query_types):
        offset = width * multiplier
        bars = ax.bar(x + offset, data[i], width, label=qtype.capitalize(),
                     edgecolor='black', linewidth=0.8)
        multiplier += 1

    ax.set_xlabel('RAGAS Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('RAGAS Metrics Breakdown by Query Type', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(query_types) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / "metrics_breakdown.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_dataset_performance(df: pd.DataFrame, output_dir: Path):
    """Create bar chart showing performance across datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by dataset
    grouped = df.groupby('dataset')['overall_score'].mean().sort_values(ascending=False)

    colors = sns.color_palette("Set2", len(grouped))
    bars = ax.bar(grouped.index, grouped.values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Overall Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Across Benchmark Datasets', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / "performance_by_dataset.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_latency_analysis(df: pd.DataFrame, output_dir: Path):
    """Create box plot for latency analysis across query types."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter successful queries only
    df_success = df[df['success'] == True]

    # Create box plot
    query_types = sorted(df_success['query_type'].unique())
    data = [df_success[df_success['query_type'] == qt]['latency_ms'].values for qt in query_types]

    box_parts = ax.boxplot(data, labels=[qt.capitalize() for qt in query_types],
                          patch_artist=True, notch=True)

    # Color the boxes
    colors = sns.color_palette("pastel", len(query_types))
    for patch, color in zip(box_parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    ax.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Retrieval Latency Distribution by Query Type', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / "latency_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create correlation heatmap for RAGAS metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select only metric columns
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision',
               'context_recall', 'overall_score', 'latency_ms']

    # Filter successful queries
    df_success = df[df['success'] == True]

    # Calculate correlation matrix
    corr_matrix = df_success[metrics].corr()

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)

    ax.set_title('Correlation Matrix of Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)

    # Format labels
    labels = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    output_file = output_dir / "correlation_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_success_rate_chart(df: pd.DataFrame, output_dir: Path):
    """Create success rate visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # By query type
    success_by_type = df.groupby('query_type')['success'].apply(
        lambda x: (x == True).sum() / len(x) * 100
    ).sort_values(ascending=False)

    colors1 = sns.color_palette("Greens_r", len(success_by_type))
    bars1 = ax1.bar(success_by_type.index, success_by_type.values,
                   color=colors1, edgecolor='black', linewidth=1.2)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Query Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Success Rate by Query Type', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # By dataset
    success_by_dataset = df.groupby('dataset')['success'].apply(
        lambda x: (x == True).sum() / len(x) * 100
    ).sort_values(ascending=False)

    colors2 = sns.color_palette("Blues_r", len(success_by_dataset))
    bars2 = ax2.bar(success_by_dataset.index, success_by_dataset.values,
                   color=colors2, edgecolor='black', linewidth=1.2)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Success Rate by Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / "success_rates.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_score_distribution(df: pd.DataFrame, output_dir: Path):
    """Create histogram showing score distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    df_success = df[df['success'] == True]

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]

        # Create histogram
        ax.hist(df_success[metric], bins=20, color=color, alpha=0.7,
               edgecolor='black', linewidth=1.2)

        # Add mean line
        mean_val = df_success[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3f}')

        ax.set_xlabel('Score', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = output_dir / "score_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def create_comprehensive_dashboard(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    df_success = df[df['success'] == True]

    # Overall metrics summary (top center)
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('off')

    summary_text = f"""
    AGENTIC GRAPHRAG - EVALUATION DASHBOARD

    Total Queries: {len(df)} | Successful: {len(df_success)} | Success Rate: {len(df_success)/len(df)*100:.1f}%

    Average Overall Score: {df_success['overall_score'].mean():.3f}
    Average Faithfulness: {df_success['faithfulness'].mean():.3f} | Average Relevancy: {df_success['answer_relevancy'].mean():.3f}
    Average Precision: {df_success['context_precision'].mean():.3f} | Average Recall: {df_success['context_recall'].mean():.3f}
    Average Latency: {df_success['latency_ms'].mean():.1f}ms
    """

    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                   fontsize=11, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Performance by query type (middle left)
    ax1 = fig.add_subplot(gs[1, 0])
    grouped = df_success.groupby('query_type')['overall_score'].mean()
    ax1.bar(range(len(grouped)), grouped.values, color=sns.color_palette("husl", len(grouped)))
    ax1.set_xticks(range(len(grouped)))
    ax1.set_xticklabels([qt[:8] for qt in grouped.index], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Avg Score', fontsize=9)
    ax1.set_title('Performance by Query Type', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Metrics radar chart (middle center)
    ax2 = fig.add_subplot(gs[1, 1], projection='polar')
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    values = [df_success[m].mean() for m in metrics]
    values += values[:1]  # complete the circle

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax2.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
    ax2.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.set_title('RAGAS Metrics', fontsize=10, fontweight='bold', pad=20)
    ax2.grid(True)

    # Dataset performance (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    dataset_perf = df_success.groupby('dataset')['overall_score'].mean()
    ax3.barh(range(len(dataset_perf)), dataset_perf.values, color=sns.color_palette("Set2", len(dataset_perf)))
    ax3.set_yticks(range(len(dataset_perf)))
    ax3.set_yticklabels(dataset_perf.index, fontsize=8)
    ax3.set_xlabel('Avg Score', fontsize=9)
    ax3.set_title('Performance by Dataset', fontsize=10, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Latency distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(df_success['latency_ms'], bins=15, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax4.axvline(df_success['latency_ms'].mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Latency (ms)', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.set_title('Latency Distribution', fontsize=10, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Score distribution (bottom center)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(df_success['overall_score'], bins=15, color='#96CEB4', alpha=0.7, edgecolor='black')
    ax5.axvline(df_success['overall_score'].mean(), color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Overall Score', fontsize=9)
    ax5.set_ylabel('Frequency', fontsize=9)
    ax5.set_title('Score Distribution', fontsize=10, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    # Success rate pie chart (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    success_counts = df['success'].value_counts()
    colors = ['#90EE90', '#FFB6C1']
    ax6.pie(success_counts.values, labels=['Success', 'Failed'],
           autopct='%1.1f%%', colors=colors, startangle=90,
           textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax6.set_title('Success Rate', fontsize=10, fontweight='bold')

    output_file = output_dir / "comprehensive_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("  AGENTIC GRAPHRAG - VISUALIZATION GENERATION")
    print("=" * 70)

    # Load results
    results_dir = Path("data/evaluation")
    print(f"\n📂 Loading results from: {results_dir}")

    results = load_results(results_dir)
    df = pd.DataFrame(results['detailed'])

    print(f"✅ Loaded {len(df)} evaluation results")

    # Create output directory for plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Generating visualizations...")
    print(f"{'─' * 70}")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # Generate all plots
    create_performance_comparison(df, plots_dir)
    create_metrics_breakdown(df, plots_dir)
    create_dataset_performance(df, plots_dir)
    create_latency_analysis(df, plots_dir)
    create_correlation_heatmap(df, plots_dir)
    create_success_rate_chart(df, plots_dir)
    create_score_distribution(df, plots_dir)
    create_comprehensive_dashboard(df, plots_dir)

    print(f"\n{'=' * 70}")
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 All plots saved in: {plots_dir.absolute()}")
    print("\n📊 Generated visualizations:")
    print("   1. performance_by_query_type.png - Bar chart of performance across query types")
    print("   2. metrics_breakdown.png - RAGAS metrics grouped by query type")
    print("   3. performance_by_dataset.png - Performance across benchmark datasets")
    print("   4. latency_analysis.png - Box plot of retrieval latency")
    print("   5. correlation_heatmap.png - Correlation matrix of metrics")
    print("   6. success_rates.png - Success rate by query type and dataset")
    print("   7. score_distributions.png - Histograms of metric distributions")
    print("   8. comprehensive_dashboard.png - All-in-one dashboard view")
    print("\n💡 Use these plots in your publication/presentation!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
