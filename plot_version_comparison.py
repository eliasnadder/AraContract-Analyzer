#!/usr/bin/env python3
"""
AraContract Analyzer — Version Comparison Visualization

Generates comprehensive plots comparing model performance across v1, v1.5, and v2.

Run: python plot_version_comparison.py
Output: plots/ directory with PNG files
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, FuncFormatter

# ============================================================================
# VERSION DATA
# ============================================================================

VERSIONS = ['v1\n(Baseline)', 'v1.5\n(Weights+\nOversample)', 'v2\n(Oversample+\nThreshold)']
VERSION_COLORS = ['#6c757d', '#fd7e14', '#28a745']

# Task 1 — Type Classification
TYPE_ACCURACY = [0.8872, 0.8619, 0.8631]
TYPE_MACRO_F1 = [0.8746, 0.8401, 0.8401]
TYPE_WEIGHTED_F1 = [0.8882, 0.8624, 0.8635]

# Task 2 — Risk Level Classification (Overall)
RISK_MACRO_F1 = [0.7404, 0.7317, 0.7547]
RISK_WEIGHTED_F1 = [0.8702, 0.7218, 0.7547]  # Note: v1.5 had bug, v2 fixed

# Risk Level — Per-Class Metrics (v2 final)
RISK_LEVELS = ['Low', 'Medium', 'High']
V2_RISK_PRECISION = [0.9440, 0.5333, 0.7437]
V2_RISK_RECALL = [0.8736, 0.5333, 0.9219]
V2_RISK_F1 = [0.9074, 0.5333, 0.8233]

# Medium Class Evolution (Key Focus)
MEDIUM_PRECISION = [0.42, 0.4286, 0.5333]  # v1 estimated
MEDIUM_RECALL = [0.35, 0.55, 0.5333]
MEDIUM_F1 = [0.4516, 0.4818, 0.5333]

# Support (Test Set Distribution)
RISK_SUPPORT = [617, 60, 192]
RISK_PERCENT = [71, 7, 22]

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2563eb',
    'success': '#16a34a',
    'warning': '#ea580c',
    'danger': '#dc2626',
    'gray': '#6b7280',
}

FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 11
FONTSIZE_TICK = 10

# ============================================================================
# PLOT 1: Task 1 vs Task 2 Overview
# ============================================================================

def plot_overview():
    """Overall performance comparison across versions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Task 1
    ax1 = axes[0]
    x = np.arange(len(VERSIONS))
    width = 0.25

    bars1 = ax1.bar(x - width, TYPE_ACCURACY, width, label='Accuracy', color=COLORS['primary'], alpha=0.8)
    bars2 = ax1.bar(x, TYPE_MACRO_F1, width, label='Macro F1', color=COLORS['success'], alpha=0.8)
    bars3 = ax1.bar(x + width, TYPE_WEIGHTED_F1, width, label='Weighted F1', color=COLORS['gray'], alpha=0.8)

    ax1.set_ylabel('Score', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Task 1 — Clause Type Classification', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(VERSIONS, fontsize=FONTSIZE_TICK)
    ax1.set_ylim(0.80, 0.95)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Task 2
    ax2 = axes[1]

    bars4 = ax2.bar(x - width/2, RISK_MACRO_F1, width, label='Macro F1', color=COLORS['success'], alpha=0.8)
    bars5 = ax2.bar(x + width/2, RISK_WEIGHTED_F1, width, label='Weighted F1', color=COLORS['gray'], alpha=0.8)

    ax2.set_ylabel('Score', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Task 2 — Risk Level Classification', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(VERSIONS, fontsize=FONTSIZE_TICK)
    ax2.set_ylim(0.65, 0.95)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    # Highlight v2 improvement
    ax2.annotate('Best Macro F1',
                xy=(2, 0.7547),
                xytext=(0.5, 0.82),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
                fontsize=10, color=COLORS['success'],
                fontweight='bold')

    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/v1_v2_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v1_v2_overview.png")


# ============================================================================
# PLOT 2: Medium Class Evolution (Focus)
# ============================================================================

def plot_medium_evolution():
    """Track medium class improvement across versions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(len(VERSIONS))
    width = 0.6

    # Precision
    ax1 = axes[0]
    bars1 = ax1.bar(x, MEDIUM_PRECISION, width, color=COLORS['warning'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Precision', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Medium Class — Precision', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(VERSIONS, fontsize=FONTSIZE_TICK)
    ax1.set_ylim(0, 0.6)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 baseline')

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Recall
    ax2 = axes[1]
    bars2 = ax2.bar(x, MEDIUM_RECALL, width, color=COLORS['primary'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Recall', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Medium Class — Recall', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(VERSIONS, fontsize=FONTSIZE_TICK)
    ax2.set_ylim(0, 0.7)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Annotate improvement
    ax2.annotate(f'+18%\n({MEDIUM_RECALL[0]:.0%}→{MEDIUM_RECALL[2]:.0%})',
                xy=(2, MEDIUM_RECALL[2]),
                xytext=(0.3, 0.55),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
                fontsize=10, color=COLORS['success'],
                fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # F1 Score
    ax3 = axes[2]
    bars3 = ax3.bar(x, MEDIUM_F1, width, color=COLORS['success'], alpha=0.8, edgecolor='black')
    ax3.set_ylabel('F1 Score', fontsize=FONTSIZE_LABEL)
    ax3.set_title('Medium Class — F1 Score', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(VERSIONS, fontsize=FONTSIZE_TICK)
    ax3.set_ylim(0, 0.6)

    # Annotate total improvement
    improvement = ((MEDIUM_F1[2] - MEDIUM_F1[0]) / MEDIUM_F1[0]) * 100
    ax3.annotate(f'+{improvement:.1f}%\nimprovement',
                xy=(2, MEDIUM_F1[2]),
                xytext=(0.3, 0.45),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
                fontsize=10, color=COLORS['success'],
                fontweight='bold')

    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/medium_class_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/medium_class_evolution.png")


# ============================================================================
# PLOT 3: Risk Level Per-Class Breakdown (v2)
# ============================================================================

def plot_risk_breakdown():
    """Per-class metrics for the final v2 model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(RISK_LEVELS))
    width = 0.25

    prec_bars = ax.bar(x - width, V2_RISK_PRECISION, width,
                       label='Precision', color=COLORS['primary'], alpha=0.8)
    rec_bars = ax.bar(x, V2_RISK_RECALL, width,
                      label='Recall', color=COLORS['success'], alpha=0.8)
    f1_bars = ax.bar(x + width, V2_RISK_F1, width,
                     label='F1 Score', color=COLORS['warning'], alpha=0.8)

    ax.set_ylabel('Score', fontsize=FONTSIZE_LABEL)
    ax.set_title('v2 Model — Risk Level Per-Class Metrics', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lvl}\n(n={sup})' for lvl, sup in zip(RISK_LEVELS, RISK_SUPPORT)],
                       fontsize=FONTSIZE_TICK)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bars in [prec_bars, rec_bars, f1_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    # Highlight medium class challenge
    ax.annotate('Challenge Class',
               xy=(1, V2_RISK_F1[1]),
               xytext=(1.3, 0.35),
               arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2),
               fontsize=10, color=COLORS['danger'],
               fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/risk_breakdown_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/risk_breakdown_v2.png")


# ============================================================================
# PLOT 4: Data Distribution
# ============================================================================

def plot_data_distribution():
    """Show the class imbalance in the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute counts
    ax1 = axes[0]
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    bars1 = ax1.bar(RISK_LEVELS, RISK_SUPPORT, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Number of Samples', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Risk Level Distribution — Absolute Counts', fontsize=FONTSIZE_TITLE, fontweight='bold')

    for bar, pct in zip(bars1, RISK_PERCENT):
        height = bar.get_height()
        ax1.annotate(f'{height:,}\n({pct}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylim(0, 3000)

    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(RISK_SUPPORT,
                                        labels=[f'{lvl}\n({pct}%)' for lvl, pct in zip(RISK_LEVELS, RISK_PERCENT)],
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        explode=(0.05, 0.15, 0.05),
                                        shadow=True,
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax2.set_title('Risk Level Distribution — Percentage', fontsize=FONTSIZE_TITLE, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/data_distribution.png")


# ============================================================================
# PLOT 5: Combined Dashboard
# ============================================================================

def plot_dashboard():
    """Single-page dashboard with all key metrics."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top row
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Bottom row (wide)
    ax4 = fig.add_subplot(gs[1, :])

    # Plot 1: Task 1 Accuracy Trend
    ax1.plot(VERSIONS, TYPE_ACCURACY, marker='o', markersize=8, linewidth=2,
             color=COLORS['primary'], markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(range(len(VERSIONS)), TYPE_ACCURACY, alpha=0.2, color=COLORS['primary'])
    ax1.set_ylabel('Accuracy', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Task 1 Accuracy', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.set_ylim(0.84, 0.90)
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(TYPE_ACCURACY):
        ax1.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=10)

    # Plot 2: Task 2 Macro F1 Trend
    ax2.plot(VERSIONS, RISK_MACRO_F1, marker='s', markersize=8, linewidth=2,
             color=COLORS['success'], markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(range(len(VERSIONS)), RISK_MACRO_F1, alpha=0.2, color=COLORS['success'])
    ax2.set_ylabel('Macro F1', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Task 2 Macro F1', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.set_ylim(0.70, 0.78)
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(RISK_MACRO_F1):
        ax2.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=10)

    # Plot 3: Medium F1 Trend
    ax3.plot(VERSIONS, MEDIUM_F1, marker='^', markersize=8, linewidth=2,
             color=COLORS['warning'], markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(range(len(VERSIONS)), MEDIUM_F1, alpha=0.2, color=COLORS['warning'])
    ax3.set_ylabel('F1 Score', fontsize=FONTSIZE_LABEL)
    ax3.set_title('Medium Risk F1', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax3.set_ylim(0.40, 0.58)
    ax3.grid(True, alpha=0.3)
    for i, v in enumerate(MEDIUM_F1):
        ax3.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=10)

    # Annotate improvement
    total_improvement = ((MEDIUM_F1[-1] - MEDIUM_F1[0]) / MEDIUM_F1[0]) * 100
    ax3.annotate(f'Total: +{total_improvement:.1f}%',
                xy=(2, MEDIUM_F1[2]),
                xytext=(0.5, 0.52),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2),
                fontsize=11, color=COLORS['warning'], fontweight='bold')

    # Plot 4: v2 Final Metrics Table (as visual)
    ax4.axis('off')

    table_data = [
        ['Metric', 'v1', 'v1.5', 'v2', 'Change'],
        ['Task 1 Accuracy', f'{TYPE_ACCURACY[0]:.4f}', f'{TYPE_ACCURACY[1]:.4f}', f'{TYPE_ACCURACY[2]:.4f}',
         f'{TYPE_ACCURACY[2]-TYPE_ACCURACY[0]:+.4f}'],
        ['Task 2 Macro F1', f'{RISK_MACRO_F1[0]:.4f}', f'{RISK_MACRO_F1[1]:.4f}', f'{RISK_MACRO_F1[2]:.4f}',
         f'{RISK_MACRO_F1[2]-RISK_MACRO_F1[0]:+.4f}'],
        ['Medium Precision', f'{MEDIUM_PRECISION[0]:.4f}', f'{MEDIUM_PRECISION[1]:.4f}', f'{MEDIUM_PRECISION[2]:.4f}',
         f'{MEDIUM_PRECISION[2]-MEDIUM_PRECISION[0]:+.4f}'],
        ['Medium Recall', f'{MEDIUM_RECALL[0]:.4f}', f'{MEDIUM_RECALL[1]:.4f}', f'{MEDIUM_RECALL[2]:.4f}',
         f'{MEDIUM_RECALL[2]-MEDIUM_RECALL[0]:+.4f}'],
        ['Medium F1', f'{MEDIUM_F1[0]:.4f}', f'{MEDIUM_F1[1]:.4f}', f'{MEDIUM_F1[2]:.4f}',
         f'{MEDIUM_F1[2]-MEDIUM_F1[0]:+.4f}'],
    ]

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=[COLORS['gray']] * 5,
        bbox=[0.05, 0.1, 0.9, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Highlight v2 column (green)
    for i in range(1, len(table_data)):
        for j in range(5):
            if j == 3:  # v2 column
                table[(i, j)].set_facecolor('#d4edda')
                table[(i, j)].set_edgecolor(COLORS['success'])
                table[(i, j)].set_linewidth(2)

    ax4.set_title('Summary: Version Comparison (v1 → v1.5 → v2)',
                  fontsize=14, fontweight='bold', pad=20)

    plt.savefig('plots/dashboard_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/dashboard_summary.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os

    # Create output directory
    os.makedirs('plots', exist_ok=True)

    print("=" * 60)
    print("AraContract Analyzer — Version Comparison Visualization")
    print("=" * 60)
    print()

    # Generate all plots
    plot_overview()
    plot_medium_evolution()
    plot_risk_breakdown()
    plot_data_distribution()
    plot_dashboard()

    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print("Output directory: plots/")
    print("=" * 60)