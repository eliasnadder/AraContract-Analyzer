#!/usr/bin/env python3
"""
AraContract Analyzer — v2 Training History Visualization

Generates plots from the v2 training history log.
Data extracted from v2_history.md

Run: python3 plot_v2_training.py
Output: plots/ directory with v2_*.png files
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# ============================================================================
# V2 TRAINING DATA (from v2_history.md)
# ============================================================================

EPOCHS = [1, 2, 3, 4, 5]

# Training Loss
TRAIN_LOSS = [2.1983, 0.9885, 0.5624, 0.3986, 0.3261]
TRAIN_TYPE_LOSS = [1.3997, 0.5557, 0.3186, 0.2238, 0.1897]
TRAIN_RISK_LOSS = [0.7985, 0.4328, 0.2438, 0.1748, 0.1364]

# Validation F1 Scores
VAL_TYPE_F1 = [0.7219, 0.7860, 0.8089, 0.8263, 0.8241]
VAL_RISK_F1 = [0.7531, 0.8624, 0.8889, 0.8752, 0.8996]

# Validation Accuracy
VAL_TYPE_ACC = [0.7167, 0.7801, 0.8060, 0.8228, 0.8215]
VAL_RISK_ACC = [0.6999, 0.8525, 0.8849, 0.8668, 0.8978]

# Weighted Avg F1 (Average of Type + Risk)
VAL_AVG_F1 = [(v + r) / 2 for v, r in zip(VAL_TYPE_F1, VAL_RISK_F1)]

# GPU/Warmup Epoch (first epoch had higher loss due to warmup)
BEST_EPOCH = 5
BEST_AVG_F1 = 0.8618

# Final Test Set Metrics (threshold-adjusted)
TEST_TYPE_ACC = 0.8631
TEST_TYPE_F1 = 0.8635
TEST_RISK_MACRO_F1 = 0.7547
TEST_RISK_ACC = 0.8608

# Per-Class Type Metrics (Test Set)
TYPE_LABELS = [
    'general\nprovisions',
    'payment\nfinancial',
    'party\nobligations_a',
    'party\nobligations_b',
    'duration\nexpiration',
    'termination',
    'penalties\ndamages',
    'dispute\nresolution'
]
TYPE_PRECISION = [0.8714, 0.9021, 0.6462, 0.6905, 0.8629, 0.8495, 0.9350, 0.9205]
TYPE_RECALL = [0.9385, 0.8663, 0.6774, 0.7838, 0.8231, 0.8316, 0.8915, 0.9643]
TYPE_F1 = [0.9037, 0.8838, 0.6614, 0.7342, 0.8425, 0.8404, 0.9127, 0.9419]
TYPE_SUPPORT = [130, 202, 62, 37, 130, 95, 129, 84]

# Per-Class Risk Metrics (Test Set, threshold-adjusted)
RISK_LABELS = ['Low', 'Medium', 'High']
RISK_PRECISION = [0.9440, 0.5333, 0.7437]
RISK_RECALL = [0.8736, 0.5333, 0.9219]
RISK_F1 = [0.9074, 0.5333, 0.8233]
RISK_SUPPORT = [617, 60, 192]

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'primary': '#2563eb',
    'success': '#16a34a',
    'warning': '#ea580c',
    'danger': '#dc2626',
    'info': '#0891b2',
    'gray': '#6b7280',
    'light_gray': '#e5e7eb',
}

FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 11
FONTSIZE_TICK = 10

# ============================================================================
# PLOT 1: Training Loss Curves
# ============================================================================

def plot_loss_curves():
    """Training loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = EPOCHS

    ax.plot(x, TRAIN_LOSS, marker='o', markersize=8, linewidth=2.5,
            color=COLORS['danger'], label='Total Loss', zorder=3)
    ax.plot(x, TRAIN_TYPE_LOSS, marker='s', markersize=6, linewidth=2,
            color=COLORS['primary'], label='Type Loss', alpha=0.8)
    ax.plot(x, TRAIN_RISK_LOSS, marker='^', markersize=6, linewidth=2,
            color=COLORS['success'], label='Risk Loss', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Loss', fontsize=FONTSIZE_LABEL)
    ax.set_title('v2 Training — Loss Curves', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(EPOCHS)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate final values
    ax.annotate(f'{TRAIN_LOSS[-1]:.4f}', xy=(5, TRAIN_LOSS[-1]), xytext=(5, 0.35),
                fontsize=9, ha='center', color=COLORS['danger'], fontweight='bold')
    ax.annotate(f'{TRAIN_TYPE_LOSS[-1]:.4f}', xy=(5, TRAIN_TYPE_LOSS[-1]), xytext=(5, 0.20),
                fontsize=9, ha='center', color=COLORS['primary'])
    ax.annotate(f'{TRAIN_RISK_LOSS[-1]:.4f}', xy=(5, TRAIN_RISK_LOSS[-1]), xytext=(5, 0.14),
                fontsize=9, ha='center', color=COLORS['success'])

    plt.tight_layout()
    plt.savefig('plots/v2_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v2_training_loss.png")


# ============================================================================
# PLOT 2: Validation F1 Scores
# ============================================================================

def plot_val_f1():
    """Validation F1 scores over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = EPOCHS

    ax.plot(x, VAL_TYPE_F1, marker='o', markersize=8, linewidth=2.5,
            color=COLORS['primary'], label='Type F1', zorder=3)
    ax.plot(x, VAL_RISK_F1, marker='s', markersize=8, linewidth=2.5,
            color=COLORS['success'], label='Risk F1', zorder=3)
    ax.plot(x, VAL_AVG_F1, marker='^', markersize=6, linewidth=2,
            color=COLORS['warning'], label='Avg F1', linestyle='--', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('F1 Score', fontsize=FONTSIZE_LABEL)
    ax.set_title('v2 Validation — F1 Scores', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(EPOCHS)
    ax.set_ylim(0.65, 0.95)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotate final values
    for i, (vf1, rf1) in enumerate(zip(VAL_TYPE_F1, VAL_RISK_F1)):
        ax.annotate(f'{vf1:.3f}', xy=(i+1, vf1), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8, color=COLORS['primary'])
        ax.annotate(f'{rf1:.3f}', xy=(i+1, rf1), xytext=(0, -12),
                    textcoords='offset points', ha='center', fontsize=8, color=COLORS['success'])

    # Highlight best epoch
    ax.axvline(x=BEST_EPOCH, color=COLORS['success'], linestyle='--', alpha=0.5,
               label=f'Best (Epoch {BEST_EPOCH})')
    ax.annotate(f'Best: {BEST_AVG_F1:.4f}',
               xy=(BEST_EPOCH, VAL_AVG_F1[BEST_EPOCH-1]),
               xytext=(3, 0.75),
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
               fontsize=10, color=COLORS['success'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/v2_val_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v2_val_f1.png")


# ============================================================================
# PLOT 3: Validation Accuracy
# ============================================================================

def plot_val_accuracy():
    """Validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = EPOCHS

    ax.plot(x, VAL_TYPE_ACC, marker='o', markersize=8, linewidth=2.5,
            color=COLORS['primary'], label='Type Accuracy', zorder=3)
    ax.plot(x, VAL_RISK_ACC, marker='s', markersize=8, linewidth=2.5,
            color=COLORS['success'], label='Risk Accuracy', zorder=3)

    ax.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE_LABEL)
    ax.set_title('v2 Validation — Accuracy', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xticks(EPOCHS)
    ax.set_ylim(0.60, 0.95)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotate final values
    for i, (vacc, racc) in enumerate(zip(VAL_TYPE_ACC, VAL_RISK_ACC)):
        ax.annotate(f'{vacc:.3f}', xy=(i+1, vacc), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8, color=COLORS['primary'])
        ax.annotate(f'{racc:.3f}', xy=(i+1, racc), xytext=(0, -12),
                    textcoords='offset points', ha='center', fontsize=8, color=COLORS['success'])

    plt.tight_layout()
    plt.savefig('plots/v2_val_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v2_val_accuracy.png")


# ============================================================================
# PLOT 4: Per-Class Type Metrics (Test Set)
# ============================================================================

def plot_type_per_class():
    """Per-class metrics for clause type classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax1 = axes[0]
    x = np.arange(len(TYPE_LABELS))
    width = 0.25

    prec_bars = ax1.bar(x - width, TYPE_PRECISION, width,
                        label='Precision', color=COLORS['primary'], alpha=0.8)
    rec_bars = ax1.bar(x, TYPE_RECALL, width,
                       label='Recall', color=COLORS['success'], alpha=0.8)
    f1_bars = ax1.bar(x + width, TYPE_F1, width,
                      label='F1 Score', color=COLORS['warning'], alpha=0.8)

    ax1.set_ylabel('Score', fontsize=FONTSIZE_LABEL)
    ax1.set_title('v2 Test Set — Type Per-Class Metrics', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(TYPE_LABELS, fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [prec_bars, rec_bars, f1_bars]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Table with support
    ax2 = axes[1]
    ax2.axis('off')

    table_data = []
    for i, label in enumerate(TYPE_LABELS):
        table_data.append([
            label.replace('\n', ' '),
            f'{TYPE_PRECISION[i]:.4f}',
            f'{TYPE_RECALL[i]:.4f}',
            f'{TYPE_F1[i]:.4f}',
            str(TYPE_SUPPORT[i])
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Class', 'Precision', 'Recall', 'F1', 'Support'],
        loc='center',
        cellLoc='center',
        colColours=[COLORS['gray']] * 5,
        bbox=[0.05, 0.1, 0.9, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    ax2.set_title('Type Classification — Detailed Metrics',
                  fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('plots/v2_type_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v2_type_per_class.png")


# ============================================================================
# PLOT 5: Per-Class Risk Metrics (Test Set)
# ============================================================================

def plot_risk_per_class():
    """Per-class metrics for risk level classification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax1 = axes[0]
    x = np.arange(len(RISK_LABELS))
    width = 0.25

    prec_bars = ax1.bar(x - width, RISK_PRECISION, width,
                        label='Precision', color=COLORS['primary'], alpha=0.8)
    rec_bars = ax1.bar(x, RISK_RECALL, width,
                       label='Recall', color=COLORS['success'], alpha=0.8)
    f1_bars = ax1.bar(x + width, RISK_F1, width,
                      label='F1 Score', color=COLORS['warning'], alpha=0.8)

    ax1.set_ylabel('Score', fontsize=FONTSIZE_LABEL)
    ax1.set_title('v2 Test Set — Risk Per-Class (Threshold-Adjusted)',
                  fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{lbl}\n(n={sup})' for lbl, sup in zip(RISK_LABELS, RISK_SUPPORT)],
                        fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Highlight medium class
    ax1.annotate('Challenge Class',
                xy=(1, RISK_F1[1]),
                xytext=(1.3, 0.35),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2),
                fontsize=10, color=COLORS['danger'], fontweight='bold')

    for bars in [prec_bars, rec_bars, f1_bars]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    # Table
    ax2 = axes[1]
    ax2.axis('off')

    table_data = [
        ['Low', f'{RISK_PRECISION[0]:.4f}', f'{RISK_RECALL[0]:.4f}', f'{RISK_F1[0]:.4f}', str(RISK_SUPPORT[0])],
        ['Medium', f'{RISK_PRECISION[1]:.4f}', f'{RISK_RECALL[1]:.4f}', f'{RISK_F1[1]:.4f}', str(RISK_SUPPORT[1])],
        ['High', f'{RISK_PRECISION[2]:.4f}', f'{RISK_RECALL[2]:.4f}', f'{RISK_F1[2]:.4f}', str(RISK_SUPPORT[2])],
    ]

    table = ax2.table(
        cellText=table_data,
        colLabels=['Class', 'Precision', 'Recall', 'F1', 'Support'],
        loc='center',
        cellLoc='center',
        colColours=[COLORS['gray']] * 5,
        bbox=[0.1, 0.2, 0.8, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Highlight medium row
    for j in range(5):
        table[(1, j)].set_facecolor('#fef3c7')
        table[(1, j)].set_edgecolor(COLORS['warning'])
        table[(1, j)].set_linewidth(2)

    ax2.set_title('Risk Classification — Detailed Metrics',
                  fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('plots/v2_risk_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v2_risk_per_class.png")


# ============================================================================
# PLOT 6: Combined v2 Training Summary Dashboard
# ============================================================================

def plot_v2_dashboard():
    """Single-page dashboard for v2 training."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    # Plot 1: Loss curves
    ax1.plot(EPOCHS, TRAIN_LOSS, marker='o', markersize=6, linewidth=2,
             color=COLORS['danger'], label='Total Loss')
    ax1.plot(EPOCHS, TRAIN_TYPE_LOSS, marker='s', markersize=5, linewidth=2,
             color=COLORS['primary'], alpha=0.7, label='Type Loss')
    ax1.plot(EPOCHS, TRAIN_RISK_LOSS, marker='^', markersize=5, linewidth=2,
             color=COLORS['success'], alpha=0.7, label='Risk Loss')
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Val F1
    ax2.plot(EPOCHS, VAL_TYPE_F1, marker='o', markersize=6, linewidth=2,
             color=COLORS['primary'], label='Type F1')
    ax2.plot(EPOCHS, VAL_RISK_F1, marker='s', markersize=6, linewidth=2,
             color=COLORS['success'], label='Risk F1')
    ax2.set_title('Validation F1', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=BEST_EPOCH, color=COLORS['success'], linestyle='--', alpha=0.5)

    # Plot 3: Val Accuracy
    ax3.plot(EPOCHS, VAL_TYPE_ACC, marker='o', markersize=6, linewidth=2,
             color=COLORS['primary'], label='Type Acc')
    ax3.plot(EPOCHS, VAL_RISK_ACC, marker='s', markersize=6, linewidth=2,
             color=COLORS['success'], label='Risk Acc')
    ax3.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Test Set Summary Table
    ax4.axis('off')

    test_metrics = [
        ['Task', 'Metric', 'Score'],
        ['Type Classification', 'Accuracy', f'{TEST_TYPE_ACC:.4f}'],
        ['Type Classification', 'Weighted F1', f'{TEST_TYPE_F1:.4f}'],
        ['Risk Classification', 'Accuracy', f'{TEST_RISK_ACC:.4f}'],
        ['Risk Classification', 'Macro F1', f'{TEST_RISK_MACRO_F1:.4f}'],
        ['', 'Best Val Avg F1', f'{BEST_AVG_F1:.4f} (Epoch {BEST_EPOCH})'],
    ]

    table = ax4.table(
        cellText=test_metrics[1:],
        colLabels=test_metrics[0],
        loc='center',
        cellLoc='center',
        colColours=[COLORS['gray']] * 3,
        bbox=[0.25, 0.15, 0.5, 0.7]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)

    ax4.set_title('v2 Final Test Set Results', fontsize=14, fontweight='bold', pad=20)

    plt.savefig('plots/v2_training_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: plots/v2_training_dashboard.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os

    os.makedirs('plots', exist_ok=True)

    print("=" * 60)
    print("AraContract Analyzer — v2 Training History Visualization")
    print("=" * 60)
    print()

    plot_loss_curves()
    plot_val_f1()
    plot_val_accuracy()
    plot_type_per_class()
    plot_risk_per_class()
    plot_v2_dashboard()

    print()
    print("=" * 60)
    print("✓ All v2 training plots generated!")
    print("Output directory: plots/")
    print("=" * 60)