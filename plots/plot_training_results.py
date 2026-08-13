#!/usr/bin/env python3
"""
Parse training_results_analysis.html (or v1_before_after_party_split.html)
and a training log / Notes.md to recreate training plots using matplotlib.

Usage:
  python scripts/plot_training_results.py \
      --html v1_before_after_party_split.html \
      --notes tt.md \
      --out outputs --show
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _js_multiline_array(text: str, name: str) -> List[float]:
    """
    Extract a named JS array that may span multiple lines and contain
    arithmetic expressions like  0.9308 - 0.9603.
    """
    m = re.search(
        rf"(?:const\s+)?{re.escape(name)}\s*=\s*\[(.*?)\];",
        text,
        re.S,
    )
    if not m:
        return []
    body = m.group(1)
    # Evaluate each comma-separated token as a Python expression
    results: List[float] = []
    for token in re.split(r",", body):
        token = token.strip()
        if not token:
            continue
        try:
            results.append(float(eval(token)))  # noqa: S307 — safe: only numeric literals
        except Exception:
            pass
    return results


def _js_array(text: str, name: str) -> List[Optional[float]]:
    """Extract a named JS array of numbers (may contain null), e.g. const foo = [1, null, 3];"""
    m = re.search(rf"(?:const\s+)?{re.escape(name)}\s*=\s*\[([^\]]+)\]", text)
    if not m:
        return []
    # Replace JS null with Python None so ast.literal_eval works
    raw = "[" + m.group(1).replace("null", "None") + "]"
    try:
        return list(ast.literal_eval(raw))
    except Exception:
        return []


def _js_string_array(text: str, name: str) -> List[str]:
    """Extract a named JS array of quoted strings."""
    m = re.search(rf"(?:const\s+)?{re.escape(name)}\s*=\s*\[([^\]]+)\]", text)
    if not m:
        return []
    raw = "[" + m.group(1) + "]"
    try:
        result = ast.literal_eval(raw)
        return [str(x) for x in result]
    except Exception:
        # fallback: grab quoted tokens
        return re.findall(r"'([^']+)'", m.group(1))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_epoch_data(html_text: str) -> Dict:
    """
    Supports two HTML formats:
      1. Original training_results_analysis.html  — epochData.type_f1 / risk_f1
      2. v1_before_after_party_split.html         — Chart.js dataset data arrays
                                                    for 'before' and 'after' epoch curves
    """
    # --- Format 1: explicit epochData block ---
    def _extract_named(name: str) -> List[float]:
        m = re.search(rf"{name}\s*:\s*\[([^\]]+)\]", html_text)
        if not m:
            return []
        try:
            return list(ast.literal_eval("[" + m.group(1) + "]"))
        except Exception:
            return []

    type_f1 = _extract_named("type_f1")
    risk_f1 = _extract_named("risk_f1")

    # --- Format 2: Chart.js epoch line chart datasets ---
    # The HTML has two datasets inside the epochChart: before (7 classes) and after (8 classes).
    # We grab both `data:` arrays that appear after the epochChart canvas definition.
    epoch_section = ""
    m_epoch = re.search(r"epochChart.*", html_text, re.S)
    if m_epoch:
        epoch_section = m_epoch.group(0)

    epoch_datasets: List[List[float]] = []
    for m in re.finditer(r"data\s*:\s*\[([0-9.,\s]+)\]", epoch_section):
        try:
            arr = list(ast.literal_eval("[" + m.group(1) + "]"))
            if arr:
                epoch_datasets.append(arr)
        except Exception:
            pass

    # epoch_datasets[0] = before (7 classes), epoch_datasets[1] = after (8 classes)
    before_epoch: List[float] = epoch_datasets[0] if len(epoch_datasets) > 0 else []
    after_epoch: List[float] = epoch_datasets[1] if len(epoch_datasets) > 1 else []

    # Epoch labels
    labels_m = re.search(r"labels\s*:\s*\[([^\]]+)\]", html_text)
    labels: List[str] = []
    if labels_m:
        try:
            raw = ast.literal_eval("[" + labels_m.group(1) + "]")
            labels = [str(x) for x in raw]
        except Exception:
            labels = re.findall(r"'([^']+)'", labels_m.group(1))

    # If we got epoch curves from format 2, use the after-split curve as type_f1
    # (it corresponds to the current 8-class model) and keep risk_f1 from notes.
    if not type_f1 and after_epoch:
        type_f1 = after_epoch

    return {
        "labels": labels,
        "type_f1": type_f1,
        "risk_f1": risk_f1,
        "before_epoch": before_epoch,
        "after_epoch": after_epoch,
    }


def parse_classes(html_text: str) -> List[Dict]:
    """
    Supports two formats:
      1. const classes = [{name:'...', p:0.9, r:0.9, f1:0.9, n:100}, ...]
      2. afterLabels / afterF1 arrays (v1_before_after_party_split.html)
    """
    # Format 1
    m = re.search(r"const classes = \[(.*?)\];", html_text, re.S)
    if m:
        block = m.group(1)
        items = re.findall(r"\{([^}]+)\}", block, re.S)
        classes = []
        for item in items:
            name_m = re.search(r"name\s*:\s*'([^']+)'", item)
            p_m = re.search(r"\bp\s*:\s*([0-9.]+)", item)
            r_m = re.search(r"\br\s*:\s*([0-9.]+)", item)
            f1_m = re.search(r"\bf1\s*:\s*([0-9.]+)", item)
            n_m = re.search(r"\bn\s*:\s*([0-9]+)", item)
            if not name_m:
                continue
            classes.append({
                "name": name_m.group(1),
                "p": float(p_m.group(1)) if p_m else None,
                "r": float(r_m.group(1)) if r_m else None,
                "f1": float(f1_m.group(1)) if f1_m else None,
                "n": int(n_m.group(1)) if n_m else None,
            })
        if classes:
            return classes

    # Format 2: afterLabels + afterF1
    after_labels = _js_string_array(html_text, "afterLabels")
    after_f1 = _js_array(html_text, "afterF1")
    if after_labels and after_f1:
        return [
            {"name": lbl.replace("\n", "_"), "p": None, "r": None, "f1": f1, "n": None}
            for lbl, f1 in zip(after_labels, after_f1)
        ]

    return []


def parse_before_after(html_text: str) -> Optional[Dict]:
    """
    Parse before/after comparison data from v1_before_after_party_split.html.
    Returns None if the data isn't present.
    """
    before_labels = _js_string_array(html_text, "before.labels") or []
    # 'before' is an object literal — grab its labels and f1 arrays differently
    m_before = re.search(r"const before\s*=\s*\{(.*?)\};", html_text, re.S)
    if m_before:
        blk = m_before.group(1)
        lbl_m = re.search(r"labels\s*:\s*\[([^\]]+)\]", blk)
        f1_m = re.search(r"f1\s*:\s*\[([^\]]+)\]", blk)
        if lbl_m:
            try:
                before_labels = [str(x) for x in ast.literal_eval("[" + lbl_m.group(1) + "]")]
            except Exception:
                before_labels = re.findall(r"'([^']+)'", lbl_m.group(1))
        before_f1: List[float] = []
        if f1_m:
            try:
                before_f1 = list(ast.literal_eval("[" + f1_m.group(1) + "]"))
            except Exception:
                before_f1 = []
    else:
        before_f1 = []

    after_labels = _js_string_array(html_text, "afterLabels")
    after_f1 = _js_array(html_text, "afterF1")
    merged_labels = _js_string_array(html_text, "mergedLabels")
    before_mapped = _js_array(html_text, "beforeMapped")
    after_mapped = _js_array(html_text, "afterMapped")
    delta_labels = _js_string_array(html_text, "deltaLabels")
    delta_values = _js_multiline_array(html_text, "deltaValues")

    if not (before_f1 or after_f1):
        return None

    return {
        "before_labels": before_labels,
        "before_f1": before_f1,
        "after_labels": after_labels,
        "after_f1": after_f1,
        "merged_labels": merged_labels,
        "before_mapped": before_mapped,
        "after_mapped": after_mapped,
        "delta_labels": delta_labels,
        "delta_values": delta_values,
    }


def parse_notes(notes_text: str) -> Dict[str, float]:
    """Parse training log or Notes.md for summary metrics."""
    out: Dict[str, float] = {}

    m = re.search(r"Best weighted F1:\s*([0-9.]+)", notes_text)
    if m:
        out["best_weighted_f1"] = float(m.group(1))

    m = re.search(r"Test type_f1:\s*([0-9.]+)\s*\|\s*risk_f1:\s*([0-9.]+)", notes_text)
    if m:
        out["test_type_f1"] = float(m.group(1))
        out["test_risk_f1"] = float(m.group(2))

    m = re.search(r"Test type_acc:\s*([0-9.]+)\s*\|\s*risk_acc:\s*([0-9.]+)", notes_text)
    if m:
        out["test_type_acc"] = float(m.group(1))
        out["test_risk_acc"] = float(m.group(2))

    # Parse per-epoch val metrics from training log lines
    val_type_f1: List[float] = []
    val_risk_f1: List[float] = []
    for line in notes_text.splitlines():
        m = re.search(r"Val type_f1:\s*([0-9.]+)\s*\|\s*risk_f1:\s*([0-9.]+)", line)
        if m:
            val_type_f1.append(float(m.group(1)))
            val_risk_f1.append(float(m.group(2)))

    if val_type_f1:
        out["val_type_f1"] = val_type_f1  # type: ignore[assignment]
    if val_risk_f1:
        out["val_risk_f1"] = val_risk_f1  # type: ignore[assignment]

    return out


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_epochs(
    type_f1: List[float],
    risk_f1: List[float],
    epoch_labels: List[str],
    out_path: Path,
    show: bool = False,
) -> None:
    n = max(len(type_f1), len(risk_f1))
    if n == 0:
        print("No epoch data to plot — skipping epoch_f1.png")
        return
    epochs = list(range(1, n + 1))
    x_labels = epoch_labels if len(epoch_labels) == n else [f"Epoch {e}" for e in epochs]

    plt.figure(figsize=(8, 4.2))
    if type_f1:
        plt.plot(epochs[: len(type_f1)], type_f1, marker="o", label="Type F1", color="#534AB7")
    if risk_f1:
        plt.plot(epochs[: len(risk_f1)], risk_f1, marker="o", label="Risk F1", color="#1D9E75")
    plt.xticks(epochs, x_labels)
    plt.ylim(0.6, 1.0)
    plt.ylabel("F1")
    plt.xlabel("Epoch")
    plt.title("Type & Risk F1 across Epochs")
    plt.grid(alpha=0.2)
    plt.legend()
    out_file = out_path / "epoch_f1.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved epoch plot to: {out_file}")


def plot_before_after_epochs(
    before: List[float],
    after: List[float],
    out_path: Path,
    show: bool = False,
) -> None:
    n = max(len(before), len(after))
    if n == 0:
        return
    epochs = list(range(1, n + 1))
    x_labels = [f"Epoch {e}" for e in epochs]

    plt.figure(figsize=(8, 4.2))
    if before:
        plt.plot(epochs[: len(before)], before, marker="o", linestyle="--",
                 label="Before split (7 classes)", color="#534AB7")
    if after:
        plt.plot(epochs[: len(after)], after, marker="o",
                 label="After split (8 classes)", color="#1D9E75")
    plt.xticks(epochs, x_labels)
    plt.ylim(0.68, 0.94)
    plt.ylabel("Val Type F1")
    plt.xlabel("Epoch")
    plt.title("Val Type F1: Before vs After party_obligations Split")
    plt.grid(alpha=0.2)
    plt.legend()
    out_file = out_path / "epoch_before_after.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved before/after epoch plot to: {out_file}")


def plot_class_f1(classes: List[Dict], out_path: Path, show: bool = False) -> None:
    names = [c["name"].replace("_", " ").replace("\n", " ") for c in classes]
    f1s = [c["f1"] for c in classes]

    x = np.arange(len(names))
    colors = ["#E24B4A" if f < 0.80 else "#1D9E75" if f >= 0.90 else "#378ADD" for f in f1s]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, f1s, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 (test set)")
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    # Legend
    patches = [
        mpatches.Patch(color="#E24B4A", label="F1 < 0.80"),
        mpatches.Patch(color="#378ADD", label="0.80 ≤ F1 < 0.90"),
        mpatches.Patch(color="#1D9E75", label="F1 ≥ 0.90"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    out_file = out_path / "class_f1.png"
    fig.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved class F1 bar chart to: {out_file}")


def plot_before_after_classes(ba: Dict, out_path: Path, show: bool = False) -> None:
    merged = ba["merged_labels"]
    bm = [v if v else np.nan for v in ba["before_mapped"]]
    am = [v if v else np.nan for v in ba["after_mapped"]]

    if not merged:
        return

    x = np.arange(len(merged))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width / 2, bm, width, label="Before split (7 classes)",
           color=(83/255, 74/255, 183/255, 0.75))
    ax.bar(x + width / 2, am, width, label="After split (8 classes)",
           color=(29/255, 158/255, 117/255, 0.75))
    ax.set_xticks(x)
    ax.set_xticklabels(merged, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1: Before vs After party_obligations Split")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_file = out_path / "class_before_after.png"
    fig.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved before/after class F1 chart to: {out_file}")


def plot_delta(ba: Dict, out_path: Path, show: bool = False) -> None:
    labels = ba["delta_labels"]
    values = ba["delta_values"]
    if not labels or not values:
        return

    colors = [(29/255, 158/255, 117/255, 0.8) if v >= 0 else (226/255, 75/255, 74/255, 0.8)
              for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Δ F1")
    ax.set_title("F1 Change per Class After party_obligations Split")
    for i, v in enumerate(values):
        ax.text(i, v + (0.002 if v >= 0 else -0.006),
                f"{'+' if v >= 0 else ''}{v:.3f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_file = out_path / "class_delta.png"
    fig.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved delta F1 chart to: {out_file}")


def plot_class_table(classes: List[Dict], out_path: Path) -> None:
    col_labels = ["class", "prec", "recall", "f1", "n"]
    table_data = [
        [
            c["name"],
            f"{c['p']:.3f}" if c["p"] is not None else "-",
            f"{c['r']:.3f}" if c["r"] is not None else "-",
            f"{c['f1']:.3f}" if c["f1"] is not None else "-",
            str(c["n"]) if c["n"] is not None else "-",
        ]
        for c in classes
    ]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(classes) + 1))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    out_file = out_path / "class_table.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved class metrics table to: {out_file}")


def plot_summary_bar(notes: Dict, out_path: Path, show: bool = False) -> None:
    """Bar chart of final test metrics from the training log."""
    metrics = {}
    for key in ("test_type_f1", "test_risk_f1", "test_type_acc", "test_risk_acc"):
        if key in notes:
            metrics[key.replace("test_", "").replace("_", " ")] = notes[key]
    if not metrics:
        return

    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#534AB7", "#1D9E75", "#378ADD", "#E2A84B"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Metrics Summary")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_file = out_path / "test_summary.png"
    fig.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved test summary bar chart to: {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training result plots from HTML + training log."
    )
    parser.add_argument("--html", type=Path, default=Path("training_results_analysis.html"))
    parser.add_argument("--notes", type=Path, default=Path("Notes.md"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if not args.html.exists():
        raise FileNotFoundError(f"HTML file not found: {args.html}")
    if not args.notes.exists():
        raise FileNotFoundError(f"Notes file not found: {args.notes}")

    args.out.mkdir(parents=True, exist_ok=True)

    html_text = args.html.read_text(encoding="utf-8")
    notes_text = args.notes.read_text(encoding="utf-8")

    epoch_data = parse_epoch_data(html_text)
    classes = parse_classes(html_text)
    ba = parse_before_after(html_text)
    notes = parse_notes(notes_text)

    print("Parsed epoch data keys:", {k: len(v) if isinstance(v, list) else v
                                       for k, v in epoch_data.items()})
    print("Parsed classes:", [c["name"] for c in classes])
    print("Before/after data:", "yes" if ba else "no")
    print("Parsed notes summary:", {k: v for k, v in notes.items() if not isinstance(v, list)})

    # --- Epoch curves ---
    # Prefer per-epoch data parsed from the training log (tt.md)
    type_f1 = notes.get("val_type_f1") or epoch_data.get("type_f1", [])  # type: ignore[assignment]
    risk_f1 = notes.get("val_risk_f1") or epoch_data.get("risk_f1", [])  # type: ignore[assignment]
    epoch_labels = epoch_data.get("labels", [])

    # If epoch labels came from the class chart (7 class names), discard them
    if len(epoch_labels) != len(type_f1) and len(epoch_labels) != len(risk_f1):
        epoch_labels = []

    plot_epochs(type_f1, risk_f1, epoch_labels, args.out, show=args.show)

    # --- Before/after epoch comparison (HTML-specific) ---
    before_ep = epoch_data.get("before_epoch", [])
    after_ep = epoch_data.get("after_epoch", [])
    if before_ep or after_ep:
        plot_before_after_epochs(before_ep, after_ep, args.out, show=args.show)

    # --- Class F1 bar chart ---
    if classes:
        plot_class_f1(classes, args.out, show=args.show)
        # Only plot table if we have precision/recall data
        if any(c["p"] is not None for c in classes):
            plot_class_table(classes, args.out)

    # --- Before/after class comparison + delta ---
    if ba:
        plot_before_after_classes(ba, args.out, show=args.show)
        plot_delta(ba, args.out, show=args.show)

    # --- Test metrics summary ---
    plot_summary_bar(notes, args.out, show=args.show)


if __name__ == "__main__":
    main()
