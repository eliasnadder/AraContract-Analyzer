#!/usr/bin/env python3
"""
Parse training_results_analysis.html and Notes.md to recreate training plots
using matplotlib and save them to an output directory.

Usage:
  python scripts/plot_training_results.py \
      --html training_results_analysis.html \
      --notes Notes.md \
      --out outputs --show
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def parse_epoch_data(html_text: str) -> Dict[str, List[float]]:
    # Find JS epochData block arrays
    def extract_array(name: str) -> List[float]:
        m = re.search(rf"{name}\s*:\s*\[([^\]]+)\]", html_text)
        if not m:
            return []
        arr_text = "[" + m.group(1) + "]"
        return list(ast.literal_eval(arr_text))

    labels_m = re.search(r"labels\s*:\s*\[([^\]]+)\]", html_text)
    if labels_m:
        labels_raw = ast.literal_eval("[" + labels_m.group(1) + "]")
        labels = [str(x) for x in labels_raw]
    else:
        labels = []

    type_f1 = extract_array("type_f1")
    risk_f1 = extract_array("risk_f1")

    return {"labels": labels, "type_f1": type_f1, "risk_f1": risk_f1}


def parse_classes(html_text: str) -> List[Dict]:
    m = re.search(r"const classes = \[(.*?)\];", html_text, re.S)
    if not m:
        return []
    block = m.group(1)
    items = re.findall(r"\{([^}]+)\}", block, re.S)
    classes = []
    for item in items:
        name_m = re.search(r"name\s*:\s*'([^']+)'", item)
        p_m = re.search(r"p\s*:\s*([0-9.]+)", item)
        r_m = re.search(r"r\s*:\s*([0-9.]+)", item)
        f1_m = re.search(r"f1\s*:\s*([0-9.]+)", item)
        n_m = re.search(r"n\s*:\s*([0-9]+)", item)
        if not name_m:
            continue
        classes.append({
            "name": name_m.group(1),
            "p": float(p_m.group(1)) if p_m else None,
            "r": float(r_m.group(1)) if r_m else None,
            "f1": float(f1_m.group(1)) if f1_m else None,
            "n": int(n_m.group(1)) if n_m else None,
        })
    return classes


def parse_notes(notes_text: str) -> Dict[str, float]:
    out = {}
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
    return out


def plot_epochs(labels: List[str], type_f1: List[float], risk_f1: List[float], out_path: Path, show: bool = False):
    epochs = list(range(1, max(len(type_f1), len(risk_f1)) + 1))
    plt.figure(figsize=(8, 4.2))
    plt.plot(epochs[: len(type_f1)], type_f1, marker='o', label='Type F1', color='#534AB7')
    plt.plot(epochs[: len(risk_f1)], risk_f1, marker='o', label='Risk F1', color='#1D9E75')
    plt.xticks(epochs, labels if labels else [f'Epoch {e}' for e in epochs])
    plt.ylim(0.6, 1.0)
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.title('Type & Risk F1 across Epochs')
    plt.grid(alpha=0.2)
    plt.legend()
    out_file = out_path / 'epoch_f1.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f'Saved epoch plot to: {out_file}')


def plot_class_f1(classes: List[Dict], out_path: Path, show: bool = False):
    names = [c['name'].replace('_', ' ') for c in classes]
    f1s = [c['f1'] for c in classes]
    precs = [c['p'] for c in classes]
    recs = [c['r'] for c in classes]

    x = np.arange(len(names))
    colors = [('#E24B4A' if f < 0.80 else '#1D9E75' if f >= 0.90 else '#378ADD') for f in f1s]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, f1s, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel('F1')
    ax.set_title('Per-class F1 (test set)')
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

    fig.tight_layout()
    out_file = out_path / 'class_f1.png'
    fig.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f'Saved class F1 bar chart to: {out_file}')


def plot_class_table(classes: List[Dict], out_path: Path):
    # Create a table image showing precision/recall/f1/n
    col_labels = ['class', 'prec', 'recall', 'f1', 'n']
    table_data = [[c['name'], f"{c['p']:.3f}" if c['p'] is not None else '-', f"{c['r']:.3f}" if c['r'] is not None else '-', f"{c['f1']:.3f}" if c['f1'] is not None else '-', str(c['n']) if c['n'] is not None else '-'] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(classes) + 1))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    out_file = out_path / 'class_table.png'
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close()
    print(f'Saved class metrics table to: {out_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--html', type=Path, default=Path('training_results_analysis.html'))
    parser.add_argument('--notes', type=Path, default=Path('Notes.md'))
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    html_path = args.html
    notes_path = args.notes
    out_path = args.out
    out_path.mkdir(parents=True, exist_ok=True)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes file not found: {notes_path}")

    html_text = html_path.read_text(encoding='utf-8')
    notes_text = notes_path.read_text(encoding='utf-8')

    epoch_data = parse_epoch_data(html_text)
    classes = parse_classes(html_text)
    notes = parse_notes(notes_text)

    print('Parsed epoch data:', epoch_data)
    print('Parsed classes:', [c['name'] for c in classes])
    print('Parsed notes summary:', notes)

    plot_epochs(epoch_data.get('labels', []), epoch_data.get('type_f1', []), epoch_data.get('risk_f1', []), out_path, show=args.show)
    if classes:
        plot_class_f1(classes, out_path, show=args.show)
        plot_class_table(classes, out_path)


if __name__ == '__main__':
    main()
