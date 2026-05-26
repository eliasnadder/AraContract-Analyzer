"""
Utility functions for label encoding/decoding, text preprocessing,
and metrics computation for the AraContract Analyzer.

Includes:
- Label encoding and decoding for type_clause and risk_level
- Basic Arabic/English text cleaning
- Classification metrics (accuracy, F1, precision, recall, confusion matrix)
- Result formatting utilities
"""

import re
import string
from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    RISK_IDX_TO_LABEL,
    RISK_LABEL_TO_IDX,
    TYPE_IDX_TO_LABEL,
    TYPE_LABEL_TO_IDX,
    METRICS_AVERAGE,
)


# ---------------------------------------------------------------------------
# Label encoding / decoding
# ---------------------------------------------------------------------------

def encode_type_label(label: str) -> int:
    """Encode a type_clause string label to its integer index."""
    return TYPE_LABEL_TO_IDX[label]


def decode_type_label(index: int) -> str:
    """Decode a type_clause integer index back to its string label."""
    return TYPE_IDX_TO_LABEL[index]


def encode_risk_label(label: str) -> int:
    """Encode a risk_level string label to its integer index."""
    return RISK_LABEL_TO_IDX[label]


def decode_risk_label(index: int) -> str:
    """Decode a risk_level integer index back to its string label."""
    return RISK_IDX_TO_LABEL[index]


def encode_labels(type_label: str, risk_label: str) -> tuple[int, int]:
    """Encode both type and risk labels simultaneously."""
    return encode_type_label(type_label), encode_risk_label(risk_label)


def decode_labels(type_idx: int, risk_idx: int) -> tuple[str, str]:
    """Decode both type and risk indices back to string labels."""
    return decode_type_label(type_idx), decode_risk_label(risk_idx)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic cleaning for Arabic and English contract text.

    Steps:
    - Normalize Arabic characters (alef variants, hamza forms, etc.)
    - Remove excessive whitespace, newlines, and redundant spaces
    - Remove zero-width characters
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize Arabic characters
    text = re.sub(r"[أإآ]", "ا", text)  # Alef variants
    text = re.sub(r"[ؤ]", "و", text)               # Hamza on waaw
    text = re.sub(r"[ة]", "ه", text)               # Ta marbuta

    # Remove zero-width characters
    text = re.sub(r"[​-‏﻿]", "", text)

    # Normalize multiple whitespaces
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def preprocess_batch(texts: List[str]) -> List[str]:
    """Apply :func:`clean_text` to a batch of strings."""
    return [clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    true_labels: Union[List[int], np.ndarray],
    pred_labels: Union[List[int], np.ndarray],
    num_classes: int,
    average: str = METRICS_AVERAGE,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        true_labels: Ground-truth label indices.
        pred_labels: Predicted label indices.
        num_classes: Total number of classes.
        average: Averaging strategy for F1 / precision / recall.

    Returns:
        Dictionary with keys: accuracy, f1, precision, recall.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels, average=average, zero_division=0),
        "precision": precision_score(
            true_labels, pred_labels, average=average, zero_division=0
        ),
        "recall": recall_score(
            true_labels, pred_labels, average=average, zero_division=0
        ),
    }
    return metrics


def compute_confusion_matrix(
    true_labels: Union[List[int], np.ndarray],
    pred_labels: Union[List[int], np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns:
        A (num_classes x num_classes) numpy array.
    """
    labels = list(range(num_classes))
    return confusion_matrix(
        np.asarray(true_labels),
        np.asarray(pred_labels),
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_prediction_result(
    predicted_type: str,
    predicted_risk: str,
    type_probs: Dict[str, float],
    risk_probs: Dict[str, float],
) -> Dict:
    """
    Format a single prediction result into a structured dictionary.

    Returns:
        Dictionary with type, risk, and probability scores.
    """
    return {
        "predicted_type_clause": predicted_type,
        "predicted_risk_level": predicted_risk,
        "type_clause_probabilities": type_probs,
        "risk_level_probabilities": risk_probs,
    }


def format_batch_results(
    predicted_types: List[str],
    predicted_risks: List[str],
    type_probabilities: List[Dict[str, float]],
    risk_probabilities: List[Dict[str, float]],
) -> List[Dict]:
    """Format a batch of prediction results."""
    results = []
    for ptype, prisk, tprobs, rprobs in zip(
        predicted_types, predicted_risks, type_probabilities, risk_probabilities
    ):
        results.append(format_prediction_result(ptype, prisk, tprobs, rprobs))
    return results