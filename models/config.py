"""
Configuration constants for the AraContract Analyzer clause classification model.

This module centralizes all hyperparameters, file paths, label mappings,
and model settings used across the training, evaluation, and inference pipelines.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data file paths
TRAIN_FILE = DATA_DIR / "aracontract_train.jsonl"
VAL_FILE = DATA_DIR / "aracontract_val.jsonl"
TEST_FILE = DATA_DIR / "aracontract_test.jsonl"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
TRANSFORMER_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
MAX_SEQ_LENGTH = 512

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_DROPOUT = 0.1
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Label mappings -- type_clause (7 classes)
# ---------------------------------------------------------------------------
TYPE_LABELS = [
    "general_provisions",
    "payment_financial",
    "party_obligations",
    "duration_expiration",
    "termination",
    "penalties_damages",
    "dispute_resolution",
]

TYPE_LABEL_TO_IDX = {label: i for i, label in enumerate(TYPE_LABELS)}
TYPE_IDX_TO_LABEL = {i: label for i, label in enumerate(TYPE_LABELS)}
NUM_TYPE_CLASSES = len(TYPE_LABELS)

# ---------------------------------------------------------------------------
# Label mappings -- risk_level (3 classes)
# ---------------------------------------------------------------------------
RISK_LABELS = ["low", "medium", "high"]
RISK_LABEL_TO_IDX = {label: i for i, label in enumerate(RISK_LABELS)}
RISK_IDX_TO_LABEL = {i: label for i, label in enumerate(RISK_LABELS)}
NUM_RISK_CLASSES = len(RISK_LABELS)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
CHECKPOINT_NAME = "aracontract_classifier"
BEST_MODEL_NAME = "best_model.pt"
LOG_INTERVAL = 50
EVAL_INTERVAL = 1
PATIENCE = 3

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
METRICS_AVERAGE = "weighted"