"""
PyTorch Dataset implementation for loading AraContract clause data from JSONL.

The :class:`ClauseDataset` reads records from JSONL files, tokenizes text
using the CAMeLBERT tokenizer, and maps string labels to integer indices.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from warnings import warn

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import (
    MAX_SEQ_LENGTH,
    RISK_LABEL_TO_IDX,
    TRANSFORMER_MODEL,
    TYPE_LABEL_TO_IDX,
)


class ClauseDataset(Dataset):
    """
    Dataset for AraContract clause classification.

    Loads records from a JSONL file and returns tokenized inputs with
    integer-encoded type and risk labels.

    Args:
        data_path: Path to the JSONL file.
        tokenizer: Pre-initialised Hugging Face tokenizer (or model name).
        max_length: Maximum token sequence length.
    """

    TYPE_LABEL_MAP = TYPE_LABEL_TO_IDX
    RISK_LABEL_MAP = RISK_LABEL_TO_IDX

    def __init__(
        self,
        data_path: str | Path,
        tokenizer=None,
        max_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
        else:
            self.tokenizer = tokenizer

        #: List of raw records loaded from the JSONL file.
        self.records: List[Dict] = self._load_records()

        # Validate label coverage
        self._check_unknown_labels()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _load_records(self) -> List[Dict]:
        """Read and parse the JSONL file into a list of dictionaries."""
        records = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _check_unknown_labels(self) -> None:
        """Warn if the dataset contains labels not present in the config."""
        for rec in self.records:
            ttype = rec.get("type_clause")
            trisk = rec.get("risk_level")
            if ttype not in self.TYPE_LABEL_MAP:
                warn(
                    f"Unknown type_clause label '{ttype}' found in record. "
                    "It will be skipped during indexing.",
                    stacklevel=1,
                )
            if trisk not in self.RISK_LABEL_MAP:
                warn(
                    f"Unknown risk_level label '{trisk}' found in record. "
                    "It will be skipped during indexing.",
                    stacklevel=1,
                )

    # ---------------------------------------------------------------------
    # Dataset protocol
    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]

        text = record.get("text", "")
        type_label_str = record.get("type_clause", "")
        risk_label_str = record.get("risk_level", "")

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        type_label = self.TYPE_LABEL_MAP.get(type_label_str, 0)
        risk_label = self.RISK_LABEL_MAP.get(risk_label_str, 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "type_label": torch.tensor(type_label, dtype=torch.long),
            "risk_label": torch.tensor(risk_label, dtype=torch.long),
        }

    # ---------------------------------------------------------------------
    # Inverse mapping access
    # ---------------------------------------------------------------------

    @classmethod
    def get_type_label_map(cls) -> Dict[str, int]:
        """Return the type_clause label-to-index mapping."""
        return cls.TYPE_LABEL_MAP.copy()

    @classmethod
    def get_risk_label_map(cls) -> Dict[str, int]:
        """Return the risk_level label-to-index mapping."""
        return cls.RISK_LABEL_MAP.copy()

    @classmethod
    def get_type_idx_to_label(cls) -> Dict[int, str]:
        """Return the type_clause index-to-label mapping."""
        return {v: k for k, v in cls.TYPE_LABEL_MAP.items()}

    @classmethod
    def get_risk_idx_to_label(cls) -> Dict[int, str]:
        """Return the risk_level index-to-label mapping."""
        return {v: k for k, v in cls.RISK_LABEL_MAP.items()}