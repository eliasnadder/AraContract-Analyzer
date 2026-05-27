# test_model_forward.py
import pytest
import torch
import sys
from pathlib import Path

# Add models directory to sys.path so we can import model and config
models_dir = Path(__file__).resolve().parent.parent
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from model import AraContractClassifier
from config import NUM_TYPE_CLASSES, NUM_RISK_CLASSES

def test_forward_shapes():
    # Instantiate classifier (loads bert model config/weights locally or from HF cache)
    # We can use a fast mock encoder or load the config without downloading full weights to save time/resources,
    # but here we load it. Let's do a quick shape check.
    model = AraContractClassifier()
    model.eval()
    
    # Create dummy inputs (batch_size=2, seq_len=16)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    
    with torch.no_grad():
        type_logits, risk_logits = model(input_ids, attention_mask)
        
    assert type_logits.shape == (batch_size, NUM_TYPE_CLASSES)
    assert risk_logits.shape == (batch_size, NUM_RISK_CLASSES)
