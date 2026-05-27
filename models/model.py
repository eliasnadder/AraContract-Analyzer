"""
Model architecture for AraContract Classifier.
Implements a multi-task BERT model for joint type and risk classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import TRANSFORMER_MODEL, NUM_TYPE_CLASSES, NUM_RISK_CLASSES
from utils import compute_metrics

class AraContractClassifier(nn.Module):
    def __init__(self, model_name: str = TRANSFORMER_MODEL, dropout_prob: float = 0.1):
        super().__init__()
        # Load the base transformer
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification heads
        self.type_classifier = nn.Linear(hidden_size, NUM_TYPE_CLASSES)
        self.risk_classifier = nn.Linear(hidden_size, NUM_RISK_CLASSES)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooler output if available, otherwise CLS token representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        
        type_logits = self.type_classifier(pooled_output)
        risk_logits = self.risk_classifier(pooled_output)
        
        return type_logits, risk_logits

    def training_step(self, input_ids, attention_mask, type_labels, risk_labels):
        type_logits, risk_logits = self(input_ids, attention_mask)
        
        type_loss = self.loss_fn(type_logits, type_labels)
        risk_loss = self.loss_fn(risk_logits, risk_labels)
        
        total_loss = type_loss + risk_loss
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "type_loss": type_loss.item(),
            "risk_loss": risk_loss.item(),
        }
        
        return total_loss, loss_dict

    def evaluate(self, dataloader, device) -> dict:
        self.eval()
        
        all_type_preds = []
        all_type_labels = []
        all_risk_preds = []
        all_risk_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                type_labels = batch["type_label"].to(device)
                risk_labels = batch["risk_label"].to(device)
                
                type_logits, risk_logits = self(input_ids, attention_mask)
                
                type_preds = torch.argmax(type_logits, dim=-1)
                risk_preds = torch.argmax(risk_logits, dim=-1)
                
                all_type_preds.extend(type_preds.cpu().numpy())
                all_type_labels.extend(type_labels.cpu().numpy())
                all_risk_preds.extend(risk_preds.cpu().numpy())
                all_risk_labels.extend(risk_labels.cpu().numpy())
                
        # Compute metrics using the helper from utils.py
        type_metrics = compute_metrics(all_type_labels, all_type_preds, NUM_TYPE_CLASSES)
        risk_metrics = compute_metrics(all_risk_labels, all_risk_preds, NUM_RISK_CLASSES)
        
        return {
            "type_f1": type_metrics["f1"],
            "risk_f1": risk_metrics["f1"],
            "type_accuracy": type_metrics["accuracy"],
            "risk_accuracy": risk_metrics["accuracy"],
        }
