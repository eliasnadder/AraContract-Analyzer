"""
Inference wrapper for AraContract Classifier.
Handles tokenization, model loading, device placement, and predictions.
Includes a rule-based fallback if the model checkpoint is missing.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import logging

from app.models.labels import (
    MODEL_TYPE_LABELS,
    RISK_LABELS,
    map_model_output_to_canonical,
    MODEL_TYPE_LABEL_TO_IDX,
    RISK_LABEL_TO_IDX
)
from app.services.risk_warnings import get_warning_for_clause

logger = logging.getLogger(__name__)

# Re-define model class structure for loading state dict
class AraContractClassifier(nn.Module):
    def __init__(self, model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.type_classifier = nn.Linear(hidden_size, len(MODEL_TYPE_LABELS))
        self.risk_classifier = nn.Linear(hidden_size, len(RISK_LABELS))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        type_logits = self.type_classifier(pooled_output)
        risk_logits = self.risk_classifier(pooled_output)
        return type_logits, risk_logits


class AraContractInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_fallback = True
        
        # Check if checkpoint exists
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading classifier model from {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-msa")
                self.model = AraContractClassifier()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.is_fallback = False
                logger.info("Classifier model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Falling back to rule-based classification.")
        else:
            logger.warning(f"Checkpoint not found at {model_path}. Using rule-based fallback classification.")

    def _predict_rule_based(self, text: str) -> Dict[str, Any]:
        """Rule-based/keyword classification fallback."""
        text_norm = text.replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
        
        # Heuristics for clause type
        type_scores = {label: 0.05 for label in MODEL_TYPE_LABELS}
        
        if any(w in text_norm for w in ["دفع", "سداد", "ليرة", "دولار", "مبلغ", "قيمة", "مالي", "رصيد", "ثمن", "دفعات"]):
            type_scores["payment_financial"] = 0.8
        elif any(w in text_norm for w in ["مدة", "سريان", "ينتهي", "تجديد", "سنة", "شهر", "فترة", "تاريخ"]):
            type_scores["duration_expiration"] = 0.8
        elif any(w in text_norm for w in ["فسخ", "انهاء", "الغاء", "ابطال", "مفسوخ"]):
            type_scores["termination"] = 0.8
        elif any(w in text_norm for w in ["غرامة", "تعويض", "جزائي", "شرط جزائي", "اضرار", "مسؤولية", "تقصير"]):
            type_scores["penalties_damages"] = 0.8
        elif any(w in text_norm for w in ["نزاع", "خلاف", "تحكيم", "محكمة", "قضاء", "صلاحية", "نزاعات"]):
            type_scores["dispute_resolution"] = 0.8
        elif any(w in text_norm for w in ["يلتزم", "يتعهد", "مسؤولية", "الطرف الاول", "الطرف الثاني", "تعهد"]):
            type_scores["party_obligations"] = 0.8
        else:
            type_scores["general_provisions"] = 0.8
            
        # Normalize type probabilities
        sum_types = sum(type_scores.values())
        type_probs = {k: v / sum_types for k, v in type_scores.items()}
        pred_type_model = max(type_probs, key=type_probs.get)
        
        # Heuristics for risk level
        risk_scores = {"low": 0.6, "medium": 0.3, "high": 0.1}
        if any(w in text_norm for w in ["دون اشعار", "دون انذار", "تلقائي", "منفرد", "بشكل منفرد", "يتحمل وحده"]):
            risk_scores = {"low": 0.1, "medium": 0.3, "high": 0.6}
        elif any(w in text_norm for w in ["غرامة", "فوائد", "تعويض", "محكمة"]):
            risk_scores = {"low": 0.2, "medium": 0.6, "high": 0.2}
            
        sum_risks = sum(risk_scores.values())
        risk_probs = {k: v / sum_risks for k, v in risk_scores.items()}
        pred_risk = max(risk_probs, key=risk_probs.get)
        
        # Map 7-class model output to canonical 8-class SRS output
        canonical_type = map_model_output_to_canonical(pred_type_model, text)
        
        # Generate warning if high risk
        warning = get_warning_for_clause(canonical_type, pred_risk)
        
        # Build canonical probability dict for the 7 raw model classes
        return {
            "predicted_type_clause": canonical_type,
            "predicted_risk_level": pred_risk,
            "type_clause_probabilities": type_probs,
            "risk_level_probabilities": risk_probs,
            "warning": warning
        }

    def predict_single(self, text: str, return_probs: bool = False) -> Dict[str, Any]:
        if self.is_fallback:
            return self._predict_rule_based(text)
            
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                type_logits, risk_logits = self.model(input_ids, attention_mask)
                
                type_probs = torch.softmax(type_logits, dim=-1).squeeze(0).cpu().numpy()
                risk_probs = torch.softmax(risk_logits, dim=-1).squeeze(0).cpu().numpy()
                
            pred_type_idx = np.argmax(type_probs)
            pred_risk_idx = np.argmax(risk_probs)
            
            pred_type_model = MODEL_TYPE_LABELS[pred_type_idx]
            pred_risk = RISK_LABELS[pred_risk_idx]
            
            # Map 7-class model output to canonical 8-class SRS output
            canonical_type = map_model_output_to_canonical(pred_type_model, text)
            
            # Get warning
            warning = get_warning_for_clause(canonical_type, pred_risk)
            
            type_probs_dict = {MODEL_TYPE_LABELS[i]: float(type_probs[i]) for i in range(len(MODEL_TYPE_LABELS))}
            risk_probs_dict = {RISK_LABELS[i]: float(risk_probs[i]) for i in range(len(RISK_LABELS))}
            
            return {
                "predicted_type_clause": canonical_type,
                "predicted_risk_level": pred_risk,
                "type_clause_probabilities": type_probs_dict,
                "risk_level_probabilities": risk_probs_dict,
                "warning": warning
            }
        except Exception as e:
            logger.error(f"Error during deep learning inference: {e}. Falling back to rule-based.")
            return self._predict_rule_based(text)

    def predict_batch(self, texts: List[str], return_probs: bool = False) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            results.append(self.predict_single(text, return_probs))
        return results
