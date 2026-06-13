"""
Inference wrapper for AraContract Classifier.
Handles tokenization, model loading, device placement, and predictions.
Includes a rule-based fallback if the model checkpoint is missing.
"""

import json
import os
import re
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
import logging

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:  # Allows fallback/unit tests to run without model deps.
    AutoTokenizer = None
    AutoModel = None

from app.models.labels import (
    MODEL_TYPE_LABELS_7,
    MODEL_TYPE_LABELS_8,
    RISK_LABELS,
    map_model_output_to_canonical,
    MODEL_TYPE_LABEL_TO_IDX,
    RISK_LABEL_TO_IDX
)
from app.services.risk_warnings import get_warning_for_clause

logger = logging.getLogger(__name__)

# Re-define model class structure for loading state dict


class AraContractClassifier(nn.Module):
    def __init__(self, model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa", num_type_classes: int = 7):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError(
                "transformers is required to load AraContractClassifier")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.type_classifier = nn.Linear(hidden_size, num_type_classes)
        self.risk_classifier = nn.Linear(hidden_size, len(RISK_LABELS))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_fallback = True
        # Default to 8-class labels (canonical)
        self.labels = MODEL_TYPE_LABELS_8
        self.risk_thresholds = {"medium_threshold": 0.5, "high_threshold": 0.5}

        # Check if checkpoint exists
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading classifier model from {model_path}...")
                if AutoTokenizer is None:
                    raise RuntimeError(
                        "transformers is required to load classifier checkpoints")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "CAMeL-Lab/bert-base-arabic-camelbert-msa")
                # Load checkpoint - handle both direct state_dict and wrapped (Colab) format
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=True)
                # Colab saves with 'model_state_dict' wrapper, extract if present
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    self.risk_thresholds.update(
                        checkpoint.get("config", {}).get("risk_thresholds", {})
                    )
                else:
                    state_dict = checkpoint

                config_path = os.path.join(
                    os.path.dirname(model_path), "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        export_config = json.load(f)
                    self.risk_thresholds.update(
                        export_config.get("risk_thresholds", {}))

                # Dynamic shape detection from checkpoint
                if "type_classifier.weight" in state_dict:
                    num_classes = state_dict["type_classifier.weight"].shape[0]
                    if num_classes == 8:
                        self.labels = MODEL_TYPE_LABELS_8
                    elif num_classes == 7:
                        self.labels = MODEL_TYPE_LABELS_7
                    else:
                        self.labels = [
                            f"class_{i}" for i in range(num_classes)]
                else:
                    self.labels = MODEL_TYPE_LABELS_7

                self.model = AraContractClassifier(
                    num_type_classes=len(self.labels))
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.is_fallback = False
                logger.info(
                    f"Classifier model loaded successfully with {len(self.labels)} classes.")
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoint: {e}. Falling back to rule-based classification.")
        else:
            logger.warning(
                f"Checkpoint not found at {model_path}. Using rule-based fallback classification.")

    def _predict_rule_based(self, text: str) -> Dict[str, Any]:
        """Rule-based/keyword classification fallback."""
        text_norm = self._normalize_arabic(text)

        # Heuristics for clause type
        type_scores = {label: 0.05 for label in self.labels}

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
            if "party_obligations" in type_scores:
                type_scores["party_obligations"] = 0.8
            else:
                if any(w in text_norm for w in ["الطرف الاول", "المؤجر", "البائع", "صاحب العمل"]):
                    type_scores["party_obligations_a"] = 0.8
                elif any(w in text_norm for w in ["الطرف الثاني", "المستاجر", "المشتري", "الموظف"]):
                    type_scores["party_obligations_b"] = 0.8
                else:
                    type_scores["party_obligations_a"] = 0.8
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

        # Map model output to canonical 8-class SRS output
        canonical_type = map_model_output_to_canonical(pred_type_model, text)
        canonical_type, pred_risk = self._apply_legal_overrides(
            text,
            canonical_type,
            pred_risk,
        )

        # Generate warning if high risk
        warning = get_warning_for_clause(canonical_type, pred_risk)

        # Build canonical probability dict for the raw model classes
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
                type_logits, risk_logits = self.model(
                    input_ids, attention_mask)

                type_probs = torch.softmax(
                    type_logits, dim=-1).squeeze(0).cpu().numpy()
                risk_probs = torch.softmax(
                    risk_logits, dim=-1).squeeze(0).cpu().numpy()

            pred_type_idx = np.argmax(type_probs)
            pred_risk_idx = self._predict_risk_index(risk_probs)

            pred_type_model = self.labels[pred_type_idx]
            pred_risk = RISK_LABELS[pred_risk_idx]

            # Map model output to canonical 8-class SRS output
            canonical_type = map_model_output_to_canonical(
                pred_type_model, text)
            canonical_type, pred_risk = self._apply_legal_overrides(
                text,
                canonical_type,
                pred_risk,
            )

            # Get warning
            warning = get_warning_for_clause(canonical_type, pred_risk)

            type_probs_dict = {self.labels[i]: float(
                type_probs[i]) for i in range(len(self.labels))}
            risk_probs_dict = {RISK_LABELS[i]: float(
                risk_probs[i]) for i in range(len(RISK_LABELS))}

            return {
                "predicted_type_clause": canonical_type,
                "predicted_risk_level": pred_risk,
                "type_clause_probabilities": type_probs_dict,
                "risk_level_probabilities": risk_probs_dict,
                "warning": warning
            }
        except Exception as e:
            logger.error(
                f"Error during deep learning inference: {e}. Falling back to rule-based.")
            return self._predict_rule_based(text)

    def predict_batch(self, texts: List[str], return_probs: bool = False) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            results.append(self.predict_single(text, return_probs))
        return results

    # def _predict_risk_index(self, risk_probs: np.ndarray) -> int:
    #     if risk_probs[1] >= self.risk_thresholds["medium_threshold"]:
    #         return 1
    #     if risk_probs[2] >= self.risk_thresholds["high_threshold"]:
    #         return 2
    #     return int(np.argmax(risk_probs))

    # الكود الصحيح — high له أولوية أعلى
    def _predict_risk_index(self, risk_probs: np.ndarray) -> int:
        if risk_probs[2] >= self.risk_thresholds["high_threshold"]:   # high أولاً
            return 2
        if risk_probs[1] >= self.risk_thresholds["medium_threshold"]:  # ثم medium
            return 1
        return int(np.argmax(risk_probs))

    def _normalize_arabic(self, text: str) -> str:
        text_norm = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text_norm = text_norm.replace("ة", "ه").replace("ى", "ي")
        text_norm = re.sub(r"[\u064b-\u065f\u0670]", "", text_norm)
        return re.sub(r"\s+", " ", text_norm).strip()

    def _apply_legal_overrides(
        self,
        text: str,
        predicted_type: str,
        predicted_risk: str,
        type_confidence=None
    ) -> tuple[str, str]:
        
        text_norm = self._normalize_arabic(text)

        if type_confidence is None or type_confidence < 0.60:
            if self._is_dispute_clause(text_norm):
                predicted_type = "dispute_resolution"
            elif self._is_party_b_obligation(text_norm):
                predicted_type = "party_obligations_b"
            elif self._is_party_a_obligation(text_norm):
                predicted_type = "party_obligations_a"
            elif self._is_financial_clause(text_norm):
                predicted_type = "payment_financial"

        if self._is_high_risk_party_b_clause(text_norm):
            return "party_obligations_b", "high"
        if self._is_high_risk_termination_clause(text_norm):
            if predicted_type not in {"party_obligations_b", "termination"}:
                predicted_type = "termination"
            return predicted_type, "high"
        if self._is_medium_risk_damages_clause(text_norm) and predicted_risk == "low":
            if predicted_type not in {"penalties_damages", "party_obligations_b", "payment_financial"}:
                predicted_type = "penalties_damages"
            return predicted_type, "medium"

        return predicted_type, predicted_risk

    def _is_financial_clause(self, text_norm: str) -> bool:
        if any(t in text_norm for t in ["موضوع العقد", "نطاقه", "محل الاستثمار"]):
            return False
    
        financial_terms = [
            "ايرادات",
            "ارباح",
            "الرصيد",
            "السلفه",
            "محاسبه",
            "المصاريف",
            "الرسوم",
            "التامين",
            "مبلغ",
            "يدفع",
            "توريد",
        ]
        return any(term in text_norm for term in financial_terms)
    
    def _is_dispute_clause(self, text_norm: str) -> bool:
        return any(term in text_norm for term in ["خلاف", "نزاع", "تحكيم", "محكم"])

    def _is_party_b_obligation(self, text_norm: str) -> bool:
        party_b_terms = ["الفريق الثاني", "الطرف الثاني", "المستثمر"]
        obligation_terms = ["يلتزم", "استلم", "عدم",
                            "يتحمل", "مسؤولي", "اعاد", "اتخاذ"]
        return any(term in text_norm for term in party_b_terms) and any(
            term in text_norm for term in obligation_terms
        )

    def _is_party_a_obligation(self, text_norm: str) -> bool:
        party_a_terms = ["الفريق الاول", "الطرف الاول", "المالك"]
        obligation_terms = ["تقع علي عاتق", "يتحمل", "يلتزم", "رسوم", "صيان"]
        return any(term in text_norm for term in party_a_terms) and any(
            term in text_norm for term in obligation_terms
        )

    def _is_high_risk_party_b_clause(self, text_norm: str) -> bool:
        has_party_b = any(term in text_norm for term in [
                          "الفريق الثاني", "الطرف الثاني", "المستثمر"])
        severe_terms = [
            "كامل المسؤوليه",
            "المسؤوليه المدنيه والجزائيه",
            "ايا كان سببها",
            "دون الحاجه لاعذار",
            "دون الحاجه لانذار",
            "دون الحاجه الي اعذار",
            "دون الحصول علي حكم",
            "من تلقاء نفسه",
            "بمجرد حلول اجلها",
        ]
        return has_party_b and any(term in text_norm for term in severe_terms)

    def _is_high_risk_termination_clause(self, text_norm: str) -> bool:
        termination_terms = ["فسخ", "مفسوخ", "انهاء", "من تلقاء نفسه"]
        unfair_terms = [
            "دون اشعار",
            "دون اخطار",
            "دون انذار",
            "دون الحاجه لاعذار",
            "دون الحصول علي حكم",
            "في اي وقت",
        ]
        return any(term in text_norm for term in termination_terms) and any(
            term in text_norm for term in unfair_terms
        )

    def _is_medium_risk_damages_clause(self, text_norm: str) -> bool:
        damage_terms = ["تعويض", "اضرار", "عطل", "ضرر", "مخالفات", "غرامات"]
        responsibility_terms = ["يتحمل", "تسديد", "ناتجه عن", "بسبب"]
        return any(term in text_norm for term in damage_terms) and any(
            term in text_norm for term in responsibility_terms
        )
