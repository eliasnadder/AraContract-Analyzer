import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# تعريف الفئات الثمانية حسب SRS المحدث
CLAUSE_TYPES = [
    'general_provisions',
    'payment_financial',
    'party_one_obligations',  # تم الفصل
    'party_two_obligations',  # تم الفصل
    'duration_expiration',
    'termination',
    'penalties_damages',
    'dispute_resolution'
]

RISK_LEVELS = ['low', 'medium', 'high']

class ImprovedAraContractClassifier(nn.Module):
    def __init__(self, model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix", num_types=8, num_risks=3):
        super(ImprovedAraContractClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # رؤوس التصنيف (Classification Heads)
        self.type_classifier = nn.Linear(self.bert.config.hidden_size, num_types)
        self.risk_classifier = nn.Linear(self.bert.config.hidden_size, num_risks)
        self.reason_classifier = nn.Linear(self.bert.config.hidden_size, 50) # لسبب الخطورة (اختياري)

        # --- تحسين 1: أوزان الخسارة لمعالجة عدم التوازن (Class Imbalance) ---
        # تم زيادة وزن termination بشكل كبير لتحسين Recall
        # تم إضافة أوزان للفئات الجديدة (party_one/two)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        type_weights = torch.tensor([
            1.0,  # general_provisions (كثيرة، لا تحتاج وزن عالي)
            1.2,  # payment_financial
            1.1,  # party_one_obligations
            1.1,  # party_two_obligations
            1.2,  # duration_expiration
            2.8,  # termination <-- تم رفع الوزن من 2.5 إلى 2.8 لمعالجة الـ Recall المنخفض جداً (0.66)
            1.4,  # penalties_damages
            1.6   # dispute_resolution
        ], dtype=torch.float).to(self.device)
        
        # أوزان لمستويات الخطورة (لمعالجة ضعف فئة medium)
        # نفترض أن التوزيع: low (كثير), medium (قليل جداً), high (متوسط)
        risk_weights = torch.tensor([
            1.0,  # low
            3.5,  # medium <-- وزن عالي جداً لأن نسبتها 3.5% فقط
            1.8   # high
        ], dtype=torch.float).to(self.device)

        self.type_loss_fn = nn.CrossEntropyLoss(weight=type_weights)
        self.risk_loss_fn = nn.CrossEntropyLoss(weight=risk_weights)
        
        print(f"✅ تم تهيئة النموذج مع {num_types} فئات للأبناد و {num_risks} مستويات للخطورة.")
        print(f"⚖️ أوزان الخسارة لـ Termination: {type_weights[5].item()}")
        print(f"⚖️ أوزان الخسارة لـ Medium Risk: {risk_weights[1].item()}")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        type_logits = self.type_classifier(pooled_output)
        risk_logits = self.risk_classifier(pooled_output)
        # reason_logits = self.reason_classifier(pooled_output) # يمكن تفعيله لاحقاً
        
        return type_logits, risk_logits

    def compute_loss(self, type_logits, risk_logits, type_labels, risk_labels):
        loss_type = self.type_loss_fn(type_logits, type_labels)
        loss_risk = self.risk_loss_fn(risk_logits, risk_labels)
        return loss_type + loss_risk

def evaluate_model_detailed(model, dataloader, device):
    """
    دالة تقييم شاملة تطبق التوصيات المذكورة:
    1. تقرير مفصل لـ Clause Types.
    2. تقرير مفصل لـ Risk Levels (لكشف مشكلة Medium).
    """
    model.eval()
    all_type_preds = []
    all_type_labels = []
    all_risk_preds = []
    all_risk_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            type_labels = batch['type_labels'].to(device)
            risk_labels = batch['risk_labels'].to(device)

            type_logits, risk_logits = model(input_ids, attention_mask)

            type_preds = torch.argmax(type_logits, dim=1)
            risk_preds = torch.argmax(risk_logits, dim=1)

            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())
            all_risk_preds.extend(risk_preds.cpu().numpy())
            all_risk_labels.extend(risk_labels.cpu().numpy())

    print("\n" + "="*50)
    print("📊 تقرير تقييم أنواع البنود (Clause Types)")
    print("="*50)
    # استخدام أسماء الفئات المحدثة
    target_names_types = [t.replace('_', ' ').title() for t in CLAUSE_TYPES]
    print(classification_report(all_type_labels, all_type_preds, target_names=target_names_types, digits=4))

    print("\n" + "="*50)
    print("⚠️ تقرير تقييم مستويات الخطورة (Risk Levels) - كشف الخلل")
    print("="*50)
    # هنا نكشف الحقيقة حول فئة Medium
    print(classification_report(all_risk_labels, all_risk_preds, target_names=['Low', 'Medium', 'High'], digits=4))
    
    # تحليل خاص لفئة Termination
    term_true = 0
    term_pred = 0
    term_tp = 0
    term_idx = CLAUSE_TYPES.index('termination')
    
    for t, p in zip(all_type_labels, all_type_preds):
        if t == term_idx: term_true += 1
        if p == term_idx: term_pred += 1
        if t == term_idx and p == term_idx: term_tp += 1
        
    recall_term = term_tp / term_true if term_true > 0 else 0
    print(f"\n🔍 تحليل عميق لفئة Termination:")
    print(f"   العدد الحقيقي: {term_true} | المتوقع: {term_pred} | الصحيح: {term_tp}")
    print(f"   Recall المحسوب يدوياً: {recall_term:.4f}")
    if recall_term < 0.75:
        print("   ⚠️ تحذير: الـ Recall لا يزال منخفضاً، قد يحتاج النموذج لمزيد من Epochs أو Data Augmentation لهذه الفئة.")
    else:
        print("   ✅ ممتاز: تحسن ملحوظ في اكتشاف بنود الفسخ!")

# ملاحظة للاستخدام:
# لاستخدام هذا الكود في الـ Notebook:
# 1. قم بتعريف الكلاس ImprovedAraContractClassifier بدلاً من القديم.
# 2. تأكد أن DataLoader يعيد labels تتوافق مع الفئات الـ 8 الجديدة.
# 3. استدعِ evaluate_model_detailed بعد التدريب.
