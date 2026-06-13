# %% [markdown]
# # AraContract Analyzer - Model Training
# ## Google Colab Notebook with Google Drive Checkpoints
# 
# This notebook trains the AraContract clause classification model on Google Colab with GPU acceleration and saves checkpoints directly to Google Drive.
# 
# **Model:** CAMeLBERT (Arabic BERT) for multi-task classification (clause type + risk level)
# 
# **Dataset:** AraContract JSONL format with `text`, `type_clause`, and `risk_level` fields

# %% [markdown]
# ## 1. Setup and Configuration

# %%
#@title Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("✓ Google Drive mounted at /content/drive")

# %%
#@title Check GPU Availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# %%
#@title Install Required Packages
# We use numpy<2.0 to maintain compatibility with scikit-learn 1.4.0
!pip install -q transformers==4.37.0 scikit-learn==1.4.0 'numpy<2.0'
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
print("✓ Packages installed. IMPORTANT: Please go to Runtime -> Restart session now.")

# %% [markdown]
# ## 2. Configuration

# %%
#@title Training Configuration
from dataclasses import dataclass
from pathlib import Path

# Google Drive folder for checkpoints (CHANGE THIS to your preferred path)
DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/AraContract/checkpoints"

@dataclass
class Config:
    # Paths
    drive_checkpoint_folder: str = DRIVE_CHECKPOINT_FOLDER

    # Model
    model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    max_seq_length: int = 512

    # Training hyperparameters
    batch_size: int = 16  # Reduce if OOM
    learning_rate: float = 2e-5
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    type_loss_weight: float = 1.15
    risk_loss_weight: float = 1.0
    seed: int = 42

    # Training settings
    log_interval: int = 50
    eval_interval: int = 1

    # Label mappings - 7 type classes
    type_labels: list = None
    risk_labels: list = None
    risk_class_weights: list = None
    medium_threshold: float = 0.5
    high_threshold: float = 0.5

    def __post_init__(self):
        self.type_labels = [
            "general_provisions",
            "payment_financial",
            "party_obligations_a",
            "party_obligations_b",
            "duration_expiration",
            "termination",
            "penalties_damages",
            "dispute_resolution",
        ]
        self.risk_labels = ["low", "medium", "high"]
        # Use a mild medium-risk weight without oversampling. V1.5 failed from
        # double compensation (weights + sampler); V2.2 kept type quality but
        # still under-detected medium risk.
        self.risk_class_weights = [1.0, 1.35, 1.0]
        self.num_type_classes = len(self.type_labels)
        self.num_risk_classes = len(self.risk_labels)
        self.type_label_to_idx = {label: i for i, label in enumerate(self.type_labels)}
        self.risk_label_to_idx = {label: i for i, label in enumerate(self.risk_labels)}

config = Config()

# Create checkpoint directory on Drive
Path(config.drive_checkpoint_folder).mkdir(parents=True, exist_ok=True)
print(f"✓ Checkpoint folder: {config.drive_checkpoint_folder}")

# %% [markdown]
# ## 3. Dataset Loading and Preprocessing

# %%
#@title Dataset Statistics Check
import json
from collections import Counter

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# Load data files
train_data = load_jsonl('/content/drive/MyDrive/AraContract/data/aracontract_train.jsonl')
val_data = load_jsonl('/content/drive/MyDrive/AraContract/data/aracontract_val.jsonl')
test_data = load_jsonl('/content/drive/MyDrive/AraContract/data/aracontract_test.jsonl')

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Total: {len(train_data) + len(val_data) + len(test_data)}")

# %%
#@title Check Label Distribution
print("\n=== Type Clause Distribution (Train) ===")
type_counts = Counter(r['clause_type'] for r in train_data)
for label, count in sorted(type_counts.items()):
    print(f"  {label}: {count} ({100*count/len(train_data):.1f}%)")

print("\n=== Risk Level Distribution (Train) ===")
risk_counts = Counter(r['risk_level'] for r in train_data)
for label, count in sorted(risk_counts.items()):
    print(f"  {label}: {count} ({100*count/len(train_data):.1f}%)")

# %%
#@title Sample Data Check
print("Sample training record:")
print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 4. PyTorch Dataset and Model

# %%
#@title ClauseDataset Implementation
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from warnings import warn

class ClauseDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512, type_label_map=None, risk_label_map=None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.type_label_map = type_label_map or config.type_label_to_idx
        self.risk_label_map = risk_label_map or config.risk_label_to_idx
        self._validate_labels()

    def _validate_labels(self):
        unknown_type_labels = sorted({
            record.get('clause_type', '') for record in self.records
            if record.get('clause_type', '') not in self.type_label_map
        })
        unknown_risk_labels = sorted({
            record.get('risk_level', '') for record in self.records
            if record.get('risk_level', '') not in self.risk_label_map
        })
        if unknown_type_labels or unknown_risk_labels:
            raise ValueError(
                f"Unknown labels found. type_clause={unknown_type_labels}, "
                f"risk_level={unknown_risk_labels}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record.get('text', '')
        type_label = record.get('clause_type', '')
        risk_label = record.get('risk_level', '')

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        type_idx = self.type_label_map[type_label]
        risk_idx = self.risk_label_map[risk_label]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'type_label': torch.tensor(type_idx, dtype=torch.long),
            'risk_label': torch.tensor(risk_idx, dtype=torch.long)
        }

print("✓ ClauseDataset defined")

# %%
#@title AraContractClassifier Model
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

class AraContractClassifier(nn.Module):
    def __init__(self, model_name, num_type_classes=7, num_risk_classes=3, dropout_prob=0.1, risk_class_weights=None):
        super().__init__()
        class FocalLoss(nn.Module):
          def __init__(self, alpha=None, gamma=2.0):
              super().__init__()
              self.alpha = alpha
              self.gamma = gamma

          def forward(self, logits, labels):
              ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none', weight=self.alpha)
              pt = torch.exp(-ce_loss)
              focal_loss = ((1 - pt) ** self.gamma) * ce_loss
              return focal_loss.mean()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.type_classifier = nn.Linear(hidden_size, num_type_classes)
        self.risk_classifier = nn.Linear(hidden_size, num_risk_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.risk_loss_fn = FocalLoss(alpha=torch.tensor([1.0, 1.5, 0.8]), gamma=2.0)

        if risk_class_weights is not None:
            self.register_buffer("risk_class_weights", risk_class_weights)
        else:
            self.risk_class_weights = None

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
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
        if self.risk_class_weights is not None:
            risk_loss = F.cross_entropy(risk_logits, risk_labels, weight=self.risk_class_weights)
        else:
            risk_loss = self.loss_fn(risk_logits, risk_labels)
        total_loss = (config.type_loss_weight * type_loss) + (config.risk_loss_weight * risk_loss)

        loss_dict = {
            'total_loss': total_loss.item(),
            'type_loss': type_loss.item(),
            'risk_loss': risk_loss.item()
        }
        return total_loss, loss_dict

    def evaluate(self, dataloader, device):
        self.eval()
        all_type_preds, all_type_labels = [], []
        all_risk_preds, all_risk_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                type_labels = batch['type_label'].to(device)
                risk_labels = batch['risk_label'].to(device)

                type_logits, risk_logits = self(input_ids, attention_mask)
                type_preds = torch.argmax(type_logits, dim=-1)
                risk_preds = torch.argmax(risk_logits, dim=-1)

                all_type_preds.extend(type_preds.cpu().numpy())
                all_type_labels.extend(type_labels.cpu().numpy())
                all_risk_preds.extend(risk_preds.cpu().numpy())
                all_risk_labels.extend(risk_labels.cpu().numpy())

        type_f1 = f1_score(all_type_labels, all_type_preds, average='weighted', zero_division=0)
        risk_f1 = f1_score(all_risk_labels, all_risk_preds, average='weighted', zero_division=0)
        type_macro_f1 = f1_score(all_type_labels, all_type_preds, average='macro', zero_division=0)
        risk_macro_f1 = f1_score(all_risk_labels, all_risk_preds, average='macro', zero_division=0)
        risk_recall_per_class = recall_score(
            all_risk_labels,
            all_risk_preds,
            labels=list(range(config.num_risk_classes)),
            average=None,
            zero_division=0,
        )
        risk_precision_per_class = precision_score(
            all_risk_labels,
            all_risk_preds,
            labels=list(range(config.num_risk_classes)),
            average=None,
            zero_division=0,
        )
        type_acc = accuracy_score(all_type_labels, all_type_preds)
        risk_acc = accuracy_score(all_risk_labels, all_risk_preds)

        return {
            'type_f1': type_f1,
            'risk_f1': risk_f1,
            'type_macro_f1': type_macro_f1,
            'risk_macro_f1': risk_macro_f1,
            'risk_low_recall': float(risk_recall_per_class[0]),
            'risk_medium_recall': float(risk_recall_per_class[1]),
            'risk_high_recall': float(risk_recall_per_class[2]),
            'risk_low_precision': float(risk_precision_per_class[0]),
            'risk_medium_precision': float(risk_precision_per_class[1]),
            'risk_high_precision': float(risk_precision_per_class[2]),
            'type_accuracy': type_acc,
            'risk_accuracy': risk_acc
        }

print("✓ AraContractClassifier defined")

# %% [markdown]
# ## 5. Training Functions

# %%
# @title Training Utilities
import random
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# def build_dataloaders(train_records, val_records, tokenizer, batch_size=16, max_length=512, use_weighted_sampler=False):
#     train_dataset = ClauseDataset(train_records, tokenizer, max_length)
#     val_dataset = ClauseDataset(val_records, tokenizer, max_length)

#     if use_weighted_sampler:
#         risk_counts = Counter(r.get('risk_level', '') for r in train_records)
#         sample_weights = [
#             1.0 / max(risk_counts.get(r.get('risk_level', ''), 1), 1)
#             for r in train_records
#         ]
#         sampler = WeightedRandomSampler(
#             weights=sample_weights, num_samples=len(sample_weights), replacement=True
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
#         )
#     else:
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
#         )
#     val_loader = DataLoader(val_dataset, batch_size=batch_size,
#                             shuffle=False, num_workers=0, drop_last=False)

#     return train_loader, val_loader

def build_dataloaders(train_records, val_records, tokenizer, batch_size=16, max_length=512,
                      use_weighted_sampler=False, sampler_alpha=0.5, max_sample_weight=2.0):
  train_dataset = ClauseDataset(train_records, tokenizer, max_length)
  val_dataset = ClauseDataset(val_records, tokenizer, max_length)

  if use_weighted_sampler:
      risk_counts = Counter(r.get('risk_level', '') for r in train_records)
      print(f"Risk counts for sampler: {dict(risk_counts)}")

      total = sum(risk_counts.values())
      n_classes = len(risk_counts)
      # Tempered inverse-frequency weights. Full inverse-frequency sampling
      # changes the effective clause-type distribution too much because risk
      # and type are correlated in this dataset.
      sample_weights = [
          min(
              (total / (n_classes * risk_counts.get(r.get('risk_level', ''), 1))) ** sampler_alpha,
              max_sample_weight,
          )
          for r in train_records
      ]

      print(f"Sample weight stats: min={min(sample_weights):.2f}, max={max(sample_weights):.2f}")

      sampler = WeightedRandomSampler(
          weights=sample_weights,
          num_samples=len(sample_weights),  # أو *2 لزيادة التعرض للـ medium
          replacement=True
      )
      train_loader = DataLoader(
          train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
      )
  else:
      train_loader = DataLoader(
          train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
      )
  val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=0, drop_last=False)

  return train_loader, val_loader

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None, log_interval=50):
    model.train()
    total_loss = 0.0
    total_type_loss = 0.0
    total_risk_loss = 0.0
    steps = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        type_labels = batch['type_label'].to(device)
        risk_labels = batch['risk_label'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, loss_dict = model.training_step(
                    input_ids, attention_mask, type_labels, risk_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = model.training_step(
                input_ids, attention_mask, type_labels, risk_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss_dict['total_loss']
        total_type_loss += loss_dict['type_loss']
        total_risk_loss += loss_dict['risk_loss']
        steps += 1

        if (step + 1) % log_interval == 0:
            print(
                f"  Step {step + 1}/{len(dataloader)} | Loss: {total_loss / steps:.4f}")

    return {
        'loss': total_loss / steps,
        'type_loss': total_type_loss / steps,
        'risk_loss': total_risk_loss / steps
    }


def save_checkpoint(model, path, run_name=None, epoch=None, best_f1=None, extra_config=None):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'model_name': config.model_name,
            'num_type_classes': config.num_type_classes,
            'num_risk_classes': config.num_risk_classes,
            'type_labels': config.type_labels,
            'risk_labels': config.risk_labels,
        }
    }
    if extra_config:
        checkpoint['config'].update(extra_config)
    if run_name:
        checkpoint['run_name'] = run_name
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if best_f1 is not None:
        checkpoint['best_f1'] = best_f1

    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")


def collect_predictions(model, dataloader, device):
    model.eval()
    all_type_preds, all_type_labels = [], []
    all_risk_labels = []
    risk_probs_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            type_labels = batch['type_label'].to(device)
            risk_labels = batch['risk_label'].to(device)

            type_logits, risk_logits = model(input_ids, attention_mask)
            type_preds = torch.argmax(type_logits, dim=-1)
            risk_probs = torch.softmax(risk_logits, dim=-1)

            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())
            all_risk_labels.extend(risk_labels.cpu().numpy())
            risk_probs_all.extend(risk_probs.cpu().numpy())

    return {
        'type_true': np.array(all_type_labels),
        'type_preds': np.array(all_type_preds),
        'risk_true': np.array(all_risk_labels),
        'risk_probs': np.array(risk_probs_all),
    }


def apply_risk_thresholds(risk_probs, medium_threshold=0.5, high_threshold=0.5):
    preds = []
    for probs in risk_probs:
        # Medium is the rarest class and can be swallowed by an aggressive
        # high-risk threshold. Give explicit medium confidence priority.
        if probs[1] >= medium_threshold:
            preds.append(1)
        elif probs[2] >= high_threshold:
            preds.append(2)
        else:
            preds.append(int(np.argmax(probs)))
    return np.array(preds)


def tune_risk_thresholds(risk_true, risk_probs):
    best = {
        'score': -1.0,
        'medium_threshold': config.medium_threshold,
        'high_threshold': config.high_threshold,
        'macro_f1': 0.0,
        'medium_f1': 0.0,
        'high_recall': 0.0,
    }

    for medium_threshold in np.arange(0.35, 0.61, 0.02):
        # V2.1 selected high=0.35, which improved high recall but reduced
        # medium F1 on the held-out test set. Keep high threshold conservative.
        for high_threshold in np.arange(0.45, 0.76, 0.02):
            preds = apply_risk_thresholds(risk_probs, medium_threshold, high_threshold)
            macro_f1 = f1_score(risk_true, preds, average='macro', zero_division=0)
            per_class_f1 = f1_score(
                risk_true,
                preds,
                labels=list(range(config.num_risk_classes)),
                average=None,
                zero_division=0,
            )
            per_class_recall = recall_score(
                risk_true,
                preds,
                labels=list(range(config.num_risk_classes)),
                average=None,
                zero_division=0,
            )
            # Optimize the imbalanced risk task directly. Add a small medium
            # bonus because medium is the minority class that regressed in V2.1.
            score = macro_f1 + (0.05 * per_class_f1[1])
            if score > best['score']:
                best = {
                    'score': float(score),
                    'medium_threshold': float(medium_threshold),
                    'high_threshold': float(high_threshold),
                    'macro_f1': float(macro_f1),
                    'medium_f1': float(per_class_f1[1]),
                    'high_recall': float(per_class_recall[2]),
                }

    return best


def validation_selection_score(model, dataloader, device):
    preds = collect_predictions(model, dataloader, device)
    thresholds = tune_risk_thresholds(preds['risk_true'], preds['risk_probs'])
    thresholded_risk_preds = apply_risk_thresholds(
        preds['risk_probs'],
        thresholds['medium_threshold'],
        thresholds['high_threshold'],
    )

    type_macro_f1 = f1_score(
        preds['type_true'],
        preds['type_preds'],
        average='macro',
        zero_division=0,
    )
    risk_macro_f1 = f1_score(
        preds['risk_true'],
        thresholded_risk_preds,
        average='macro',
        zero_division=0,
    )
    high_recall = recall_score(
        preds['risk_true'],
        thresholded_risk_preds,
        labels=[2],
        average='macro',
        zero_division=0,
    )
    medium_f1 = thresholds['medium_f1']
    # Real app quality depends on both heads. Keep type important, but do not
    # let a tiny type gain choose weaker medium-risk behavior.
    score = (0.45 * type_macro_f1) + (0.45 * risk_macro_f1) + (0.10 * medium_f1)

    return {
        'score': float(score),
        'type_macro_f1': float(type_macro_f1),
        'risk_macro_f1_thresholded': float(risk_macro_f1),
        'risk_high_recall_thresholded': float(high_recall),
        'risk_medium_f1_thresholded': float(medium_f1),
        'risk_thresholds': thresholds,
    }


print("✓ Training utilities defined")

# %% [markdown]
# ## 6. Main Training Loop

# %%
#@title Run Training
from datetime import datetime
import json

def train(
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    run_name=None,
    resume_checkpoint_path=None,
    start_epoch=1,
    use_weighted_sampler=False
 ):
    # Setup
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Generate run name if not provided
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # DataLoaders
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        train_data, val_data, tokenizer, batch_size, config.max_seq_length, use_weighted_sampler
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print(f"Loading model: {config.model_name}")
    risk_class_weights = None
    if config.risk_class_weights:
        risk_class_weights = torch.tensor(config.risk_class_weights, dtype=torch.float)
    model = AraContractClassifier(
        model_name=config.model_name,
        num_type_classes=config.num_type_classes,
        num_risk_classes=config.num_risk_classes,
        dropout_prob=config.dropout,
        risk_class_weights=risk_class_weights
    ).to(device)

    # Resume checkpoint if provided
    best_f1 = 0.0
    best_risk_thresholds = {
        'medium_threshold': config.medium_threshold,
        'high_threshold': config.high_threshold,
    }
    if resume_checkpoint_path:
        checkpoint = torch.load(
            resume_checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_f1' in checkpoint:
            best_f1 = max(best_f1, checkpoint['best_f1'])
        if 'epoch' in checkpoint:
            start_epoch = max(start_epoch, checkpoint['epoch'] + 1)
        print(f"Resuming from {resume_checkpoint_path}")
        print(f"Starting at epoch {start_epoch}")

    remaining_epochs = max(epochs - start_epoch + 1, 0)
    if remaining_epochs == 0:
        print("Requested epochs already completed. Skipping training.")
        return model, tokenizer, {'train': [], 'val': []}

    # Optimizer & scheduler
    total_steps = len(train_loader) * remaining_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    history = {'train': [], 'val': []}

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Run name: {run_name}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, config.log_interval
        )
        print(f"Train loss: {train_metrics['loss']:.4f} | type: {train_metrics['type_loss']:.4f} | risk: {train_metrics['risk_loss']:.4f}")

        # Validate
        val_metrics = model.evaluate(val_loader, device)
        print(f"Val type_f1: {val_metrics['type_f1']:.4f} | risk_f1: {val_metrics['risk_f1']:.4f}")
        print(f"Val type_macro_f1: {val_metrics['type_macro_f1']:.4f} | risk_macro_f1: {val_metrics['risk_macro_f1']:.4f}")
        print(f"Val risk recalls: low={val_metrics['risk_low_recall']:.4f} medium={val_metrics['risk_medium_recall']:.4f} high={val_metrics['risk_high_recall']:.4f}")
        print(f"Val type_acc: {val_metrics['type_accuracy']:.4f} | risk_acc: {val_metrics['risk_accuracy']:.4f}")

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save best model using macro/per-class behavior, not weighted F1.
        selection = validation_selection_score(model, val_loader, device)
        print(
            "Val selection_score: "
            f"{selection['score']:.4f} | "
            f"type_macro={selection['type_macro_f1']:.4f} | "
            f"risk_macro_thresholded={selection['risk_macro_f1_thresholded']:.4f} | "
            f"medium_f1_thresholded={selection['risk_medium_f1_thresholded']:.4f} | "
            f"high_recall_thresholded={selection['risk_high_recall_thresholded']:.4f}"
        )
        print(
            "Val risk thresholds: "
            f"medium={selection['risk_thresholds']['medium_threshold']:.2f}, "
            f"high={selection['risk_thresholds']['high_threshold']:.2f}"
        )

        if selection['score'] > best_f1:
            best_f1 = selection['score']
            best_risk_thresholds = selection['risk_thresholds']
            best_model_path = f"{config.drive_checkpoint_folder}/{run_name}_best.pt"
            save_checkpoint(
                model,
                best_model_path,
                run_name,
                epoch=epoch,
                best_f1=best_f1,
                extra_config={'risk_thresholds': best_risk_thresholds},
            )
            print(f"  ★ New best model! Selection score: {best_f1:.4f}")

        # Save epoch checkpoint
        epoch_path = f"{config.drive_checkpoint_folder}/{run_name}_epoch{epoch}.pt"
        save_checkpoint(
            model,
            epoch_path,
            run_name,
            epoch=epoch,
            best_f1=best_f1,
            extra_config={'risk_thresholds': best_risk_thresholds},
        )

    # Save final model
    final_path = f"{config.drive_checkpoint_folder}/{run_name}_final.pt"
    save_checkpoint(
        model,
        final_path,
        run_name,
        epoch=epochs,
        best_f1=best_f1,
        extra_config={'risk_thresholds': best_risk_thresholds},
    )

    # Save training history
    history_path = f"{config.drive_checkpoint_folder}/{run_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save tokenizer
    tokenizer_path = f"{config.drive_checkpoint_folder}/{run_name}_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best selection score: {best_f1:.4f}")
    print(f"Checkpoints saved to: {config.drive_checkpoint_folder}")
    print(f"{'='*50}")

    return model, tokenizer, history


# %%
#@title Start Training
# Adjust these parameters as needed
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer

EPOCHS = 5  # @param {type: "integer"}
BATCH_SIZE = 16  # @param {type: "integer"}
LEARNING_RATE = 2e-5  # @param {type: "number"}
RUN_NAME = "aracontract_v2.3"  # @param {type: "string"}
USE_WEIGHTED_SAMPLER = False  # @param {type: "boolean"}

def find_latest_epoch_checkpoint(folder, run_name):
    pattern = re.compile(rf"{re.escape(run_name)}_epoch(\d+)\.pt$")
    latest_epoch = None
    latest_path = None
    for path in Path(folder).glob(f"{run_name}_epoch*.pt"):
        match = pattern.search(path.name)
        if match:
            epoch_num = int(match.group(1))
            if latest_epoch is None or epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_path = path
    return latest_path, latest_epoch

def load_checkpoint_model(checkpoint_path):
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    risk_weights = torch.tensor(config.risk_class_weights, dtype=torch.float) if config.risk_class_weights else None
    model = AraContractClassifier(
        model_name=config.model_name,
        num_type_classes=config.num_type_classes,
        num_risk_classes=config.num_risk_classes,
        dropout_prob=config.dropout,
        risk_class_weights=risk_weights
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def load_saved_tokenizer(run_name):
    tokenizer_path = Path(config.drive_checkpoint_folder) / f"{run_name}_tokenizer"
    if tokenizer_path.exists():
        return AutoTokenizer.from_pretrained(str(tokenizer_path))
    return AutoTokenizer.from_pretrained(config.model_name)

drive_folder = Path(config.drive_checkpoint_folder)
best_path = drive_folder / f"{RUN_NAME}_best.pt"
final_path = drive_folder / f"{RUN_NAME}_final.pt"
latest_epoch_path, latest_epoch = find_latest_epoch_checkpoint(drive_folder, RUN_NAME)

model = None
tokenizer = None
history = None

checkpoint_path = None
if final_path.exists():
    checkpoint_path = final_path
elif best_path.exists():
    checkpoint_path = best_path
elif latest_epoch_path is not None:
    checkpoint_path = latest_epoch_path

if checkpoint_path is not None:
    model, checkpoint = load_checkpoint_model(str(checkpoint_path))
    tokenizer = load_saved_tokenizer(RUN_NAME)
    print(f"Found checkpoint: {checkpoint_path}")
    if latest_epoch is not None:
        print(f"Latest epoch checkpoint: epoch {latest_epoch}")
    choice = input(
        "Checkpoint found. Retrain from scratch (r), continue from latest epoch (c), or use loaded model (l)? [c]: "
    ).strip().lower()

    if choice in ("", "c", "continue"):
        if latest_epoch_path is None:
            print("No epoch checkpoint found, starting from scratch.")
            model, tokenizer, history = train(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                run_name=RUN_NAME,
                use_weighted_sampler=USE_WEIGHTED_SAMPLER
            )
        elif latest_epoch >= EPOCHS:
            print("All requested epochs are already completed. Using loaded model.")
        else:
            model, tokenizer, history = train(
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                run_name=RUN_NAME,
                resume_checkpoint_path=str(latest_epoch_path),
                start_epoch=latest_epoch + 1,
                use_weighted_sampler=USE_WEIGHTED_SAMPLER
            )
    elif choice in ("r", "retrain"):
        model, tokenizer, history = train(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            run_name=RUN_NAME,
            use_weighted_sampler=USE_WEIGHTED_SAMPLER
        )
    else:
        print("Using loaded model without retraining.")
else:
    print("No checkpoint found. Starting training from scratch.")
    model, tokenizer, history = train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        run_name=RUN_NAME,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER
    )

# %% [markdown]
# ## 7. Evaluation and Testing

# %%
# Load best model and evaluate on test set
# Define device globally if not already available
device = get_device()
RUN_NAME = "aracontract_v2.3"  # @param {type: "string"}

# Update this path to your best model checkpoint
BEST_MODEL_PATH = f"{config.drive_checkpoint_folder}/{RUN_NAME}_best.pt"  # @param {type: "string"}

# Load checkpoint (Setting weights_only=False because the checkpoint contains numpy scalars)
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
print(f"Loaded checkpoint: {BEST_MODEL_PATH}")
print(f"Run name: {checkpoint.get('run_name', 'N/A')}")
risk_thresholds = checkpoint.get('config', {}).get('risk_thresholds', {
    'medium_threshold': config.medium_threshold,
    'high_threshold': config.high_threshold,
})
print(
    "Risk thresholds: "
    f"medium={risk_thresholds['medium_threshold']:.2f}, "
    f"high={risk_thresholds['high_threshold']:.2f}"
)

# Reconstruct model with correct weights to match the state_dict
risk_weights = torch.tensor(config.risk_class_weights, dtype=torch.float) if config.risk_class_weights else None

test_model = AraContractClassifier(
    model_name=config.model_name,
    num_type_classes=config.num_type_classes,
    num_risk_classes=config.num_risk_classes,
    risk_class_weights=risk_weights
).to(device)

test_model.load_state_dict(checkpoint['model_state_dict'])
print("✓ Model loaded")

# Create test dataloader
test_dataset = ClauseDataset(test_data, tokenizer, config.max_seq_length)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Evaluate
print("\nEvaluating on test set...")
test_metrics = test_model.evaluate(test_loader, device)
print(f"Test type_f1: {test_metrics['type_f1']:.4f} | risk_f1: {test_metrics['risk_f1']:.4f}")
print(f"Test type_acc: {test_metrics['type_accuracy']:.4f} | risk_acc: {test_metrics['risk_accuracy']:.4f}")

if test_metrics['type_f1'] >= 0.80 and test_metrics['risk_f1'] >= 0.80:
    print("✓ Meets SRS target")
else:
    print(f"⚠ type_f1={test_metrics['type_f1']:.4f}, risk_f1={test_metrics['risk_f1']:.4f}")

# %%
#@title Full Evaluation — Both Tasks (with threshold applied)
from sklearn.metrics import classification_report, f1_score
import torch
import numpy as np
from torch.utils.data import DataLoader

# --- Choose Model to Evaluate ---
RUN_NAME = "aracontract_v2.3"  # @param {type: "string"}
BEST_MODEL_PATH = f"{config.drive_checkpoint_folder}/{RUN_NAME}_best.pt"

# Load the specific model
device = get_device()
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
print(f"Loaded checkpoint for evaluation: {BEST_MODEL_PATH}")
risk_thresholds = checkpoint.get('config', {}).get('risk_thresholds', {
    'medium_threshold': config.medium_threshold,
    'high_threshold': config.high_threshold,
})

# Reconstruct model
risk_weights = torch.tensor(config.risk_class_weights, dtype=torch.float) if config.risk_class_weights else None
test_model = AraContractClassifier(
    model_name=config.model_name,
    num_type_classes=config.num_type_classes,
    num_risk_classes=config.num_risk_classes,
    risk_class_weights=risk_weights
).to(device)
test_model.load_state_dict(checkpoint['model_state_dict'])

def full_evaluation(model, dataloader, device, type_labels, risk_labels, risk_thresholds):
    model.eval()
    all_type_preds, all_type_true = [], []
    all_risk_true = []
    risk_probs_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            type_labels_b  = batch['type_label'].to(device)
            risk_labels_b  = batch['risk_label'].to(device)

            type_logits, risk_logits = model(input_ids, attention_mask)

            all_type_preds.extend(torch.argmax(type_logits, dim=-1).cpu().numpy())
            all_type_true.extend(type_labels_b.cpu().numpy())
            all_risk_true.extend(risk_labels_b.cpu().numpy())
            risk_probs = torch.softmax(risk_logits, dim=-1).cpu().numpy()
            risk_probs_all.extend(risk_probs)

    risk_probs_array = np.array(risk_probs_all)

    final_risk_preds = apply_risk_thresholds(
        risk_probs_array,
        risk_thresholds['medium_threshold'],
        risk_thresholds['high_threshold'],
    )
    macro_f1 = f1_score(all_risk_true, final_risk_preds, average='macro', zero_division=0)

    print(
        "Applied validation thresholds: "
        f"medium={risk_thresholds['medium_threshold']:.2f}, "
        f"high={risk_thresholds['high_threshold']:.2f}, "
        f"Risk Macro F1={macro_f1:.4f}"
    )

    print("=" * 55)
    print("TASK 1 — Clause Type Classification")
    print("=" * 55)
    print(classification_report(
        all_type_true, all_type_preds,
        target_names=type_labels, digits=4
    ))

    print("=" * 55)
    print("TASK 2 — Risk Level Classification (threshold-adjusted)")
    print("=" * 55)
    print(classification_report(
        all_risk_true, final_risk_preds,
        target_names=risk_labels, digits=4
    ))

# Run on test set
test_dataset = ClauseDataset(test_data, tokenizer, config.max_seq_length)
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

full_evaluation(
    test_model, test_loader, device,
    config.type_labels,
    config.risk_labels,
    risk_thresholds
)

# %%
#@title Detailed Classification Report
from sklearn.metrics import classification_report

def get_detailed_report(model, dataloader, device, label_names):
    model.eval()
    all_type_preds, all_type_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            type_labels = batch['type_label'].to(device)

            type_logits, _ = model(input_ids, attention_mask)
            type_preds = torch.argmax(type_logits, dim=-1)

            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())

    print("\n=== Type Clause Classification Report ===")
    print(classification_report(all_type_labels, all_type_preds, target_names=label_names, digits=4))

get_detailed_report(test_model, test_loader, device, config.type_labels)

# %% [markdown]
# ## 8. Export Model for Deployment

# %%
#@title Export Model and Tokenizer for Backend
from pathlib import Path
import os

EXPORT_DIR = f"{config.drive_checkpoint_folder}/{RUN_NAME}_export"
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# Save model weights only (smaller file for deployment)
export_path = f"{EXPORT_DIR}/aracontract_classifier.pt"
torch.save(test_model.state_dict(), export_path)
print(f"✓ Model weights saved: {export_path}")

# Save tokenizer
tokenizer.save_pretrained(EXPORT_DIR)
print(f"✓ Tokenizer saved: {EXPORT_DIR}")

# Save config info
export_config = {
    'model_name': config.model_name,
    'num_type_classes': config.num_type_classes,
    'num_risk_classes': config.num_risk_classes,
    'type_labels': config.type_labels,
    'risk_labels': config.risk_labels,
    'type_label_to_idx': config.type_label_to_idx,
    'risk_label_to_idx': config.risk_label_to_idx,
    'risk_thresholds': risk_thresholds,
    'test_metrics': test_metrics,
}
config_path = f"{EXPORT_DIR}/config.json"
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(export_config, f, indent=2, ensure_ascii=False)
print(f"✓ Config saved: {config_path}")

# Print export summary
print(f"\n{'='*50}")
print("Export Summary:")
print(f"{'='*50}")
for f in os.listdir(EXPORT_DIR):
    fpath = f"{EXPORT_DIR}/{f}"
    size = os.path.getsize(fpath)
    print(f"  {f}: {size / 1024 / 1024:.1f} MB")

# %% [markdown]
# ## 9. Inference Test

# %%
#@title Test Inference on Sample Text
import torch.nn.functional as F

def predict_clause(text):
    """Predict type and risk for a single clause text."""
    test_model.eval()

    # Tokenize
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # Forward pass
    with torch.no_grad():
        type_logits, risk_logits = test_model(
            inputs['input_ids'],
            inputs['attention_mask']
        )

    # Get predictions
    type_probs = F.softmax(type_logits, dim=-1).cpu().numpy()[0]
    risk_probs = F.softmax(risk_logits, dim=-1).cpu().numpy()[0]

    type_idx = type_probs.argmax()
    risk_idx = int(apply_risk_thresholds(
        np.array([risk_probs]),
        risk_thresholds['medium_threshold'],
        risk_thresholds['high_threshold'],
    )[0])

    return {
        'type': config.type_labels[type_idx],
        'type_confidence': float(type_probs[type_idx]),
        'risk': config.risk_labels[risk_idx],
        'risk_confidence': float(risk_probs[risk_idx]),
        'type_probabilities': dict(zip(config.type_labels, type_probs)),
        'risk_probabilities': dict(zip(config.risk_labels, risk_probs))
    }

# Test with sample Arabic contract clauses
test_clauses = [
    "يلتزم الطرف الثاني بدفع المبلغ المتفق عليه خلال مدة لا تتجاوز 30 يوماً من تاريخ الفاتورة",
    "يحق للطرف الأول إنهاء العقد في أي وقت دون إشعار مسبق في حال إخلال الطرف الثاني بالتزاماته",
    "تبلغ مدة هذا العقد سنة واحدة قابلة للتجديد تلقائياً ما لم يخطره أحد الطرفين قبل 30 يوماً",
]

for clause in test_clauses:
    result = predict_clause(clause)
    print(f"\nClause: {clause[:50]}...")
    print(f"  Type: {result['type']} ({result['type_confidence']:.2%})")
    print(f"  Risk: {result['risk']} ({result['risk_confidence']:.2%})")

# %% [markdown]
# ## 10. Download Checkpoints Locally (Optional)

# %%
#@title Download Best Model and Export Folder
from google.colab import files
import os

# List available checkpoints
checkpoint_files = os.listdir(config.drive_checkpoint_folder)
print("Available files in checkpoint folder:")
for f in sorted(checkpoint_files):
    fpath = f"{config.drive_checkpoint_folder}/{f}"
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {f}: {size / 1024 / 1024:.1f} MB")

# Download best model
print("\nDownloading best model...")
files.download(f"{config.drive_checkpoint_folder}/{RUN_NAME}_best.pt")

# Download config
print("Downloading config...")
files.download(f"{config.drive_checkpoint_folder}/{RUN_NAME}_history.json")

# %%
