"""
Training script for the AraContract clause classification model.

Loads data, builds DataLoaders, instantiates the model, runs the training
loop, validates each epoch with weighted F1, and saves the best checkpoint.
Does NOT execute training automatically.
"""

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    BEST_MODEL_NAME,
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    EVAL_INTERVAL,
    LOG_INTERVAL,
    MAX_SEQ_LENGTH,
    TRANSFORMER_MODEL,
    VAL_FILE,
    TRAIN_FILE,
)
from dataset import ClauseDataset
from model import AraContractClassifier


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Auto-select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    tokenizer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    train_dataset = ClauseDataset(
        TRAIN_FILE, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH
    )
    val_dataset = ClauseDataset(
        VAL_FILE, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
    """Run one training epoch and return average losses."""
    model.train()
    total_loss = 0.0
    total_type_loss = 0.0
    total_risk_loss = 0.0
    steps = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        type_labels = batch["type_label"].to(device)
        risk_labels = batch["risk_label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, loss_dict = model.training_step(
                    input_ids, attention_mask, type_labels, risk_labels
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = model.training_step(
                input_ids, attention_mask, type_labels, risk_labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss_dict["total_loss"]
        total_type_loss += loss_dict["type_loss"]
        total_risk_loss += loss_dict["risk_loss"]
        steps += 1

        if (step + 1) % LOG_INTERVAL == 0:
            print(
                f"  Step {step + 1}/{len(dataloader)} | "
                f"Loss: {total_loss / steps:.4f}"
            )

    return {
        "loss": total_loss / steps,
        "type_loss": total_type_loss / steps,
        "risk_loss": total_risk_loss / steps,
    }


def validate(model, dataloader, device) -> Dict[str, float]:
    """Evaluate the model on the validation set and return metrics."""
    metrics = model.evaluate(dataloader, device)
    return metrics


def save_checkpoint(model, path: Path) -> None:
    """Save a model checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Entry point for training the AraContract classifier.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
        learning_rate: Peak learning rate for the optimizer.
        weight_decay: Weight decay applied to parameters.
        warmup_ratio: Fraction of training steps used for linear warmup.
        seed: Random seed for reproducibility.
        checkpoint_path: Optional path to resume from a saved checkpoint.
    """
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    train_loader, val_loader = build_dataloaders(tokenizer, batch_size)

    # Model
    model = AraContractClassifier().to(device)

    # Resume from checkpoint
    if checkpoint_path is not None:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Optimizer & scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_f1 = 0.0
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )
        print(
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"type_loss: {train_metrics['type_loss']:.4f} | "
            f"risk_loss: {train_metrics['risk_loss']:.4f}"
        )

        # Validation
        if epoch % EVAL_INTERVAL == 0:
            val_metrics = validate(model, val_loader, device)
            print(
                f"  Val type_f1: {val_metrics['type_f1']:.4f} | "
                f"risk_f1: {val_metrics['risk_f1']:.4f}"
            )

            avg_f1 = (val_metrics["type_f1"] + val_metrics["risk_f1"]) / 2
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                save_path = CHECKPOINT_DIR / BEST_MODEL_NAME
                save_checkpoint(model, save_path)
                print(f"  Saved new best model to {save_path}")

    print(f"\nTraining complete. Best weighted F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
