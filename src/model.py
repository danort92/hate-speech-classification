# src/model.py
# Training loop, evaluation, and model utilities

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


def build_model(model_name, num_labels, device):
    """Load a HuggingFace sequence classification model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model.to(device)


def get_class_weights(labels_array, device):
    """Compute balanced class weights from label array."""
    classes = np.unique(labels_array)
    weights = compute_class_weight("balanced", classes=classes, y=labels_array)
    return torch.tensor(weights, dtype=torch.float).to(device)


def evaluate(model, loader, device, criterion=None):
    """
    Run inference on a DataLoader.
    Returns: macro F1, avg loss, predictions list, labels list
    """
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if criterion is not None:
                total_loss += criterion(outputs.logits, labels).item()
            else:
                # fallback: use model's built-in loss
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += out.loss.item()

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1       = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / len(loader)
    return f1, avg_loss, all_preds, all_labels


def train(model, train_loader, val_loader, device,
          epochs=3, lr=2e-5, criterion=None):
    """
    Fine-tune model with AdamW + linear scheduler + warmup.

    Args:
        model        : HuggingFace model
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        device       : torch device
        epochs       : number of training epochs
        lr           : learning rate
        criterion    : optional weighted loss (nn.CrossEntropyLoss)

    Returns:
        best_model_state : state_dict of best checkpoint (by val F1)
        history          : dict with train_loss, val_loss, val_f1 per epoch
    """
    n_steps  = len(train_loader) * epochs
    n_warmup = int(0.1 * n_steps)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                               num_warmup_steps=n_warmup,
                               num_training_steps=n_steps)

    history          = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1      = 0.0
    best_model_state = None

    print(f"Training for {epochs} epochs | {n_steps} total steps\n")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()

            if criterion is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss    = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss             = total_train_loss / len(train_loader)
        val_f1, avg_val_loss, _, _ = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)

        print(f"\n── Epoch {epoch+1}/{epochs} ──")
        print(f"   Train loss : {avg_train_loss:.4f}")
        print(f"   Val loss   : {avg_val_loss:.4f}")
        print(f"   Val F1     : {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"   ✓ New best model (F1={best_val_f1:.4f})")
        print()

    print(f"Training complete. Best Val F1: {best_val_f1:.4f}")
    return best_model_state, history
