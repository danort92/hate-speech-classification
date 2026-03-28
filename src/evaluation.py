# src/evaluation.py
# Metrics, visualizations, and robustness evaluation

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)

LABEL_NAMES = {0: "hate", 1: "offensive", 2: "normal"}
COLORS      = {"hate": "#e63946", "offensive": "#f4a261", "normal": "#2a9d8f"}


# ── Inference ─────────────────────────────────────────────

class _SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


def predict(model, tokenizer, texts, labels, device, batch_size=32):
    """Run inference and return predictions and softmax probabilities."""
    dataset = _SimpleDataset(texts, labels, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out   = model(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=-1)
            all_preds.extend(probs.argmax(dim=-1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


# ── Plots ─────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["hate", "offensive", "normal"],
                yticklabels=["hate", "offensive", "normal"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {save_path}")


def plot_training_curves(history, save_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train", color="#457b9d")
    axes[0].plot(epochs, history["val_loss"],   marker="o", label="Val",   color="#e63946")
    axes[0].set_title("Loss per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(epochs, history["val_f1"], marker="o", color="#2a9d8f")
    axes[1].set_title("Validation Macro F1", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {save_path}")


def plot_robustness(results, save_path="robustness.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_colors = ["#2a9d8f"] + ["#e63946"] * (len(results) - 1)
    bars = ax.bar(results.keys(), results.values(),
                  color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.axhline(results["clean"], color="gray", linestyle="--", linewidth=1, label="Baseline")
    ax.set_ylim(0, 1); ax.set_ylabel("Macro F1")
    ax.set_title("Robustness to Obfuscation", fontweight="bold")
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {save_path}")


def plot_robustness_comparison(baseline, improved, save_path="robustness_comparison.png"):
    x      = np.arange(len(baseline))
    width  = 0.35
    keys   = list(baseline.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, [baseline[k] for k in keys], width,
                label="Baseline", color="#457b9d", alpha=0.85)
    b2 = ax.bar(x + width/2, [improved[k]  for k in keys], width,
                label="Improved",  color="#2a9d8f", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylim(0, 1); ax.set_ylabel("Macro F1")
    ax.set_title("Baseline vs Improved — Robustness", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {save_path}")


# ── Robustness evaluation ─────────────────────────────────

def robustness_eval(model, tokenizer, texts, labels, obfuscation_fns, device):
    """
    Evaluate model on clean and obfuscated test sets.

    Args:
        obfuscation_fns : dict {name: fn} from src.obfuscation

    Returns:
        results      : dict {condition: macro_f1}
        all_preds    : dict {condition: preds array}
        obf_texts    : dict {condition: obfuscated texts}
    """
    random.seed(42)
    results, all_preds_dict, obf_texts_dict = {}, {}, {}

    # Clean baseline
    preds, _ = predict(model, tokenizer, texts, labels, device)
    f1 = f1_score(labels, preds, average="macro")
    results["clean"]           = f1
    all_preds_dict["clean"]    = preds
    obf_texts_dict["clean"]    = texts
    print(f"  clean          → Macro F1: {f1:.4f}")

    for name, fn in obfuscation_fns.items():
        obf  = [fn(t) for t in texts]
        p, _ = predict(model, tokenizer, obf, labels, device)
        f1   = f1_score(labels, p, average="macro")
        drop = results["clean"] - f1
        results[name]        = f1
        all_preds_dict[name] = p
        obf_texts_dict[name] = obf
        print(f"  {name:<16} → Macro F1: {f1:.4f}  (drop: {drop:+.4f})")

    return results, all_preds_dict, obf_texts_dict


def per_class_breakdown(model, tokenizer, texts, labels, obf_texts_dict, device):
    """Print per-class F1 for each condition."""
    print("\n── Per-class F1 breakdown ──")
    header = f"{'Condition':<18} {'hate':>8} {'offensive':>10} {'normal':>8} {'macro':>8}"
    print(header)
    print("─" * len(header))
    for name, txt_list in obf_texts_dict.items():
        p, _      = predict(model, tokenizer, txt_list, labels, device)
        per_class = f1_score(labels, p, average=None)
        macro     = f1_score(labels, p, average="macro")
        print(f"  {name:<16} {per_class[0]:>8.4f} {per_class[1]:>10.4f} {per_class[2]:>8.4f} {macro:>8.4f}")


def failure_analysis(clean_preds, obf_preds, labels, clean_texts, obf_texts, n=5):
    """Show examples correct on clean but wrong after obfuscation."""
    failures = []
    for i, (cp, op, tl) in enumerate(zip(clean_preds, obf_preds, labels)):
        if cp == tl and op != tl:
            failures.append({
                "original":   clean_texts[i],
                "obfuscated": obf_texts[i],
                "true":       LABEL_NAMES[tl],
                "pred_clean": LABEL_NAMES[cp],
                "pred_obf":   LABEL_NAMES[op],
            })

    print(f"\n✓ Failure cases: {len(failures)} ({len(failures)/len(labels)*100:.1f}% of test set)\n")
    for i, f in enumerate(random.sample(failures, min(n, len(failures)))):
        print(f"── Example {i+1} ──")
        print(f"  Original   : {f['original'][:100]}")
        print(f"  Obfuscated : {f['obfuscated'][:100]}")
        print(f"  True       : {f['true']}")
        print(f"  Clean pred : {f['pred_clean']}  ✓")
        print(f"  Obf pred   : {f['pred_obf']}  ✗\n")

    return failures
