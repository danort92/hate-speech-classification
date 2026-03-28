# src/preprocessing.py
# Text cleaning and PyTorch Dataset class for DistilBERT

import re
import torch
from torch.utils.data import Dataset

MAX_LEN = 128


def clean_text(text):
    """
    Light cleaning for social media text.
    Replaces URLs and @mentions, strips # from hashtags, collapses whitespace.
    Intentionally minimal — DistilBERT is pretrained on raw text.
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "[URL]",   text)
    text = re.sub(r"@\w+",           "[USER]",   text)
    text = re.sub(r"#(\w+)",         r"\1",      text)
    text = re.sub(r"\s+",            " ",        text).strip()
    return text


def apply_cleaning(train_df, val_df, test_df):
    """Apply clean_text to all splits and add clean_text column."""
    for df in [train_df, val_df, test_df]:
        df["clean_text"] = df["text"].apply(clean_text)
    return train_df, val_df, test_df


class HateDataset(Dataset):
    """
    PyTorch Dataset for HateXplain.
    Tokenizes text on the fly using a HuggingFace tokenizer.

    Args:
        df         : DataFrame with clean_text and label columns
        tokenizer  : HuggingFace tokenizer
        max_len    : max token length (default 128)
        augment_fn : optional function (str -> str) applied to hate/offensive texts
    """
    def __init__(self, df, tokenizer, max_len=MAX_LEN, augment_fn=None):
        self.texts      = df["clean_text"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        if self.augment_fn is not None and label in [0, 1]:
            import random
            if random.random() < 0.5:
                text = self.augment_fn(text)

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long)
        }
