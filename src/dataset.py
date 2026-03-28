# src/dataset.py
# Download and process the HateXplain dataset from GitHub

import json
import urllib.request
import pandas as pd
from collections import Counter

BASE_URL  = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
LABEL_MAP = {"hatespeech": 0, "offensive": 1, "normal": 2}
LABEL_NAMES = {0: "hate", 1: "offensive", 2: "normal"}


def _download_json(filename):
    url = BASE_URL + filename
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode())


def _majority_vote(label_list):
    count = Counter(label_list)
    top   = count.most_common(1)[0]
    return top[0] if top[1] >= 2 else None


def _build_dataframe(post_ids, data):
    rows = []
    for pid in post_ids:
        if pid not in data:
            continue
        post   = data[pid]
        labels = [a["label"] for a in post["annotators"]]
        vote   = _majority_vote(labels)
        if vote is None:
            continue
        text      = " ".join(post["post_tokens"])
        label_int = LABEL_MAP[vote]
        rows.append({
            "post_id":    pid,
            "text":       text,
            "label":      label_int,
            "label_name": LABEL_NAMES[label_int]
        })
    return pd.DataFrame(rows)


def download_hatexplain():
    """
    Download HateXplain from GitHub and return train, val, test DataFrames.
    Labels: 0=hate, 1=offensive, 2=normal (majority vote from 3 annotators).
    Ambiguous posts (no majority) are discarded.

    Returns:
        train_df, val_df, test_df : pd.DataFrame
    """
    print("Downloading dataset from GitHub...")
    data       = _download_json("dataset.json")
    split_info = _download_json("post_id_divisions.json")
    print(f"✓ Total posts: {len(data)}")

    train_df = _build_dataframe(split_info["train"], data)
    val_df   = _build_dataframe(split_info["val"],   data)
    test_df  = _build_dataframe(split_info["test"],  data)

    print(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df
