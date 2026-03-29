# Hate Speech Classification with Robustness Testing

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)

Fine-tuning DistilBERT and hateBERT on the HateXplain dataset for 3-class hate speech detection, with robustness evaluation against common text obfuscation tactics.

**MSc Responsible AI — NLP Project | OPIT University**

---

## Project Structure

```
├── notebooks/
│   ├── notebook_tfidf_lr.ipynb      # TF-IDF + Logistic Regression (classical ML)
│   ├── notebook_biLSTM.ipynb        # BiLSTM (deep learning)
│   ├── notebook_distilbert.ipynb    # full pipeline with DistilBERT
│   └── notebook_hatebert.ipynb      # full pipeline with hateBERT
├── requirements.txt
└── README.md
```

Each notebook is fully self-contained — all code (dataset download, preprocessing, training, evaluation, obfuscation, plotting) is defined inline so it runs directly on Google Colab without additional setup.

## Dataset

[HateXplain](https://github.com/hate-alert/HateXplain) — ~20,000 posts from Twitter and Gab, annotated by 3 workers each. Labels resolved via majority vote.

| Split | Size |
|-------|------|
| Train | ~15,000 |
| Val   | ~1,900  |
| Test  | ~1,900  |

**Classes:** `hate` (0) · `offensive` (1) · `normal` (2)

## Model

Four models are compared, spanning classical ML to transformers:

| Model | Type | Notes |
|-------|------|-------|
| **TF-IDF + LR** | Classical ML | Logistic Regression on TF-IDF features (unigrams + bigrams) |
| **BiLSTM** | Deep Learning | Bidirectional LSTM with learned word embeddings (128d) |
| **DistilBERT** | Transformer | `distilbert-base-uncased` — general-purpose, lightweight |
| **hateBERT** | Transformer | `GroNLP/hateBERT` — BERT re-trained on abusive Reddit content |

Each model is evaluated in two configurations:
- **Baseline:** standard training (LR: default weights; BiLSTM/transformers: early stopping on val loss)
- **Improved:** balanced class weights (LR: `class_weight='balanced'`; transformers: + adversarial augmentation)

## Results

### Clean Test Set

| Model                   | Hate F1 | Offensive F1 | Normal F1 | Macro F1 |
|-------------------------|---------|--------------|-----------|----------|
| TF-IDF + LR — Baseline  | 0.732   | 0.486        | 0.731     | 0.650    |
| TF-IDF + LR — Improved  | 0.728   | 0.511        | 0.717     | 0.652    |
| BiLSTM — Baseline        | 0.699   | 0.432        | 0.687     | 0.606    |
| BiLSTM — Improved        | 0.692   | 0.456        | 0.666     | 0.604    |
| DistilBERT — Baseline   | 0.765   | 0.518        | 0.748     | 0.677    |
| DistilBERT — Improved   | 0.770   | 0.461        | 0.742     | 0.658    |
| hateBERT — Baseline     | 0.772   | 0.521        | 0.743     | 0.679    |
| hateBERT — Improved     | 0.764   | 0.479        | 0.754     | 0.666    |

### Robustness to Obfuscation

| Condition   | LR Baseline | LR Improved | BiLSTM Baseline | BiLSTM Improved | DistilBERT Baseline | DistilBERT Improved | hateBERT Baseline | hateBERT Improved |
|-------------|-------------|-------------|-----------------|-----------------|---------------------|---------------------|-------------------|-------------------|
| Clean       | 0.650       | 0.652       | 0.606           | 0.604           | 0.677               | 0.658               | 0.679             | 0.666             |
| Leet-speak  | 0.445 (−0.205) | 0.460 (−0.192) | 0.272 (−0.334) | 0.274 (−0.330) | 0.365 (−0.312)      | 0.379 (−0.279)      | 0.398 (−0.281)    | 0.382 (−0.297)    |
| Punctuation | 0.584 (−0.066) | 0.586 (−0.066) | 0.526 (−0.080) | 0.532 (−0.072) | 0.580 (−0.097)      | 0.503 (−0.155)      | 0.607 (−0.072)    | 0.468 (−0.211)    |
| Char repeat | 0.587 (−0.063) | 0.601 (−0.051) | 0.540 (−0.066) | 0.540 (−0.064) | 0.620 (−0.057)      | 0.635 (−0.023)      | 0.657 (−0.022)    | 0.649 (−0.030)    |
| Combined    | 0.447 (−0.203) | 0.459 (−0.193) | 0.266 (−0.340) | 0.274 (−0.330) | 0.371 (−0.306)      | 0.337 (−0.321)      | 0.395 (−0.284)    | 0.361 (−0.318)    |

## How to Run

> **Prerequisites:** Python 3.9+ and an internet connection (the dataset is downloaded automatically).
> A GPU is recommended for training (~15 min on GPU, ~2 h on CPU). Google Colab provides a free GPU.

Each notebook is independent and self-contained. Pick any notebook and run it — no dependencies between them.

| Notebook | Model | Estimated time | GPU needed | Open in Colab |
|----------|-------|----------------|------------|---------------|
| `notebook_tfidf_lr.ipynb` | TF-IDF + LR | ~2 min | No | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/hate-speech-classification/blob/main/notebooks/notebook_tfidf_lr.ipynb) |
| `notebook_biLSTM.ipynb` | BiLSTM | ~10 min | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/hate-speech-classification/blob/main/notebooks/notebook_biLSTM.ipynb) |
| `notebook_distilbert.ipynb` | DistilBERT | ~30 min | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/hate-speech-classification/blob/main/notebooks/notebook_distilbert.ipynb) |
| `notebook_hatebert.ipynb` | hateBERT | ~45 min | Yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/hate-speech-classification/blob/main/notebooks/notebook_hatebert.ipynb) |

### Option 1 — Google Colab (recommended)

1. Click one of the **Open in Colab** badges above
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells in order: **Runtime → Run all**

> **Note:** the first cell installs dependencies automatically. After installation, Colab may ask you to restart the runtime — confirm and then run all cells again.

### Option 2 — Local setup

```bash
# 1. Clone the repository
git clone https://github.com/danort92/hate-speech-classification.git
cd hate-speech-classification

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and open a notebook
jupyter notebook notebooks/notebook_distilbert.ipynb
# or: jupyter notebook notebooks/notebook_hatebert.ipynb
```

### What happens when you run a notebook

Each notebook is split into sequential sections:

| Section | What it does | Estimated time |
|---------|--------------|----------------|
| **Configuration** | Sets hyperparameters (model, batch size, lr) | instant |
| **Download & EDA** | Downloads HateXplain, exploratory analysis with plots | ~1 min |
| **Preprocessing** | Text cleaning, tokenization, DataLoader creation | ~1 min |
| **Baseline training** | Fine-tunes the model (early stopping) | ~10–20 min (GPU) |
| **Improved training** | Class weights + adversarial augmentation | ~15–25 min (GPU) |
| **Robustness eval** | Tests on obfuscated text (leet-speak, punctuation, etc.) | ~3 min |

> **Tip:** if you just want to see the results without retraining, the plots and metrics are already visible in the notebook output cells on GitHub.

## Key Findings

- **hateBERT baseline is the best overall model** (macro F1 0.679), outperforming DistilBERT (0.677), TF-IDF + LR (0.650), and BiLSTM (0.606)
- **TF-IDF + LR is surprisingly competitive** — only 0.03 F1 behind the transformers, and it trains in seconds with no GPU
- **BiLSTM is the weakest model** (macro F1 0.606) — word-level embeddings trained from scratch on ~15k samples lack the representational power of pre-trained transformers or the lexical coverage of TF-IDF n-grams
- `offensive` is the hardest class for all models (F1 ~0.43–0.52) due to semantic overlap with both `hate` and `normal`
- **BiLSTM is the least robust to leet-speak** (drop −0.334) — word-level tokenization treats every obfuscated word as OOV, losing all semantic information
- **TF-IDF + LR is the most robust to leet-speak** (drop −0.205 vs −0.281 hateBERT, −0.312 DistilBERT, −0.334 BiLSTM) — character-level n-grams partially survive obfuscation, while WordPiece and word-level tokenizers break on substituted characters
- **hateBERT is more robust than DistilBERT** on all conditions — domain-specific pre-training provides a stronger prior that survives surface-level text manipulation
- For transformers, the improved models (class weights + adversarial augmentation) **degrade robustness** — they collapse the offensive class (recall ~0.50 → ~0.39) as obfuscated examples teach the model to default to `normal`
- For TF-IDF + LR, balanced class weights **do improve robustness** across all conditions (+0.01–0.014), showing that the augmentation strategy, not class weights, is the problem with transformer improved models
- **Threshold tuning on the baseline** (no re-training) boosts hate recall across all models:

| Model | Default Hate Recall | Tuned Hate Recall | Threshold Used | Hate Precision Cost | Macro F1 Cost |
|-------|--------------------|-----------------------------|----------------|---------------------|---------------|
| TF-IDF + LR | 0.785 | 0.904 | 0.225 | −0.108 | −0.041 |
| BiLSTM | 0.749 | 0.889 | 0.150 | −0.129 | — |
| DistilBERT | 0.860 | 0.904 | 0.225 | −0.046 | −0.020 |
| hateBERT | 0.828 | 0.904 | 0.150 | −0.098 | −0.035 |

## Analysis & Limitations

### What each "Improved" model actually changes

Not all improved models use the same strategy:

| Model | Improved strategy |
|-------|------------------|
| **TF-IDF + LR** | `class_weight='balanced'` only — no augmentation (TF-IDF vectors are computed once, augmenting text post-vectorisation would have no effect on the feature space) |
| **BiLSTM** | `class_weight` (via weighted loss) + adversarial augmentation (50 % of hate/offensive training posts obfuscated with `combined_obfuscation`) |
| **DistilBERT** | Same as BiLSTM: weighted loss + adversarial augmentation |
| **hateBERT** | Same as BiLSTM: weighted loss + adversarial augmentation |

This difference is itself an interesting finding: TF-IDF + LR's robustness *improves* with balanced weights alone (+0.01–0.014 across all conditions), while the three neural models that add augmentation see robustness *degrade*. This isolates the augmentation strategy — not class weighting — as the source of the regression.

### Why adversarial augmentation does not close the robustness gap

For BiLSTM, DistilBERT and hateBERT, the improved model applies `combined_obfuscation` to 50 % of hate/offensive training posts. Despite this, robustness gains are modest or negative. The root cause is a **train/val distribution mismatch in the early stopping criterion**: the validation loss — which determines when training stops and which checkpoint is saved — is computed on the *clean* validation set. The model is therefore selected based on clean-text performance, not robustness. Training sees obfuscated examples, but the checkpoint that gets saved is the one that minimises loss on unperturbed text, which may not be the checkpoint that generalises best to obfuscated inputs.

A stricter fix would evaluate a combined metric (e.g. average of clean and obfuscated F1) at each epoch and use that for early stopping. This was left out of scope to keep the pipeline readable, but is worth noting as a design trade-off.

### Overfitting: why validation loss rises after epoch 1–2

The validation loss curve shows a characteristic BERT fine-tuning pattern: training loss decreases monotonically while validation loss reaches a minimum at epoch 1–2 and then increases slightly. This reflects two concurrent phenomena:

1. **Catastrophic forgetting** — as fine-tuning continues, the model's pre-trained representations are progressively overwritten. The linear learning rate schedule decays slowly, so later epochs still apply non-trivial updates that degrade generalisation.
2. **Dataset size** — HateXplain (~15k training samples) is relatively small for a 66M-parameter model. After 1–2 epochs the model has already memorised a significant portion of the training distribution.

Early stopping (patience=2) mitigates this by restoring the best-val-loss checkpoint, but the underlying tension between training signal and generalisation remains. Reducing `LR` to `1e-5` or adding a stronger warmup would likely flatten the val loss curve without requiring architectural changes.

## References

Mathew, B. et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
