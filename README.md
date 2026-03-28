# Hate Speech Classification with Robustness Testing

Fine-tuning DistilBERT and hateBERT on the HateXplain dataset for 3-class hate speech detection, with robustness evaluation against common text obfuscation tactics.

**MSc Responsible AI — NLP Project | OPIT University**

---

## Project Structure

```
├── notebooks/
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

Two models are compared:

| Model | Base | Notes |
|-------|------|-------|
| **DistilBERT** | `distilbert-base-uncased` | General-purpose, lightweight |
| **hateBERT** | `GroNLP/hateBERT` | BERT re-trained on abusive Reddit content |

Each model is evaluated in two configurations:
- **Baseline:** 3 epochs, AdamW (lr=2e-5), linear scheduler with 10% warmup
- **Improved:** 4 epochs + balanced class weights + adversarial augmentation on hate/offensive posts

## Results

### Clean Test Set

| Model                   | Hate F1 | Offensive F1 | Normal F1 | Macro F1 |
|-------------------------|---------|--------------|-----------|----------|
| DistilBERT — Baseline   | 0.765   | 0.518        | 0.748     | 0.677    |
| DistilBERT — Improved   | 0.770   | 0.461        | 0.742     | 0.658    |
| hateBERT — Baseline     | 0.772   | 0.521        | 0.743     | 0.679    |
| hateBERT — Improved     | 0.764   | 0.479        | 0.754     | 0.666    |

### Robustness to Obfuscation

| Condition   | DistilBERT Baseline | DistilBERT Improved | hateBERT Baseline | hateBERT Improved |
|-------------|---------------------|---------------------|-------------------|-------------------|
| Clean       | 0.677               | 0.658               | 0.679             | 0.666             |
| Leet-speak  | 0.365 (−0.312)      | 0.379 (−0.279)      | 0.398 (−0.281)    | 0.382 (−0.297)    |
| Punctuation | 0.580 (−0.097)      | 0.503 (−0.155)      | 0.607 (−0.072)    | 0.468 (−0.211)    |
| Char repeat | 0.620 (−0.057)      | 0.635 (−0.023)      | 0.657 (−0.022)    | 0.649 (−0.030)    |
| Combined    | 0.371 (−0.306)      | 0.337 (−0.321)      | 0.395 (−0.284)    | 0.361 (−0.318)    |

## How to Run

> **Prerequisites:** Python 3.9+ and an internet connection (the dataset is downloaded automatically).
> A GPU is recommended for training (~15 min on GPU, ~2 h on CPU). Google Colab provides a free GPU.

There are two independent notebooks — one per model. Each is self-contained and runs the full pipeline (EDA → training → robustness evaluation).

| Notebook | Model | Estimated time (T4 GPU) | Open in Colab |
|----------|-------|-------------------------|---------------|
| `notebook_distilbert.ipynb` | DistilBERT | ~30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/hate-speech-classification/blob/main/notebooks/notebook_distilbert.ipynb) |
| `notebook_hatebert.ipynb` | hateBERT | ~45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/hate-speech-classification/blob/main/notebooks/notebook_hatebert.ipynb) |

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

- **hateBERT baseline is the best overall model** (macro F1 0.679), outperforming DistilBERT (0.677) thanks to pre-training on abusive Reddit content
- `offensive` is the hardest class for both models (F1 ~0.52) due to semantic overlap with both `hate` and `normal`
- **hateBERT is more robust to obfuscation across all conditions** — leet-speak drop is −0.281 vs −0.312 for DistilBERT, punctuation −0.072 vs −0.097, char repeat −0.022 vs −0.057. Domain-specific pre-training provides a stronger prior that survives surface-level text manipulation
- **Leet-speak and combined obfuscation are the most damaging**, causing ~0.28–0.31 F1 drops on both models — the tokenizer cannot recognise key offensive words when characters are substituted (e.g. `h4t3`)
- The improved models (class weights + adversarial augmentation) **do not improve robustness on either model** — they degrade clean F1 and collapse the offensive class (recall drops from ~0.50 to ~0.39), as obfuscated training examples teach the model to default to `normal` when keywords are unrecognisable
- **Threshold tuning on the baseline** (no re-training) boosts DistilBERT hate recall from 0.860 to 0.904 with only −0.046 precision cost — a more effective strategy than adversarial augmentation for prioritising recall

## Analysis & Limitations

### Why the improved model does not fully close the robustness gap

The improved model applies `combined_obfuscation` to 50% of hate/offensive training posts (adversarial augmentation). Despite this, robustness gains are modest. The root cause is a **train/val distribution mismatch in the early stopping criterion**: the validation loss — which determines when training stops and which checkpoint is saved — is computed on the *clean* validation set. The model is therefore selected based on clean-text performance, not robustness. Training sees obfuscated examples, but the checkpoint that gets saved is the one that minimises loss on unperturbed text, which may not be the checkpoint that generalises best to obfuscated inputs.

A stricter fix would evaluate a combined metric (e.g. average of clean and obfuscated F1) at each epoch and use that for early stopping. This was left out of scope to keep the pipeline readable, but is worth noting as a design trade-off.

### Overfitting: why validation loss rises after epoch 1–2

The validation loss curve shows a characteristic BERT fine-tuning pattern: training loss decreases monotonically while validation loss reaches a minimum at epoch 1–2 and then increases slightly. This reflects two concurrent phenomena:

1. **Catastrophic forgetting** — as fine-tuning continues, the model's pre-trained representations are progressively overwritten. The linear learning rate schedule decays slowly, so later epochs still apply non-trivial updates that degrade generalisation.
2. **Dataset size** — HateXplain (~15k training samples) is relatively small for a 66M-parameter model. After 1–2 epochs the model has already memorised a significant portion of the training distribution.

Early stopping (patience=2) mitigates this by restoring the best-val-loss checkpoint, but the underlying tension between training signal and generalisation remains. Reducing `LR` to `1e-5` or adding a stronger warmup would likely flatten the val loss curve without requiring architectural changes.

## References

Mathew, B. et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
