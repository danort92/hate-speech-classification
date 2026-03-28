# Hate Speech Classification with Robustness Testing

Fine-tuning DistilBERT and hateBERT on the HateXplain dataset for 3-class hate speech detection, with robustness evaluation against common text obfuscation tactics.

**MSc Responsible AI — NLP Project | OPIT University**

---

## Project Structure

```
├── notebooks/
│   ├── notebook_distilbert.ipynb    # full pipeline with DistilBERT
│   └── notebook_hatebert.ipynb      # full pipeline with hateBERT
├── src/
│   ├── dataset.py        # HateXplain download and majority-vote labelling
│   ├── preprocessing.py  # text cleaning and PyTorch HateDataset class
│   ├── model.py          # training loop, evaluation, class weights
│   ├── obfuscation.py    # leet-speak, punctuation insertion, char repeat
│   └── evaluation.py     # metrics, confusion matrix, robustness plots
├── requirements.txt
└── README.md
```

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
| DistilBERT — Baseline   | 0.769   | 0.538        | 0.750     | 0.686    |
| DistilBERT — Improved   | —       | —            | —         | —        |
| hateBERT — Baseline     | —       | —            | —         | —        |
| hateBERT — Improved     | —       | —            | —         | —        |

### Robustness to Obfuscation

| Condition   | DistilBERT Baseline | DistilBERT Improved | hateBERT Baseline | hateBERT Improved |
|-------------|---------------------|---------------------|-------------------|-------------------|
| Clean       | 0.686               | —                   | —                 | —                 |
| Leet-speak  | 0.347 (−0.339)      | —                   | —                 | —                 |
| Punctuation | 0.586 (−0.100)      | —                   | —                 | —                 |
| Char repeat | 0.640 (−0.046)      | —                   | —                 | —                 |
| Combined    | 0.357 (−0.329)      | —                   | —                 | —                 |

## How to Run

> **Prerequisites:** Python 3.9+ and an internet connection (the dataset is downloaded automatically).
> A GPU is recommended for training (~15 min on GPU, ~2 h on CPU). Google Colab provides a free GPU.

There are two independent notebooks — one per model. Each is self-contained and runs the full pipeline (EDA → training → robustness evaluation).

| Notebook | Model | Estimated time on Colab free (T4 GPU) |
|----------|-------|---------------------------------------|
| `notebook_distilbert.ipynb` | DistilBERT | ~30 min |
| `notebook_hatebert.ipynb` | hateBERT | ~45 min |

### Option 1 — Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **File → Open notebook → GitHub** and paste:
   ```
   https://github.com/danort92/hate-speech-classification
   ```
3. Select the notebook you want to run (`notebook_distilbert.ipynb` or `notebook_hatebert.ipynb`)
4. Enable GPU: **Runtime → Change runtime type → T4 GPU**
5. Run all cells in order: **Runtime → Run all**

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

- DistilBERT achieves **macro F1 0.686** on the clean test set
- `offensive` is the hardest class (F1 0.538) due to semantic overlap with both `hate` and `normal`
- **Leet-speak causes a −0.34 F1 drop** — the model cannot recognise key offensive words when characters are substituted (e.g. `h4t3`)
- The improved model uses adversarial augmentation during training to close this gap
- 31% of test posts that were correctly classified become misclassified as `normal` after combined obfuscation

## References

Mathew, B. et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
