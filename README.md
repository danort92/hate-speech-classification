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
| hateBERT — Baseline     | 0.768   | 0.545        | 0.717     | 0.677    |
| hateBERT — Improved     | —       | —            | —         | 0.676    |

### Robustness to Obfuscation

| Condition   | DistilBERT Baseline | DistilBERT Improved | hateBERT Baseline | hateBERT Improved |
|-------------|---------------------|---------------------|-------------------|-------------------|
| Clean       | 0.686               | —                   | 0.677             | 0.676             |
| Leet-speak  | 0.347 (−0.339)      | —                   | 0.425 (−0.252)    | 0.382 (−0.294)    |
| Punctuation | 0.586 (−0.100)      | —                   | 0.600 (−0.077)    | 0.412 (−0.264)    |
| Char repeat | 0.640 (−0.046)      | —                   | 0.609 (−0.068)    | 0.468 (−0.208)    |
| Combined    | 0.357 (−0.329)      | —                   | 0.415 (−0.262)    | 0.352 (−0.324)    |

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

- hateBERT achieves **macro F1 0.677** on the clean test set; `offensive` is the hardest class (F1 0.545) due to semantic overlap with both `hate` and `normal`
- **Leet-speak is the most damaging obfuscation** — it causes a −0.252 F1 drop on the baseline, as the model cannot recognise key offensive words when characters are substituted (e.g. `h4t3`)
- The improved model (class weights + adversarial augmentation) **does not improve robustness** — it actually degrades performance on punctuation (−0.174) and char repeat (−0.173) compared to baseline, while clean F1 stays flat (0.676 vs 0.677)
- This is caused by a **train/val distribution mismatch**: early stopping selects the checkpoint with the best *clean* validation loss, not the most robust one (see Analysis & Limitations)
- Combined obfuscation remains the hardest condition, dropping F1 from 0.677 to 0.415 (baseline) and 0.352 (improved)

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
