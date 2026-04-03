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
- **Improved:** balanced class weights (`class_weight='balanced'` for LR; weighted loss for neural models)

## Results

### Clean Test Set

| Model                          | Hate F1 | Offensive F1 | Normal F1 | Macro F1 |
|--------------------------------|---------|--------------|-----------|----------|
| TF-IDF + LR — Baseline         | 0.732   | 0.486        | 0.731     | 0.650    |
| TF-IDF + LR — Improved         | 0.728   | 0.511        | 0.717     | 0.652    |
| BiLSTM — Baseline              | 0.702   | 0.413        | 0.700     | 0.605    |
| BiLSTM — Improved              | 0.691   | 0.500        | 0.653     | 0.614    |
| BiLSTM — Targeted aug. (4c)    | 0.657   | 0.371        | 0.642     | 0.557    |
| DistilBERT — Baseline          | 0.757   | 0.483        | 0.746     | 0.662    |
| DistilBERT — Improved          | 0.738   | 0.545        | 0.681     | 0.654    |
| DistilBERT — Targeted aug.(4c) | 0.760   | 0.467        | 0.753     | 0.660    |
| hateBERT — Baseline            | 0.772   | 0.512        | 0.755     | 0.680    |
| hateBERT — Improved            | 0.771   | 0.543        | 0.732     | 0.682    |
| hateBERT — Targeted aug. (4c)  | 0.769   | 0.486        | 0.762     | 0.672    |

### Robustness to Targeted Obfuscation (BiLSTM)

Obfuscation is applied only to words in the **hate lexicon** (words appearing ≥70 % of the time in hate/offensive posts, min 5 occurrences) — matching realistic adversarial behaviour where users obfuscate slurs, not neutral context words.

| Model / Config                       | Clean F1 | Obfuscated F1 | Drop    |
|--------------------------------------|----------|---------------|---------|
| Baseline (4a — no aug)               | 0.605    | 0.399         | −0.206  |
| Improved (4b — class weights only)   | 0.614    | 0.435         | −0.180  |
| Task 4c — AUG_RATE=0.2, N_PASSES=3  | 0.572    | 0.433         | −0.140  |
| Task 4c — AUG_RATE=0.5, N_PASSES=3  | 0.572    | 0.433         | −0.140  |
| Task 4c — AUG_RATE=0.5, N_PASSES=10 | 0.547    | 0.458         | −0.090  |

The recommended configuration is **AUG_RATE=0.5, N_PASSES=3**: best Pareto point on the robustness/clean-F1 trade-off. Increasing to N_PASSES=10 gains only 0.002 robustness at a cost of −0.010 clean F1; Vocabulary coverage saturates at N=3 because leet substitutions are a finite character set.

### Robustness to Targeted Obfuscation (DistilBERT and hateBERT)

| Model / Config                        | Clean F1 | Obfuscated F1 | Drop    |
|---------------------------------------|----------|---------------|---------|
| DistilBERT — Baseline (4a)            | 0.662    | 0.529         | −0.133  |
| DistilBERT — Improved (4b)            | 0.654    | 0.519         | −0.135  |
| DistilBERT — Targeted aug. (4c)       | 0.660    | 0.570         | −0.090  |
| hateBERT — Baseline (4a)              | 0.680    | 0.514         | −0.166  |
| hateBERT — Improved (4b)              | 0.682    | 0.580         | −0.102  |
| hateBERT — Targeted aug. (4c)         | 0.672    | 0.599         | **−0.073** |

### Robustness to Full Obfuscation (TF-IDF + LR — reference)

TF-IDF + LR uses character-level n-grams and is evaluated against full-sentence obfuscation (all tokens).

| Condition   | LR Baseline | LR Improved |
|-------------|-------------|-------------|
| Clean       | 0.650       | 0.652       |
| Leet-speak  | 0.445 (−0.205) | 0.460 (−0.192) |
| Punctuation | 0.584 (−0.066) | 0.586 (−0.066) |
| Char repeat | 0.587 (−0.063) | 0.601 (−0.051) |
| Combined    | 0.447 (−0.203) | 0.459 (−0.193) |

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
| **Improved training** | Balanced class weights | ~15–25 min (GPU) |
| **Robustness eval** | Tests on obfuscated text (leet-speak, punctuation, etc.) | ~3 min |

> **Tip:** if you just want to see the results without retraining, the plots and metrics are already visible in the notebook output cells on GitHub.

## Key Findings

- **hateBERT improved is the best overall model** (macro F1 0.682), outperforming DistilBERT (0.662), TF-IDF + LR (0.650), and BiLSTM (0.605)
- **TF-IDF + LR is surprisingly competitive** — only 0.032 F1 behind the best hateBERT result, and it trains in seconds with no GPU
- **BiLSTM is the weakest model** (macro F1 0.605) — word-level embeddings trained from scratch on ~15k samples lack the representational power of pre-trained transformers or the lexical coverage of TF-IDF n-grams
- `offensive` is the hardest class for all models (F1 ~0.41–0.53) due to semantic overlap with both `hate` and `normal`
- **BiLSTM is the least robust** (drop −0.206 under targeted obfuscation) — word-level tokenization maps every obfuscated token to `<UNK>`, destroying the signal of discriminative words entirely
- **hateBERT is the most robust under targeted obfuscation after mitigation** (drop −0.166 baseline, −0.073 with Task 4c) — domain-specific pre-training on abusive text provides a stronger prior for recognising obfuscated slurs via WordPiece subword decomposition
- **Targeted augmentation (Task 4c) behaves differently across architectures**: for BiLSTM it trades clean F1 for robustness (OOV problem partially mitigated but at distribution-shift cost); for DistilBERT and hateBERT it improves robustness with negligible clean F1 change, because WordPiece already handles obfuscated tokens without collapsing to `<UNK>`
- For BiLSTM, class weights give a small clean-text improvement (+0.009 macro F1) with no meaningful robustness change — confirming that robustness is an architectural property, not a training-strategy property
- For BERT models, class weights shift offensive F1 up (+0.06 DistilBERT, +0.03 hateBERT) with small macro-F1 changes, reflecting the already-strong pre-training prior
- **Threshold tuning on the baseline** (no re-training) boosts hate recall across all models:

| Model | Default Hate Recall | Tuned Hate Recall | Threshold Used | Hate Precision Cost | Macro F1 Cost |
|-------|--------------------|-----------------------------|----------------|---------------------|---------------|
| TF-IDF + LR | 0.785 | 0.904 | 0.225 | −0.108 | −0.041 |
| BiLSTM | 0.721 | 0.827 | 0.150 | −0.114 | −0.036 |
| DistilBERT | 0.833 | 0.906 | 0.225 | −0.046 | −0.010 |
| hateBERT | 0.803 | 0.879 | 0.150 | −0.098 | −0.040 |

## Analysis & Limitations

### What each "Improved" model actually changes

All models use the same improved strategy: **balanced class weights** to address the class imbalance in HateXplain.

| Model | Improved strategy |
|-------|------------------|
| **TF-IDF + LR** | `class_weight='balanced'` |
| **BiLSTM** | Weighted loss (`CrossEntropyLoss` with balanced class weights) |
| **DistilBERT** | Weighted loss (`CrossEntropyLoss` with balanced class weights) |
| **hateBERT** | Weighted loss (`CrossEntropyLoss` with balanced class weights) |

### Robustness evaluation

The robustness evaluation is **diagnostic**: models are trained on clean text and tested on artificially obfuscated variants (leet-speak, punctuation insertion, character repetition). The evaluation measures how fragile each architecture is under surface-level text perturbation.

The results reflect inherent architectural differences (targeted obfuscation, hate lexicon words only):

| Model | Baseline drop | Task 4c drop |
|-------|--------------|-------------|
| BiLSTM | −0.206 | −0.140 |
| DistilBERT | −0.133 | −0.090 |
| hateBERT | −0.166 | **−0.073** |

- **BiLSTM** is the most fragile: its fixed word-level vocabulary maps every obfuscated token to `<UNK>`, losing the entire signal of discriminative hate words
- **BERT models** are more robust: WordPiece decomposes `n1gg3r` into subword pieces rather than a single `<UNK>`, preserving partial semantic signal through the attention layers
- **hateBERT > DistilBERT** on robustness — domain-specific pre-training on abusive Reddit content provides a stronger prior for recognising hate-loaded subword sequences

### What targeted augmentation (Task 4c) achieves and what it costs

Task 4c tests whether a more principled augmentation — obfuscating only words that are discriminative for hate speech (the **hate lexicon**) and including their variants in the vocabulary — can improve robustness under realistic adversarial conditions.

**Parameters and their role:**
- `AUG_RATE` (default 0.5): fraction of hate/offensive samples augmented per epoch. This is the dominant parameter — it directly controls the distribution shift and therefore how much clean F1 is lost.
- `N_PASSES` (default 3): number of random-seed passes used to populate the extended vocabulary with obfuscated variants. This saturates quickly (leet substitutions are a finite set) and has negligible effect beyond N=3.

**Pareto frontier (AUG_RATE=0.5, N_PASSES=3 is the recommended setting):**

| AUG_RATE | N_PASSES | Clean F1 | Obf F1 | Drop |
|----------|----------|----------|--------|------|
| — | — | 0.605 | 0.399 | −0.206 |
| 0.2 | 3 | 0.572 | 0.433 | −0.140 |
| 0.5 | 3 | 0.557 | 0.456 | **−0.100** |
| 0.5 | 10 | 0.547 | 0.458 | −0.090 |

**Architecture determines the trade-off:**

| Architecture | Task 4c effect on clean F1 | Task 4c effect on robustness |
|---|---|---|
| BiLSTM | −0.048 (0.605→0.557) | +0.066 drop reduction |
| DistilBERT | −0.002 (0.662→0.660) | +0.043 drop reduction |
| hateBERT | −0.008 (0.680→0.672) | +0.093 drop reduction |

For BiLSTM, Task 4c adds obfuscated forms to the vocabulary, but the surrounding context words still become `<UNK>` — so robustness only partially improves and clean F1 suffers. For BERT models, vocabulary extension is not needed (WordPiece handles it), so targeted augmentation during fine-tuning improves robustness almost for free.

**The remaining bottleneck for BiLSTM:** a genuine fix requires character-level or subword representations (fastText, BERT) that survive token-level substitution without mapping to `<UNK>`.

### Overfitting: why validation loss rises after epoch 1–2

The validation loss curve shows a characteristic BERT fine-tuning pattern: training loss decreases monotonically while validation loss reaches a minimum at epoch 1–2 and then increases slightly. This reflects two concurrent phenomena:

1. **Catastrophic forgetting** — as fine-tuning continues, the model's pre-trained representations are progressively overwritten. The linear learning rate schedule decays slowly, so later epochs still apply non-trivial updates that degrade generalisation.
2. **Dataset size** — HateXplain (~15k training samples) is relatively small for a 66M-parameter model. After 1–2 epochs the model has already memorised a significant portion of the training distribution.

Early stopping (patience=2) mitigates this by restoring the best-val-loss checkpoint, but the underlying tension between training signal and generalisation remains. Reducing `LR` to `1e-5` or adding a stronger warmup would likely flatten the val loss curve without requiring architectural changes.

## References

Mathew, B. et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
