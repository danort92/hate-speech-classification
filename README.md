# Hate Speech Classification with Robustness Testing

Fine-tuning DistilBERT on the HateXplain dataset for 3-class hate speech detection, with robustness evaluation against common text obfuscation tactics.

**MSc Responsible AI — NLP Project | OPIT University**

---

## Project Structure

```
├── notebooks/
│   └── hate_speech_pipeline.ipynb   # full pipeline (EDA → training → evaluation)
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

- **Architecture:** DistilBERT (`distilbert-base-uncased`) with a 3-class classification head
- **Baseline:** 3 epochs, AdamW (lr=2e-5), linear scheduler with 10% warmup
- **Improved:** 4 epochs + class weights (balanced) + adversarial augmentation on hate/offensive posts

## Results

### Clean Test Set

| Model    | Hate F1 | Offensive F1 | Normal F1 | Macro F1 |
|----------|---------|--------------|-----------|----------|
| Baseline | 0.769   | 0.538        | 0.750     | 0.686    |
| Improved | —       | —            | —         | —        |

### Robustness to Obfuscation (Baseline)

| Condition   | Macro F1 | Drop    |
|-------------|----------|---------|
| Clean       | 0.686    | —       |
| Leet-speak  | 0.347    | −0.339  |
| Punctuation | 0.586    | −0.100  |
| Char repeat | 0.640    | −0.046  |
| Combined    | 0.357    | −0.329  |

## How to Run

> **Prerequisiti:** Python 3.9+ e una connessione internet (il dataset viene scaricato automaticamente).
> Una GPU è consigliata per il training (~15 min su GPU, ~2 h su CPU). Google Colab offre una GPU gratuita.

### Opzione 1 — Google Colab (consigliato)

1. Apri [Google Colab](https://colab.research.google.com/)
2. Seleziona **File → Apri notebook → GitHub** e incolla:
   ```
   https://github.com/danort92/hate-speech-classification
   ```
3. Seleziona `notebooks/hate_speech_pipeline.ipynb`
4. Abilita la GPU: **Runtime → Cambia tipo di runtime → T4 GPU**
5. Esegui tutte le celle in ordine: **Runtime → Esegui tutto**

> **Nota:** la prima cella installa le dipendenze automaticamente. Dopo l'installazione, Colab potrebbe chiederti di riavviare il runtime — conferma e poi esegui nuovamente tutte le celle.

### Opzione 2 — In locale

```bash
# 1. Clona la repository
git clone https://github.com/danort92/hate-speech-classification.git
cd hate-speech-classification

# 2. Crea un ambiente virtuale (consigliato)
python3 -m venv venv
source venv/bin/activate        # su Windows: venv\Scripts\activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Avvia Jupyter e apri il notebook
jupyter notebook notebooks/hate_speech_pipeline.ipynb
```

### Cosa succede quando esegui il notebook

Il notebook è suddiviso in sezioni sequenziali:

| Sezione | Cosa fa | Tempo stimato |
|---------|---------|---------------|
| **Configurazione** | Imposta iperparametri (modello, batch size, lr) | istantaneo |
| **Download e EDA** | Scarica HateXplain, analisi esplorativa con grafici | ~1 min |
| **Preprocessing** | Pulizia testo, tokenizzazione, creazione dei DataLoader | ~1 min |
| **Training baseline** | Fine-tuning DistilBERT (3 epoche) | ~10 min (GPU) |
| **Training improved** | Modello con class weights + data augmentation | ~15 min (GPU) |
| **Robustness eval** | Test su testo offuscato (leet-speak, punteggiatura, ecc.) | ~3 min |

> **Tip:** se vuoi solo vedere i risultati senza riaddestrare, i grafici e le metriche sono già visibili nelle celle di output del notebook su GitHub.

## Key Findings

- DistilBERT achieves **macro F1 0.686** on the clean test set
- `offensive` is the hardest class (F1 0.538) due to semantic overlap with both `hate` and `normal`
- **Leet-speak causes a −0.34 F1 drop** — the model cannot recognise key offensive words when characters are substituted (e.g. `h4t3`)
- The improved model uses adversarial augmentation during training to close this gap
- 31% of test posts that were correctly classified become misclassified as `normal` after combined obfuscation

## References

Mathew, B. et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
