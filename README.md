# AI_support_tickets

# AI Support Tickets Classification (3-Class) — BERT Fine-tuning & Error Analysis

This resume-focused NLP project fine-tunes a pretrained Transformer (BERT/DistilBERT-family) to classify customer support tickets into **3 classes**. The repo emphasizes **controlled tuning strategies (ablation)**, **validation/test evaluation**, and **error analysis**, rather than production deployment.

---

## Highlights
- Compared **fine-tuning strategies** (ablation):
  - Full fine-tuning (encoder + classifier head)
  - Head-only training (encoder frozen)
  - Partial unfreeze (last 2 encoder layers + head)
- Reported **Accuracy + Macro-F1** (Macro-F1 highlights per-class performance)
- Implemented evaluation workflow (confusion matrix / classification report / error analysis)
- Best configuration: **Full fine-tuning with LR=3e-5**  
  - **Test Acc: 0.7722**
  - **Test Macro-F1: 0.7636**

---

## Results

### Fine-tuning Strategy & Learning Rate (Validation)

| Exp ID | Description                            | Tuning Strategy               | LR (enc/head) | Val Acc | Val Macro-F1 | Notes           |
| -----: | -------------------------------------- | ----------------------------- | ------------- | ------: | -----------: | --------------- |
|     A1 | Full fine-tuning (baseline)            | Full FT                       | 2e-5 / 2e-5   |  0.7521 |       0.7466 | baseline        |
|     A2 | Frozen encoder (train classifier only) | Head-only (encoder frozen)    | — / 1e-3      |  0.3774 |       0.3779 | underfits badly |
|   A2.5 | Partial unfreeze (last 2 layers)       | Unfreeze last 2 layers + head | 2e-5 / 1e-3   |  0.6134 |       0.6108 | big gain vs A2  |
|     B2 | Full fine-tuning, LR=3e-5              | Full FT                       | 3e-5 / 3e-5   |  0.7759 |       0.7701 | best overall    |
|    B3* | Full fine-tuning, LR=5e-5 (optional)   | Full FT                       | 5e-5 / 5e-5   |         |              | optional        |

### Final Test Results

| Exp ID | Model (final)               | Test Acc | Test Macro-F1 | Notes |
| -----: | --------------------------- | -------: | ------------: | ----- |
|     A1 | Full fine-tuning (baseline) |   0.7580 |        0.7512 | baseline |
|     A2 | Head-only (frozen encoder)  |   0.3880 |        0.3888 | underfits |
|   A2.5 | Partial unfreeze (last 2)   |   0.6323 |        0.6270 | improves vs A2 |
|     B2 | Full fine-tuning (LR=3e-5)  |   0.7722 |        0.7636 | **best** |

**Recommended final model:** **B2 (Full fine-tuning, LR=3e-5)**

---

## Key Takeaways
- **Head-only training underfits** severely on this dataset, indicating the classifier head alone cannot adapt the pretrained representation sufficiently.
- **Partial unfreezing** provides a large gain over head-only, showing that adapting the last layers helps capture task-specific signals.
- **Full fine-tuning performs best**, and learning rate is a meaningful lever (3e-5 > 2e-5 in this setup).

---

## Repository Structure

```text
.
├── .claude/
├── data/
├── notebooks/
│   ├── 05_error_analysis.ipynb
│   └── 06_bert_error_analysis.ipynb
├── src/
│   ├── model/
│   │   └── __init__.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── features.py
│   ├── text_preprocess.py
│   └── train_nn.py
├── .gitignore
├── BERT_IMPROVEMENTS.md
├── CODE_REVIEW.md
├── ISSUES_FOUND.md
├── README.md
├── environment.yml
├── fix_environment.sh
└── requirements.txt

Setup
Option A — Conda
conda env create -f environment.yml
conda activate <your-env-name>

Option B — pip + venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


If you hit environment issues:

bash fix_environment.sh

How to Run
1) Notebook workflow (recommended)

This repository is notebook-driven for experimentation and analysis.

jupyter lab
# or
jupyter notebook


Suggested order:

notebooks/05_error_analysis.ipynb — evaluation utilities + error analysis workflow

notebooks/06_bert_error_analysis.ipynb — BERT fine-tuning focused error analysis

2) Script workflow (src/)

If you prefer running scripts directly, the main entry points are:

src/train_nn.py — training / fine-tuning

src/evaluate.py — evaluation (metrics, confusion matrix, reports)

Example (template — adjust arguments based on your implementation):

python -m src.train_nn
python -m src.evaluate


If your scripts require arguments (model name, LR, epochs, etc.), check the docstrings or the top of each file and mirror the parameters used in the notebooks.

Evaluation Artifacts (Recommended Outputs)

For a resume-grade ML project, the following artifacts are typically included:

Confusion Matrix (test)

Classification report (per-class Precision / Recall / F1)

Per-class F1 bar chart

Error analysis: representative misclassified samples + failure mode summary

These are commonly generated in notebooks/05_* and notebooks/06_*.

Documentation Notes

BERT_IMPROVEMENTS.md — experiment ideas, tuning notes, and improvement plan

CODE_REVIEW.md — code review notes and refactor decisions

ISSUES_FOUND.md — known issues/pitfalls and fixes





