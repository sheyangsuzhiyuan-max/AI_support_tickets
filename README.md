# AI Support Tickets Classification

This NLP project classifies customer support tickets into **3 priority classes** (high, medium, low) using multiple approaches from traditional ML to state-of-the-art transformers.

**Two Project Modes:**
1. **📚 CA6000 Assignment** - Complete report generation for coursework
2. **🔬 BERT Fine-tuning** - Hyperparameter experiments

---

## Quick Start

### Setup Environment

```bash
# Option A - Conda
conda env create -f environment.yml
conda activate <your-env-name>

# Option B - pip + venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📚 CA6000 Assignment Mode

### Purpose
Generate a complete assignment report comparing Logistic Regression, TextCNN, and BERT models.

### Run Assignment

```bash
# One-click: automatically trains models on first run, then generates report
python run_assignment.py

# Quick test mode
python run_assignment.py --quick
```

**First Run:** Will automatically train all models (~40 minutes total):
- Logistic Regression (<1 min)
- TextCNN (~10 min)
- BERT (~30 min)

**Subsequent Runs:** Uses existing models, generates report in seconds.

### What It Does

**Automated Report Generation:**
1. ✅ **Data Import & Inspection** - Load 28K support tickets, show samples
2. ✅ **Data Cleaning** - Error detection, text preprocessing, quality checks
3. ✅ **Statistical Analysis** - Mean/median/variance, class distribution, text length stats
4. ✅ **Model Training** - Logistic Regression, TextCNN, BERT
5. ✅ **Evaluation** - Accuracy, F1, confusion matrices, error analysis
6. ✅ **AI Assistant Usage** - Document how Claude/AI tools were used
7. ✅ **Conclusion** - Summary of findings and recommendations

**Output:**
- `CA6000_Assignment_Report.md` - Complete markdown report (ready to submit)
- Console output with progress and key metrics

**Report Includes:**
- ✓ Dataset source and import process
- ✓ Error checking and cleaning steps
- ✓ Statistical summary (mean, variance, distribution)
- ✓ Three neural network models (LogReg, CNN, BERT)
- ✓ Training process and evaluation
- ✓ Final accuracy and performance metrics
- ✓ AI coding assistant usage documentation

### Assignment Requirements Checklist

Per CA6000 specification:

- ✅ Source of dataset and import method
- ✅ Error checking and detection (NaN, outliers, value errors)
- ✅ Data cleaning with Pandas functions
- ✅ Summary statistics (mean/median, variance)
- ✅ Neural network models for prediction
- ✅ Training process and evaluation
- ✅ Model accuracy and performance
- ✅ AI assistant usage summary

**Due Date:** 31-December-2025

---

## 🔬 BERT Fine-tuning Mode

### Purpose
Explore BERT fine-tuning hyperparameters for optimal performance.

### Run BERT Fine-tuning

```bash
# Run all BERT fine-tuning experiments
python run_bert_finetuning.py

# Quick test (1 epoch)
python run_bert_finetuning.py --quick

# Run specific experiments only
python run_bert_finetuning.py --exp EXP1 EXP2
```

### Experiment Configurations

| ID | Description | LR | Epochs | Freeze |
|----|-------------|----|----- --|--------|
| EXP1 | Baseline | 2e-5 | 3 | No |
| EXP2 | Higher LR | 3e-5 | 3 | No |
| EXP3 | Highest LR | 5e-5 | 3 | No |
| EXP4 | Frozen encoder | 2e-5 | 3 | Yes |
| EXP5 | More epochs | 3e-5 | 5 | No |

### Output Files

- `data/bert_experiments_<timestamp>.json` - Detailed results for all epochs
- `data/bert_experiments_summary_<timestamp>.csv` - Summary table
- Console: Real-time progress and best configuration

### Features

- ✅ **No Model Saving** - Only records metrics (saves disk space)
- ✅ **Automatic Comparison** - Ranks experiments by validation F1
- ✅ **GPU Memory Management** - Clears cache between experiments
- ✅ **Progress Tracking** - tqdm progress bars for training
- ✅ **Structured Logging** - JSON format for easy analysis

---

## Project Structure

```
.
├── run_assignment.py              # 📚 CA6000 Assignment entry point
├── run_bert_finetuning.py         # 🔬 BERT fine-tuning entry point
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_baseline_ml.ipynb       # Logistic regression baseline
│   ├── 03_cnn_model.ipynb         # TextCNN model
│   ├── 04_bert_model.ipynb        # BERT training (for assignment)
│   └── 05_bert_finetuning.ipynb   # BERT fine-tuning experiments
│
├── src/
│   ├── model/
│   │   ├── bert_model.py          # BERT implementation (online loading)
│   │   ├── text_cnn.py            # TextCNN implementation
│   │   ├── baseline_logreg.joblib # Saved LogReg model
│   │   ├── textcnn.pt             # Saved CNN checkpoint
│   │   └── bert_finetuned.pt      # Saved BERT checkpoint
│   ├── data_utils.py              # Data loading utilities
│   ├── text_preprocess.py         # Text cleaning functions
│   ├── features.py                # TF-IDF feature extraction
│   └── evaluate.py                # Evaluation metrics
│
├── data/
│   ├── raw/                       # Original data
│   └── processed/                 # Processed splits
│
├── cleanup.sh                     # Clean local models and temp files
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Key Features

### 1. Online Model Loading ☁️
- BERT models load from **HuggingFace Hub** (no local downloads needed)
- First run downloads and caches to `~/.cache/huggingface/`
- Saves ~512MB disk space
- Easier server deployment

### 2. Two Clear Entry Points 🚪

**Assignment Mode (`run_assignment.py`):**
- Generates complete CA6000 report
- Evaluates existing trained models
- Documents all required sections
- Ready for submission

**BERT Fine-tuning Mode (`run_bert_finetuning.py`):**
- Runs systematic BERT experiments
- Compares different hyperparameters
- Records results without saving models
- Optimizes for best configuration

### 3. Comprehensive Evaluation 📊
- Accuracy, F1 (Macro/Weighted), Precision, Recall
- Per-class metrics
- Confusion matrices
- Error analysis
- Model comparison tables

---

## Model Performance

### Assignment Models (Test Set)

| Model | Accuracy | F1 Macro | F1 Weighted | Notes |
|-------|----------|----------|-------------|-------|
| Logistic Regression | ~64% | ~64% | ~65% | TF-IDF baseline |
| TextCNN | ~61% | ~55% | ~58% | Needs tuning |
| BERT (DistilBERT) | **~75-77%** | **~75-76%** | **~76%** | Best overall |

### BERT Fine-tuning Results (Validation Set)

| Exp | Config | Val Acc | Val F1 | Notes |
|-----|--------|---------|--------|-------|
| B2 | LR=3e-5, Full FT, 3 epochs | 0.7759 | 0.7701 | **Recommended** |
| A1 | LR=2e-5, Full FT, 3 epochs | 0.7521 | 0.7466 | Baseline |
| A2.5 | Partial unfreeze, 2e-5 | 0.6134 | 0.6108 | Better than frozen |
| A2 | Frozen encoder | 0.3774 | 0.3779 | Underfits |

**Key Finding:** Full fine-tuning with LR=3e-5 achieves best results.

---

## Usage Examples

### For Assignment Submission

```bash
# 1. Generate report
python run_assignment.py

# 2. Review report
cat CA6000_Assignment_Report.md

# 3. (Optional) Convert to PDF
pandoc CA6000_Assignment_Report.md -o CA6000_Report.pdf

# 4. Submit report + code
```

### For BERT Fine-tuning

```bash
# On server with GPU
python run_bert_finetuning.py

# Check results
cat data/bert_experiments_summary_*.csv

# Run more experiments with best LR
python run_bert_finetuning.py --exp EXP2 EXP5
```

### For Interactive Exploration

```bash
# Jupyter notebook for assignment (BERT training)
jupyter notebook notebooks/04_bert_model.ipynb

# Jupyter notebook for BERT fine-tuning experiments
jupyter notebook notebooks/05_bert_finetuning.ipynb
```

---

## Cleanup

Remove local model files and temporary data:

```bash
./cleanup.sh
```

**Deletes:**
- Local BERT model folders (~512MB)
- Temporary CSV files
- Python cache
- Jupyter checkpoints

**Keeps:**
- Trained model checkpoints (*.pt, *.joblib)
- Source code
- Data files

---

## Server Deployment

### For Assignment

```bash
# Upload to server
scp -r . user@server:~/project/

# Run on server (one command!)
ssh user@server
cd ~/project
python run_assignment.py

# Download report
exit
scp user@server:~/project/CA6000_Assignment_Report.md .
```

**Note:** First run will take ~40 minutes to train all models automatically.

### For BERT Fine-tuning

```bash
# Run experiments in background
nohup python run_bert_finetuning.py > experiments.log 2>&1 &

# Monitor progress
tail -f experiments.log

# Download results
scp user@server:~/project/data/bert_experiments_*.csv .
```

---

## Key Takeaways

### From Assignment

1. **Data Quality Matters** - Clean, consistent data is crucial
2. **Baseline First** - LogReg provides surprisingly strong results
3. **Progressive Complexity** - Start simple, add complexity only if needed
4. **Evaluation Rigor** - Use multiple metrics, not just accuracy

### From BERT Fine-tuning

1. **Full Fine-tuning > Frozen** - BERT benefits from end-to-end training
2. **Learning Rate is Critical** - 3e-5 > 2e-5 for this task
3. **Diminishing Returns** - More epochs don't always help
4. **Pre-training Advantage** - Transfer learning provides huge boost

---

---

## Requirements

**Core:**
- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- Scikit-learn
- Pandas, NumPy

**For Assignment:**
- No GPU required (uses pre-trained models)
- ~8GB RAM
- ~5 minutes runtime

**For BERT Fine-tuning:**
- GPU strongly recommended (CUDA)
- ~16GB RAM
- ~2-3 hours for all experiments

See [requirements.txt](requirements.txt) for full dependencies.

---

## License

This is an educational project for CA6000 coursework.

---

## Notes

### Assignment Mode
- ✅ Uses existing trained models (no GPU needed)
- ✅ Generates markdown report automatically
- ✅ Covers all CA6000 requirements
- ✅ Ready for submission

### BERT Fine-tuning Mode
- ✅ Trains new models (GPU recommended)
- ✅ Records experiments systematically
- ✅ Doesn't save checkpoints (saves space)
- ✅ Finds optimal hyperparameters

---

**Happy learning! 🚀**

**For Questions:**
- Assignment: Review `CA6000_Assignment_Report.md` after running
- BERT Fine-tuning: Check `data/bert_experiments_summary_*.csv` for results
