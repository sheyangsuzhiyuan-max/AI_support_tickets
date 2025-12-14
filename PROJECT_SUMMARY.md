# Project Summary

## 📚 CA6000 Assignment - AI Support Tickets Classification

### Two Entry Points

#### 1. Assignment Mode (作业)
```bash
python run_assignment.py
```
- **Purpose:** Generate complete CA6000 report
- **Output:** `CA6000_Assignment_Report.md`
- **Time:** ~5 minutes
- **GPU:** Not required
- **Auto-cleanup:** Yes

#### 2. Personal Research Mode (个人项目)
```bash
python run_personal_project.py
```
- **Purpose:** BERT fine-tuning experiments
- **Output:** `data/bert_experiments_*.csv`
- **Time:** ~2-3 hours
- **GPU:** Required
- **Experiments:** 5 different configurations

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run assignment (generates report)
python run_assignment.py

# 3. Check report
cat CA6000_Assignment_Report.md
```

---

## What's Changed

✅ **Cleaned Up:**
- Removed redundant documentation files
- Single README with clear instructions
- Two focused entry points

✅ **Fixed:**
- Added seaborn to dependencies
- Auto-cleanup after report generation
- Online BERT model loading

✅ **Features:**
- Assignment report auto-generation
- Systematic BERT experiments
- No GPU needed for assignment
- Comprehensive evaluation metrics

---

## File Structure

```
.
├── run_assignment.py          # 📚 Assignment entry
├── run_personal_project.py    # 🔬 Research entry
├── README.md                  # Complete documentation
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
└── src/                       # Source code
```

---

## Next Steps

### For Assignment Submission
1. Run `python run_assignment.py`
2. Review `CA6000_Assignment_Report.md`
3. Submit report + code

### For Personal Research
1. Upload to server with GPU
2. Run `python run_personal_project.py`
3. Check results in `data/bert_experiments_summary_*.csv`

---

**Documentation:** See [README.md](README.md) for full details
