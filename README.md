# CDS-Based Corporate Distress Prediction

## Research Question

Does machine learning extract incremental predictive value from CDS data beyond simple heuristic rules?

This project investigates whether modern machine learning techniques can predict corporate financial distress 12 months in advance using Credit Default Swap (CDS) spreads, accounting fundamentals, and equity market data. We compare a naive CDS-based benchmark to supervised machine learning models (XGBoost, LightGBM) to isolate gains from model sophistication versus information expansion.

**Key Finding:** Model sophistication accounts for 77% of performance gains, while information expansion contributes 23%.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/karagullealtan37/Corporate-Distress-Forecasting-via-CDS-Fundamentals.git
cd Corporate-Distress-Forecasting-via-CDS-Fundamentals
```

The repository includes all data files, code, and dependencies.

### 2. Install Dependencies

**Required:** Python 3.11+ (tested with Python 3.11.2)

**Option A: Using pip (recommended for local development)**
```bash
pip3 install -r requirements.txt
```

**Option B: Using conda (recommended for Nuvolos/EPFL)**
```bash
conda env create -f environment.yml
conda activate cds-distress-prediction
```

---

## Usage

### Run the Complete Pipeline

```bash
python3 main.py
```

This executes:
1. **Pipeline Steps (1-15):** Data processing, feature engineering, model training, evaluation (~8-10 min)
2. **XGBoost Experiments (5 total):** Overfitting reduction, feature selection, cross-validation (~2-5 min)

**Total Runtime:** ~10-15 minutes on MacBook Pro M1

**Note:** Experiments reuse preprocessed data from the pipeline, which is why they run faster. All models are trained from scratch each time.

### Alternative Commands

```bash
# Run only pipeline steps (no experiments)
python3 main.py --pipeline-only

# Run only XGBoost experiments (requires pipeline outputs)
python3 main.py --experiments-only

# Show help
python3 main.py --help
```

---

## Expected Output

After running `python3 main.py`, you will find:

### Models
- `output/models/xgboost_optimized.pkl` - Pipeline XGBoost model
- `output/experiments/models/xgboost_strong_reg_top10_shap.pkl` - **Recommended model** (Top 10 SHAP features)

### Results
- `output/step13_evaluation_results.csv` - Pipeline evaluation metrics
- `output/experiments/*.csv` - Experiment results

### Figures
- `report/figures/step14_model_comparison.png` - Model comparison
- `report/figures/step15_shap_summary_xgboost.png` - Feature importance
- `report/figures/experiments/xgboost_overfitting_comparison.png` - Overfitting journey
- `report/figures/experiments/three_way_roc_comparison.png` - Incremental gains
- `report/figures/experiments/three_way_cv_comparison.png` - Cross-validation

### Performance Summary

**Recommended Model (XGBoost Top 10 Features):**
- Test AUC: 0.636 (+58.2% vs naive baseline)
- Precision: 30.0%, Recall: 72.0%, F1: 0.420
- Features: 10 (66% reduction from 29)

**Key Finding:** Model sophistication (naive → XGBoost CDS-only) accounts for 77% of gains (+44.8%), while information expansion (CDS-only → Top 10) contributes 23% (+9.3%)

---

## Project Structure

```
CDS Project/
├── main.py                  # Main entry point (runs full pipeline)
├── README.md                # Reproducibility and execution instructions
├── requirements.txt         # pip-based Python dependencies
├── environment.yml          # Conda environment specification
│
├── src/                     # Core 15-step deterministic pipeline
│   ├── step1_data_inspection.py
│   ├── step2_data_quality.py
│   ├── step3_data_cleaning.py
│   ├── step4_data_merging.py
│   ├── step5_preprocessing.py
│   ├── ...
│   └── step15_explainability.py
│
├── experiments/             # Exploratory and diagnostic experiments
│   ├── exp1_reduce_overfitting.py
│   ├── exp1b_reduce_overfitting_xgboost.py
│   ├── exp1c_strong_reg_top10_shap.py
│   ├── ...
│   └── exp6_combined_optimization.py
│
├── data/                    # Processed datasets (Compustat, Capital IQ)
├── output/                  # Serialized models, preprocessors, figures
└── report/figures/          # Figures used in the final report
```

---

## Dependencies

**Python 3.11+** (tested with Python 3.11.2) with the following packages:
- pandas 2.0+
- numpy 1.24+
- scikit-learn 1.3+
- xgboost 2.0+
- lightgbm 4.0+
- shap 0.42+
- matplotlib 3.7+
- seaborn 0.12+

All dependencies are listed in `requirements.txt`.

**GitHub Repository:** https://github.com/karagullealtan37/Corporate-Distress-Forecasting-via-CDS-Fundamentals

---

## Reproducibility

- **Random seed:** All random operations use `random_state=42`
- **Fixed splits:** Train/test split (2010-2020 vs 2021-2023) is hardcoded
- **Serialized transformations:** Preprocessing fitted on train, applied to test
- **Runtime:** ~5-10 minutes tested on MacBook Pro M1

---

## Author

Altan Karagulle  
Master of Science in Finance (MScF)  
University of Lausanne

Advanced Programming (DSAP) - Final Project  
December 2025
