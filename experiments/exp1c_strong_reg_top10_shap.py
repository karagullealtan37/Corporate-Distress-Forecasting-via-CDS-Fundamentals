"""
EXPERIMENT 1C: Strong Regularization XGBoost with Top 10 SHAP Features

Goal: Train the Strong Regularization XGBoost model using only the top 10
      most important features identified by SHAP analysis.

Strategy:
    1. Load SHAP feature importance from previous analysis
    2. Select top 10 features
    3. Train Strong Regularization XGBoost with these features
    4. Compare performance to full model

Outputs:
    - Model: output/experiments/models/xgboost_strong_reg_top10_shap.pkl
    - Results: output/experiments/strong_reg_top10_shap_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_MODELS_DIR = EXP_OUTPUT_DIR / 'models'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Create directories
EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_top_shap_features(n_features=10):
    """
    Load top N features from SHAP analysis.
    """
    print(f"Loading top {n_features} SHAP features...")
    
    # Load SHAP values
    shap_df = pd.read_csv(OUTPUT_DIR / 'shap_values_xgboost.csv')
    
    # Get top N features by mean absolute SHAP value
    top_features = shap_df.nlargest(n_features, 'mean_abs_shap')['feature'].tolist()
    
    print(f"\nTop {n_features} SHAP Features:")
    print("-" * 60)
    for i, (_, row) in enumerate(shap_df.nlargest(n_features, 'mean_abs_shap').iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25s} (SHAP: {row['mean_abs_shap']:.4f})")
    print()
    
    return top_features


def load_data(features):
    """
    Load preprocessed train/test data with selected features.
    """
    print("Loading train/test data...")
    
    # Load datasets
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    # Load preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load full feature list to get indices
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    all_features = feature_list['feature'].tolist()
    
    # Get indices of selected features
    feature_indices = [all_features.index(f) for f in features if f in all_features]
    
    # Prepare data with all features first (for preprocessing)
    X_train_full = train_df[all_features].copy()
    y_train = train_df['distress_flag'].copy()
    X_test_full = test_df[all_features].copy()
    y_test = test_df['distress_flag'].copy()
    
    # Apply preprocessing to all features
    X_train_preprocessed = scaler.transform(imputer.transform(X_train_full))
    X_test_preprocessed = scaler.transform(imputer.transform(X_test_full))
    
    # Select only the top SHAP features
    X_train = pd.DataFrame(
        X_train_preprocessed[:, feature_indices],
        columns=features,
        index=X_train_full.index
    )
    X_test = pd.DataFrame(
        X_test_preprocessed[:, feature_indices],
        columns=features,
        index=X_test_full.index
    )
    
    print(f"  ✓ Train: {X_train.shape}")
    print(f"  ✓ Test: {X_test.shape}")
    print(f"  ✓ Features: {len(features)} (reduced from {len(all_features)})")
    print(f"  ✓ Train distress rate: {y_train.mean()*100:.1f}%")
    print(f"  ✓ Test distress rate: {y_test.mean()*100:.1f}%\n")
    
    return X_train, X_test, y_train, y_test


def train_strong_reg_model(X_train, y_train):
    """
    Train Strong Regularization XGBoost model.
    """
    print_section("TRAINING STRONG REGULARIZATION XGBOOST")
    
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ XGBoost not installed. Install with: pip install xgboost")
        return None
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}\n")
    
    # Strong Regularization parameters (from exp1b)
    params = {
        'n_estimators': 60,
        'max_depth': 3,
        'learning_rate': 0.03,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 10,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 3.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    print("Training with Strong Regularization parameters:")
    print(f"  max_depth: {params['max_depth']}")
    print(f"  n_estimators: {params['n_estimators']}")
    print(f"  learning_rate: {params['learning_rate']}")
    print(f"  gamma: {params['gamma']}")
    print(f"  reg_alpha: {params['reg_alpha']}")
    print(f"  reg_lambda: {params['reg_lambda']}")
    print()
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully\n")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance.
    """
    print_section("MODEL EVALUATION")
    
    # Train predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred)
    }
    
    results['auc_gap'] = results['train_auc'] - results['test_auc']
    results['f1_gap'] = results['train_f1'] - results['test_f1']
    
    # Print results
    print("PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Metric':<20s} {'Train':<12s} {'Test':<12s} {'Gap':<12s}")
    print("-" * 80)
    print(f"{'AUC':<20s} {results['train_auc']:<12.4f} {results['test_auc']:<12.4f} {results['auc_gap']:<12.4f}")
    print(f"{'F1 Score':<20s} {results['train_f1']:<12.4f} {results['test_f1']:<12.4f} {results['f1_gap']:<12.4f}")
    print(f"{'Precision':<20s} {results['train_precision']:<12.4f} {results['test_precision']:<12.4f}")
    print(f"{'Recall':<20s} {results['train_recall']:<12.4f} {results['test_recall']:<12.4f}")
    print(f"{'Accuracy':<20s} {results['train_accuracy']:<12.4f} {results['test_accuracy']:<12.4f}")
    print("-" * 80)
    
    return results


def compare_with_full_model(results):
    """
    Compare with full feature model from exp1b.
    """
    print_section("COMPARISON WITH FULL MODEL")
    
    print("Strong Regularization Performance:")
    print("-" * 80)
    print(f"{'Model':<30s} {'Test AUC':<12s} {'Test F1':<12s} {'AUC Gap':<12s}")
    print("-" * 80)
    print(f"{'Full Model (29 features)':<30s} {'0.6263':<12s} {'0.3930':<12s} {'0.0624':<12s}")
    print(f"{'Top 10 SHAP (this model)':<30s} {results['test_auc']:<12.4f} {results['test_f1']:<12.4f} {results['auc_gap']:<12.4f}")
    print("-" * 80)
    
    # Calculate differences
    auc_diff = results['test_auc'] - 0.6263
    f1_diff = results['test_f1'] - 0.3930
    gap_diff = results['auc_gap'] - 0.0624
    
    print(f"\nDifference (Top 10 SHAP - Full Model):")
    print(f"  Test AUC: {auc_diff:+.4f} ({auc_diff/0.6263*100:+.1f}%)")
    print(f"  Test F1:  {f1_diff:+.4f} ({f1_diff/0.3930*100:+.1f}%)")
    print(f"  AUC Gap:  {gap_diff:+.4f}")
    
    if abs(auc_diff) < 0.01:
        print("\n✅ EXCELLENT: Performance maintained with 65% fewer features!")
    elif auc_diff > -0.02:
        print("\n✅ GOOD: Minimal performance loss with significant feature reduction")
    else:
        print("\n⚠️  WARNING: Significant performance degradation")


def main():
    """
    Main execution.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1C: STRONG REG XGBOOST WITH TOP 10 SHAP FEATURES".center(80))
    print("="*80)
    
    # Load top SHAP features
    top_features = load_top_shap_features(n_features=10)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(top_features)
    
    # Train model
    model = train_strong_reg_model(X_train, y_train)
    
    if model is None:
        return
    
    # Evaluate
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Compare with full model
    compare_with_full_model(results)
    
    # Save model
    model_file = EXP_MODELS_DIR / 'xgboost_strong_reg_top10_shap.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved: {model_file}")
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df['model'] = 'Strong Reg Top 10 SHAP'
    results_df['n_features'] = len(top_features)
    results_file = EXP_OUTPUT_DIR / 'strong_reg_top10_shap_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✓ Results saved: {results_file}")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print(f"\nModel: {model_file}")
    print(f"Results: {results_file}")
    print(f"\nNext: Compare with other feature selection methods\n")


if __name__ == "__main__":
    main()
