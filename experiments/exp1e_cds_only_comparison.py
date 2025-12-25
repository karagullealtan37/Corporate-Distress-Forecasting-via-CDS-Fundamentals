"""
EXPERIMENT 1E: CDS-Only Model and Three-Way Benchmark Comparison

Goal: Train XGBoost Strong Regularization using ONLY CDS-derived features,
      then compare three models to isolate sources of predictive gain:
      1. Naive CDS Threshold (heuristic baseline)
      2. XGBoost CDS-Only (ML on same information)
      3. XGBoost Top 10 SHAP (ML with fundamentals)

This comparison answers:
    - Does ML extract more from CDS data than heuristics?
    - Do fundamentals add value beyond CDS?

Outputs:
    - Model: output/experiments/models/xgboost_cds_only.pkl
    - Results: output/experiments/cds_only_comparison_results.csv
    - Figure: report/figures/experiments/three_way_roc_comparison.png
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

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve
import xgboost as xgb

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


def load_data():
    """Load preprocessed train/test data."""
    print("Loading train/test data...")
    
    # Load datasets
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    # Load feature list
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    all_features = feature_list['feature'].tolist()
    
    # Load preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Test: {test_df.shape}")
    print(f"  ✓ All features: {len(all_features)}\n")
    
    return train_df, test_df, all_features, imputer, scaler


def get_cds_features(all_features):
    """Extract CDS-derived features only."""
    cds_features = [f for f in all_features if 'cds' in f.lower()]
    
    print(f"CDS-derived features ({len(cds_features)}):")
    for f in cds_features:
        print(f"  - {f}")
    print()
    
    return cds_features


def prepare_data(train_df, test_df, features):
    """Prepare train/test data with selected features."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Extract features and target
    X_train = train_df[features].copy()
    y_train = train_df['distress_flag'].copy()
    X_test = test_df[features].copy()
    y_test = test_df['distress_flag'].copy()
    
    # Fit new preprocessors on selected features only
    imputer_new = SimpleImputer(strategy='median')
    scaler_new = StandardScaler()
    
    # Apply preprocessing
    X_train_imputed = imputer_new.fit_transform(X_train)
    X_test_imputed = imputer_new.transform(X_test)
    
    X_train = pd.DataFrame(
        scaler_new.fit_transform(X_train_imputed),
        columns=features,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler_new.transform(X_test_imputed),
        columns=features,
        index=X_test.index
    )
    
    return X_train, X_test, y_train, y_test


def train_cds_only_model(X_train, y_train):
    """Train Strong Regularization XGBoost with CDS features only."""
    print_section("TRAINING CDS-ONLY XGBOOST MODEL")
    
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
    
    print("Training XGBoost with Strong Regularization...")
    print(f"  Features: CDS-derived only ({X_train.shape[1]} features)")
    print(f"  max_depth: {params['max_depth']}")
    print(f"  n_estimators: {params['n_estimators']}")
    print(f"  learning_rate: {params['learning_rate']}")
    print()
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully\n")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance."""
    # Train predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'model': model_name,
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'y_test_proba': y_test_proba
    }
    
    results['auc_gap'] = results['train_auc'] - results['test_auc']
    
    return results


def create_naive_cds_baseline(test_df):
    """Create naive CDS threshold baseline predictions."""
    # Use median CDS spread as threshold (from step14)
    cds_values = test_df['cds_spread_lag1'].fillna(test_df['cds_spread_lag1'].median())
    median_cds = cds_values.median()
    
    # Predict distress if CDS > median
    y_pred_naive = (cds_values > median_cds).astype(int)
    y_proba_naive = cds_values / cds_values.max()
    
    return y_pred_naive, y_proba_naive, median_cds


def load_top10_model_results(test_df):
    """Load predictions from Top 10 SHAP model."""
    print("Loading Top 10 SHAP model...")
    
    # Load model
    model_file = EXP_MODELS_DIR / 'xgboost_strong_reg_top10_shap.pkl'
    
    if not model_file.exists():
        print("⚠️  Top 10 model not found. Run exp1c first.")
        return None
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load top features
    shap_df = pd.read_csv(OUTPUT_DIR / 'shap_values_xgboost.csv')
    top_features = shap_df.nlargest(10, 'mean_abs_shap')['feature'].tolist()
    
    # Load preprocessors
    with open(MODELS_DIR / 'imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare test data
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    all_features = feature_list['feature'].tolist()
    feature_indices = [all_features.index(f) for f in top_features if f in all_features]
    
    X_test_full = test_df[all_features].copy()
    X_test_preprocessed = scaler.transform(imputer.transform(X_test_full))
    X_test = X_test_preprocessed[:, feature_indices]
    
    # Predict
    y_proba_top10 = model.predict_proba(X_test)[:, 1]
    
    print(f"  ✓ Loaded Top 10 model ({len(top_features)} features)\n")
    
    return y_proba_top10


def plot_three_way_roc_comparison(y_test, y_proba_naive, y_proba_cds, y_proba_top10, 
                                   auc_naive, auc_cds, auc_top10):
    """Create three-way ROC curve comparison."""
    print_section("GENERATING THREE-WAY ROC COMPARISON")
    
    plt.close('all')
    plt.clf()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: ROC Curves
    # Naive CDS
    fpr_naive, tpr_naive, _ = roc_curve(y_test, y_proba_naive)
    ax1.plot(fpr_naive, tpr_naive, linewidth=2.5, label=f'Naive CDS Threshold (AUC={auc_naive:.3f})',
             color='red', linestyle='--', alpha=0.8)
    
    # CDS-Only ML
    fpr_cds, tpr_cds, _ = roc_curve(y_test, y_proba_cds)
    ax1.plot(fpr_cds, tpr_cds, linewidth=2.5, label=f'XGBoost CDS-Only (AUC={auc_cds:.3f})',
             color='orange', alpha=0.8)
    
    # Top 10 SHAP
    if y_proba_top10 is not None:
        fpr_top10, tpr_top10, _ = roc_curve(y_test, y_proba_top10)
        ax1.plot(fpr_top10, tpr_top10, linewidth=2.5, label=f'XGBoost Top 10 (AUC={auc_top10:.3f})',
                 color='green', alpha=0.8)
    
    # Random classifier
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random (AUC=0.500)')
    
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax1.set_title('ROC Curves: Three-Way Model Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: AUC Comparison Bar Chart
    models = ['Naive CDS\nThreshold', 'XGBoost\nCDS-Only', 'XGBoost\nTop 10']
    aucs = [auc_naive, auc_cds, auc_top10 if y_proba_top10 is not None else 0]
    colors = ['red', 'orange', 'green']
    
    bars = ax2.bar(models, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement percentages
    if y_proba_top10 is not None:
        # CDS-Only vs Naive
        imp1 = ((auc_cds - auc_naive) / auc_naive * 100)
        ax2.text(1, auc_cds/2, f'+{imp1:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Top 10 vs Naive
        imp2 = ((auc_top10 - auc_naive) / auc_naive * 100)
        ax2.text(2, auc_top10/2, f'+{imp2:.1f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax2.set_ylabel('AUC', fontweight='bold', fontsize=12)
    ax2.set_title('Out-of-Sample AUC Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim([0, 0.8])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.4, linewidth=1.5, label='Random')
    
    plt.tight_layout()
    
    # Save
    output_file = EXP_FIGURES_DIR / 'three_way_roc_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def print_comparison_summary(results_naive, results_cds, results_top10):
    """Print detailed comparison summary."""
    print_section("THREE-WAY COMPARISON SUMMARY")
    
    print("Out-of-Sample Performance (Test Set 2021-2023):")
    print("-" * 90)
    print(f"{'Model':<30s} {'AUC':<12s} {'F1':<12s} {'Precision':<12s} {'Recall':<12s}")
    print("-" * 90)
    
    print(f"{'1. Naive CDS Threshold':<30s} {results_naive['auc']:<12.4f} {results_naive['f1']:<12.4f} "
          f"{results_naive['precision']:<12.4f} {results_naive['recall']:<12.4f}")
    
    print(f"{'2. XGBoost CDS-Only':<30s} {results_cds['test_auc']:<12.4f} {results_cds['test_f1']:<12.4f} "
          f"{results_cds['test_precision']:<12.4f} {results_cds['test_recall']:<12.4f}")
    
    if results_top10:
        print(f"{'3. XGBoost Top 10 SHAP':<30s} {results_top10['auc']:<12.4f} {results_top10['f1']:<12.4f} "
              f"{results_top10['precision']:<12.4f} {results_top10['recall']:<12.4f}")
    
    print("-" * 90)
    
    # Calculate improvements
    print("\nIMPROVEMENT ANALYSIS:")
    print("-" * 90)
    
    # CDS-Only vs Naive
    auc_imp1 = ((results_cds['test_auc'] - results_naive['auc']) / results_naive['auc'] * 100)
    f1_imp1 = ((results_cds['test_f1'] - results_naive['f1']) / results_naive['f1'] * 100)
    
    print(f"\nXGBoost CDS-Only vs Naive Threshold:")
    print(f"  AUC improvement:  {results_cds['test_auc'] - results_naive['auc']:+.4f} ({auc_imp1:+.1f}%)")
    print(f"  F1 improvement:   {results_cds['test_f1'] - results_naive['f1']:+.4f} ({f1_imp1:+.1f}%)")
    print(f"  → Gain from MODEL SOPHISTICATION (same CDS information)")
    
    if results_top10:
        # Top 10 vs CDS-Only
        auc_imp2 = ((results_top10['auc'] - results_cds['test_auc']) / results_cds['test_auc'] * 100)
        f1_imp2 = ((results_top10['f1'] - results_cds['test_f1']) / results_cds['test_f1'] * 100)
        
        print(f"\nXGBoost Top 10 vs CDS-Only:")
        print(f"  AUC improvement:  {results_top10['auc'] - results_cds['test_auc']:+.4f} ({auc_imp2:+.1f}%)")
        print(f"  F1 improvement:   {results_top10['f1'] - results_cds['test_f1']:+.4f} ({f1_imp2:+.1f}%)")
        print(f"  → Gain from ADDING FUNDAMENTALS")
        
        # Top 10 vs Naive (total)
        auc_imp_total = ((results_top10['auc'] - results_naive['auc']) / results_naive['auc'] * 100)
        f1_imp_total = ((results_top10['f1'] - results_naive['f1']) / results_naive['f1'] * 100)
        
        print(f"\nXGBoost Top 10 vs Naive (Total Improvement):")
        print(f"  AUC improvement:  {results_top10['auc'] - results_naive['auc']:+.4f} ({auc_imp_total:+.1f}%)")
        print(f"  F1 improvement:   {results_top10['f1'] - results_naive['f1']:+.4f} ({f1_imp_total:+.1f}%)")
    
    print("-" * 90)
    
    # Key insight
    print("\n✅ KEY INSIGHT:")
    print(f"   Model sophistication accounts for {auc_imp1:.1f}% AUC improvement")
    if results_top10:
        print(f"   Fundamentals add incremental {auc_imp2:.1f}% AUC improvement")
        print(f"   → ML extracts {auc_imp1/(auc_imp1+auc_imp2)*100:.0f}% of total gain from CDS data alone")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("EXPERIMENT 1E: CDS-ONLY MODEL & THREE-WAY COMPARISON".center(80))
    print("="*80)
    
    # Load data
    train_df, test_df, all_features, imputer, scaler = load_data()
    
    # Get CDS features
    cds_features = get_cds_features(all_features)
    
    # Prepare CDS-only data
    X_train_cds, X_test_cds, y_train, y_test = prepare_data(
        train_df, test_df, cds_features
    )
    
    # Train CDS-only model
    model_cds = train_cds_only_model(X_train_cds, y_train)
    
    # Evaluate CDS-only model
    print_section("EVALUATING CDS-ONLY MODEL")
    results_cds = evaluate_model(model_cds, X_train_cds, y_train, X_test_cds, y_test, 
                                  'XGBoost CDS-Only')
    
    print(f"CDS-Only Model Performance:")
    print(f"  Train AUC: {results_cds['train_auc']:.4f}")
    print(f"  Test AUC:  {results_cds['test_auc']:.4f}")
    print(f"  AUC Gap:   {results_cds['auc_gap']:.4f}")
    print(f"  Test F1:   {results_cds['test_f1']:.4f}")
    
    # Save CDS-only model
    model_file = EXP_MODELS_DIR / 'xgboost_cds_only.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_cds, f)
    print(f"\n✓ Model saved: {model_file}")
    
    # Create naive baseline
    print_section("CREATING NAIVE CDS BASELINE")
    y_pred_naive, y_proba_naive, median_cds = create_naive_cds_baseline(test_df)
    
    results_naive = {
        'model': 'Naive CDS Threshold',
        'auc': roc_auc_score(y_test, y_proba_naive),
        'f1': f1_score(y_test, y_pred_naive),
        'precision': precision_score(y_test, y_pred_naive),
        'recall': recall_score(y_test, y_pred_naive),
        'threshold': median_cds
    }
    
    print(f"Naive CDS Threshold: {median_cds:.2f} bps")
    print(f"  Test AUC: {results_naive['auc']:.4f}")
    print(f"  Test F1:  {results_naive['f1']:.4f}")
    
    # Load Top 10 model results
    print_section("LOADING TOP 10 SHAP MODEL")
    y_proba_top10 = load_top10_model_results(test_df)
    
    results_top10 = None
    if y_proba_top10 is not None:
        y_pred_top10 = (y_proba_top10 > 0.5).astype(int)
        results_top10 = {
            'model': 'XGBoost Top 10 SHAP',
            'auc': roc_auc_score(y_test, y_proba_top10),
            'f1': f1_score(y_test, y_pred_top10),
            'precision': precision_score(y_test, y_pred_top10),
            'recall': recall_score(y_test, y_pred_top10)
        }
        print(f"Top 10 Model Performance:")
        print(f"  Test AUC: {results_top10['auc']:.4f}")
        print(f"  Test F1:  {results_top10['f1']:.4f}")
    
    # Create three-way ROC comparison
    plot_three_way_roc_comparison(
        y_test, y_proba_naive, results_cds['y_test_proba'], y_proba_top10,
        results_naive['auc'], results_cds['test_auc'], 
        results_top10['auc'] if results_top10 else 0
    )
    
    # Print comparison summary
    print_comparison_summary(results_naive, results_cds, results_top10)
    
    # Save results
    comparison_results = pd.DataFrame([
        {
            'model': 'Naive CDS Threshold',
            'n_features': 1,
            'test_auc': results_naive['auc'],
            'test_f1': results_naive['f1'],
            'test_precision': results_naive['precision'],
            'test_recall': results_naive['recall']
        },
        {
            'model': 'XGBoost CDS-Only',
            'n_features': len(cds_features),
            'test_auc': results_cds['test_auc'],
            'test_f1': results_cds['test_f1'],
            'test_precision': results_cds['test_precision'],
            'test_recall': results_cds['test_recall']
        }
    ])
    
    if results_top10:
        comparison_results = pd.concat([comparison_results, pd.DataFrame([{
            'model': 'XGBoost Top 10 SHAP',
            'n_features': 10,
            'test_auc': results_top10['auc'],
            'test_f1': results_top10['f1'],
            'test_precision': results_top10['precision'],
            'test_recall': results_top10['recall']
        }])], ignore_index=True)
    
    results_file = EXP_OUTPUT_DIR / 'cds_only_comparison_results.csv'
    comparison_results.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print(f"\nModel: {model_file}")
    print(f"Results: {results_file}")
    print(f"Figure: {EXP_FIGURES_DIR / 'three_way_roc_comparison.png'}\n")


if __name__ == "__main__":
    main()
