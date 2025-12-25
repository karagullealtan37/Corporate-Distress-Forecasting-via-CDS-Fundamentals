"""
EXPERIMENT 1D: Cross-Validation for Final Model

Final Model: Strong Regularization XGBoost with Top 10 SHAP Features

Goal: Validate the final model using time-series cross-validation to ensure
      robust performance across different time periods.

Configuration:
    - Model: XGBoost with Strong Regularization
    - Features: Top 10 SHAP features
    - CV: 5-fold TimeSeriesSplit (preserves temporal order)
    
Outputs:
    - Results: output/experiments/final_model_cv_results.csv
    - Figure: report/figures/experiments/final_model_cv_performance.png
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score
)
import xgboost as xgb

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Create directories
EXP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_top_shap_features(n_features=10):
    """Load top N features from SHAP analysis."""
    shap_df = pd.read_csv(OUTPUT_DIR / 'shap_values_xgboost.csv')
    top_features = shap_df.nlargest(n_features, 'mean_abs_shap')['feature'].tolist()
    return top_features


def load_data():
    """Load training data."""
    print("Loading training data...")
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Date range: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  ✓ Distress rate: {train_df['distress_flag'].mean():.1%}\n")
    return train_df


def perform_time_series_cv(train_df, features, n_splits=5):
    """
    Perform time-series cross-validation.
    
    Uses TimeSeriesSplit which preserves temporal order:
    - Fold 1: Train on earliest data, validate on next period
    - Fold 2: Train on earliest + fold 1, validate on next period
    - etc.
    """
    print_section(f"TIME-SERIES CROSS-VALIDATION ({n_splits} FOLDS)")
    
    print(f"Model: Strong Regularization XGBoost")
    print(f"Features: Top {len(features)} SHAP features")
    print(f"CV Strategy: TimeSeriesSplit (preserves temporal order)")
    print()
    
    # Sort by date to ensure temporal order
    train_df = train_df.sort_values('date').reset_index(drop=True)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Storage for results
    cv_results = {
        'fold': [],
        'train_size': [],
        'val_size': [],
        'train_period': [],
        'val_period': [],
        'train_distress_rate': [],
        'val_distress_rate': [],
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Strong Regularization parameters
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
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}/{n_splits}".center(80))
        print(f"{'='*80}\n")
        
        # Split data
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        
        # Extract features and target
        X_train_cv = train_fold[features].copy()
        y_train_cv = train_fold['distress_flag'].copy()
        X_val_cv = val_fold[features].copy()
        y_val_cv = val_fold['distress_flag'].copy()
        
        dates_train = train_fold['date']
        dates_val = val_fold['date']
        
        # Print fold info
        train_period = f"{dates_train.min()} to {dates_train.max()}"
        val_period = f"{dates_val.min()} to {dates_val.max()}"
        
        print(f"Train period: {train_period}")
        print(f"Val period:   {val_period}")
        print(f"Train size:   {len(X_train_cv):,} observations")
        print(f"Val size:     {len(X_val_cv):,} observations")
        print(f"Train distress rate: {y_train_cv.mean():.1%}")
        print(f"Val distress rate:   {y_val_cv.mean():.1%}")
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train_cv)
        X_val_imputed = imputer.transform(X_val_cv)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        # Calculate scale_pos_weight for this fold
        scale_pos_weight = (y_train_cv == 0).sum() / (y_train_cv == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        # Train model
        print(f"\nTraining XGBoost (Strong Regularization)...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train_cv, verbose=False)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_val_cv, y_val_proba)
        accuracy = accuracy_score(y_val_cv, y_val_pred)
        precision = precision_score(y_val_cv, y_val_pred, zero_division=0)
        recall = recall_score(y_val_cv, y_val_pred)
        f1 = f1_score(y_val_cv, y_val_pred)
        
        # Store results
        cv_results['fold'].append(fold)
        cv_results['train_size'].append(len(X_train_cv))
        cv_results['val_size'].append(len(X_val_cv))
        cv_results['train_period'].append(train_period)
        cv_results['val_period'].append(val_period)
        cv_results['train_distress_rate'].append(y_train_cv.mean())
        cv_results['val_distress_rate'].append(y_val_cv.mean())
        cv_results['auc'].append(auc)
        cv_results['accuracy'].append(accuracy)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        cv_results['f1'].append(f1)
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"  AUC:       {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
    
    return pd.DataFrame(cv_results)


def plot_cv_results(results_df):
    """Create visualization of CV results."""
    print_section("GENERATING VISUALIZATIONS")
    
    plt.close('all')
    plt.clf()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Metrics by Fold
    ax1 = fig.add_subplot(gs[0, :])
    
    x = results_df['fold']
    width = 0.15
    x_pos = np.arange(len(x))
    
    ax1.bar(x_pos - 2*width, results_df['auc'], width, label='AUC', alpha=0.8, color='steelblue')
    ax1.bar(x_pos - width, results_df['f1'], width, label='F1', alpha=0.8, color='darkorange')
    ax1.bar(x_pos, results_df['precision'], width, label='Precision', alpha=0.8, color='green')
    ax1.bar(x_pos + width, results_df['recall'], width, label='Recall', alpha=0.8, color='red')
    ax1.bar(x_pos + 2*width, results_df['accuracy'], width, label='Accuracy', alpha=0.8, color='purple')
    
    ax1.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title('Cross-Validation Performance by Fold', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Fold {i}' for i in x])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Add mean line
    for metric, color in [('auc', 'steelblue'), ('f1', 'darkorange')]:
        mean_val = results_df[metric].mean()
        ax1.axhline(y=mean_val, color=color, linestyle='--', alpha=0.5, linewidth=2)
    
    # Plot 2: AUC Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.plot(results_df['fold'], results_df['auc'], marker='o', linewidth=2, 
             markersize=10, color='steelblue', label='AUC')
    ax2.axhline(y=results_df['auc'].mean(), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean: {results_df["auc"].mean():.4f}')
    
    ax2.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax2.set_ylabel('AUC', fontweight='bold', fontsize=12)
    ax2.set_title('AUC Across Folds', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 0.75])
    
    # Plot 3: F1 Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.plot(results_df['fold'], results_df['f1'], marker='s', linewidth=2,
             markersize=10, color='darkorange', label='F1')
    ax3.axhline(y=results_df['f1'].mean(), color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {results_df["f1"].mean():.4f}')
    
    ax3.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax3.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
    ax3.set_title('F1 Score Across Folds', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.2, 0.6])
    
    plt.tight_layout()
    
    # Save
    output_file = EXP_FIGURES_DIR / 'final_model_cv_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def print_summary(results_df):
    """Print summary statistics."""
    print_section("CROSS-VALIDATION SUMMARY")
    
    print("Performance Across All Folds:")
    print("-" * 80)
    print(f"{'Metric':<15s} {'Mean':<12s} {'Std':<12s} {'Min':<12s} {'Max':<12s}")
    print("-" * 80)
    
    for metric in ['auc', 'f1', 'precision', 'recall', 'accuracy']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        
        print(f"{metric.upper():<15s} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
    
    print("-" * 80)
    
    # Stability analysis
    print("\nStability Analysis:")
    auc_std = results_df['auc'].std()
    f1_std = results_df['f1'].std()
    
    if auc_std < 0.02:
        print(f"  ✅ EXCELLENT: AUC very stable across folds (std: {auc_std:.4f})")
    elif auc_std < 0.05:
        print(f"  ✅ GOOD: AUC reasonably stable across folds (std: {auc_std:.4f})")
    else:
        print(f"  ⚠️  WARNING: AUC varies significantly across folds (std: {auc_std:.4f})")
    
    if f1_std < 0.05:
        print(f"  ✅ EXCELLENT: F1 very stable across folds (std: {f1_std:.4f})")
    elif f1_std < 0.10:
        print(f"  ✅ GOOD: F1 reasonably stable across folds (std: {f1_std:.4f})")
    else:
        print(f"  ⚠️  WARNING: F1 varies significantly across folds (std: {f1_std:.4f})")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("EXPERIMENT 1D: FINAL MODEL CROSS-VALIDATION".center(80))
    print("="*80)
    print("\nFinal Model: Strong Regularization XGBoost + Top 10 SHAP Features")
    
    # Load top SHAP features
    top_features = load_top_shap_features(n_features=10)
    print(f"\nTop 10 SHAP Features: {', '.join(top_features)}")
    
    # Load data
    train_df = load_data()
    
    # Perform cross-validation
    results_df = perform_time_series_cv(train_df, top_features, n_splits=5)
    
    # Plot results
    plot_cv_results(results_df)
    
    # Print summary
    print_summary(results_df)
    
    # Save results
    results_file = EXP_OUTPUT_DIR / 'final_model_cv_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    
    print("\n" + "="*80)
    print("✅ CROSS-VALIDATION COMPLETE".center(80))
    print("="*80)
    print(f"\nMean AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"Mean F1:  {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(f"\nResults: {results_file}")
    print(f"Figure: {EXP_FIGURES_DIR / 'final_model_cv_performance.png'}\n")


if __name__ == "__main__":
    main()
