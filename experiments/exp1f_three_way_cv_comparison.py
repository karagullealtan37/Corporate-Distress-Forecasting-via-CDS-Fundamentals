"""
EXPERIMENT 1F: Three-Way Cross-Validation Comparison

Goal: Perform time-series cross-validation for all three models to assess
      temporal stability and robustness:
      1. Naive CDS Threshold
      2. XGBoost CDS-Only
      3. XGBoost Top 10 SHAP

Outputs:
    - Results: output/experiments/three_way_cv_results.csv
    - Figure: report/figures/experiments/three_way_cv_comparison.png
      (Multi-panel: CV performance, stability, fold-by-fold, feature importance)
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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
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


def load_data_and_features():
    """Load training data and feature sets."""
    print("Loading training data and feature sets...")
    
    # Load training data
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    
    # Load all features
    feature_list = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')
    all_features = feature_list['feature'].tolist()
    
    # Get CDS features
    cds_features = [f for f in all_features if 'cds' in f.lower()]
    
    # Get Top 10 SHAP features
    shap_df = pd.read_csv(OUTPUT_DIR / 'shap_values_xgboost.csv')
    top10_features = shap_df.nlargest(10, 'mean_abs_shap')['feature'].tolist()
    
    print(f"  ✓ Train data: {train_df.shape}")
    print(f"  ✓ CDS features: {len(cds_features)}")
    print(f"  ✓ Top 10 features: {len(top10_features)}\n")
    
    return train_df, cds_features, top10_features


def perform_cv_naive_threshold(train_df, n_splits=5):
    """Cross-validate naive CDS threshold model."""
    print_section(f"CV: NAIVE CDS THRESHOLD ({n_splits} FOLDS)")
    
    # Sort by date
    train_df = train_df.sort_values('date').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df), 1):
        val_fold = train_df.iloc[val_idx].copy()
        
        # Naive threshold: median CDS
        cds_values = val_fold['cds_spread_lag1'].fillna(val_fold['cds_spread_lag1'].median())
        median_cds = cds_values.median()
        
        y_val = val_fold['distress_flag']
        y_pred = (cds_values > median_cds).astype(int)
        y_proba = cds_values / cds_values.max()
        
        # Calculate metrics
        results.append({
            'fold': fold,
            'model': 'Naive CDS',
            'auc': roc_auc_score(y_val, y_proba),
            'f1': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred)
        })
        
        print(f"Fold {fold}: AUC={results[-1]['auc']:.4f}, F1={results[-1]['f1']:.4f}")
    
    return pd.DataFrame(results)


def perform_cv_ml_model(train_df, features, model_name, n_splits=5):
    """Cross-validate ML model with given features."""
    print_section(f"CV: {model_name.upper()} ({n_splits} FOLDS)")
    
    print(f"Features: {len(features)}")
    print(f"Model: Strong Regularization XGBoost\n")
    
    # Sort by date
    train_df = train_df.sort_values('date').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
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
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df), 1):
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        
        # Prepare data
        X_train = train_fold[features].copy()
        y_train = train_fold['distress_flag'].copy()
        X_val = val_fold[features].copy()
        y_val = val_fold['distress_flag'].copy()
        
        # Preprocess
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_scaled = scaler.transform(imputer.transform(X_val))
        
        # Train model
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        results.append({
            'fold': fold,
            'model': model_name,
            'auc': roc_auc_score(y_val, y_proba),
            'f1': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred)
        })
        
        print(f"Fold {fold}: AUC={results[-1]['auc']:.4f}, F1={results[-1]['f1']:.4f}")
    
    return pd.DataFrame(results)


def create_multi_panel_visualization(results_df, cds_features, top10_features):
    """Create comprehensive multi-panel visualization."""
    print_section("GENERATING MULTI-PANEL VISUALIZATION")
    
    plt.close('all')
    plt.clf()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Color scheme
    colors = {'Naive CDS': 'red', 'CDS-Only': 'orange', 'Top 10': 'green'}
    
    # ========== PANEL 1: AUC by Fold (Line Plot) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    for model in ['Naive CDS', 'CDS-Only', 'Top 10']:
        data = results_df[results_df['model'] == model]
        ax1.plot(data['fold'], data['auc'], marker='o', linewidth=2.5, 
                markersize=8, label=model, color=colors[model], alpha=0.8)
    
    ax1.set_xlabel('Fold', fontweight='bold', fontsize=11)
    ax1.set_ylabel('AUC', fontweight='bold', fontsize=11)
    ax1.set_title('AUC Across CV Folds', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 0.75])
    
    # ========== PANEL 2: F1 by Fold (Line Plot) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    for model in ['Naive CDS', 'CDS-Only', 'Top 10']:
        data = results_df[results_df['model'] == model]
        ax2.plot(data['fold'], data['f1'], marker='s', linewidth=2.5,
                markersize=8, label=model, color=colors[model], alpha=0.8)
    
    ax2.set_xlabel('Fold', fontweight='bold', fontsize=11)
    ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    ax2.set_title('F1 Score Across CV Folds', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.0, 0.5])
    
    # ========== PANEL 3: Mean Performance Comparison (Bar Chart) ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    mean_auc = results_df.groupby('model')['auc'].mean()
    models_order = ['Naive CDS', 'CDS-Only', 'Top 10']
    mean_values = [mean_auc[m] for m in models_order]
    bar_colors = [colors[m] for m in models_order]
    
    bars = ax3.bar(range(len(models_order)), mean_values, color=bar_colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mean_values)):
        ax3.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xticks(range(len(models_order)))
    ax3.set_xticklabels(models_order, fontsize=10)
    ax3.set_ylabel('Mean AUC', fontweight='bold', fontsize=11)
    ax3.set_title('Mean CV Performance', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 0.7])
    
    # ========== PANEL 4: Stability Analysis (Box Plot) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    data_for_box = [results_df[results_df['model'] == m]['auc'].values 
                    for m in models_order]
    
    bp = ax4.boxplot(data_for_box, labels=models_order, patch_artist=True,
                     widths=0.6, showmeans=True)
    
    for patch, color in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_ylabel('AUC', fontweight='bold', fontsize=11)
    ax4.set_title('AUC Stability (Box Plot)', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0.3, 0.75])
    
    # ========== PANEL 5: Precision-Recall Trade-off ==========
    ax5 = fig.add_subplot(gs[1, 1])
    
    for model in models_order:
        data = results_df[results_df['model'] == model]
        ax5.scatter(data['recall'], data['precision'], s=150, 
                   color=colors[model], alpha=0.7, edgecolors='black',
                   linewidth=2, label=model)
        
        # Add mean point
        mean_recall = data['recall'].mean()
        mean_precision = data['precision'].mean()
        ax5.scatter(mean_recall, mean_precision, s=300, marker='*',
                   color=colors[model], edgecolors='black', linewidth=2)
    
    ax5.set_xlabel('Recall', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax5.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold', pad=10)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 0.8])
    ax5.set_ylim([0, 0.5])
    
    # ========== PANEL 6: Coefficient of Variation (Stability Metric) ==========
    ax6 = fig.add_subplot(gs[1, 2])
    
    cv_values = []
    for model in models_order:
        data = results_df[results_df['model'] == model]['auc']
        cv = (data.std() / data.mean()) * 100  # CV in percentage
        cv_values.append(cv)
    
    bars = ax6.bar(range(len(models_order)), cv_values, color=bar_colors,
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, cv_values):
        ax6.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax6.set_xticks(range(len(models_order)))
    ax6.set_xticklabels(models_order, fontsize=10)
    ax6.set_ylabel('Coefficient of Variation (%)', fontweight='bold', fontsize=11)
    ax6.set_title('AUC Stability (Lower = More Stable)', fontsize=12, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=10, color='green', linestyle='--', alpha=0.5, linewidth=2, label='10% threshold')
    ax6.legend(fontsize=9)
    
    # ========== PANEL 7: Feature Count Comparison ==========
    ax7 = fig.add_subplot(gs[2, 0])
    
    feature_counts = [1, len(cds_features), len(top10_features)]
    
    bars = ax7.barh(range(len(models_order)), feature_counts, color=bar_colors,
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, feature_counts)):
        ax7.text(val, bar.get_y() + bar.get_height()/2.,
                f'  {val} features',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax7.set_yticks(range(len(models_order)))
    ax7.set_yticklabels(models_order, fontsize=10)
    ax7.set_xlabel('Number of Features', fontweight='bold', fontsize=11)
    ax7.set_title('Model Complexity', fontsize=12, fontweight='bold', pad=10)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # ========== PANEL 8: Performance vs Complexity ==========
    ax8 = fig.add_subplot(gs[2, 1])
    
    mean_aucs = [mean_auc[m] for m in models_order]
    
    scatter = ax8.scatter(feature_counts, mean_aucs, s=300, c=bar_colors,
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, model in enumerate(models_order):
        ax8.annotate(model, (feature_counts[i], mean_aucs[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=bar_colors[i], alpha=0.3))
    
    ax8.set_xlabel('Number of Features', fontweight='bold', fontsize=11)
    ax8.set_ylabel('Mean AUC', fontweight='bold', fontsize=11)
    ax8.set_title('Performance vs Complexity', fontsize=12, fontweight='bold', pad=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0.35, 0.65])
    
    # ========== PANEL 9: Summary Statistics Table ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary table
    summary_data = []
    for model in models_order:
        data = results_df[results_df['model'] == model]
        summary_data.append([
            model,
            f"{data['auc'].mean():.4f}",
            f"{data['auc'].std():.4f}",
            f"{data['f1'].mean():.4f}",
            f"{data['precision'].mean():.4f}",
            f"{data['recall'].mean():.4f}"
        ])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Model', 'AUC', 'Std', 'F1', 'Prec', 'Rec'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i, color in enumerate(bar_colors, 1):
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 0)].set_alpha(0.3)
    
    ax9.set_title('CV Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Three-Way Cross-Validation Comparison: Comprehensive Analysis',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_file = EXP_FIGURES_DIR / 'three_way_cv_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def print_summary(results_df):
    """Print comprehensive summary."""
    print_section("CROSS-VALIDATION SUMMARY")
    
    print("Mean Performance Across All Folds:")
    print("-" * 90)
    print(f"{'Model':<20s} {'AUC':<12s} {'Std':<12s} {'F1':<12s} {'Precision':<12s} {'Recall':<12s}")
    print("-" * 90)
    
    for model in ['Naive CDS', 'CDS-Only', 'Top 10']:
        data = results_df[results_df['model'] == model]
        print(f"{model:<20s} {data['auc'].mean():<12.4f} {data['auc'].std():<12.4f} "
              f"{data['f1'].mean():<12.4f} {data['precision'].mean():<12.4f} {data['recall'].mean():<12.4f}")
    
    print("-" * 90)
    
    # Stability analysis
    print("\nSTABILITY ANALYSIS (Coefficient of Variation):")
    print("-" * 60)
    for model in ['Naive CDS', 'CDS-Only', 'Top 10']:
        data = results_df[results_df['model'] == model]['auc']
        cv = (data.std() / data.mean()) * 100
        
        if cv < 10:
            status = "✅ EXCELLENT"
        elif cv < 20:
            status = "✅ GOOD"
        else:
            status = "⚠️  HIGH VARIANCE"
        
        print(f"{model:<20s} CV = {cv:5.1f}%  {status}")
    
    print("-" * 60)
    
    # Improvement analysis
    print("\nIMPROVEMENT OVER NAIVE BASELINE:")
    print("-" * 60)
    
    naive_auc = results_df[results_df['model'] == 'Naive CDS']['auc'].mean()
    
    for model in ['CDS-Only', 'Top 10']:
        model_auc = results_df[results_df['model'] == model]['auc'].mean()
        improvement = ((model_auc - naive_auc) / naive_auc) * 100
        print(f"{model:<20s} +{improvement:5.1f}% AUC improvement")
    
    print("-" * 60)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("EXPERIMENT 1F: THREE-WAY CROSS-VALIDATION COMPARISON".center(80))
    print("="*80)
    
    # Load data
    train_df, cds_features, top10_features = load_data_and_features()
    
    # Perform CV for all three models
    results_naive = perform_cv_naive_threshold(train_df, n_splits=5)
    results_cds = perform_cv_ml_model(train_df, cds_features, 'CDS-Only', n_splits=5)
    results_top10 = perform_cv_ml_model(train_df, top10_features, 'Top 10', n_splits=5)
    
    # Combine results
    results_df = pd.concat([results_naive, results_cds, results_top10], ignore_index=True)
    
    # Create visualization
    create_multi_panel_visualization(results_df, cds_features, top10_features)
    
    # Print summary
    print_summary(results_df)
    
    # Save results
    results_file = EXP_OUTPUT_DIR / 'three_way_cv_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print(f"\nResults: {results_file}")
    print(f"Figure: {EXP_FIGURES_DIR / 'three_way_cv_comparison.png'}")
    print("\nMulti-panel visualization includes:")
    print("  1. AUC by fold (line plot)")
    print("  2. F1 by fold (line plot)")
    print("  3. Mean performance (bar chart)")
    print("  4. Stability analysis (box plot)")
    print("  5. Precision-Recall trade-off")
    print("  6. Coefficient of variation")
    print("  7. Feature count comparison")
    print("  8. Performance vs complexity")
    print("  9. Summary statistics table\n")


if __name__ == "__main__":
    main()
