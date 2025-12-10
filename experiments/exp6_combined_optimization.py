"""
EXPERIMENT 6: Combined Optimization - The Complete Model

Combines:
1. Medium Regularization (from exp1)
2. Temporal Change Features (from exp5)
3. Optimal Threshold (from exp4)

Goal: Achieve maximum recall with temporal features + optimal threshold
Expected: Recall ~0.75-0.80 (75-80% of distressed firms caught)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score, 
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
EXP_OUTPUT_DIR = OUTPUT_DIR / 'experiments'
EXP_MODELS_DIR = EXP_OUTPUT_DIR / 'models'
EXP_FIGURES_DIR = PROJECT_ROOT / 'report' / 'figures' / 'experiments'

# Create directories
EXP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def create_temporal_features(df):
    """
    Create temporal change features (same as exp5).
    """
    print("Creating temporal change features...")
    
    df = df.sort_values(['gvkey', 'date']).copy()
    grouped = df.groupby('gvkey')
    
    new_features = []
    
    # 1. Debt-to-Assets change
    if 'debt_to_assets' in df.columns:
        df['debt_to_assets_change_1y'] = grouped['debt_to_assets'].diff(4)
        new_features.append('debt_to_assets_change_1y')
    
    # 2. Altman Z-Score change
    if 'altman_z_score' in df.columns:
        df['altman_z_change_1y'] = grouped['altman_z_score'].diff(4)
        new_features.append('altman_z_change_1y')
    
    # 3. ROA change
    if 'roa' in df.columns:
        df['roa_change_1y'] = grouped['roa'].diff(4)
        new_features.append('roa_change_1y')
    
    # 4. Current ratio change
    if 'current_ratio' in df.columns:
        df['current_ratio_change_1y'] = grouped['current_ratio'].diff(4)
        new_features.append('current_ratio_change_1y')
    
    # 5. CDS spread changes
    if 'cds_spread_lag1' in df.columns and 'cds_spread_lag4' in df.columns:
        df['cds_spread_lag2'] = grouped['cds_spread_lag1'].shift(1)
        df['cds_spread_change_1q'] = df['cds_spread_lag1'] - df['cds_spread_lag2']
        df['cds_spread_change_1y'] = df['cds_spread_lag1'] - df['cds_spread_lag4']
        new_features.extend(['cds_spread_change_1q', 'cds_spread_change_1y'])
    
    # 6. Return changes
    if 'return_1m' in df.columns:
        df['return_1m_prev'] = grouped['return_1m'].shift(1)
        df['return_1m_change'] = df['return_1m'] - df['return_1m_prev']
        new_features.append('return_1m_change')
    
    if 'return_lag1' in df.columns:
        df['return_lag1_change'] = grouped['return_lag1'].diff(1)
        new_features.append('return_lag1_change')
    
    # 7. Volatility changes
    if 'volatility_3m' in df.columns:
        df['volatility_3m_change'] = grouped['volatility_3m'].diff(1)
        new_features.append('volatility_3m_change')
    
    if 'volatility_12m' in df.columns:
        df['volatility_12m_change'] = grouped['volatility_12m'].diff(4)
        new_features.append('volatility_12m_change')
    
    print(f"✓ Created {len(new_features)} temporal features")
    
    return df, new_features


def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal threshold using F1-maximization.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return best_threshold


def main():
    print_section("EXPERIMENT 6: COMBINED OPTIMIZATION - THE COMPLETE MODEL")
    
    # ============================================================================
    # STEP 1: Load data and create temporal features
    # ============================================================================
    print_section("STEP 1: LOAD DATA & CREATE TEMPORAL FEATURES")
    
    # Load original data
    train_df = pd.read_csv(OUTPUT_DIR / 'train_data.csv', low_memory=False)
    test_df = pd.read_csv(OUTPUT_DIR / 'test_data.csv', low_memory=False)
    
    print(f"Original train: {train_df.shape}")
    print(f"Original test: {test_df.shape}")
    
    # Load original features
    original_features = pd.read_csv(OUTPUT_DIR / 'ml_feature_list.csv')['feature'].tolist()
    print(f"Original features: {len(original_features)}")
    
    # Create temporal features
    train_df, new_features = create_temporal_features(train_df)
    test_df, _ = create_temporal_features(test_df)
    
    # Combined feature list
    all_features = original_features + new_features
    print(f"Total features: {len(all_features)} ({len(original_features)} original + {len(new_features)} temporal)")
    
    # Prepare data
    X_train = train_df[all_features].copy()
    y_train = train_df['distress_flag'].copy()
    X_test = test_df[all_features].copy()
    y_test = test_df['distress_flag'].copy()
    
    # Handle NaN from temporal features (fill with 0 = no change when no history)
    print(f"\nNaN counts before filling:")
    print(f"  Train: {X_train.isna().sum().sum()} NaN values")
    print(f"  Test: {X_test.isna().sum().sum()} NaN values")
    
    # Fill NaN in temporal features with 0 (no change when no history available)
    for feat in new_features:
        X_train[feat] = X_train[feat].fillna(0)
        X_test[feat] = X_test[feat].fillna(0)
    
    print(f"\nAfter filling temporal NaN with 0:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # ============================================================================
    # STEP 2: Train model with Medium Regularization + Temporal Features
    # ============================================================================
    print_section("STEP 2: TRAIN MODEL (MEDIUM REGULARIZATION + TEMPORAL FEATURES)")
    
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = pd.DataFrame(
        scaler.fit_transform(imputer.fit_transform(X_train)),
        columns=all_features,
        index=X_train.index
    )
    
    X_test_processed = pd.DataFrame(
        scaler.transform(imputer.transform(X_test)),
        columns=all_features,
        index=X_test.index
    )
    
    # Medium Regularization config (from exp1)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # 2^4 - 1
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 80,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'verbose': -1
    }
    
    print("Training LightGBM with:")
    print("  ✓ Medium Regularization")
    print("  ✓ Temporal Change Features")
    print(f"  ✓ Class imbalance handling (scale_pos_weight={params['scale_pos_weight']:.2f})")
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_processed, y_train)
    
    print("\n✓ Training complete")
    
    # ============================================================================
    # STEP 3: Evaluate with default threshold (0.50)
    # ============================================================================
    print_section("STEP 3: EVALUATE WITH DEFAULT THRESHOLD (0.50)")
    
    y_train_prob = model.predict_proba(X_train_processed)[:, 1]
    y_test_prob = model.predict_proba(X_test_processed)[:, 1]
    
    y_train_pred_default = (y_train_prob >= 0.5).astype(int)
    y_test_pred_default = (y_test_prob >= 0.5).astype(int)
    
    # Metrics with default threshold
    train_auc_default = roc_auc_score(y_train, y_train_prob)
    test_auc_default = roc_auc_score(y_test, y_test_prob)
    test_recall_default = recall_score(y_test, y_test_pred_default)
    test_precision_default = precision_score(y_test, y_test_pred_default)
    test_f1_default = f1_score(y_test, y_test_pred_default)
    
    print("Performance with DEFAULT threshold (0.50):")
    print(f"  Train AUC: {train_auc_default:.4f}")
    print(f"  Test AUC: {test_auc_default:.4f}")
    print(f"  Test Recall: {test_recall_default:.4f}")
    print(f"  Test Precision: {test_precision_default:.4f}")
    print(f"  Test F1: {test_f1_default:.4f}")
    
    cm_default = confusion_matrix(y_test, y_test_pred_default)
    tn, fp, fn, tp = cm_default.ravel()
    total_distressed = y_test.sum()
    
    print(f"\nBusiness Impact (Default 0.50):")
    print(f"  Distressed firms caught: {tp} / {total_distressed} ({tp/total_distressed*100:.1f}%)")
    print(f"  Distressed firms missed: {fn} ({fn/total_distressed*100:.1f}%)")
    
    # ============================================================================
    # STEP 4: Find optimal threshold (on TRAINING data - unbiased!)
    # ============================================================================
    print_section("STEP 4: OPTIMIZE THRESHOLD ON TRAINING DATA (UNBIASED)")
    
    optimal_threshold = find_optimal_threshold(y_train, y_train_prob)
    print(f"Optimal threshold (F1-maximized on TRAIN): {optimal_threshold:.4f}")
    print("Note: Threshold optimized on training data, will apply to test data")
    
    # Predictions with optimal threshold
    y_test_pred_optimal = (y_test_prob >= optimal_threshold).astype(int)
    
    # Metrics with optimal threshold
    test_recall_optimal = recall_score(y_test, y_test_pred_optimal)
    test_precision_optimal = precision_score(y_test, y_test_pred_optimal)
    test_f1_optimal = f1_score(y_test, y_test_pred_optimal)
    
    print(f"\nPerformance with OPTIMAL threshold ({optimal_threshold:.4f}):")
    print(f"  Test AUC: {test_auc_default:.4f} (unchanged)")
    print(f"  Test Recall: {test_recall_optimal:.4f}")
    print(f"  Test Precision: {test_precision_optimal:.4f}")
    print(f"  Test F1: {test_f1_optimal:.4f}")
    
    cm_optimal = confusion_matrix(y_test, y_test_pred_optimal)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
    
    print(f"\nBusiness Impact (Optimal {optimal_threshold:.4f}):")
    print(f"  Distressed firms caught: {tp_opt} / {total_distressed} ({tp_opt/total_distressed*100:.1f}%)")
    print(f"  Distressed firms missed: {fn_opt} ({fn_opt/total_distressed*100:.1f}%)")
    print(f"  Additional firms caught: +{tp_opt - tp} ({(tp_opt-tp)/tp*100:+.1f}%)")
    
    # ============================================================================
    # STEP 5: Compare to baseline
    # ============================================================================
    print_section("STEP 5: COMPARISON TO BASELINE")
    
    # Load baseline results (from exp4)
    baseline_recall = 0.464  # Original LightGBM
    baseline_caught = 679
    
    print("COMPLETE COMPARISON:")
    print("="*80)
    print(f"{'Model':<45} {'Recall':<10} {'Caught':<10} {'Improvement':<15}")
    print("="*80)
    print(f"{'1. Baseline (Original LightGBM)':<45} {baseline_recall:<10.3f} {baseline_caught:<10} {'baseline':<15}")
    print(f"{'2. + Temporal Features (default 0.50)':<45} {test_recall_default:<10.3f} {tp:<10} {f'+{(test_recall_default-baseline_recall)/baseline_recall*100:.1f}%':<15}")
    print(f"{'3. + Optimal Threshold (FINAL MODEL)':<45} {test_recall_optimal:<10.3f} {tp_opt:<10} {f'+{(test_recall_optimal-baseline_recall)/baseline_recall*100:.1f}%':<15}")
    print("="*80)
    
    print(f"\n🎯 FINAL MODEL IMPROVEMENT:")
    print(f"   Recall: {baseline_recall:.3f} → {test_recall_optimal:.3f} (+{(test_recall_optimal-baseline_recall)/baseline_recall*100:.1f}%)")
    print(f"   Firms caught: {baseline_caught} → {tp_opt} (+{tp_opt-baseline_caught})")
    print(f"   AUC: 0.6431 → {test_auc_default:.4f} (+{(test_auc_default-0.6431)/0.6431*100:.1f}%)")
    
    # ============================================================================
    # STEP 6: Visualizations
    # ============================================================================
    print_section("STEP 6: GENERATE VISUALIZATIONS")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    x = np.arange(2)
    width = 0.35
    
    caught = [tp, tp_opt]
    missed = [fn, fn_opt]
    
    ax1.bar(x - width/2, caught, width, label='Caught (TP)', color='green', alpha=0.7, edgecolor='black')
    ax1.bar(x + width/2, missed, width, label='Missed (FN)', color='red', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Number of Distressed Firms', fontweight='bold', fontsize=12)
    ax1.set_title('Business Impact: Caught vs Missed Distressed Firms', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Default (0.50)', f'Optimal ({optimal_threshold:.4f})'])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (c, m) in enumerate(zip(caught, missed)):
        ax1.text(i - width/2, c + 20, f'{c}\\n({c/total_distressed*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.text(i + width/2, m + 20, f'{m}\\n({m/total_distressed*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Recall Improvement Journey
    ax2 = fig.add_subplot(gs[0, 2])
    
    recalls = [baseline_recall, test_recall_default, test_recall_optimal]
    labels = ['Baseline', '+ Temporal\\nFeatures', '+ Optimal\\nThreshold']
    colors = ['steelblue', 'darkorange', 'gold']
    
    bars = ax2.bar(labels, recalls, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Recall', fontweight='bold', fontsize=12)
    ax2.set_title('Recall Improvement Journey', fontweight='bold', fontsize=13, pad=10)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.55, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.55)')
    ax2.axhline(y=0.65, color='darkgreen', linestyle='--', alpha=0.5, linewidth=2, label='Stretch (0.65)')
    ax2.legend(fontsize=9)
    
    for bar, val in zip(bars, recalls):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 3: Precision-Recall Curve
    ax3 = fig.add_subplot(gs[1, 0])
    
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_test_prob)
    
    ax3.plot(recall_curve, precision_curve, linewidth=3, color='steelblue', label='PR Curve')
    ax3.scatter([test_recall_default], [test_precision_default], s=200, color='darkorange', 
               edgecolor='black', linewidth=2, zorder=5, label=f'Default (0.50)')
    ax3.scatter([test_recall_optimal], [test_precision_optimal], s=200, color='gold', 
               edgecolor='black', linewidth=2, zorder=5, label=f'Optimal ({optimal_threshold:.4f})')
    ax3.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax3.set_title('Precision-Recall Curve', fontweight='bold', fontsize=13, pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: ROC Curve
    ax4 = fig.add_subplot(gs[1, 1])
    
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    
    ax4.plot(fpr, tpr, linewidth=3, color='steelblue', label=f'AUC = {test_auc_default:.4f}')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3, label='Random')
    ax4.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax4.set_title('ROC Curve', fontweight='bold', fontsize=13, pad=10)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Feature Importance (Top 20)
    ax5 = fig.add_subplot(gs[1, 2])
    
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    colors_feat = ['green' if f in new_features else 'steelblue' for f in importance_df['feature']]
    
    ax5.barh(importance_df['feature'], importance_df['importance'], 
            color=colors_feat, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Importance', fontweight='bold', fontsize=11)
    ax5.set_title('Top 20 Features (Temporal in Green)', fontweight='bold', fontsize=12, pad=10)
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.tick_params(axis='y', labelsize=9)
    
    # Plot 6: Confusion Matrix - Default
    ax6 = fig.add_subplot(gs[2, 0])
    
    cm_display = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax6,
               xticklabels=['Pred: No', 'Pred: Yes'],
               yticklabels=['True: No', 'True: Yes'])
    ax6.set_title(f'Confusion Matrix\\nDefault (0.50)', fontweight='bold', fontsize=12, pad=10)
    
    # Plot 7: Confusion Matrix - Optimal
    ax7 = fig.add_subplot(gs[2, 1])
    
    cm_display_opt = np.array([[tn_opt, fp_opt], [fn_opt, tp_opt]])
    sns.heatmap(cm_display_opt, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax7,
               xticklabels=['Pred: No', 'Pred: Yes'],
               yticklabels=['True: No', 'True: Yes'])
    ax7.set_title(f'Confusion Matrix\\nOptimal ({optimal_threshold:.4f})', fontweight='bold', fontsize=12, pad=10)
    
    # Plot 8: Metrics Comparison
    ax8 = fig.add_subplot(gs[2, 2])
    
    metrics = ['Recall', 'Precision', 'F1-Score']
    default_vals = [test_recall_default, test_precision_default, test_f1_default]
    optimal_vals = [test_recall_optimal, test_precision_optimal, test_f1_optimal]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax8.bar(x - width/2, default_vals, width, label='Default (0.50)', 
           color='darkorange', alpha=0.7, edgecolor='black')
    ax8.bar(x + width/2, optimal_vals, width, label=f'Optimal ({optimal_threshold:.4f})', 
           color='gold', alpha=0.7, edgecolor='black')
    
    ax8.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax8.set_title('Metrics Comparison', fontweight='bold', fontsize=13, pad=10)
    ax8.set_xticks(x)
    ax8.set_xticklabels(metrics)
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_ylim([0, 1])
    
    plt.suptitle('FINAL MODEL: Medium Regularization + Temporal Features + Optimal Threshold', 
                fontweight='bold', fontsize=16, y=0.995)
    
    plt.savefig(EXP_FIGURES_DIR / 'combined_optimization_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved")
    
    # ============================================================================
    # STEP 7: Save final model
    # ============================================================================
    print_section("STEP 7: SAVE FINAL MODEL")
    
    final_model_config = {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'features': all_features,
        'original_features': original_features,
        'temporal_features': new_features,
        'optimal_threshold': optimal_threshold,
        'params': params,
        'performance': {
            'test_auc': test_auc_default,
            'test_recall_default': test_recall_default,
            'test_recall_optimal': test_recall_optimal,
            'test_precision_optimal': test_precision_optimal,
            'test_f1_optimal': test_f1_optimal,
            'firms_caught': tp_opt,
            'firms_missed': fn_opt,
            'total_distressed': int(total_distressed)
        }
    }
    
    model_path = EXP_MODELS_DIR / 'lightgbm_final_complete_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_model_config, f)
    
    print(f"✓ Final model saved: {model_path}")
    
    # Save results CSV
    results_df = pd.DataFrame({
        'model': ['Baseline', 'Baseline + Temporal (0.50)', 'FINAL (Temporal + Optimal Threshold)'],
        'test_auc': [0.6431, test_auc_default, test_auc_default],
        'test_recall': [baseline_recall, test_recall_default, test_recall_optimal],
        'test_precision': [0.342, test_precision_default, test_precision_optimal],
        'test_f1': [0.394, test_f1_default, test_f1_optimal],
        'firms_caught': [baseline_caught, tp, tp_opt],
        'firms_missed': [784, fn, fn_opt],
        'threshold': [0.50, 0.50, optimal_threshold]
    })
    
    results_df.to_csv(EXP_OUTPUT_DIR / 'combined_optimization_results.csv', index=False)
    print(f"✓ Results saved: {EXP_OUTPUT_DIR / 'combined_optimization_results.csv'}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print_section("✅ COMBINED OPTIMIZATION COMPLETE - THE FINAL MODEL")
    
    print("🎯 FINAL MODEL CONFIGURATION:")
    print("  ✓ Medium Regularization (low overfitting)")
    print(f"  ✓ {len(original_features)} original + {len(new_features)} temporal features")
    print(f"  ✓ Optimal threshold: {optimal_threshold:.4f}")
    
    print(f"\n📊 FINAL PERFORMANCE:")
    print(f"  Test AUC: {test_auc_default:.4f}")
    print(f"  Test Recall: {test_recall_optimal:.4f} (catches {tp_opt/total_distressed*100:.1f}% of distressed firms)")
    print(f"  Test Precision: {test_precision_optimal:.4f}")
    print(f"  Test F1-Score: {test_f1_optimal:.4f}")
    
    print(f"\n🚀 IMPROVEMENT vs BASELINE:")
    print(f"  Recall: {baseline_recall:.3f} → {test_recall_optimal:.3f} (+{(test_recall_optimal-baseline_recall)/baseline_recall*100:.1f}%)")
    print(f"  Firms caught: {baseline_caught} → {tp_opt} (+{tp_opt-baseline_caught} firms)")
    print(f"  AUC: 0.6431 → {test_auc_default:.4f} (+{(test_auc_default-0.6431)/0.6431*100:.1f}%)")
    
    print(f"\n💡 USAGE IN NOTEBOOK:")
    print(f"```python")
    print(f"with open('{model_path}', 'rb') as f:")
    print(f"    config = pickle.load(f)")
    print(f"")
    print(f"model = config['model']")
    print(f"threshold = config['optimal_threshold']  # {optimal_threshold:.4f}")
    print(f"features = config['features']  # {len(all_features)} features")
    print(f"")
    print(f"# Predict")
    print(f"y_prob = model.predict_proba(X_test)[:, 1]")
    print(f"y_pred = (y_prob >= threshold).astype(int)")
    print(f"```")
    
    print(f"\n This is my FINAL, COMPLETE, OPTIMIZED model!")
    print(f"   Ready for dissertation and deployment! 🎓🚀")


if __name__ == "__main__":
    main()
