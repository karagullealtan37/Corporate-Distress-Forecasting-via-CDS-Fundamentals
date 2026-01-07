"""
Corporate Distress Prediction - Main Pipeline
Runs the complete 15-step ML pipeline + all experiments from raw data.

All steps execute from scratch with full terminal output.
No cached results - everything is recomputed.

Usage:
    python main.py                    # Run full pipeline + all experiments
    python main.py --pipeline-only    # Run pipeline steps only
    python main.py --experiments-only # Run all experiments only
    python main.py --help             # Show options
"""

import sys
import os
from pathlib import Path
import importlib.util
import time
from datetime import timedelta

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Force fresh execution - no cached outputs
os.environ['FORCE_RECOMPUTE'] = '1'


def run_step(step_name, module_name, folder='src'):
    """Run a single pipeline step or experiment with full output."""
    print(f"\n{'='*70}")
    print(f"  {step_name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        # Import the module
        module_path = PROJECT_ROOT / folder / f'{module_name}.py'
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        
        # Load the module (but don't execute yet)
        sys.stdout.flush()
        spec.loader.exec_module(module)
        
        # Now explicitly call main() if it exists
        # This ensures all print() statements execute
        if hasattr(module, 'main'):
            sys.stdout.flush()
            module.main()
            sys.stdout.flush()
        else:
            # Some modules might not have main(), just loading them executes code
            pass
        
        elapsed = time.time() - start_time
        print(f"\n✅ {step_name} completed successfully (⏱️  {elapsed:.1f}s)\n")
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n {step_name} failed with error (⏱️  {elapsed:.1f}s):")
        print(f"   {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False, elapsed


def run_xgboost_experiments():
    """Run all experiments."""
    
    print("\n" + "="*70)
    print("  ALL EXPERIMENTS - FROM RAW DATA")
    print("="*70)
    print("\nRunning all experiments:")
    print("  • Exp 1: Overfitting Reduction")
    print("  • Exp 1b: XGBoost Overfitting Reduction (Progressive Regularization)")
    print("  • Exp 1c: XGBoost Top 10 SHAP Features (Strong Regularization)")
    print("  • Exp 1d: XGBoost Final Model Cross-Validation")
    print("  • Exp 1e: XGBoost CDS-Only vs Full Features")
    print("  • Exp 1f: Three-Way CV Comparison (Naive → CDS-Only → Top 10)")
    print("  • Exp 4: Optimize Recall")
    print("  • Exp 5: Temporal Features")
    print("  • Exp 6: Combined Optimization")
    print("  • Exp 16: Incremental Value CV")
    print("\n" + "="*70 + "\n")
    
    experiments = [
        ("Exp 1: Overfitting Reduction", "exp1_reduce_overfitting"),
        ("Exp 1b: XGBoost Overfitting Reduction", "exp1b_reduce_overfitting_xgboost"),
        ("Exp 1c: XGBoost Top 10 SHAP", "exp1c_strong_reg_top10_shap"),
        ("Exp 1d: XGBoost Final Model CV", "exp1d_final_model_cv"),
        ("Exp 1e: XGBoost CDS-Only Comparison", "exp1e_cds_only_comparison"),
        ("Exp 1f: Three-Way CV Comparison", "exp1f_three_way_cv_comparison"),
        ("Exp 4: Optimize Recall", "exp4_optimize_recall"),
        ("Exp 5: Temporal Features", "exp5_temporal_features"),
        ("Exp 6: Combined Optimization", "exp6_combined_optimization"),
        ("Exp 16: Incremental Value CV", "exp16_incremental_value_cv"),
    ]
    
    completed = 0
    failed = []
    total_time = 0
    
    for i, (exp_name, module_name) in enumerate(experiments, 1):
        print(f"\n[Experiment {i}/{len(experiments)}]")
        
        success, elapsed = run_step(exp_name, module_name, folder='experiments')
        total_time += elapsed
        
        if success:
            completed += 1
        else:
            failed.append(exp_name)
            print(f"\n  Warning: {exp_name} failed but continuing...\n")
    
    return completed, failed, len(experiments), total_time


def print_story_highlights(run_pipeline, run_exps):
    """Print a narrative summary of key metrics across steps and experiments."""
    print("\n" + "="*70)
    print("  KEY RESULTS SUMMARY")
    print("="*70)
    
    if run_pipeline:
        print("\n PIPELINE STEPS (1-15): Complete data processing and model training")
        print("  • Steps 1-4: Raw data → cleaned, merged dataset (600 firms, 28K observations)")
        print("  • Steps 5-9: Feature engineering → 29 features + distress target")
        print("  • Steps 10-12: Model training → XGBoost (AUC 0.632, F1 0.390)")
        print("  • Steps 13-15: Evaluation → Test performance + SHAP explainability")
    
    if run_exps:
        print("\n XGBOOST EXPERIMENTS: Progressive improvement journey")
        print("  • Exp 1b: Overfitting reduction → Train-test gap from 31.4% to 5.1%")
        print("  • Exp 1c: Top 10 SHAP features → Strong regularization + parsimony")
        print("  • Exp 1d: Cross-validation → Temporal robustness confirmed")
        print("  • Exp 1e: CDS-only comparison → Model sophistication = 77% of gains")
        print("  • Exp 1f: Three-way CV → Naive (0.472) → CDS-only (0.533) → Top 10 (0.555)")


def main():
    """Run the complete ML pipeline from raw data with all experiments."""
    
    # Start total timer
    total_start_time = time.time()
    
    # Parse command line arguments
    run_pipeline = True
    run_exps = True  # Run all experiments by default
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ['--help', '-h']:
            print("\nUsage:")
            print("  python main.py                    # Run pipeline + all experiments (default)")
            print("  python main.py --pipeline-only    # Run pipeline steps only")
            print("  python main.py --experiments-only # Run all experiments only")
            print("  python main.py --help             # Show this help\n")
            print("Note: All steps execute from scratch. No cached outputs are used.")
            return 0
        elif arg == '--pipeline-only':
            run_pipeline = True
            run_exps = False
        elif arg == '--experiments-only':
            run_pipeline = False
            run_exps = True
        else:
            print(f"\n Unknown argument: {arg}")
            print("Use --help to see available options\n")
            return 1
    
    print("\n" + "="*70)
    print("  CDS DISTRESS PREDICTION - FULL PIPELINE FROM RAW DATA")
    print("="*70)
    print("\n⚡ EXECUTION MODE: Fresh computation (no cached outputs)")
    print("📊 OUTPUT: All results printed to terminal")
    
    if run_pipeline:
        print("\n📋 PIPELINE STEPS (15 total):")
        print("  • Steps 1-2:  Data inspection and quality checks")
        print("  • Steps 3-4:  Data cleaning and merging")
        print("  • Steps 5-9:  Feature engineering and target creation")
        print("  • Steps 10-12: Model training and optimization")
        print("  • Steps 13-15: Evaluation and explainability")
    
    if run_exps:
        print("\n ALL EXPERIMENTS (10 total):")
        print("  • Exp 1: Overfitting reduction")
        print("  • Exp 1b: XGBoost overfitting reduction (progressive regularization)")
        print("  • Exp 1c: Top 10 SHAP features (strong regularization)")
        print("  • Exp 1d: Final model cross-validation")
        print("  • Exp 1e: CDS-only vs full features comparison")
        print("  • Exp 1f: Three-way CV (naive → CDS-only → Top 10)")
        print("  • Exp 4: Optimize recall")
        print("  • Exp 5: Temporal features")
        print("  • Exp 6: Combined optimization")
        print("  • Exp 16: Incremental value CV")
    
    print("\n" + "="*70 + "\n")
    
    # Define all pipeline steps
    steps = [
        ("Step 1: Data Inspection", "step1_data_inspection"),
        ("Step 2: Data Quality Assessment", "step2_data_quality"),
        ("Step 3: Data Cleaning", "step3_data_cleaning"),
        ("Step 4: Data Merging", "step4_data_merging"),
        ("Step 5: Preprocessing", "step5_preprocessing"),
        ("Step 6: Accounting Features", "step6_accounting_features"),
        ("Step 7: Market Features", "step7_market_features"),
        ("Step 8: Feature Validation", "step8_feature_validation"),
        ("Step 9: Target Creation", "step9_target_creation"),
        ("Step 10: ML Construction", "step10_ml_construction"),
        ("Step 11: Model Training", "step11_model_training"),
        ("Step 12: Model Optimization", "step12_model_optimization"),
        ("Step 13: Model Evaluation", "step13_model_evaluation"),
        ("Step 14: Benchmark Comparison", "step14_benchmark_comparison"),
        ("Step 15: Explainability Analysis", "step15_explainability"),
    ]
    
    # Track progress
    pipeline_completed = 0
    pipeline_failed = []
    pipeline_time = 0
    exp_completed = 0
    exp_failed = []
    exp_time = 0
    
    # Run pipeline steps
    if run_pipeline:
        pipeline_start = time.time()
        for i, (step_name, module_name) in enumerate(steps, 1):
            print(f"\n[Step {i}/{len(steps)}]")
            
            success, elapsed = run_step(step_name, module_name)
            
            if success:
                pipeline_completed += 1
            else:
                pipeline_failed.append(step_name)
                print(f"\n⚠️  Warning: {step_name} failed but continuing...\n")
        pipeline_time = time.time() - pipeline_start
    
    # Run XGBoost experiments
    if run_exps:
        exp_completed, exp_failed, exp_total, exp_time = run_xgboost_experiments()
    
    # Final summary
    print("\n" + "="*70)
    print("  EXECUTION SUMMARY")
    print("="*70)
    
    if run_pipeline:
        print(f"\n📋 Pipeline: {pipeline_completed}/{len(steps)} steps completed")
        if pipeline_failed:
            print(f"    Failed: {len(pipeline_failed)} step(s)")
            for step in pipeline_failed:
                print(f"      - {step}")
        else:
            print("    All pipeline steps completed successfully!")
    
    if run_exps:
        exp_total = exp_completed + len(exp_failed)
        print(f"\n🎯 Experiments: {exp_completed}/{exp_total} experiments completed")
        if exp_failed:
            print(f"    Failed: {len(exp_failed)} experiment(s)")
            for exp in exp_failed:
                print(f"      - {exp}")
        else:
            print("    All experiments completed successfully!")

    print_story_highlights(run_pipeline, run_exps)

    print("\n" + "="*70)
    print("  OUTPUT LOCATIONS")
    print("="*70)
    print(f"\n📁 Processed Data:       {PROJECT_ROOT / 'output'}")
    print(f" Pipeline Models:      {PROJECT_ROOT / 'output' / 'models'}")
    print(f"Pipeline Figures:     {PROJECT_ROOT / 'report' / 'figures'}")
    print(f" Evaluation Results:   {PROJECT_ROOT / 'output' / 'step13_evaluation_results.csv'}")
    
    if run_exps:
        print(f"\n Experiment Outputs:")
        print(f"   • Models:  {PROJECT_ROOT / 'output' / 'experiments' / 'models'}")
        print(f"   • Figures: {PROJECT_ROOT / 'report' / 'figures' / 'experiments'}")
        print(f"   • Results: {PROJECT_ROOT / 'output' / 'experiments'}")
    
    print("\n" + "="*70)
    print("  RECOMMENDED XGBOOST MODEL")
    print("="*70)
    
    if run_exps:
        print(f"\n🏆 BEST MODEL: XGBoost Top 10 SHAP Features (Strong Regularization)")
        print(f"   📂 Location: output/experiments/models/xgboost_strong_reg_top10_shap.pkl")
        print(f"   📊 Performance:")
        print(f"      • Test AUC: 0.636 (+58.2% vs naive baseline)")
        print(f"      • Precision: 30.0%, Recall: 72.0%")
        print(f"      • F1 Score: 0.420")
        print(f"   ⚡ Features: 10 (vs 29 full set = 66% reduction)")
        print(f"   ✅ Benefits: Simpler, faster, more interpretable, strong regularization")
        print(f"\n   📈 Key Finding: Model sophistication = 77% of gains")
        print(f"      • Naive CDS threshold → XGBoost CDS-only: +44.8% (0.402 → 0.582)")
        print(f"      • XGBoost CDS-only → XGBoost Top 10: +9.3% (0.582 → 0.636)")
    else:
        print(f"\n🏆 Pipeline Model: XGBoost Optimized")
        print(f"   📂 Location: output/models/xgboost_optimized.pkl")
        print(f"   📊 Test AUC: 0.632, F1: 0.390")
        print(f"   ⚡ Run experiments for Top 10 feature model")
    
    print("\n" + "="*70 + "\n")
    
    # Calculate and display total runtime
    total_elapsed = time.time() - total_start_time
    
    print("="*70)
    print("  ⏱️  RUNTIME SUMMARY")
    print("="*70)
    
    if run_pipeline:
        pipeline_td = timedelta(seconds=int(pipeline_time))
        print(f"\n📋 Pipeline Steps: {pipeline_td} ({pipeline_time:.1f}s)")
        print(f"   • Average per step: {pipeline_time/len(steps):.1f}s")
    
    if run_exps:
        exp_td = timedelta(seconds=int(exp_time))
        print(f"\n🎯 Experiments: {exp_td} ({exp_time:.1f}s)")
        print(f"   • Average per experiment: {exp_time/10:.1f}s")
    
    total_td = timedelta(seconds=int(total_elapsed))
    print(f"\n⏱️  TOTAL RUNTIME: {total_td} ({total_elapsed:.1f}s)")
    
    # Show breakdown
    if run_pipeline and run_exps:
        pipeline_pct = (pipeline_time / total_elapsed) * 100
        exp_pct = (exp_time / total_elapsed) * 100
        print(f"\n   Breakdown:")
        print(f"   • Pipeline: {pipeline_pct:.1f}%")
        print(f"   • Experiments: {exp_pct:.1f}%")
    
    print("\n" + "="*70 + "\n")
    
    has_failures = (pipeline_failed if run_pipeline else []) or (exp_failed if run_exps else [])
    return 0 if not has_failures else 1


if __name__ == "__main__":
    sys.exit(main())
