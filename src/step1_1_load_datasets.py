"""
STEP 1.1: Load All Datasets

This script loads all 4 datasets required for the CDS prediction ML project:
    1. Compustat (Fundamentals - Quarterly)
    2. CRSP (Market data - Monthly)
    3. CDS Prices (Monthly CDS spreads)
    4. Mapping (GVKEY ↔ IQT ↔ Company Name)

Objectives:
    - Load each dataset with appropriate encoding/separator handling
    - Perform basic validation (non-empty, expected columns)
    - Print loading summary
    - Return all datasets for next steps

Outputs:
    - Console: Loading status and basic info for each dataset
    - Returns: 4 pandas DataFrames (compustat, crsp, cds, mapping)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_with_encoding(filepath, encodings=['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'], sep=',', **kwargs):
    """
    Try loading CSV with multiple encodings and separators.
    
    Args:
        filepath: Path to CSV file
        encodings: List of encodings to try
        sep: Primary separator to try first
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        pandas DataFrame
    """
    separators = [sep, ';', ',']  # Try specified separator first, then others
    
    for encoding in encodings:
        for separator in separators:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=separator, low_memory=False, **kwargs)
                # Check if we actually got multiple columns (not all in one column)
                if len(df.columns) > 1:
                    print(f"    ✓ Loaded with {encoding} encoding and '{separator}' separator")
                    print(f"    ✓ Shape: {len(df):,} rows × {len(df.columns)} columns")
                    return df
            except Exception as e:
                continue
    
    raise ValueError(f"Could not load {filepath} with any encoding/separator combination")


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_compustat():
    """
    Load Compustat fundamentals data (quarterly).
    
    Returns:
        pandas DataFrame with Compustat data
    """
    print("1. Loading Compustat (Fundamentals - Quarterly)...")
    filepath = DATA_DIR / 'CDS firms fundamentals Quarterly.csv'
    
    compustat = load_with_encoding(filepath)
    
    # Basic validation
    expected_cols = ['gvkey', 'datadate', 'atq', 'ltq', 'niq', 'saleq']
    missing_cols = [col for col in expected_cols if col not in compustat.columns]
    
    if missing_cols:
        print(f"    ⚠️  Warning: Missing expected columns: {missing_cols}")
    else:
        print(f"    ✓ All expected key columns present")
    
    print(f"    ✓ Columns: {list(compustat.columns[:10])}...")
    print()
    
    return compustat


def load_crsp():
    """
    Load CRSP market data (monthly).
    
    Returns:
        pandas DataFrame with CRSP data
    """
    print("2. Loading CRSP (Market Data - Monthly)...")
    filepath = DATA_DIR / 'Security prices.csv'
    
    crsp = pd.read_csv(filepath, low_memory=False)
    
    print(f"    ✓ Shape: {len(crsp):,} rows × {len(crsp.columns)} columns")
    
    # Basic validation
    expected_cols = ['gvkey', 'datadate', 'prccm', 'trt1m']
    missing_cols = [col for col in expected_cols if col not in crsp.columns]
    
    if missing_cols:
        print(f"    ⚠️  Warning: Missing expected columns: {missing_cols}")
    else:
        print(f"    ✓ All expected key columns present")
    
    print(f"    ✓ Columns: {list(crsp.columns)}")
    print()
    
    return crsp


def load_cds():
    """
    Load CDS prices data (monthly, wide format).
    
    Returns:
        pandas DataFrame with CDS data
    """
    print("3. Loading CDS Prices (Monthly CDS Spreads)...")
    filepath = DATA_DIR / 'CDS Prices.csv'
    
    cds = pd.read_csv(filepath, low_memory=False)
    
    print(f"    ✓ Shape: {len(cds):,} rows × {len(cds.columns)} columns")
    print(f"    ✓ Format: Wide format (firms as rows, dates as columns)")
    print(f"    ✓ First column (firm IDs): {cds.columns[0]}")
    print(f"    ✓ Date columns: {cds.columns[1]} to {cds.columns[-1]}")
    print()
    
    return cds


def load_mapping():
    """
    Load GVKEY ↔ IQT ↔ Company Name mapping.
    
    Returns:
        pandas DataFrame with mapping data
    """
    print("4. Loading Mapping (GVKEY ↔ IQT ↔ Company Name)...")
    filepath = DATA_DIR / 'GVKEY IQT Name matching .csv'
    
    # Load without header since file has no header row
    mapping = load_with_encoding(filepath, header=None)
    
    # Assign column names
    if len(mapping.columns) >= 3:
        mapping.columns = ['gvkey', 'iqt', 'company_name'] + [f'col_{i}' for i in range(3, len(mapping.columns))]
        print(f"    ✓ Assigned column names: {list(mapping.columns[:3])}")
    
    print()
    
    return mapping


def validate_datasets(compustat, crsp, cds, mapping):
    """
    Perform basic validation checks on all datasets.
    
    Args:
        compustat: Compustat DataFrame
        crsp: CRSP DataFrame
        cds: CDS DataFrame
        mapping: Mapping DataFrame
    """
    print_section("VALIDATION SUMMARY")
    
    # Check non-empty
    datasets = {
        'Compustat': compustat,
        'CRSP': crsp,
        'CDS': cds,
        'Mapping': mapping
    }
    
    all_valid = True
    for name, df in datasets.items():
        if df is None or len(df) == 0:
            print(f"✗ {name}: EMPTY or FAILED TO LOAD")
            all_valid = False
        else:
            print(f"✓ {name}: {len(df):,} rows × {len(df.columns)} columns")
    
    print()
    
    if all_valid:
        print("✅ All datasets loaded successfully!")
    else:
        print("⚠️  Some datasets failed to load properly")
    
    print()


def main():
    """
    Main execution function for Step 1.1.
    """
    print_section("STEP 1.1: LOAD ALL DATASETS")
    
    # Load all datasets
    compustat = load_compustat()
    crsp = load_crsp()
    cds = load_cds()
    mapping = load_mapping()
    
    # Validate
    validate_datasets(compustat, crsp, cds, mapping)
    
    print("="*80)
    print("✅ STEP 1.1 COMPLETE: All datasets loaded successfully")
    print("="*80)
    print("\nNext: Run step1_2_basic_profiling.py")
    print()
    
    return compustat, crsp, cds, mapping


if __name__ == "__main__":
    compustat, crsp, cds, mapping = main()
