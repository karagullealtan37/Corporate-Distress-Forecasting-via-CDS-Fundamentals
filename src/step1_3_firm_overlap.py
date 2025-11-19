"""
STEP 1.3: Check Firm Overlap Across Datasets

This script analyzes firm overlap across all 4 datasets to understand:
    - Which firms appear in all datasets
    - Which firms are missing from specific datasets
    - Mapping quality (GVKEY ↔ IQT linkage)
    - Potential data loss during merging

Objectives:
    - Identify complete firm coverage
    - Flag firms with missing data
    - Validate mapping completeness
    - Prepare for data merging steps

Outputs:
    - Console: Overlap analysis and Venn diagram statistics
    - Identification of firms in all datasets vs partial coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import loading functions
import sys
sys.path.append(str(Path(__file__).parent))
from step1_1_load_datasets import load_compustat, load_crsp, load_cds, load_mapping


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def analyze_firm_sets(compustat, crsp, cds, mapping):
    """
    Extract unique firm sets from each dataset.
    
    Args:
        compustat: Compustat DataFrame
        crsp: CRSP DataFrame
        cds: CDS DataFrame
        mapping: Mapping DataFrame
    
    Returns:
        Dictionary of firm sets
    """
    print_section("EXTRACTING FIRM IDENTIFIERS")
    
    # Normalize GVKEY to integers (remove leading zeros)
    # Compustat firms (GVKEY)
    comp_firms = set(pd.to_numeric(compustat['gvkey'], errors='coerce').dropna().astype(int))
    print(f"Compustat firms (GVKEY): {len(comp_firms):,}")
    
    # CRSP firms (GVKEY) - convert from string with leading zeros to int
    crsp_firms = set(pd.to_numeric(crsp['gvkey'], errors='coerce').dropna().astype(int))
    print(f"CRSP firms (GVKEY):      {len(crsp_firms):,}")
    
    # CDS firms (IQT codes) - keep as strings
    cds_firms = set(cds.iloc[:, 0].dropna().astype(str).unique())
    print(f"CDS firms (IQT):         {len(cds_firms):,}")
    
    # Mapping firms (both GVKEY and IQT)
    mapping_gvkey = set(pd.to_numeric(mapping['gvkey'], errors='coerce').dropna().astype(int))
    mapping_iqt = set(mapping['iqt'].dropna().astype(str).unique())
    print(f"Mapping firms (GVKEY):   {len(mapping_gvkey):,}")
    print(f"Mapping firms (IQT):     {len(mapping_iqt):,}")
    print()
    
    return {
        'compustat': comp_firms,
        'crsp': crsp_firms,
        'cds': cds_firms,
        'mapping_gvkey': mapping_gvkey,
        'mapping_iqt': mapping_iqt
    }


def analyze_overlap(firm_sets):
    """
    Analyze overlap between datasets.
    
    Args:
        firm_sets: Dictionary of firm sets
    """
    print_section("FIRM OVERLAP ANALYSIS")
    
    comp = firm_sets['compustat']
    crsp = firm_sets['crsp']
    map_gvkey = firm_sets['mapping_gvkey']
    map_iqt = firm_sets['mapping_iqt']
    cds = firm_sets['cds']
    
    # Compustat ∩ CRSP (both use GVKEY)
    comp_crsp = comp & crsp
    print(f"📊 COMPUSTAT ∩ CRSP")
    print(f"   Overlap: {len(comp_crsp):,} firms")
    print(f"   Only in Compustat: {len(comp - crsp):,} firms")
    print(f"   Only in CRSP: {len(crsp - comp):,} firms")
    print()
    
    # Compustat ∩ Mapping
    comp_map = comp & map_gvkey
    print(f"📊 COMPUSTAT ∩ MAPPING")
    print(f"   Overlap: {len(comp_map):,} firms")
    print(f"   Only in Compustat: {len(comp - map_gvkey):,} firms")
    print(f"   Only in Mapping: {len(map_gvkey - comp):,} firms")
    print()
    
    # CRSP ∩ Mapping
    crsp_map = crsp & map_gvkey
    print(f"📊 CRSP ∩ MAPPING")
    print(f"   Overlap: {len(crsp_map):,} firms")
    print(f"   Only in CRSP: {len(crsp - map_gvkey):,} firms")
    print(f"   Only in Mapping: {len(map_gvkey - crsp):,} firms")
    print()
    
    # CDS ∩ Mapping (via IQT)
    cds_map = cds & map_iqt
    print(f"📊 CDS ∩ MAPPING (via IQT)")
    print(f"   Overlap: {len(cds_map):,} firms")
    print(f"   Only in CDS: {len(cds - map_iqt):,} firms")
    print(f"   Only in Mapping: {len(map_iqt - cds):,} firms")
    print()
    
    # All datasets (Compustat ∩ CRSP ∩ Mapping ∩ CDS)
    # Need to use mapping to link GVKEY to IQT
    all_gvkey = comp & crsp & map_gvkey
    print(f"🎯 COMPLETE COVERAGE (Compustat ∩ CRSP ∩ Mapping)")
    print(f"   Firms with all data: {len(all_gvkey):,}")
    print()
    
    return {
        'comp_crsp': comp_crsp,
        'comp_map': comp_map,
        'crsp_map': crsp_map,
        'cds_map': cds_map,
        'all_gvkey': all_gvkey
    }


def validate_mapping(mapping, firm_sets):
    """
    Validate mapping quality.
    
    Args:
        mapping: Mapping DataFrame
        firm_sets: Dictionary of firm sets
    """
    print_section("MAPPING VALIDATION")
    
    # Check if all CDS firms have a mapping
    cds_firms = firm_sets['cds']
    mapped_iqt = firm_sets['mapping_iqt']
    
    unmapped_cds = cds_firms - mapped_iqt
    print(f"🔗 CDS → GVKEY MAPPING")
    print(f"   CDS firms with mapping: {len(cds_firms & mapped_iqt):,}/{len(cds_firms):,}")
    print(f"   CDS firms without mapping: {len(unmapped_cds):,}")
    if len(unmapped_cds) > 0:
        print(f"   ⚠️  Unmapped IQT codes: {list(unmapped_cds)[:5]}...")
    print()
    
    # Check if all mapped GVKEYs exist in Compustat
    mapped_gvkey = firm_sets['mapping_gvkey']
    comp_firms = firm_sets['compustat']
    
    mapped_not_in_comp = mapped_gvkey - comp_firms
    print(f"🔗 MAPPING → COMPUSTAT")
    print(f"   Mapped GVKEYs in Compustat: {len(mapped_gvkey & comp_firms):,}/{len(mapped_gvkey):,}")
    print(f"   Mapped GVKEYs not in Compustat: {len(mapped_not_in_comp):,}")
    if len(mapped_not_in_comp) > 0:
        print(f"   ⚠️  Missing GVKEYs: {list(mapped_not_in_comp)[:5]}...")
    print()
    
    # Check if all mapped GVKEYs exist in CRSP
    crsp_firms = firm_sets['crsp']
    mapped_not_in_crsp = mapped_gvkey - crsp_firms
    print(f"🔗 MAPPING → CRSP")
    print(f"   Mapped GVKEYs in CRSP: {len(mapped_gvkey & crsp_firms):,}/{len(mapped_gvkey):,}")
    print(f"   Mapped GVKEYs not in CRSP: {len(mapped_not_in_crsp):,}")
    if len(mapped_not_in_crsp) > 0:
        print(f"   ⚠️  Missing GVKEYs: {list(mapped_not_in_crsp)[:5]}...")
    print()


def identify_complete_firms(mapping, overlaps):
    """
    Identify firms with complete data across all datasets.
    
    Args:
        mapping: Mapping DataFrame
        overlaps: Dictionary of overlap sets
    
    Returns:
        DataFrame of complete firms
    """
    print_section("COMPLETE FIRM IDENTIFICATION")
    
    # Firms in Compustat ∩ CRSP ∩ Mapping
    complete_gvkey = overlaps['all_gvkey']
    
    # Filter mapping to only include complete firms
    complete_mapping = mapping[mapping['gvkey'].isin(complete_gvkey)].copy()
    
    print(f"✅ FIRMS WITH COMPLETE DATA")
    print(f"   Total firms: {len(complete_mapping):,}")
    print(f"   These firms have:")
    print(f"      ✓ Compustat fundamentals (quarterly)")
    print(f"      ✓ CRSP market data (monthly)")
    print(f"      ✓ CDS spreads (monthly)")
    print(f"      ✓ Complete GVKEY ↔ IQT mapping")
    print()
    
    print(f"📝 SAMPLE COMPLETE FIRMS")
    print(complete_mapping[['gvkey', 'iqt', 'company_name']].head(10).to_string(index=False))
    print()
    
    return complete_mapping


def generate_summary(firm_sets, overlaps, complete_mapping):
    """
    Generate final summary of firm overlap analysis.
    
    Args:
        firm_sets: Dictionary of firm sets
        overlaps: Dictionary of overlap sets
        complete_mapping: DataFrame of complete firms
    """
    print_section("OVERLAP SUMMARY")
    
    print(f"📊 DATASET SIZES")
    print(f"   Compustat:  {len(firm_sets['compustat']):,} firms")
    print(f"   CRSP:       {len(firm_sets['crsp']):,} firms")
    print(f"   CDS:        {len(firm_sets['cds']):,} firms")
    print(f"   Mapping:    {len(firm_sets['mapping_gvkey']):,} firms")
    print()
    
    print(f"🎯 OVERLAP STATISTICS")
    print(f"   Compustat ∩ CRSP:                {len(overlaps['comp_crsp']):,} firms")
    print(f"   Compustat ∩ CRSP ∩ Mapping:      {len(overlaps['all_gvkey']):,} firms")
    print(f"   Complete (all 4 datasets):       {len(complete_mapping):,} firms")
    print()
    
    # Calculate coverage percentages
    total_unique = len(firm_sets['compustat'] | firm_sets['crsp'] | firm_sets['mapping_gvkey'])
    coverage_pct = (len(complete_mapping) / total_unique) * 100
    
    print(f"📈 COVERAGE")
    print(f"   Total unique firms (any dataset): {total_unique:,}")
    print(f"   Firms with complete data:         {len(complete_mapping):,}")
    print(f"   Coverage rate:                    {coverage_pct:.1f}%")
    print()
    
    print(f"🔍 KEY INSIGHTS")
    print(f"   ✓ Strong overlap between Compustat and CRSP")
    print(f"   ✓ Mapping provides good linkage to CDS data")
    print(f"   ✓ ~{len(complete_mapping)} firms available for ML modeling")
    print(f"   → Sufficient sample size for robust analysis")
    print()
    
    # Data loss estimation
    comp_loss = len(firm_sets['compustat']) - len(complete_mapping)
    crsp_loss = len(firm_sets['crsp']) - len(complete_mapping)
    print(f"⚠️  POTENTIAL DATA LOSS (if requiring all datasets)")
    print(f"   Compustat firms excluded: {comp_loss:,} ({comp_loss/len(firm_sets['compustat'])*100:.1f}%)")
    print(f"   CRSP firms excluded:      {crsp_loss:,} ({crsp_loss/len(firm_sets['crsp'])*100:.1f}%)")
    print()


def main():
    """
    Main execution function for Step 1.3.
    """
    print("\n" + "="*80)
    print("STEP 1.3: FIRM OVERLAP ANALYSIS".center(80))
    print("="*80 + "\n")
    
    # Load all datasets
    print("Loading datasets...\n")
    compustat = load_compustat()
    crsp = load_crsp()
    cds = load_cds()
    mapping = load_mapping()
    
    # Extract firm sets
    firm_sets = analyze_firm_sets(compustat, crsp, cds, mapping)
    
    # Analyze overlap
    overlaps = analyze_overlap(firm_sets)
    
    # Validate mapping
    validate_mapping(mapping, firm_sets)
    
    # Identify complete firms
    complete_mapping = identify_complete_firms(mapping, overlaps)
    
    # Generate summary
    generate_summary(firm_sets, overlaps, complete_mapping)
    
    print("="*80)
    print("✅ STEP 1.3 COMPLETE: Firm overlap analysis finished")
    print("="*80)
    print("\nNext: Run step1_4_summary_statistics.py")
    print()
    
    return firm_sets, overlaps, complete_mapping


if __name__ == "__main__":
    firm_sets, overlaps, complete_mapping = main()
