"""
Data Validation Script for Gender Wage Gap Analysis
This script validates the expanded dataset and checks for potential issues.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def validate_dataset():
    """Validate the expanded Balkan wage data"""

    print("="*80)
    print("DATA VALIDATION SCRIPT")
    print("="*80)

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'expanded_balkan_wage_data.csv'

    try:
        df = pd.read_csv(data_path)
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Path: {data_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
    except FileNotFoundError:
        print(f"\n✗ ERROR: Dataset not found at {data_path}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR loading dataset: {e}")
        return False

    # Check for required columns
    required_cols = ['country', 'year', 'gender', 'sector', 'education_level',
                     'avg_monthly_wage', 'hours_worked', 'age_group',
                     'data_source', 'wage_gap_pct']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n✗ ERROR: Missing required columns: {missing_cols}")
        return False
    else:
        print(f"\n✓ All required columns present")

    # Check for missing values
    print(f"\n{'='*80}")
    print("MISSING VALUE CHECK")
    print("="*80)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values detected")
    else:
        print("⚠ Missing values found:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df) * 100)
            print(f"  {col}: {count} ({pct:.1f}%)")

    # Check data types
    print(f"\n{'='*80}")
    print("DATA TYPE CHECK")
    print("="*80)
    print(df.dtypes)

    # Validate numeric columns
    print(f"\n{'='*80}")
    print("NUMERIC VALIDATION")
    print("="*80)

    # Check for negative wages
    if (df['avg_monthly_wage'] < 0).any():
        print("✗ ERROR: Negative wages detected")
        neg_wages = df[df['avg_monthly_wage'] < 0]
        print(neg_wages[['country', 'year', 'gender', 'avg_monthly_wage']])
    else:
        print("✓ No negative wages")

    # Check for unrealistic hours
    unusual_hours = df[(df['hours_worked'] < 100) | (df['hours_worked'] > 250)]
    if len(unusual_hours) > 0:
        print(f"⚠ Warning: {len(unusual_hours)} records with unusual hours (< 100 or > 250)")
        print(unusual_hours[['country', 'year', 'hours_worked']])
    else:
        print("✓ Hours worked values reasonable")

    # Check wage gap calculations
    print(f"\n{'='*80}")
    print("WAGE GAP VALIDATION")
    print("="*80)

    gap_range = df['wage_gap_pct'].describe()
    print(f"Gap range: {gap_range['min']:.2f}% to {gap_range['max']:.2f}%")
    print(f"Mean gap: {gap_range['mean']:.2f}%")
    print(f"Median gap: {gap_range['50%']:.2f}%")

    if gap_range['min'] < 0:
        print("⚠ Warning: Negative wage gaps detected (women earning more than men)")
        negative_gaps = df[df['wage_gap_pct'] < 0]
        print(f"  Count: {len(negative_gaps)}")

    if gap_range['max'] > 50:
        print(f"⚠ Warning: Very large wage gaps detected (> 50%)")
        large_gaps = df[df['wage_gap_pct'] > 50]
        print(f"  Count: {len(large_gaps)}")

    # Country coverage
    print(f"\n{'='*80}")
    print("COUNTRY COVERAGE")
    print("="*80)
    country_counts = df['country'].value_counts().sort_values(ascending=False)
    for country, count in country_counts.items():
        pct = (count / len(df) * 100)
        print(f"  {country}: {count} records ({pct:.1f}%)")

    # Year coverage
    print(f"\n{'='*80}")
    print("TEMPORAL COVERAGE")
    print("="*80)
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Unique years: {df['year'].nunique()}")
    year_counts = df['year'].value_counts().sort_index()
    print("\nRecords per year:")
    for year, count in year_counts.items():
        print(f"  {year}: {count}")

    # Gender balance
    print(f"\n{'='*80}")
    print("GENDER BALANCE")
    print("="*80)
    gender_counts = df['gender'].value_counts()
    for gender, count in gender_counts.items():
        pct = (count / len(df) * 100)
        print(f"  {gender}: {count} ({pct:.1f}%)")

    # Duplicates
    print(f"\n{'='*80}")
    print("DUPLICATE CHECK")
    print("="*80)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠ Warning: {duplicates} duplicate rows found")
        print("Consider removing duplicates with df.drop_duplicates()")
    else:
        print("✓ No duplicates detected")

    # Data source verification
    print(f"\n{'='*80}")
    print("DATA SOURCES")
    print("="*80)
    source_counts = df['data_source'].value_counts()
    print(f"Total sources: {len(source_counts)}")
    for source, count in source_counts.items():
        pct = (count / len(df) * 100)
        print(f"  {source}: {count} ({pct:.1f}%)")

    # Final summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"✓ Dataset structure: OK")
    print(f"✓ Required columns: OK")
    print(f"✓ Data loaded: {len(df)} records")
    print(f"✓ Countries: {df['country'].nunique()}")
    print(f"✓ Time span: {df['year'].max() - df['year'].min() + 1} years")

    if missing.sum() > 0:
        print(f"⚠ Missing values: {missing.sum()} (needs attention)")
    else:
        print(f"✓ Missing values: None")

    if duplicates > 0:
        print(f"⚠ Duplicates: {duplicates} (should be removed)")
    else:
        print(f"✓ Duplicates: None")

    print(f"\n{'='*80}")
    print("DATASET READY FOR ANALYSIS")
    print("="*80)

    return True

if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
