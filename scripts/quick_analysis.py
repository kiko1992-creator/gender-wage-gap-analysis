"""
Quick Analysis Script
=====================
A simple Python script to quickly analyze gender wage gaps from the command line.

Usage:
    python scripts/quick_analysis.py

Requirements:
    pandas, numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load the cleaned wage data."""
    data_path = Path(__file__).parent.parent / 'data' / 'cleaned' / 'macedonia_wage_cleaned.csv'

    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please run the data cleaning notebook first (02_data_cleaning_solutions.ipynb)")
        return None

    return pd.read_csv(data_path)


def calculate_overall_gap(df):
    """Calculate overall gender wage gap."""
    male_avg = df[df['gender'] == 'Male']['avg_monthly_wage'].mean()
    female_avg = df[df['gender'] == 'Female']['avg_monthly_wage'].mean()
    gap = ((male_avg - female_avg) / male_avg) * 100

    return {
        'male_avg': male_avg,
        'female_avg': female_avg,
        'gap_percent': gap,
        'gap_absolute': male_avg - female_avg
    }


def analyze_by_country(df):
    """Analyze wage gap by country."""
    results = []

    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        male = country_data[country_data['gender'] == 'Male']['avg_monthly_wage'].mean()
        female = country_data[country_data['gender'] == 'Female']['avg_monthly_wage'].mean()
        gap = ((male - female) / male) * 100

        results.append({
            'Country': country,
            'Male Avg': round(male, 2),
            'Female Avg': round(female, 2),
            'Gap (%)': round(gap, 2)
        })

    return pd.DataFrame(results).sort_values('Gap (%)', ascending=False)


def analyze_by_sector(df):
    """Analyze wage gap by sector."""
    results = []

    for sector in df['sector'].unique():
        sector_data = df[df['sector'] == sector]
        male = sector_data[sector_data['gender'] == 'Male']['avg_monthly_wage'].mean()
        female = sector_data[sector_data['gender'] == 'Female']['avg_monthly_wage'].mean()
        gap = ((male - female) / male) * 100

        results.append({
            'Sector': sector,
            'Male Avg': round(male, 2),
            'Female Avg': round(female, 2),
            'Gap (%)': round(gap, 2)
        })

    return pd.DataFrame(results).sort_values('Gap (%)', ascending=False)


def analyze_by_education(df):
    """Analyze wage gap by education level."""
    results = []

    for edu in df['education_level'].unique():
        edu_data = df[df['education_level'] == edu]
        male = edu_data[edu_data['gender'] == 'Male']['avg_monthly_wage'].mean()
        female = edu_data[edu_data['gender'] == 'Female']['avg_monthly_wage'].mean()
        gap = ((male - female) / male) * 100

        results.append({
            'Education': edu,
            'Male Avg': round(male, 2),
            'Female Avg': round(female, 2),
            'Gap (%)': round(gap, 2)
        })

    return pd.DataFrame(results).sort_values('Gap (%)', ascending=False)


def print_report(df):
    """Print a comprehensive analysis report."""
    print("=" * 70)
    print("GENDER WAGE GAP ANALYSIS REPORT")
    print("=" * 70)

    # Overall statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 70)
    overall = calculate_overall_gap(df)
    print(f"Male Average Wage:     {overall['male_avg']:.2f} MKD")
    print(f"Female Average Wage:   {overall['female_avg']:.2f} MKD")
    print(f"Absolute Difference:   {overall['gap_absolute']:.2f} MKD")
    print(f"Percentage Gap:        {overall['gap_percent']:.2f}%")

    # Dataset info
    print(f"\nDataset Size:          {len(df)} records")
    print(f"Countries:             {df['country'].nunique()}")
    print(f"Years Covered:         {df['year'].min()} - {df['year'].max()}")

    # By country
    print("\n🌍 ANALYSIS BY COUNTRY")
    print("-" * 70)
    country_analysis = analyze_by_country(df)
    print(country_analysis.to_string(index=False))

    # By sector
    print("\n🏢 ANALYSIS BY SECTOR")
    print("-" * 70)
    sector_analysis = analyze_by_sector(df)
    print(sector_analysis.to_string(index=False))

    # By education
    print("\n🎓 ANALYSIS BY EDUCATION LEVEL")
    print("-" * 70)
    edu_analysis = analyze_by_education(df)
    print(edu_analysis.to_string(index=False))

    # North Macedonia specific
    macedonia = df[df['country'] == 'North Macedonia']
    if len(macedonia) > 0:
        print("\n🇲🇰 NORTH MACEDONIA FOCUS")
        print("-" * 70)
        mk_overall = calculate_overall_gap(macedonia)
        print(f"Overall Gap:           {mk_overall['gap_percent']:.2f}%")
        print(f"Sample Size:           {len(macedonia)} records")

        # By year
        print("\nTrend Over Time:")
        for year in sorted(macedonia['year'].unique()):
            year_data = macedonia[macedonia['year'] == year]
            year_gap = calculate_overall_gap(year_data)
            print(f"  {year}: {year_gap['gap_percent']:.2f}%")

    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)


def main():
    """Main function."""
    print("\n🚀 Loading data...\n")

    df = load_data()

    if df is None:
        return

    print_report(df)

    # Save report to file
    output_path = Path(__file__).parent.parent / 'data' / 'cleaned' / 'quick_report.txt'
    print(f"\n💾 Saving report to: {output_path}")

    import sys
    from io import StringIO

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print_report(df)
    report_content = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print("✓ Report saved successfully!")


if __name__ == "__main__":
    main()
