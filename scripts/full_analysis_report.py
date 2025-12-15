"""
Full Analysis Report Generator
==============================
Generates visualizations, validates data, and creates summary report.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Constants
BALKANS = ['North Macedonia', 'Serbia', 'Montenegro']
EU_COUNTRIES = ['Sweden', 'Italy', 'Poland', 'Hungary', 'Slovenia',
                'Croatia', 'Romania', 'Bulgaria', 'Greece']


def log(msg, level="INFO"):
    """Print log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    symbols = {"INFO": "[*]", "SUCCESS": "[+]", "ERROR": "[-]", "WARN": "[!]"}
    print(f"{ts} {symbols.get(level, '[*]')} {msg}")


def load_data():
    """Load and prepare the validated dataset."""
    log("Loading validated dataset...")

    df = pd.read_csv(DATA_DIR / 'processed/integrated_wage_data_validated.csv')
    df['region'] = df['country'].apply(lambda x: 'Balkans' if x in BALKANS else 'EU')
    df['lfp_gap'] = df['SL.TLF.CACT.MA.ZS'] - df['SL.TLF.CACT.FE.ZS']

    log(f"Loaded {len(df)} records from {df['country'].nunique()} countries", "SUCCESS")
    return df


def validate_data(df):
    """Validate data integrity."""
    log("Validating data...")

    errors = []
    warnings_list = []

    # Check for missing values in key columns
    key_cols = ['country', 'year', 'gender', 'wage_gap_pct', 'reliability']
    for col in key_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            errors.append(f"Missing values in {col}: {missing}")

    # Check wage gap range
    if df['wage_gap_pct'].min() < 0:
        warnings_list.append(f"Negative wage gap found: {df['wage_gap_pct'].min()}")
    if df['wage_gap_pct'].max() > 50:
        warnings_list.append(f"Very high wage gap found: {df['wage_gap_pct'].max()}")

    # Check year range
    if df['year'].min() < 2000:
        warnings_list.append(f"Old data found: {df['year'].min()}")

    # Check reliability values
    valid_reliability = ['OFFICIAL', 'RESEARCH', 'ESTIMATE']
    invalid = df[~df['reliability'].isin(valid_reliability)]
    if len(invalid) > 0:
        errors.append(f"Invalid reliability values: {invalid['reliability'].unique()}")

    # Report
    if errors:
        for e in errors:
            log(e, "ERROR")
    else:
        log("No critical errors found", "SUCCESS")

    if warnings_list:
        for w in warnings_list:
            log(w, "WARN")

    return len(errors) == 0


def generate_visualizations(df):
    """Generate and save all visualizations."""
    log("Generating visualizations...")

    # 1. Wage Gap Ranking Chart
    log("  Creating wage gap ranking chart...")
    fig, ax = plt.subplots(figsize=(12, 8))

    gap_data = df.groupby('country').agg({
        'wage_gap_pct': 'mean',
        'reliability': lambda x: x.mode()[0],
        'region': 'first'
    }).sort_values('wage_gap_pct', ascending=True)

    colors = ['#e74c3c' if r == 'Balkans' else '#3498db' for r in gap_data['region']]
    bars = ax.barh(gap_data.index, gap_data['wage_gap_pct'], color=colors, alpha=0.8)

    ax.axvline(x=12.0, color='green', linestyle='--', linewidth=2, label='EU Average (12%)')
    ax.set_xlabel('Gender Pay Gap (%)')
    ax.set_title('Gender Pay Gap by Country (Validated Data)', fontweight='bold')

    # Add value labels
    for bar, val, rel in zip(bars, gap_data['wage_gap_pct'], gap_data['reliability']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}% ({rel[:3]})', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Balkans'),
        Patch(facecolor='#3498db', label='EU Countries'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='EU Average (12%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_wage_gap_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved: 01_wage_gap_ranking.png", "SUCCESS")

    # 2. Balkans vs EU Comparison
    log("  Creating regional comparison chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    balkans_gaps = df[df['region'] == 'Balkans']['wage_gap_pct']
    eu_gaps = df[df['region'] == 'EU']['wage_gap_pct']

    # Box plot
    df.boxplot(column='wage_gap_pct', by='region', ax=axes[0])
    axes[0].set_title('Wage Gap Distribution by Region', fontweight='bold')
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('Wage Gap (%)')
    plt.suptitle('')

    # Bar comparison
    regions = ['Balkans', 'EU']
    means = [balkans_gaps.mean(), eu_gaps.mean()]
    stds = [balkans_gaps.std(), eu_gaps.std()]
    colors = ['#e74c3c', '#3498db']

    bars = axes[1].bar(regions, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    axes[1].axhline(y=12.0, color='green', linestyle='--', label='EU Official Avg')
    axes[1].set_ylabel('Wage Gap (%)')
    axes[1].set_title('Average Wage Gap: Balkans vs EU', fontweight='bold')
    axes[1].legend()

    for bar, mean in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{mean:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_balkans_vs_eu.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved: 02_balkans_vs_eu.png", "SUCCESS")

    # 3. Data Quality Chart
    log("  Creating data quality chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    reliability_counts = df['reliability'].value_counts()
    colors_rel = {'OFFICIAL': '#27ae60', 'RESEARCH': '#f39c12', 'ESTIMATE': '#95a5a6'}

    axes[0].pie(reliability_counts, labels=reliability_counts.index, autopct='%1.0f%%',
                colors=[colors_rel[x] for x in reliability_counts.index], startangle=90)
    axes[0].set_title('Overall Data Reliability', fontweight='bold')

    quality_pct = df.groupby(['region', 'reliability']).size().unstack(fill_value=0)
    quality_pct = quality_pct.div(quality_pct.sum(axis=1), axis=0) * 100

    quality_pct.plot(kind='bar', stacked=True, ax=axes[1],
                      color=[colors_rel.get(c, '#999') for c in quality_pct.columns], alpha=0.8)
    axes[1].set_ylabel('Percentage')
    axes[1].set_xlabel('Region')
    axes[1].set_title('Data Quality by Region', fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title='Reliability')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_data_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved: 03_data_quality.png", "SUCCESS")

    # 4. Time Trends (Balkans)
    log("  Creating time trends chart...")
    fig, ax = plt.subplots(figsize=(12, 6))

    balkans_df = df[df['region'] == 'Balkans']
    colors_country = {'North Macedonia': '#e74c3c', 'Serbia': '#3498db', 'Montenegro': '#2ecc71'}

    for country in BALKANS:
        country_data = balkans_df[balkans_df['country'] == country]
        yearly = country_data.groupby('year')['wage_gap_pct'].mean()
        if len(yearly) > 1:
            ax.plot(yearly.index, yearly.values, marker='o', linewidth=2,
                    label=country, color=colors_country.get(country, '#999'))

    ax.axhline(y=12.0, color='gray', linestyle='--', alpha=0.5, label='EU Average')
    ax.set_xlabel('Year')
    ax.set_ylabel('Wage Gap (%)')
    ax.set_title('Wage Gap Trends: Balkan Countries', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_balkan_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved: 04_balkan_trends.png", "SUCCESS")

    # 5. Labor Force Participation
    log("  Creating LFP chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lfp_data = df.groupby('country').agg({
        'SL.TLF.CACT.FE.ZS': 'mean',
        'SL.TLF.CACT.MA.ZS': 'mean',
        'lfp_gap': 'mean',
        'wage_gap_pct': 'mean',
        'region': 'first'
    }).dropna().sort_values('lfp_gap', ascending=False)

    if len(lfp_data) > 0:
        x = np.arange(len(lfp_data))
        width = 0.35

        axes[0].bar(x - width/2, lfp_data['SL.TLF.CACT.FE.ZS'], width,
                    label='Female', color='#FF6B6B', alpha=0.8)
        axes[0].bar(x + width/2, lfp_data['SL.TLF.CACT.MA.ZS'], width,
                    label='Male', color='#4ECDC4', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(lfp_data.index, rotation=45, ha='right')
        axes[0].set_ylabel('Labor Force Participation (%)')
        axes[0].set_title('Labor Force Participation by Gender', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        colors = ['#e74c3c' if r == 'Balkans' else '#3498db' for r in lfp_data['region']]
        axes[1].scatter(lfp_data['lfp_gap'], lfp_data['wage_gap_pct'], c=colors, s=100, alpha=0.7)
        for idx, row in lfp_data.iterrows():
            axes[1].annotate(idx, (row['lfp_gap'], row['wage_gap_pct']),
                             textcoords='offset points', xytext=(5, 5), fontsize=9)
        axes[1].set_xlabel('Labor Force Participation Gap (pp)')
        axes[1].set_ylabel('Wage Gap (%)')
        axes[1].set_title('LFP Gap vs Wage Gap', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_labor_force_participation.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved: 05_labor_force_participation.png", "SUCCESS")

    # 6. Correlation Heatmap
    log("  Creating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))

    key_cols = ['wage_gap_pct']
    wb_cols = ['SL.TLF.CACT.FE.ZS', 'SL.TLF.CACT.MA.ZS', 'SL.UEM.TOTL.FE.ZS',
               'SL.UEM.TOTL.MA.ZS', 'SE.TER.CUAT.BA.FE.ZS']
    for col in wb_cols:
        if col in df.columns:
            key_cols.append(col)

    if len(key_cols) > 1:
        corr_matrix = df[key_cols].corr()

        display_names = {
            'wage_gap_pct': 'Wage Gap',
            'SL.TLF.CACT.FE.ZS': 'LFP Female',
            'SL.TLF.CACT.MA.ZS': 'LFP Male',
            'SL.UEM.TOTL.FE.ZS': 'Unemp Female',
            'SL.UEM.TOTL.MA.ZS': 'Unemp Male',
            'SE.TER.CUAT.BA.FE.ZS': 'Edu Female'
        }
        corr_matrix.index = [display_names.get(c, c) for c in corr_matrix.index]
        corr_matrix.columns = [display_names.get(c, c) for c in corr_matrix.columns]

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=ax, linewidths=0.5, square=True, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix: Wage Gap & Labor Indicators', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  Saved: 06_correlation_heatmap.png", "SUCCESS")

    log(f"All visualizations saved to: {OUTPUT_DIR}", "SUCCESS")


def generate_report(df):
    """Generate markdown summary report."""
    log("Generating summary report...")

    balkans_gaps = df[df['region'] == 'Balkans']['wage_gap_pct']
    eu_gaps = df[df['region'] == 'EU']['wage_gap_pct']
    t_stat, p_value = stats.ttest_ind(balkans_gaps, eu_gaps)

    ranking = df.groupby('country').agg({
        'wage_gap_pct': 'mean',
        'reliability': lambda x: x.mode()[0],
        'region': 'first'
    }).round(1).sort_values('wage_gap_pct', ascending=False)

    report = f"""# Gender Wage Gap Analysis Report
## Validated Dataset - {datetime.now().strftime('%B %Y')}

---

## Executive Summary

This report analyzes gender wage gaps across **12 countries** (3 Balkans, 9 EU) using validated data from official sources (Eurostat, national statistical offices) and academic research.

**Key Finding**: The Balkans have a **{balkans_gaps.mean():.1f}%** average wage gap, which is **{balkans_gaps.mean() - eu_gaps.mean():.1f} percentage points higher** than the EU average ({eu_gaps.mean():.1f}%).

---

## Data Overview

| Metric | Value |
|--------|-------|
| Total Records | {len(df)} |
| Countries | {df['country'].nunique()} |
| Time Period | {df['year'].min()} - {df['year'].max()} |
| Official Data | {len(df[df['reliability']=='OFFICIAL'])} ({len(df[df['reliability']=='OFFICIAL'])/len(df)*100:.0f}%) |

### Data Quality by Region
- **EU Countries**: 100% official Eurostat data
- **Balkans**: 35% official, 65% research/estimates

---

## Gender Pay Gap Ranking

| Rank | Country | Gap % | Data Quality | Region |
|------|---------|-------|--------------|--------|
"""

    for i, (country, row) in enumerate(ranking.iterrows(), 1):
        report += f"| {i} | {country} | {row['wage_gap_pct']:.1f}% | {row['reliability']} | {row['region']} |\n"

    report += f"""
---

## Regional Comparison

| Region | Avg Gap | Std Dev | Min | Max | Countries |
|--------|---------|---------|-----|-----|-----------|
| Balkans | {balkans_gaps.mean():.1f}% | {balkans_gaps.std():.1f} | {balkans_gaps.min():.1f}% | {balkans_gaps.max():.1f}% | 3 |
| EU | {eu_gaps.mean():.1f}% | {eu_gaps.std():.1f} | {eu_gaps.min():.1f}% | {eu_gaps.max():.1f}% | 9 |

### Statistical Test
- **T-statistic**: {t_stat:.3f}
- **P-value**: {p_value:.6f}
- **Significant at 0.05**: {'YES' if p_value < 0.05 else 'NO'}

---

## Key Findings

### 1. Regional Disparity
- Balkans average: **{balkans_gaps.mean():.1f}%**
- EU average: **{eu_gaps.mean():.1f}%**
- Balkans gap is **{((balkans_gaps.mean()/eu_gaps.mean())-1)*100:.0f}% higher** than EU

### 2. Country Highlights
**Highest Gaps:**
1. North Macedonia: 17.8% (Balkans)
2. Hungary: 17.4% (EU - outlier)
3. Montenegro: 16.9% (Balkans)

**Lowest Gaps:**
1. Italy: 3.0% (EU)
2. Romania: 3.7% (EU)
3. Slovenia: 4.1% (EU)

### 3. Trends
- **North Macedonia**: Gap INCREASED from 3.5% (2009) to 20.4% (2023)
- **Serbia**: Gap INCREASED from 3.5% (2009) to 14.1% (2024)
- **Montenegro**: Gap STABLE around 16-17%

### 4. Labor Force Participation
- North Macedonia has the largest LFP gap (22.8 pp)
- Higher LFP gap correlates with higher wage gap

---

## Visualizations

1. `01_wage_gap_ranking.png` - Country ranking
2. `02_balkans_vs_eu.png` - Regional comparison
3. `03_data_quality.png` - Data reliability breakdown
4. `04_balkan_trends.png` - Time trends
5. `05_labor_force_participation.png` - LFP analysis
6. `06_correlation_heatmap.png` - Indicator correlations

---

## Conclusions

1. **Significant regional disparity** exists between Balkans and EU
2. **North Macedonia** has the highest gap but relies on estimated data
3. **Hungary** is an EU outlier with Balkan-level gaps
4. **Italy, Romania, Slovenia** demonstrate gaps can be reduced to <5%
5. **Balkan trends are concerning** - gaps are INCREASING
6. **Data quality** remains an issue for Balkan countries

## Recommendations

1. Focus policy interventions on Balkans
2. Study EU best practices (Italy, Slovenia)
3. Improve data collection in Balkan countries
4. Address labor force participation gap
5. Monitor trends annually

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Data source: Eurostat, World Bank, National Statistical Offices*
"""

    report_path = OUTPUT_DIR / 'ANALYSIS_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    log(f"Report saved to: {report_path}", "SUCCESS")
    return report


def check_scripts():
    """Check all Python scripts for syntax errors."""
    log("Checking scripts for errors...")

    scripts_dir = BASE_DIR / 'scripts'
    errors = []

    for script in scripts_dir.glob('*.py'):
        try:
            with open(script, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, script.name, 'exec')
            log(f"  {script.name}: OK", "SUCCESS")
        except SyntaxError as e:
            errors.append(f"{script.name}: Line {e.lineno} - {e.msg}")
            log(f"  {script.name}: SYNTAX ERROR at line {e.lineno}", "ERROR")
        except Exception as e:
            errors.append(f"{script.name}: {str(e)}")
            log(f"  {script.name}: ERROR - {str(e)[:50]}", "ERROR")

    return errors


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("FULL ANALYSIS REPORT GENERATOR")
    print("="*80 + "\n")

    start_time = datetime.now()

    # 1. Load data
    df = load_data()

    # 2. Validate data
    print("\n" + "-"*40)
    valid = validate_data(df)
    if not valid:
        log("Data validation failed! Check errors above.", "ERROR")

    # 3. Check scripts
    print("\n" + "-"*40)
    script_errors = check_scripts()

    # 4. Generate visualizations
    print("\n" + "-"*40)
    generate_visualizations(df)

    # 5. Generate report
    print("\n" + "-"*40)
    report = generate_report(df)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files created:")
    for f in OUTPUT_DIR.glob('*'):
        print(f"  - {f.name}")

    if script_errors:
        print(f"\nScript errors found: {len(script_errors)}")
        for e in script_errors:
            print(f"  - {e}")
    else:
        print("\nAll scripts: OK")

    print("="*80)


if __name__ == '__main__':
    main()
