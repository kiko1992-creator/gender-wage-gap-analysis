# Data Directory

## Structure

```
data/
├── raw/                    # Original, unmodified data files
├── reference/              # External reference data (Eurostat, etc.)
├── processed/              # Cleaned, validated, ready-to-use data
└── README.md               # This file
```

## Data Files

### /processed/ (Primary data for analysis)

| File | Records | Description |
|------|---------|-------------|
| `validated_wage_data.csv` | 146 | Main dataset: 12 countries, 2009-2024 |
| `balkan_wage_data_cleaned.csv` | ~50 | Balkan countries only subset |
| `ml_features.csv` | 12 | Country-level features for ML |
| `ml_features_clustered.csv` | 12 | ML features with cluster assignments |
| `country_summary_validated.csv` | 12 | Aggregated country statistics |

### /raw/ (Original data)

| File | Description |
|------|-------------|
| `expanded_balkan_wage_data.csv` | Original Balkan wage data |

### /reference/ (External sources)

| File | Source | Description |
|------|--------|-------------|
| `official_gpg_data.csv` | Eurostat | Official gender pay gap data |

## Data Dictionary

### validated_wage_data.csv

| Column | Type | Description |
|--------|------|-------------|
| `country` | string | Country name |
| `year` | int | Year of observation |
| `gender` | string | 'Male' or 'Female' |
| `sector` | string | Economic sector |
| `education_level` | string | Education level |
| `avg_monthly_wage` | float | Average monthly wage (normalized) |
| `hours_worked` | int | Hours worked per month |
| `age_group` | string | Age group category |
| `data_source` | string | Original data source |
| `wage_gap_pct` | float | Gender wage gap percentage |
| `notes` | string | Additional notes |
| `reliability` | string | OFFICIAL, RESEARCH, or ESTIMATE |

### ml_features_clustered.csv

| Column | Type | Description |
|--------|------|-------------|
| `country` | string | Country name |
| `region` | string | 'Balkans' or 'EU' |
| `gap_mean` | float | Average wage gap (%) |
| `female_lfp` | float | Female labor force participation (%) |
| `male_lfp` | float | Male labor force participation (%) |
| `lfp_gap` | float | LFP gap (male - female) |
| `gdp_per_capita` | float | GDP per capita (USD) |
| `unemployment` | float | Unemployment rate (%) |
| `cluster` | int | K-Means cluster assignment |

## Data Sources

| Source | URL | Data Type |
|--------|-----|-----------|
| Eurostat | ec.europa.eu/eurostat | EU wage gap statistics |
| State Statistical Office MK | stat.gov.mk | North Macedonia data |
| Statistical Office Serbia | stat.gov.rs | Serbia data |
| World Bank | data.worldbank.org | GDP, labor indicators |
| ILO | ilostat.ilo.org | Labor force participation |

## Countries Included

### Balkans (n=3)
- North Macedonia
- Serbia
- Montenegro

### EU (n=9)
- Bulgaria, Croatia, Greece, Hungary
- Italy, Poland, Romania, Slovenia, Sweden

## Data Quality

- **Reliability ratings**: OFFICIAL (Eurostat, national offices), RESEARCH (academic), ESTIMATE (projections)
- **Validation**: Cross-referenced against official Eurostat figures (SDG_05_20)
- **Excluded**: Albania, Kosovo, Bosnia (insufficient reliable data)

## Usage

```python
import pandas as pd

# Load main dataset
df = pd.read_csv('data/processed/validated_wage_data.csv')

# Load ML features
df_ml = pd.read_csv('data/processed/ml_features_clustered.csv')
```

---
*Last updated: December 2025*
