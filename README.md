# Gender Wage Gap Analysis: Balkans vs European Union

## Comprehensive Statistical & Machine Learning Analysis (2009-2024)

A rigorous data science investigation comparing gender wage disparities between **Balkan countries** and **EU member states**, featuring multivariate regression, clustering, time series forecasting, and an interactive dashboard.

**Author:** Kiril Mickovski
**Status:** Publication-Ready | ML Analysis Complete | Interactive Dashboard Available

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)]()
[![License](https://img.shields.io/badge/License-Educational-orange)]()

---

## Key Findings

| Metric | Balkans | EU | Difference |
|--------|---------|-----|------------|
| **Average Wage Gap** | 15.7% | 9.0% | **+6.7 pp** |
| **Unemployment Rate** | 13.1% | 5.9% | +7.2 pp |
| **Female LFP** | 44.7% | 62.1% | -17.4 pp |

**Statistical Significance:** p = 0.017 (t-test)

### Top Insights

1. **Unemployment is the #1 predictor** of wage gaps (30.7% feature importance)
2. **Balkans trend upward** while EU remains stable
3. **North Macedonia** increasing by +1 pp/year (27.8% forecast by 2030)
4. **Italy's paradox** explained: 2.2% gap masks low female participation

---

## Interactive Dashboard

Launch the Streamlit dashboard to explore the data interactively:

```bash
cd gender-wage-gap-analysis
streamlit run app.py
```

**Features:**
- Country comparison with radar charts
- Regional analysis (Balkans vs EU)
- Time series visualization
- ML model insights (regression, clustering, PCA)
- Oaxaca-Blinder decomposition results
- Data explorer with download options

---

## Project Structure

```
gender-wage-gap-analysis/
├── app.py                          # Streamlit dashboard
├── data/
│   ├── cleaned/
│   │   └── validated_wage_data.csv # Main dataset (146 records, 12 countries)
│   ├── ml_country_data.csv         # Country-level features
│   └── ml_country_data_clustered.csv
├── notebooks/
│   └── 06_enriched_data_exploration.ipynb
├── output/
│   ├── publication_figures/        # Publication-ready charts (PNG + PDF)
│   ├── ML_ANALYSIS_REPORT.md       # Detailed ML findings
│   ├── PUBLICATION_CHECKLIST.md    # Thesis correction guide
│   └── ml_*.png                    # ML visualizations
├── scripts/
│   ├── comprehensive_data_pipeline.py
│   ├── clean_and_add_eu.py
│   └── full_analysis_report.py
├── requirements.txt
└── README.md
```

---

## Countries Analyzed

### Balkans (n=3)
| Country | Gap | Trend | Data Source |
|---------|-----|-------|-------------|
| North Macedonia | 17.8% | Increasing | State Statistical Office |
| Montenegro | 16.9% | Stable | Research estimate |
| Serbia | 13.8% | Increasing | Statistical Office Serbia |

### EU Member States (n=9)
| Country | Gap | Source |
|---------|-----|--------|
| Hungary | 17.8% | Eurostat |
| Greece | 13.6% | Eurostat |
| Bulgaria | 13.5% | Eurostat |
| Sweden | 11.3% | Eurostat |
| Croatia | 10.0% | Eurostat |
| Poland | 7.8% | Eurostat |
| Slovenia | 5.4% | Eurostat |
| Romania | 3.8% | Eurostat |
| Italy | 2.2% | Eurostat |

---

## Machine Learning Analysis

### 1. Multivariate Regression (R² = 0.80)

| Variable | Coefficient | p-value | Interpretation |
|----------|-------------|---------|----------------|
| Unemployment | +1.74 | 0.002 | 1% unemployment = +1.74pp gap |
| GDP per capita | -0.0005 | 0.004 | Higher GDP = lower gap |
| Female LFP | +0.95 | 0.010 | Selection effect |

### 2. K-Means Clustering (k=3)

The algorithm **naturally separated Balkans** into a high-gap cluster without region labels:
- **Cluster 0** (Green): Mid-range EU countries
- **Cluster 1** (Red): High-gap countries (Balkans + Greece)
- **Cluster 2** (Blue): Sweden (unique profile)

### 3. Random Forest Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Unemployment | 30.7% |
| 2 | LFP Gap | 21.0% |
| 3 | GDP per capita | 15.6% |
| 4 | Log GDP | 12.7% |
| 5 | Male LFP | 10.6% |
| 6 | Female LFP | 9.5% |

### 4. PCA Analysis

- **PC1 (65.3%):** Economic Development dimension
- **PC2 (19.0%):** Gender Inequality dimension
- Balkans cluster in low-PC1, high-PC2 quadrant

### 5. Time Series Forecasting

| Country | Trend | 2030 Forecast |
|---------|-------|---------------|
| North Macedonia | +0.96 pp/year | 27.8% |
| Serbia | +0.60 pp/year | 16.8% |
| Montenegro | +0.18 pp/year | 18.8% |

### 6. Oaxaca-Blinder Decomposition

- **78% Explained** by observable factors (unemployment, GDP, LFP)
- **22% Unexplained** (structural/discrimination effects)
- Main contributor: Unemployment (+4.88 pp)

---

## Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/gender-wage-gap-analysis.git
cd gender-wage-gap-analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Or explore with Jupyter
jupyter notebook
```

### Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn, plotly
- streamlit
- jupyter

---

## Data Sources

| Source | Data Type | Countries |
|--------|-----------|-----------|
| **Eurostat** | Official GPG (SDG_05_20) | EU members |
| **State Statistical Office MK** | Detailed wage data | North Macedonia |
| **Statistical Office Serbia** | Gender pay statistics | Serbia |
| **World Bank** | GDP, Labor indicators | All |
| **ILO** | Labor force participation | All |

**Citation:**
```
Eurostat. (2023). Gender pay gap in unadjusted form [SDG_05_20].
Retrieved from https://ec.europa.eu/eurostat/databrowser/view/SDG_05_20
```

---

## Output Files

### Publication Figures (300 DPI)
- `Fig1_country_ranking.png/pdf` - All 12 countries ranked
- `Fig2_regional_comparison.png/pdf` - Balkans vs EU with significance
- `Fig3_balkan_trends.png/pdf` - Time series 2009-2024
- `Fig4_eu_comparison.png/pdf` - EU countries comparison

### ML Visualizations
- `ml_regression_plots.png` - Scatter plots with trend lines
- `ml_kmeans_clustering.png` - Cluster visualization
- `ml_random_forest.png` - Feature importance
- `ml_pca_analysis.png` - Biplot
- `ml_time_series_forecast.png` - 2025-2030 projections
- `ml_oaxaca_blinder.png` - Decomposition analysis
- `ml_complete_dashboard.png` - Summary dashboard

### Reports
- `ML_ANALYSIS_REPORT.md` - Detailed ML methodology and findings
- `PUBLICATION_CHECKLIST.md` - Thesis correction guide
- `ANALYSIS_REPORT.md` - Full statistical analysis

---

## Usage

### Run Full Analysis Pipeline
```python
python scripts/comprehensive_data_pipeline.py
```

### Generate Publication Charts
```python
python scripts/full_analysis_report.py
```

### Launch Interactive Dashboard
```bash
streamlit run app.py
```

### Explore in Jupyter
```bash
jupyter notebook notebooks/06_enriched_data_exploration.ipynb
```

---

## Methodology

### Data Cleaning
- Removed unreliable countries (Albania, Kosovo, BiH) due to data gaps
- Validated against official Eurostat figures
- Standardized to unadjusted hourly wage gap measure

### Statistical Tests
- Independent samples t-test (Balkans vs EU)
- Pearson correlation analysis
- Linear regression with OLS

### Machine Learning
- K-Means clustering with elbow method
- Random Forest for feature importance
- PCA for dimensionality reduction
- Linear trend models for forecasting

---

## Key Policy Implications

1. **Address unemployment first** - strongest lever for reducing wage gaps
2. **Balkans need targeted intervention** - gaps diverging from EU
3. **Low female LFP masks issues** - Italy example shows selection effects
4. **EU accession** - current trends complicate alignment with EU standards

---

## License

Educational and research use. Data sourced from public statistical offices.

---

## Acknowledgments

- Eurostat for comprehensive gender pay gap data
- National Statistical Offices of North Macedonia and Serbia
- World Bank Gender Data Portal
- Academic researchers studying Balkan labor markets

---

**Last Updated:** December 2025
**Analysis Version:** 2.0
**Contact:** Kiril Mickovski
