# Gender Wage Gap Analysis

Statistical analysis and visualization of gender wage disparities in Balkan countries compared to the European Union.

## Overview

This project analyzes wage gap trends using data from official statistical sources, providing interactive visualizations and statistical insights.

## Features

- **Interactive Dashboard**: 9 analytical pages covering different aspects
- **Statistical Analysis**: Regression models, clustering, and forecasting
- **Data Visualization**: Interactive charts with Plotly
- **Data Export**: Download analysis results and visualizations

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## Project Structure

```
├── app.py                      # Main Streamlit dashboard
├── data/processed/             # Cleaned datasets
├── scripts/                    # Analysis scripts
├── tests/                      # Automated tests
└── requirements.txt            # Dependencies
```

## Dashboard Pages

1. **Overview** - Country rankings and KPIs
2. **Country Profiles** - Detailed country analysis
3. **Country Comparison** - Side-by-side comparisons
4. **Regional Analysis** - Statistical comparison
5. **Time Series** - Forecasting and trends
6. **What-If Analysis** - Scenario modeling
7. **ML Insights** - Machine learning analysis
8. **Oaxaca-Blinder** - Wage gap decomposition
9. **Data Explorer** - Interactive data tables

## Data Sources

- Eurostat
- World Bank
- ILO
- National Statistical Offices

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Testing

```bash
pytest tests/ -v
```

## License

Available for research and educational purposes.
