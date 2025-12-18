# Project Restructure Plan
## Gender Wage Gap Analysis - Production-Ready Codebase

**Author:** Kiril Mickovski
**Date:** December 2025
**Purpose:** Clean up codebase before adding advanced ML techniques

---

## Current State Analysis

### Files Inventory

| Location | File | Lines | Purpose | Action |
|----------|------|-------|---------|--------|
| `/` | `app.py` | 1008 | Streamlit dashboard | KEEP (will enhance) |
| `/scripts/` | `comprehensive_data_pipeline.py` | ~600 | Data fetching/processing | KEEP |
| `/scripts/` | `full_analysis_report.py` | ~500 | Generate visualizations | KEEP |
| `/scripts/` | `clean_and_add_eu.py` | 163 | One-time data cleaning | ARCHIVE (already ran) |
| `/data/` | `ml_country_data.csv` | - | ML features | MOVE to `/data/processed/` |
| `/data/` | `ml_country_data_clustered.csv` | - | ML with clusters | MOVE to `/data/processed/` |
| `/notebooks/` | `06_enriched_data_exploration.ipynb` | - | Only notebook | KEEP (renumber to 01) |
| `/output/` | Various `.png` files | - | Generated charts | KEEP |

### Codex Branch Files (to merge)

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `scripts/time_series_models.py` | 295 | Time series forecasting | MERGE & ENHANCE |
| `scripts/__init__.py` | 1 | Package init | MERGE |
| `output/time_series_model_report.md` | 16 | Model report | MERGE |
| Changes to `app.py` | +192 | Enhanced time series page | MERGE |

---

## Proposed New Structure

```
gender-wage-gap-analysis/
├── app.py                              # Main Streamlit dashboard
├── requirements.txt                    # Dependencies (updated)
├── README.md                           # Project overview
├── PROJECT_SUMMARY.md                  # Detailed methodology
├── IMPROVEMENT_ROADMAP.md              # Future enhancements
│
├── .devcontainer/                      # Development container
│   └── devcontainer.json
│
├── .streamlit/                         # Streamlit config
│   └── config.toml
│
├── data/
│   ├── raw/                            # Original data files
│   │   └── expanded_balkan_wage_data.csv
│   │
│   ├── reference/                      # External reference data
│   │   └── official_gpg_data.csv
│   │
│   ├── processed/                      # Processed/aggregated data
│   │   ├── validated_wage_data.csv     # Main dataset (146 records)
│   │   ├── country_summary.csv         # Country-level aggregates
│   │   ├── ml_features.csv             # ML feature matrix
│   │   └── ml_features_clustered.csv   # With cluster assignments
│   │
│   └── README.md                       # Data dictionary
│
├── scripts/
│   ├── __init__.py                     # Package init
│   │
│   ├── data_pipeline.py                # Data fetching & processing
│   │                                   # (renamed from comprehensive_data_pipeline.py)
│   │
│   ├── visualization.py                # Chart generation
│   │                                   # (renamed from full_analysis_report.py)
│   │
│   ├── time_series.py                  # Time series forecasting
│   │                                   # (from Codex branch, enhanced)
│   │
│   └── statistics.py                   # Statistical utilities
│                                       # (NEW - for future ML work)
│
├── notebooks/
│   └── 01_data_exploration.ipynb       # Main exploration notebook
│                                       # (renumbered from 06)
│
├── output/
│   ├── figures/                        # Publication-ready figures
│   │   └── *.png, *.pdf
│   │
│   ├── reports/                        # Generated reports
│   │   ├── ANALYSIS_REPORT.md
│   │   ├── ML_ANALYSIS_REPORT.md
│   │   └── time_series_model_report.md
│   │
│   └── publication/                    # Publication materials
│       └── PUBLICATION_CHECKLIST.md
│
├── archive/                            # Archived/deprecated files
│   └── clean_and_add_eu.py             # One-time script (already ran)
│
└── tests/                              # Unit tests (future)
    └── README.md
```

---

## File Operations

### Phase 1: Merge Codex Branch

1. **Merge `time_series_models.py`** into `/scripts/time_series.py`
   - Keep all functionality
   - Add module docstring with theory

2. **Merge app.py changes**
   - Enhanced Time Series page with:
     - Interactive forecast horizon slider
     - Model comparison (Linear, ARIMA, ETS)
     - Confidence bands (80%, 95%)
     - Cross-validation metrics

3. **Add `statsmodels>=0.14.0`** to requirements.txt

### Phase 2: Reorganize Data

```bash
# Move ML data files
mv data/ml_country_data.csv data/processed/ml_features.csv
mv data/ml_country_data_clustered.csv data/processed/ml_features_clustered.csv

# Move validated data
mv data/cleaned/validated_wage_data.csv data/processed/
```

**Update all imports in app.py:**
```python
# Old
data_path = APP_DIR / 'data' / 'cleaned' / 'validated_wage_data.csv'
data_path = APP_DIR / 'data' / 'ml_country_data_clustered.csv'

# New
data_path = APP_DIR / 'data' / 'processed' / 'validated_wage_data.csv'
data_path = APP_DIR / 'data' / 'processed' / 'ml_features_clustered.csv'
```

### Phase 3: Organize Scripts

**Rename files:**
```bash
mv scripts/comprehensive_data_pipeline.py scripts/data_pipeline.py
mv scripts/full_analysis_report.py scripts/visualization.py
mv scripts/clean_and_add_eu.py archive/
```

**Create `/scripts/__init__.py`:**
```python
"""
Gender Wage Gap Analysis - Scripts Module
==========================================
Reusable utilities for data processing, visualization, and ML.
"""

from .data_pipeline import DataPipeline
from .visualization import generate_report
from .time_series import (
    standardize_time_series,
    forecast_country_series,
    evaluate_models_for_country,
    choose_best_model,
)

__all__ = [
    'DataPipeline',
    'generate_report',
    'standardize_time_series',
    'forecast_country_series',
    'evaluate_models_for_country',
    'choose_best_model',
]
```

### Phase 4: Organize Output

```bash
# Create subdirectories
mkdir -p output/figures output/reports output/publication

# Move files
mv output/*.png output/figures/
mv output/publication_figures/* output/figures/
mv output/*.md output/reports/
mv output/PUBLICATION_CHECKLIST.md output/publication/
```

### Phase 5: Cleanup

**Remove:**
- `/data/cleaned/` folder (moving contents to `/data/processed/`)
- Duplicate files
- Empty directories

**Archive:**
- `clean_and_add_eu.py` (one-time script, already executed)

---

## Impact on Imports

### app.py Changes Required

```python
# Line 145: Update data paths
def load_main_data():
    data_path = APP_DIR / 'data' / 'processed' / 'validated_wage_data.csv'  # Changed

def load_country_data():
    data_path = APP_DIR / 'data' / 'processed' / 'ml_features_clustered.csv'  # Changed

# Line 24: Update imports (after Codex merge)
from scripts.time_series import (
    choose_best_model,
    evaluate_models_for_country,
    forecast_country_series,
    prepare_country_series,
    STATS_MODELS_AVAILABLE,
    standardize_time_series,
)
```

---

## Data Files Consolidation

### Current Data Files (7 files, scattered)

| File | Location | Records | Status |
|------|----------|---------|--------|
| `validated_wage_data.csv` | `/data/cleaned/` | 146 | PRIMARY - move |
| `balkan_wage_data_cleaned.csv` | `/data/cleaned/` | ~50 | SUBSET - can remove |
| `ml_country_data.csv` | `/data/` | 12 | MOVE |
| `ml_country_data_clustered.csv` | `/data/` | 12 | MOVE |
| `country_summary_validated.csv` | `/data/processed/` | 12 | KEEP |
| `integrated_wage_data_validated.csv` | `/data/processed/` | ~150 | REDUNDANT? |
| `expanded_balkan_wage_data.csv` | `/data/raw/` | ~100 | RAW - keep |

### Proposed Consolidation

| File | New Location | Purpose |
|------|--------------|---------|
| `validated_wage_data.csv` | `/data/processed/` | Main dataset |
| `ml_features.csv` | `/data/processed/` | ML feature matrix |
| `ml_features_clustered.csv` | `/data/processed/` | ML + clusters |
| `country_summary.csv` | `/data/processed/` | Aggregated stats |
| `expanded_balkan_wage_data.csv` | `/data/raw/` | Original raw data |
| `official_gpg_data.csv` | `/data/reference/` | External reference |

---

## Verification Checklist

After restructure, verify:

- [ ] `streamlit run app.py` works
- [ ] All data loads correctly
- [ ] Time Series page shows forecasts
- [ ] Chart exports work
- [ ] No broken imports

---

## Commit Strategy

1. **Commit 1:** Merge Codex branch (time series models)
2. **Commit 2:** Reorganize data folder structure
3. **Commit 3:** Rename and organize scripts
4. **Commit 4:** Organize output folder
5. **Commit 5:** Update README and documentation
6. **Commit 6:** Add tests placeholder

Or: Single squash commit with full restructure

---

## Questions for You

1. **Data files:** Should we keep `balkan_wage_data_cleaned.csv` or is `validated_wage_data.csv` sufficient?

2. **Old notebooks (01-05):** Were they deleted intentionally or lost? Should we recreate them?

3. **Output organization:** Do you want ML figures in a separate subfolder from publication figures?

4. **Archive vs Delete:** Should we archive `clean_and_add_eu.py` or delete it entirely?

---

## Ready to Proceed?

Once you approve this plan, I will:
1. Merge the Codex branch
2. Execute all file reorganizations
3. Update imports in app.py
4. Verify everything works
5. Create a clean commit

Let me know your thoughts and any modifications!
