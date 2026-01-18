# Streamlit Cloud Deployment Guide

## Quick Fix for Current Deployment Issue

### Step 1: Verify Streamlit Cloud Settings

Go to your Streamlit Cloud dashboard and ensure:

1. **Repository**: `kiko1992-creator/gender-wage-gap-analysis`
2. **Branch**: `claude/streamlit-production-optimization-BvCxo` (or master after merge)
3. **Main file path**: `app.py`

### Step 2: Required Files Checklist

✅ All files present on this branch:
- `app.py` - Main dashboard entry point
- `requirements.txt` - All dependencies including psycopg2-binary
- `database_connection.py` - PostgreSQL connection with fallback
- `sample_data.py` - Fallback data for cloud deployment
- `.streamlit/config.toml` - Streamlit configuration
- `pages/` directory with 9 advanced pages (09-17)

### Step 3: Environment Variables (if needed)

If you want to connect to a remote PostgreSQL database on Streamlit Cloud:

1. Go to App Settings → Secrets
2. Add:
```toml
[postgres]
host = "your-postgres-host.com"
port = "5432"
database = "practice_db"
user = "postgres"
password = "your-password"
```

**Note**: Current setup works WITHOUT database - uses sample_data.py fallback automatically.

## File Structure

```
gender-wage-gap-analysis/
├── app.py                          # Main entry point (pages 1-8)
├── database_connection.py          # PostgreSQL connection with fallback
├── sample_data.py                  # Fallback data for cloud
├── requirements.txt                # Dependencies
├── .streamlit/
│   └── config.toml                 # Theme and settings
├── pages/
│   ├── 09_🇪🇺_EU27_Database.py    # PostgreSQL integration page
│   ├── 10_🎨_Advanced_Visualizations.py
│   ├── 11_📊_Advanced_Statistics.py
│   ├── 12_🎯_Causal_Inference.py   # DiD, IV, Synthetic Control
│   ├── 13_📈_Panel_Econometrics.py # FE, RE, GMM
│   ├── 14_🤖_ML_Economics.py       # LASSO, DML, Random Forest
│   ├── 15_🎲_Bayesian_Methods.py   # Bayesian regression, MCMC
│   ├── 16_📉_Time_Series.py        # ARIMA, unit roots
│   └── 17_📚_Week1_OLS_Tutorial.py # Interactive learning module
├── scripts/                        # Analysis scripts
└── data/                           # Data files
```

## Deployment URL

After deployment, your app will be available at:
`https://[your-app-name].streamlit.app`

## Common Issues & Fixes

### Issue 1: "No module named 'psycopg2'"
**Fix**: Already fixed - `psycopg2-binary` is in requirements.txt

### Issue 2: "Database connection failed"
**Fix**: App automatically falls back to sample_data.py - no action needed

### Issue 3: "File not found: 1_🏠_Home.py"
**Fix**: Change main file path to `app.py` in Streamlit Cloud settings

### Issue 4: "Import error from scripts.time_series"
**Fix**: All scripts are included in this branch - redeploy

## Local Testing Before Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Test pages work
# Navigate to http://localhost:8501
# Click through all pages (1-17)
```

## Production Deployment (Alternative to Streamlit Cloud)

See `INFRASTRUCTURE.md` for Docker, AWS, and other deployment options.
