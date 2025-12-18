# Time Series Model Report

**Data source:** `data/cleaned/validated_wage_data.csv` (country-year wage gap %).

Models evaluated with 1-year rolling-origin cross-validation on each available country-level series:
- Linear trend regression (baseline)
- Drift-based ARIMA(1,1,0) approximation (statsmodels not installed)
- Holt's linear trend fallback (statsmodels not installed)

Metrics use MAE and MAPE; best model is selected by lowest MAE with MAPE as a tiebreaker.

| Country | Observations | Best Model | MAE | MAPE (%) |
| --- | --- | --- | --- | --- |
| Montenegro | 8 | LINEAR | 2.28 | 12.9 |
| North Macedonia | 15 | LINEAR | 1.33 | 7.87 |
| Serbia | 16 | LINEAR | 3.0 | 25.56 |