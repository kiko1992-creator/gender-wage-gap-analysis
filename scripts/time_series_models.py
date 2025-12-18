from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATS_MODELS_AVAILABLE = True
except ImportError:
    ARIMA = None  # type: ignore
    ExponentialSmoothing = None  # type: ignore
    STATS_MODELS_AVAILABLE = False


def standardize_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize time series data by aggregating to country-year averages
    and ensuring the expected column names/types are present.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated wage gap data.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns ['country', 'year', 'wage_gap_pct'] sorted by year.
    """
    ts_df = (
        df.groupby(["country", "year"])["wage_gap_pct"]
        .mean()
        .reset_index()
        .sort_values(["country", "year"])
    )
    ts_df["year"] = ts_df["year"].astype(int)
    ts_df["wage_gap_pct"] = ts_df["wage_gap_pct"].astype(float)
    return ts_df


def prepare_country_series(ts_df: pd.DataFrame, country: str) -> pd.Series:
    """
    Create a continuous, interpolated yearly series for a given country.

    Missing years between the first and last available observation are filled
    using linear interpolation to support time-series models that expect
    equally spaced data.
    """
    country_df = (
        ts_df[ts_df["country"] == country]
        .groupby("year")["wage_gap_pct"]
        .mean()
        .sort_index()
    )
    if country_df.empty:
        return pd.Series(dtype=float, name="wage_gap_pct")

    full_years = range(int(country_df.index.min()), int(country_df.index.max()) + 1)
    reindexed = country_df.reindex(full_years)
    interpolated = reindexed.interpolate(limit_direction="both")
    return pd.Series(
        interpolated.values,
        index=pd.Index(full_years, name="year"),
        name="wage_gap_pct",
    )


def _safe_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    y_true_adj = np.where(y_true_arr == 0, 1e-3, y_true_arr)
    return float(mean_absolute_percentage_error(y_true_adj, y_pred_arr))


def _linear_forecast(
    years: np.ndarray,
    values: np.ndarray,
    steps: int,
    ci_levels: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    slope, intercept = np.polyfit(years, values, 1)
    future_years = np.arange(years[-1] + 1, years[-1] + steps + 1)
    preds = slope * future_years + intercept
    residuals = values - (slope * years + intercept)
    resid_std = np.std(residuals, ddof=1) if residuals.size > 1 else 0.0

    intervals = {}
    for level in ci_levels:
        z = norm.ppf((1 + level) / 2)
        delta = z * resid_std
        lower = preds - delta
        upper = preds + delta
        intervals[level] = (lower, upper)
    return future_years, preds, intervals


def _arima_forecast(
    values: np.ndarray,
    base_year: int,
    steps: int,
    ci_levels: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    if not STATS_MODELS_AVAILABLE or ARIMA is None:
        # Fallback to a drift-based random walk approximation when statsmodels is unavailable
        diffs = np.diff(values)
        drift = diffs.mean() if diffs.size else 0.0
        future_years = np.arange(base_year + 1, base_year + steps + 1)
        preds = values[-1] + drift * np.arange(1, steps + 1)
        resid_std = np.std(diffs, ddof=1) if diffs.size > 1 else 0.0
        intervals = {}
        for level in ci_levels:
            z = norm.ppf((1 + level) / 2)
            delta = z * resid_std
            intervals[level] = (preds - delta, preds + delta)
        return future_years, preds, intervals

    model = ARIMA(values, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit()
    forecast_res = fit.get_forecast(steps=steps)
    preds = forecast_res.predicted_mean.to_numpy()

    intervals = {}
    for level in ci_levels:
        ci = forecast_res.conf_int(alpha=1 - level)
        intervals[level] = (ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy())

    future_years = np.arange(base_year + 1, base_year + steps + 1)
    return future_years, preds, intervals


def _ets_forecast(
    values: np.ndarray,
    base_year: int,
    steps: int,
    ci_levels: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    if not STATS_MODELS_AVAILABLE or ExponentialSmoothing is None:
        # Simple Holt's linear trend fallback
        alpha, beta = 0.5, 0.3
        level = values[0]
        trend = values[1] - values[0] if values.size > 1 else 0.0
        fitted: List[float] = []
        for val in values:
            prev_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            fitted.append(level)

        preds = level + trend * np.arange(1, steps + 1)
        resid_std = np.std(values - np.array(fitted), ddof=1) if values.size > 2 else 0.0
        intervals = {}
        for level_conf in ci_levels:
            z = norm.ppf((1 + level_conf) / 2)
            delta = z * resid_std
            intervals[level_conf] = (preds - delta, preds + delta)
        future_years = np.arange(base_year + 1, base_year + steps + 1)
        return future_years, preds, intervals

    model = ExponentialSmoothing(values, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit(optimized=True)
    preds = fit.forecast(steps)

    resid_std = np.std(fit.resid, ddof=1) if fit.resid.size > 1 else 0.0
    intervals = {}
    for level in ci_levels:
        z = norm.ppf((1 + level) / 2)
        delta = z * resid_std
        intervals[level] = (preds - delta, preds + delta)

    future_years = np.arange(base_year + 1, base_year + steps + 1)
    return future_years, preds, intervals


def forecast_country_series(
    series: pd.Series,
    model_type: str,
    horizon: int = 6,
    ci_levels: Tuple[float, float] = (0.8, 0.95),
) -> pd.DataFrame:
    """
    Forecast the wage gap for a country using the specified model.

    Parameters
    ----------
    series : pd.Series
        Yearly wage gap series indexed by year.
    model_type : str
        One of ['linear', 'arima', 'ets'].
    horizon : int
        Number of years to forecast ahead.
    ci_levels : tuple
        Confidence levels for interval estimation.
    """
    if series.empty:
        return pd.DataFrame()

    values = series.values.astype(float)
    years = series.index.to_numpy(dtype=int)
    base_year = int(years[-1])

    if model_type == "linear":
        future_years, preds, intervals = _linear_forecast(years, values, horizon, ci_levels)
    elif model_type == "arima":
        future_years, preds, intervals = _arima_forecast(values, base_year, horizon, ci_levels)
    elif model_type == "ets":
        future_years, preds, intervals = _ets_forecast(values, base_year, horizon, ci_levels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    forecast_df = pd.DataFrame(
        {
            "year": future_years,
            "forecast": preds,
            "model": model_type,
        }
    )

    for level, (lower, upper) in intervals.items():
        pct = int(level * 100)
        forecast_df[f"lower_{pct}"] = np.clip(lower, 0, None)
        forecast_df[f"upper_{pct}"] = np.clip(upper, 0, None)

    return forecast_df


def _rolling_origin_cv(
    series: pd.Series,
    model_type: str,
    max_splits: int = 3,
    min_train_size: int = 5,
) -> Optional[Dict[str, float]]:
    if series.size <= min_train_size:
        return None

    n_splits = min(max_splits, series.size - min_train_size)
    if n_splits <= 0:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)
    values = series.values.astype(float)
    years = series.index.to_numpy(dtype=int)

    preds: List[float] = []
    actuals: List[float] = []

    for train_idx, test_idx in tscv.split(values):
        if len(train_idx) < min_train_size:
            continue
        train_values = values[train_idx]
        train_years = years[train_idx]
        try:
            forecast_df = forecast_country_series(
                pd.Series(train_values, index=train_years, name="wage_gap_pct"),
                model_type,
                horizon=len(test_idx),
            )
        except Exception:
            continue

        preds.extend(forecast_df["forecast"].tolist())
        actuals.extend(values[test_idx].tolist())

    if not preds:
        return None

    mae = mean_absolute_error(actuals, preds)
    mape = _safe_mape(actuals, preds)
    return {"mae": float(mae), "mape": float(mape)}


def evaluate_models_for_country(series: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate candidate models using rolling-origin cross-validation.

    Returns a dictionary keyed by model_type with MAE and MAPE metrics.
    """
    results = {}
    for model_type in ["linear", "arima", "ets"]:
        metrics = _rolling_origin_cv(series, model_type)
        if metrics:
            results[model_type] = metrics
    return results


def choose_best_model(metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
    if not metrics:
        return None
    return min(metrics.items(), key=lambda item: (item[1]["mae"], item[1]["mape"]))[0]
