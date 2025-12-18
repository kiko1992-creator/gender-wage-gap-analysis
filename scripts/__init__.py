"""
Gender Wage Gap Analysis - Scripts Module
==========================================

Reusable utilities for data processing, visualization, and machine learning.

Modules:
--------
- data_pipeline: Data fetching and processing from multiple sources
- visualization: Chart generation and report creation
- time_series: Time series forecasting (Linear, ARIMA, ETS)
- statistics: Statistical utilities and ML methods (planned)

Usage:
------
    from scripts.time_series import (
        standardize_time_series,
        forecast_country_series,
        evaluate_models_for_country,
        choose_best_model,
    )
"""

from .time_series import (
    standardize_time_series,
    prepare_country_series,
    forecast_country_series,
    evaluate_models_for_country,
    choose_best_model,
    STATS_MODELS_AVAILABLE,
)

__all__ = [
    'standardize_time_series',
    'prepare_country_series',
    'forecast_country_series',
    'evaluate_models_for_country',
    'choose_best_model',
    'STATS_MODELS_AVAILABLE',
]

__version__ = '2.0.0'
__author__ = 'Kiril Mickovski'
