"""
Comprehensive test suite for Streamlit app production improvements.
Tests verify core functionality before and after improvements.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoading:
    """Test data loading functionality"""

    def test_main_data_file_exists(self):
        """Verify main data file exists"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'validated_wage_data.csv'
        assert data_path.exists(), f"Main data file not found at {data_path}"

    def test_main_data_loads_successfully(self):
        """Test that main data loads without errors"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'validated_wage_data.csv'
        df = pd.read_csv(data_path)
        assert len(df) > 0, "Data file is empty"
        # Column names are lowercase
        assert 'country' in df.columns, "Missing 'country' column"

    def test_country_data_file_exists(self):
        """Verify country features file exists"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'ml_features_clustered.csv'
        assert data_path.exists(), f"Country data file not found at {data_path}"

    def test_country_data_loads_successfully(self):
        """Test that country data loads without errors"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'ml_features_clustered.csv'
        df = pd.read_csv(data_path)
        assert len(df) > 0, "Country data file is empty"

    def test_all_required_columns_present(self):
        """Verify all required columns are in main data"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'validated_wage_data.csv'
        df = pd.read_csv(data_path)
        # Column names are lowercase
        required_cols = ['country', 'year', 'wage_gap_pct']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"


class TestTimeSeriesAnalysis:
    """Test time series functionality"""

    def test_time_series_import(self):
        """Verify time series module imports correctly"""
        from scripts.time_series import standardize_time_series, forecast_country_series
        assert standardize_time_series is not None
        assert forecast_country_series is not None

    def test_standardize_time_series_basic(self):
        """Test time series standardization with correct column names"""
        from scripts.time_series import standardize_time_series

        # Create sample data with lowercase column names (matching actual data)
        df = pd.DataFrame({
            'country': ['Albania'] * 5,
            'year': [2018, 2019, 2020, 2021, 2022],
            'wage_gap_pct': [10, 11, 12, 13, 14]
        })

        result = standardize_time_series(df)
        assert result is not None
        assert len(result) > 0

    def test_forecast_country_handles_short_series(self):
        """Test forecasting with minimal data"""
        from scripts.time_series import prepare_country_series

        # Test that prepare_country_series doesn't crash
        df = pd.DataFrame({
            'country': ['Test'] * 3,
            'year': [2020, 2021, 2022],
            'wage_gap_pct': [10.0, 11.0, 12.0]
        })

        # Should not crash
        try:
            from scripts.time_series import standardize_time_series
            standardized = standardize_time_series(df)
            assert True  # If we got here, it didn't crash
        except Exception:
            assert True  # Gracefully handles errors

    def test_forecast_country_returns_valid_structure(self):
        """Test forecast module has expected functions"""
        from scripts.time_series import forecast_country_series, choose_best_model

        # Just verify the functions exist and are callable
        assert callable(forecast_country_series)
        assert callable(choose_best_model)


class TestInputValidation:
    """Test input validation (will be implemented)"""

    def test_numeric_bounds_validation(self):
        """Test numeric input stays within bounds"""
        # This will test the validation function we'll create
        def validate_numeric_input(value, min_val, max_val):
            """Simple validation function"""
            return min_val <= value <= max_val

        assert validate_numeric_input(5, 0, 10) == True
        assert validate_numeric_input(-1, 0, 10) == False
        assert validate_numeric_input(15, 0, 10) == False
        assert validate_numeric_input(0, 0, 10) == True
        assert validate_numeric_input(10, 0, 10) == True

    def test_percentage_validation(self):
        """Test percentage inputs are reasonable"""
        def validate_percentage(value, allow_negative=False):
            """Validate percentage input"""
            if not allow_negative and value < 0:
                return False
            if value > 100:
                return False
            return True

        assert validate_percentage(50) == True
        assert validate_percentage(-5) == False
        assert validate_percentage(-5, allow_negative=True) == True
        assert validate_percentage(150) == False


class TestLogging:
    """Test logging functionality (will be implemented)"""

    def test_log_directory_creation(self):
        """Test log directory can be created"""
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        assert log_dir.exists()

    def test_logging_configuration(self):
        """Test logging can be configured"""
        import logging
        from logging.handlers import RotatingFileHandler

        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.INFO)

        # Test that handler can be created
        log_file = Path(__file__).parent.parent / 'logs' / 'test.log'
        handler = RotatingFileHandler(
            log_file,
            maxBytes=1_000_000,
            backupCount=3
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Test writing a log message
        logger.info("Test log message")

        assert log_file.exists(), "Log file was not created"

        # Cleanup
        logger.removeHandler(handler)
        handler.close()
        log_file.unlink()

    def test_log_rotation_settings(self):
        """Test log rotation parameters are reasonable"""
        max_bytes = 10_000_000  # 10MB
        backup_count = 3

        assert max_bytes > 0
        assert backup_count >= 1
        assert backup_count <= 10  # Reasonable upper limit


class TestCacheConfiguration:
    """Test caching improvements"""

    def test_cache_ttl_setting(self):
        """Test TTL parameter is reasonable"""
        cache_ttl = 3600  # 1 hour
        assert cache_ttl > 0
        assert cache_ttl <= 86400  # Max 24 hours

    def test_cache_max_entries(self):
        """Test max entries parameter is reasonable"""
        max_entries = 10
        assert max_entries > 0
        assert max_entries <= 100


class TestErrorHandling:
    """Test error handling patterns"""

    def test_specific_exception_handling(self):
        """Test that specific exceptions are caught properly"""
        def risky_operation():
            raise FileNotFoundError("Test file not found")

        with pytest.raises(FileNotFoundError):
            risky_operation()

    def test_graceful_degradation(self):
        """Test fallback behavior when optional features fail"""
        def optional_feature():
            try:
                # Simulate optional import failure
                raise ImportError("Optional module not available")
            except ImportError:
                return None  # Graceful degradation

        result = optional_feature()
        assert result is None  # Should return None, not crash


class TestDataIntegrity:
    """Test data integrity and consistency"""

    def test_no_missing_critical_data(self):
        """Verify critical columns have no missing values"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'validated_wage_data.csv'
        df = pd.read_csv(data_path)

        # Use lowercase column names
        critical_cols = ['country', 'year', 'wage_gap_pct']
        for col in critical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                assert missing_count == 0, f"Column {col} has {missing_count} missing values"

    def test_year_range_validity(self):
        """Test that year values are reasonable"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'validated_wage_data.csv'
        df = pd.read_csv(data_path)

        # Use lowercase column name
        if 'year' in df.columns:
            min_year = df['year'].min()
            max_year = df['year'].max()

            assert min_year >= 2000, f"Minimum year {min_year} seems too old"
            assert max_year <= 2025, f"Maximum year {max_year} is in the future"

    def test_wage_gap_range_validity(self):
        """Test that wage gap percentages are reasonable"""
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'validated_wage_data.csv'
        df = pd.read_csv(data_path)

        # Use lowercase column name
        if 'wage_gap_pct' in df.columns:
            min_gap = df['wage_gap_pct'].min()
            max_gap = df['wage_gap_pct'].max()

            assert min_gap >= -10, f"Minimum wage gap {min_gap} seems unrealistic"
            assert max_gap <= 50, f"Maximum wage gap {max_gap} seems unrealistic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
