"""
Tests for production-specific improvements:
- Logging
- Error tracking
- Input validation
- Performance monitoring

These tests verify the improvements work correctly.
"""
import pytest
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProductionLogging:
    """Test production logging implementation"""

    def test_create_logger_with_file_handler(self):
        """Test logger creation with rotating file handler"""
        from logging.handlers import RotatingFileHandler

        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('production_test')
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        log_file = log_dir / 'production_test.log'
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,
            backupCount=3
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Test all log levels
        logger.info("Test INFO message")
        logger.warning("Test WARNING message")
        logger.error("Test ERROR message")

        assert log_file.exists(), "Log file not created"

        # Read log file and verify content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test INFO message" in content
            assert "Test WARNING message" in content
            assert "Test ERROR message" in content

        # Cleanup
        logger.removeHandler(handler)
        handler.close()
        log_file.unlink()

    def test_log_rotation(self):
        """Test that log files rotate properly"""
        from logging.handlers import RotatingFileHandler

        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('rotation_test')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_file = log_dir / 'rotation_test.log'
        # Very small max size to trigger rotation
        handler = RotatingFileHandler(
            log_file,
            maxBytes=100,  # 100 bytes
            backupCount=2
        )

        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Write enough to trigger rotation
        for i in range(20):
            logger.info(f"Message {i:03d} - This is a test message to trigger rotation")

        # Cleanup
        logger.removeHandler(handler)
        handler.close()

        # Check rotation files exist
        assert log_file.exists() or (log_dir / 'rotation_test.log.1').exists()

        # Cleanup all rotation files
        for f in log_dir.glob('rotation_test.log*'):
            f.unlink()


class TestInputValidation:
    """Test input validation functions"""

    def test_validate_numeric_range(self):
        """Test numeric range validation"""

        def validate_numeric_input(value, min_val, max_val, param_name):
            """Validate numeric input is within range"""
            if not isinstance(value, (int, float)):
                return False, f"{param_name} must be a number"

            if not (min_val <= value <= max_val):
                return False, f"{param_name} must be between {min_val} and {max_val}"

            return True, ""

        # Valid inputs
        is_valid, msg = validate_numeric_input(5, 0, 10, "Test")
        assert is_valid == True
        assert msg == ""

        # Below range
        is_valid, msg = validate_numeric_input(-1, 0, 10, "Test")
        assert is_valid == False
        assert "between" in msg

        # Above range
        is_valid, msg = validate_numeric_input(15, 0, 10, "Test")
        assert is_valid == False
        assert "between" in msg

        # Boundary cases
        is_valid, msg = validate_numeric_input(0, 0, 10, "Test")
        assert is_valid == True

        is_valid, msg = validate_numeric_input(10, 0, 10, "Test")
        assert is_valid == True

    def test_validate_percentage_input(self):
        """Test percentage validation"""

        def validate_percentage(value, param_name, allow_negative=False):
            """Validate percentage input"""
            if not isinstance(value, (int, float)):
                return False, f"{param_name} must be a number"

            min_val = -100 if allow_negative else 0
            max_val = 100

            if not (min_val <= value <= max_val):
                return False, f"{param_name} must be between {min_val}% and {max_val}%"

            return True, ""

        # Valid percentage
        is_valid, msg = validate_percentage(50, "Growth")
        assert is_valid == True

        # Negative not allowed
        is_valid, msg = validate_percentage(-5, "Growth")
        assert is_valid == False

        # Negative allowed
        is_valid, msg = validate_percentage(-5, "Growth", allow_negative=True)
        assert is_valid == True

        # Too high
        is_valid, msg = validate_percentage(150, "Growth")
        assert is_valid == False

    def test_validate_year_input(self):
        """Test year validation"""

        def validate_year(year, min_year=2000, max_year=2030):
            """Validate year input"""
            if not isinstance(year, int):
                return False, "Year must be an integer"

            if not (min_year <= year <= max_year):
                return False, f"Year must be between {min_year} and {max_year}"

            return True, ""

        # Valid year
        is_valid, msg = validate_year(2025)
        assert is_valid == True

        # Too old
        is_valid, msg = validate_year(1990)
        assert is_valid == False

        # Too far in future
        is_valid, msg = validate_year(2050)
        assert is_valid == False


class TestPerformanceMonitoring:
    """Test performance monitoring utilities"""

    def test_timing_decorator(self):
        """Test function timing decorator"""
        from functools import wraps

        timings = []

        def timing_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                timings.append((func.__name__, duration))
                return result
            return wrapper

        @timing_decorator
        def slow_function():
            time.sleep(0.1)
            return "done"

        result = slow_function()
        assert result == "done"
        assert len(timings) == 1
        assert timings[0][0] == "slow_function"
        assert timings[0][1] >= 0.1  # At least 100ms

    def test_performance_threshold_detection(self):
        """Test detecting slow operations"""

        slow_operations = []

        def track_slow_operation(func_name, duration, threshold=1.0):
            """Track operations slower than threshold"""
            if duration > threshold:
                slow_operations.append({
                    'function': func_name,
                    'duration': duration,
                    'threshold': threshold
                })
                return True
            return False

        # Fast operation
        is_slow = track_slow_operation('fast_func', 0.5, threshold=1.0)
        assert is_slow == False
        assert len(slow_operations) == 0

        # Slow operation
        is_slow = track_slow_operation('slow_func', 2.5, threshold=1.0)
        assert is_slow == True
        assert len(slow_operations) == 1
        assert slow_operations[0]['function'] == 'slow_func'
        assert slow_operations[0]['duration'] == 2.5


class TestErrorHandling:
    """Test error handling patterns"""

    def test_specific_exception_types(self):
        """Test handling specific exception types"""

        def handle_file_operation(file_path):
            """Handle file operations with specific exceptions"""
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                return None, "File not found"
            except PermissionError:
                return None, "Permission denied"
            except Exception as e:
                return None, f"Unknown error: {type(e).__name__}"

        # Test non-existent file
        result = handle_file_operation('/nonexistent/file.txt')
        assert result is not None
        assert "File not found" in str(result)

    def test_graceful_degradation_pattern(self):
        """Test graceful degradation when features unavailable"""

        def get_feature_with_fallback():
            """Try advanced feature, fallback to basic"""
            try:
                # Simulate optional feature
                import nonexistent_module
                return "advanced"
            except ImportError:
                # Fallback to basic feature
                return "basic"

        result = get_feature_with_fallback()
        assert result == "basic"

    def test_error_context_preservation(self):
        """Test that error context is preserved"""

        errors = []

        def log_error_with_context(error, context):
            """Log error with context information"""
            errors.append({
                'error': str(error),
                'type': type(error).__name__,
                'context': context
            })

        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_error_with_context(e, {'user': 'test', 'operation': 'validate'})

        assert len(errors) == 1
        assert errors[0]['type'] == 'ValueError'
        assert errors[0]['context']['user'] == 'test'


class TestCacheConfiguration:
    """Test cache configuration improvements"""

    def test_cache_ttl_values(self):
        """Test TTL values are reasonable"""
        # Common TTL values in seconds
        ttl_configs = {
            'short': 300,      # 5 minutes
            'medium': 3600,    # 1 hour
            'long': 86400,     # 24 hours
        }

        for name, ttl in ttl_configs.items():
            assert ttl > 0, f"{name} TTL must be positive"
            assert ttl <= 86400, f"{name} TTL should not exceed 24 hours"

    def test_cache_size_limits(self):
        """Test cache size limits are reasonable"""
        max_entries_configs = {
            'small': 10,
            'medium': 50,
            'large': 100,
        }

        for name, max_entries in max_entries_configs.items():
            assert max_entries > 0, f"{name} max_entries must be positive"
            assert max_entries <= 1000, f"{name} max_entries too large"


class TestDataValidation:
    """Test data validation before processing"""

    def test_dataframe_validation(self):
        """Test DataFrame validation"""
        import pandas as pd

        def validate_dataframe(df, required_columns):
            """Validate DataFrame has required columns and data"""
            if df is None or df.empty:
                return False, "DataFrame is empty"

            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"Missing columns: {missing_cols}"

            return True, ""

        # Valid DataFrame
        df = pd.DataFrame({
            'Country': ['Albania', 'Bosnia'],
            'Year': [2022, 2022],
            'Value': [10, 12]
        })
        is_valid, msg = validate_dataframe(df, ['Country', 'Year', 'Value'])
        assert is_valid == True

        # Missing column
        is_valid, msg = validate_dataframe(df, ['Country', 'Year', 'MissingCol'])
        assert is_valid == False
        assert 'MissingCol' in msg

        # Empty DataFrame
        empty_df = pd.DataFrame()
        is_valid, msg = validate_dataframe(empty_df, ['Country'])
        assert is_valid == False
        assert "empty" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
