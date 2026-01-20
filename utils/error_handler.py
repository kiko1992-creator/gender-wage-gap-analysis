"""
Production-Ready Error Handling
Graceful error management for Streamlit apps

Features:
- User-friendly error messages
- Detailed logging for debugging
- Error recovery suggestions
- Exception categorization
"""

import streamlit as st
import logging
import traceback
from functools import wraps
from typing import Callable, Any, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base exception for application errors"""
    pass


class DataError(AppError):
    """Data-related errors (missing files, invalid formats)"""
    pass


class DatabaseError(AppError):
    """Database connection or query errors"""
    pass


class AnalysisError(AppError):
    """Errors during statistical analysis"""
    pass


# ============================================================
# ERROR DISPLAY HELPERS
# ============================================================

def show_error(
    message: str,
    error: Optional[Exception] = None,
    show_details: bool = False,
    icon: str = "❌"
):
    """
    Display user-friendly error message in Streamlit

    Args:
        message: User-friendly error message
        error: Original exception (optional)
        show_details: Show technical details
        icon: Error icon
    """
    st.error(f"{icon} {message}")

    if error and show_details:
        with st.expander("🔍 Technical Details"):
            st.code(str(error))
            st.code(traceback.format_exc())

    # Log to console
    if error:
        logger.error(f"{message}: {error}", exc_info=True)
    else:
        logger.error(message)


def show_warning(message: str, icon: str = "⚠️"):
    """Display warning message"""
    st.warning(f"{icon} {message}")
    logger.warning(message)


def show_info(message: str, icon: str = "ℹ️"):
    """Display info message"""
    st.info(f"{icon} {message}")
    logger.info(message)


# ============================================================
# ERROR RECOVERY SUGGESTIONS
# ============================================================

def suggest_recovery(error_type: str):
    """
    Provide recovery suggestions based on error type

    Args:
        error_type: Type of error encountered
    """
    suggestions = {
        'database': [
            "✓ Check if PostgreSQL is running",
            "✓ Verify database connection settings",
            "✓ App will use sample data as fallback"
        ],
        'data': [
            "✓ Verify data file exists and is readable",
            "✓ Check data format matches expected schema",
            "✓ Try uploading a different file"
        ],
        'analysis': [
            "✓ Check if data contains required columns",
            "✓ Verify data has enough observations",
            "✓ Try different parameters or time period"
        ],
        'import': [
            "✓ Check if required Python packages are installed",
            "✓ Verify requirements.txt is up to date",
            "✓ Try restarting the application"
        ]
    }

    if error_type in suggestions:
        st.info("**Suggested Solutions:**")
        for suggestion in suggestions[error_type]:
            st.write(suggestion)


# ============================================================
# DECORATORS FOR ERROR HANDLING
# ============================================================

def handle_errors(
    fallback_value: Any = None,
    error_message: str = "An error occurred",
    show_details: bool = False
):
    """
    Decorator to catch and handle errors gracefully

    Usage:
        @handle_errors(fallback_value=pd.DataFrame(), error_message="Failed to load data")
        def load_data():
            return pd.read_csv('file.csv')

    Args:
        fallback_value: Value to return if error occurs
        error_message: User-friendly error message
        show_details: Show technical error details
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DatabaseError as e:
                show_error(f"{error_message}: Database connection failed", e, show_details)
                suggest_recovery('database')
                return fallback_value
            except DataError as e:
                show_error(f"{error_message}: Data error", e, show_details)
                suggest_recovery('data')
                return fallback_value
            except AnalysisError as e:
                show_error(f"{error_message}: Analysis error", e, show_details)
                suggest_recovery('analysis')
                return fallback_value
            except ImportError as e:
                show_error(f"{error_message}: Missing dependency", e, show_details)
                suggest_recovery('import')
                return fallback_value
            except Exception as e:
                show_error(f"{error_message}: Unexpected error", e, show_details)
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                return fallback_value
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    fallback_value: Any = None,
    error_message: str = "Operation failed"
) -> Any:
    """
    Safely execute a function with error handling

    Args:
        func: Function to execute
        fallback_value: Value to return on error
        error_message: User-friendly error message

    Returns:
        Function result or fallback value
    """
    try:
        return func()
    except Exception as e:
        show_error(error_message, e)
        return fallback_value


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_dataframe(df, required_columns: list, min_rows: int = 1) -> bool:
    """
    Validate DataFrame has required structure

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required

    Returns:
        True if valid, raises DataError otherwise
    """
    if df is None or df.empty:
        raise DataError("DataFrame is empty or None")

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise DataError(f"Missing required columns: {missing_cols}")

    if len(df) < min_rows:
        raise DataError(f"Insufficient data: {len(df)} rows (minimum: {min_rows})")

    return True


def validate_numeric_column(df, column: str) -> bool:
    """
    Validate column contains numeric data

    Args:
        df: DataFrame to check
        column: Column name

    Returns:
        True if valid, raises DataError otherwise
    """
    if column not in df.columns:
        raise DataError(f"Column '{column}' not found")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise DataError(f"Column '{column}' must be numeric")

    return True


# ============================================================
# CONTEXT MANAGERS FOR ERROR HANDLING
# ============================================================

class ErrorContext:
    """
    Context manager for handling errors in specific sections

    Usage:
        with ErrorContext("Loading data"):
            df = load_data()
    """

    def __init__(self, operation: str, show_spinner: bool = True):
        self.operation = operation
        self.show_spinner = show_spinner
        self.spinner = None

    def __enter__(self):
        if self.show_spinner:
            self.spinner = st.spinner(f"{self.operation}...")
            self.spinner.__enter__()
        logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.spinner:
            self.spinner.__exit__(exc_type, exc_val, exc_tb)

        if exc_type is None:
            logger.info(f"Completed: {self.operation}")
            return True

        # Handle specific exception types
        error_msg = f"{self.operation} failed"

        if issubclass(exc_type, DatabaseError):
            show_error(error_msg, exc_val)
            suggest_recovery('database')
        elif issubclass(exc_type, DataError):
            show_error(error_msg, exc_val)
            suggest_recovery('data')
        elif issubclass(exc_type, AnalysisError):
            show_error(error_msg, exc_val)
            suggest_recovery('analysis')
        else:
            show_error(f"{error_msg}: Unexpected error", exc_val)

        # Suppress the exception (don't propagate)
        return True


# ============================================================
# STREAMLIT-SPECIFIC ERROR HANDLERS
# ============================================================

def handle_streamlit_errors():
    """
    Global error handler for Streamlit apps
    Call this at the top of your main app
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        """Custom exception handler"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't catch keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        st.error("⚠️ An unexpected error occurred. The app will try to recover.")

    # Install the custom exception handler
    sys.excepthook = exception_handler


# ============================================================
# IMPORT GUARD
# ============================================================

import pandas as pd  # Import here to avoid circular imports
