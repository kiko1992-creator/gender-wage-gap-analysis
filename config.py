"""
Production Configuration Management
Environment-aware settings for development, staging, and production

Features:
- Environment-based configuration
- Secure secrets management
- Performance tuning per environment
- Feature flags
"""

import os
from dataclasses import dataclass
from typing import Optional
import streamlit as st


@dataclass
class BaseConfig:
    """Base configuration with defaults"""

    # App metadata
    APP_NAME: str = "Gender Wage Gap Analysis"
    APP_VERSION: str = "2.0.0"
    APP_AUTHOR: str = "Kiril Mickovski"

    # Database settings
    DB_MIN_CONNECTIONS: int = 1
    DB_MAX_CONNECTIONS: int = 5
    DB_CONNECTION_TIMEOUT: int = 30

    # Cache settings (in seconds)
    CACHE_TTL_SHORT: int = 600  # 10 minutes
    CACHE_TTL_MEDIUM: int = 3600  # 1 hour
    CACHE_TTL_LONG: int = 86400  # 24 hours

    # Performance settings
    MAX_UPLOAD_SIZE_MB: int = 200
    ENABLE_PROFILING: bool = False
    LAZY_LOAD_PAGES: bool = True

    # Feature flags
    ENABLE_DATABASE: bool = True
    ENABLE_DARK_MODE: bool = True
    ENABLE_DATA_EXPORT: bool = True
    ENABLE_ADVANCED_STATS: bool = True

    # UI settings
    PAGE_ICON: str = "📊"
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "expanded"

    # Data settings
    DEFAULT_YEAR: int = 2023
    MIN_YEAR: int = 2020
    MAX_YEAR: int = 2023

    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_DEBUG_INFO: bool = False


@dataclass
class DevelopmentConfig(BaseConfig):
    """Development environment settings"""

    # More verbose logging
    LOG_LEVEL: str = "DEBUG"
    ENABLE_DEBUG_INFO: bool = True
    ENABLE_PROFILING: bool = True

    # Shorter cache for faster iteration
    CACHE_TTL_SHORT: int = 60  # 1 minute
    CACHE_TTL_MEDIUM: int = 300  # 5 minutes
    CACHE_TTL_LONG: int = 600  # 10 minutes

    # Smaller connection pool for dev
    DB_MAX_CONNECTIONS: int = 3


@dataclass
class ProductionConfig(BaseConfig):
    """Production environment settings"""

    # Strict logging
    LOG_LEVEL: str = "WARNING"
    ENABLE_DEBUG_INFO: bool = False
    ENABLE_PROFILING: bool = False

    # Longer cache for performance
    CACHE_TTL_SHORT: int = 900  # 15 minutes
    CACHE_TTL_MEDIUM: int = 7200  # 2 hours
    CACHE_TTL_LONG: int = 86400  # 24 hours

    # Larger connection pool for production load
    DB_MAX_CONNECTIONS: int = 10


@dataclass
class CloudConfig(BaseConfig):
    """Streamlit Cloud deployment settings"""

    # Cloud-optimized settings
    ENABLE_DATABASE: bool = False  # Use sample data on cloud
    DB_MAX_CONNECTIONS: int = 2  # Minimal connections

    # Medium cache duration
    CACHE_TTL_SHORT: int = 600  # 10 minutes
    CACHE_TTL_MEDIUM: int = 3600  # 1 hour
    CACHE_TTL_LONG: int = 43200  # 12 hours

    # Disable profiling on cloud
    ENABLE_PROFILING: bool = False
    LOG_LEVEL: str = "INFO"


class Config:
    """
    Smart configuration loader
    Automatically selects correct config based on environment
    """

    _instance = None
    _config = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Load config based on environment"""
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Detect environment and load appropriate config"""
        env = os.getenv('APP_ENV', 'development').lower()

        if env == 'production':
            self._config = ProductionConfig()
        elif env == 'cloud':
            self._config = CloudConfig()
        elif env == 'development':
            self._config = DevelopmentConfig()
        else:
            # Default to base config
            self._config = BaseConfig()

        # Check if running on Streamlit Cloud
        if self._is_streamlit_cloud():
            self._config = CloudConfig()

    @staticmethod
    def _is_streamlit_cloud() -> bool:
        """Detect if running on Streamlit Cloud"""
        # Streamlit Cloud sets HOSTNAME env var
        hostname = os.getenv('HOSTNAME', '')
        return 'streamlit' in hostname.lower() or os.getenv('STREAMLIT_SHARING_MODE') is not None

    def get(self, key: str, default=None):
        """Get configuration value"""
        return getattr(self._config, key, default)

    def get_all(self) -> dict:
        """Get all configuration as dict"""
        return {k: v for k, v in self._config.__dict__.items() if not k.startswith('_')}

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return isinstance(self._config, ProductionConfig)

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return isinstance(self._config, DevelopmentConfig)

    @property
    def is_cloud(self) -> bool:
        """Check if running on cloud"""
        return isinstance(self._config, CloudConfig)

    @property
    def environment(self) -> str:
        """Get current environment name"""
        if self.is_production:
            return "production"
        elif self.is_cloud:
            return "cloud"
        elif self.is_development:
            return "development"
        else:
            return "unknown"


# Global config instance
config = Config()


# ============================================================
# STREAMLIT PAGE CONFIGURATION HELPER
# ============================================================

def configure_streamlit_page(page_title: Optional[str] = None):
    """
    Configure Streamlit page with production settings
    Call this at the top of every page

    Args:
        page_title: Optional custom page title (default: from config)
    """
    title = page_title or config.get('APP_NAME')

    st.set_page_config(
        page_title=title,
        page_icon=config.get('PAGE_ICON'),
        layout=config.get('LAYOUT'),
        initial_sidebar_state=config.get('SIDEBAR_STATE')
    )

    # Add debug info if enabled
    if config.get('ENABLE_DEBUG_INFO'):
        with st.sidebar:
            with st.expander("🔧 Debug Info", expanded=False):
                st.write(f"**Environment:** {config.environment}")
                st.write(f"**Version:** {config.get('APP_VERSION')}")
                st.write(f"**Cache TTL:** {config.get('CACHE_TTL_MEDIUM')}s")
                st.write(f"**Database:** {'Enabled' if config.get('ENABLE_DATABASE') else 'Sample Data'}")


# ============================================================
# ENVIRONMENT VARIABLES HELPER
# ============================================================

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get secret from Streamlit secrets or environment variables
    Streamlit Cloud compatible

    Args:
        key: Secret key to retrieve
        default: Default value if not found

    Returns:
        Secret value or default
    """
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets.get(key, default)
    except:
        pass

    # Fallback to environment variables
    return os.getenv(key, default)


# ============================================================
# PERFORMANCE MONITORING
# ============================================================

def should_enable_profiling() -> bool:
    """Check if performance profiling should be enabled"""
    return config.get('ENABLE_PROFILING', False)


def get_cache_ttl(level: str = 'medium') -> int:
    """
    Get cache TTL for specified level

    Args:
        level: 'short', 'medium', or 'long'

    Returns:
        Cache TTL in seconds
    """
    level_map = {
        'short': 'CACHE_TTL_SHORT',
        'medium': 'CACHE_TTL_MEDIUM',
        'long': 'CACHE_TTL_LONG'
    }
    key = level_map.get(level, 'CACHE_TTL_MEDIUM')
    return config.get(key, 3600)
