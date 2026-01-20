"""
Production-Ready Database Connection Pool
Uses connection pooling for better performance and resource management

Features:
- Thread-safe connection pooling
- Automatic connection recycling
- Health checks
- Graceful fallback to sample data
- Production-ready error handling
"""

import psycopg2
from psycopg2 import pool
import pandas as pd
from typing import Optional
import sample_data
import os
import streamlit as st
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabasePool:
    """Thread-safe database connection pool with automatic fallback"""

    _instance = None
    _pool = None

    def __new__(cls):
        """Singleton pattern to ensure one pool across the app"""
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize connection pool if not already done"""
        if self._pool is None:
            self._initialize_pool()

    def _initialize_pool(self):
        """Create connection pool with retry logic"""
        try:
            # Check for Docker environment
            if os.getenv('POSTGRES_HOST'):
                logger.info("Initializing database pool for Docker environment...")
                self._pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,  # Limit connections for production
                    dbname=os.getenv('POSTGRES_DB', 'practice_db'),
                    user=os.getenv('POSTGRES_USER', 'postgres'),
                    password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
                    host=os.getenv('POSTGRES_HOST', 'postgres'),
                    port=os.getenv('POSTGRES_PORT', '5432')
                )
                logger.info("✅ Database pool initialized successfully")
                return
        except Exception as e:
            logger.warning(f"Docker pool initialization failed: {e}")

        # Try Unix socket (Linux/Mac development)
        try:
            logger.info("Trying Unix socket connection...")
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                dbname="practice_db",
                user="postgres",
                host="/var/run/postgresql"
            )
            logger.info("✅ Unix socket pool initialized")
            return
        except Exception as e:
            logger.warning(f"Unix socket pool failed: {e}")

        # Try TCP/IP localhost (Windows)
        try:
            logger.info("Trying TCP/IP connection...")
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                dbname="practice_db",
                user="postgres",
                password="",
                host="localhost",
                port="5432"
            )
            logger.info("✅ TCP/IP pool initialized")
            return
        except Exception as e:
            logger.warning(f"TCP/IP pool failed: {e}")

        # No database available - will use sample data
        logger.info("⚠️  No database available - using sample data fallback")
        self._pool = None

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Automatically returns connection to pool

        Usage:
            with db_pool.get_connection() as conn:
                df = pd.read_sql(query, conn)
        """
        if self._pool is None:
            yield None
            return

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Connection error: {e}")
            yield None
        finally:
            if conn:
                self._pool.putconn(conn)

    def close_all(self):
        """Close all connections in pool - call on app shutdown"""
        if self._pool:
            self._pool.closeall()
            logger.info("Database pool closed")

    def is_available(self) -> bool:
        """Check if database is available"""
        return self._pool is not None


# Global pool instance
db_pool = DatabasePool()


# ============================================================
# PRODUCTION-READY QUERY FUNCTIONS WITH CACHING
# ============================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_all_countries_2023():
    """
    Get all 27 EU countries with 2023 wage gap data
    Cached for production performance
    """
    with db_pool.get_connection() as conn:
        if not conn:
            logger.info("Using sample data for countries 2023")
            return sample_data.get_sample_countries_2023()

        try:
            query = """
                SELECT
                    c.country_name,
                    c.region,
                    c.population,
                    c.gdp_billions,
                    w.wage_gap_percent
                FROM wage_gap_data w
                JOIN eu_countries c ON w.country_code = c.country_code
                WHERE w.year = 2023
                ORDER BY w.wage_gap_percent DESC
            """
            df = pd.read_sql_query(query, conn)
            logger.info(f"✅ Loaded {len(df)} countries from database")
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return sample_data.get_sample_countries_2023()


@st.cache_data(ttl=3600)
def get_country_trend(country_name: str):
    """
    Get 4-year trend for a specific country
    Cached per country for performance
    """
    with db_pool.get_connection() as conn:
        if not conn:
            return sample_data.get_sample_country_trend(country_name)

        try:
            query = """
                SELECT
                    w.year,
                    w.wage_gap_percent
                FROM wage_gap_data w
                JOIN eu_countries c ON w.country_code = c.country_code
                WHERE c.country_name = %s
                ORDER BY w.year
            """
            df = pd.read_sql_query(query, conn, params=(country_name,))
            return df
        except Exception as e:
            logger.error(f"Trend query failed for {country_name}: {e}")
            return sample_data.get_sample_country_trend(country_name)


@st.cache_data(ttl=3600)
def get_regional_comparison():
    """
    Get regional averages for 2023
    Cached for performance
    """
    with db_pool.get_connection() as conn:
        if not conn:
            return sample_data.get_sample_regional_comparison()

        try:
            query = """
                SELECT
                    c.region,
                    COUNT(DISTINCT c.country_code) as num_countries,
                    ROUND(AVG(w.wage_gap_percent), 2) as avg_gap,
                    ROUND(MIN(w.wage_gap_percent), 2) as min_gap,
                    ROUND(MAX(w.wage_gap_percent), 2) as max_gap
                FROM wage_gap_data w
                JOIN eu_countries c ON w.country_code = c.country_code
                WHERE w.year = 2023
                GROUP BY c.region
                ORDER BY avg_gap DESC
            """
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            logger.error(f"Regional query failed: {e}")
            return sample_data.get_sample_regional_comparison()


@st.cache_data(ttl=3600)
def get_improvement_rankings():
    """
    Get countries ranked by improvement (2020 → 2023)
    Cached for performance
    """
    with db_pool.get_connection() as conn:
        if not conn:
            return sample_data.get_sample_improvement_rankings()

        try:
            query = """
                SELECT
                    c.country_name,
                    MAX(CASE WHEN w.year = 2020 THEN w.wage_gap_percent END) as gap_2020,
                    MAX(CASE WHEN w.year = 2023 THEN w.wage_gap_percent END) as gap_2023,
                    ROUND(
                        MAX(CASE WHEN w.year = 2023 THEN w.wage_gap_percent END) -
                        MAX(CASE WHEN w.year = 2020 THEN w.wage_gap_percent END),
                        2
                    ) as change
                FROM wage_gap_data w
                JOIN eu_countries c ON w.country_code = c.country_code
                GROUP BY c.country_name
                HAVING
                    MAX(CASE WHEN w.year = 2020 THEN w.wage_gap_percent END) IS NOT NULL
                    AND MAX(CASE WHEN w.year = 2023 THEN w.wage_gap_percent END) IS NOT NULL
                ORDER BY change
            """
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            logger.error(f"Rankings query failed: {e}")
            return sample_data.get_sample_improvement_rankings()


@st.cache_data(ttl=3600)
def get_all_wage_data():
    """
    Get complete wage gap dataset for advanced analysis
    Cached with 1-hour TTL
    """
    with db_pool.get_connection() as conn:
        if not conn:
            logger.info("Using sample data for complete dataset")
            return sample_data.get_sample_all_data()

        try:
            query = """
                SELECT
                    c.country_name,
                    c.country_code,
                    c.region,
                    w.*
                FROM wage_gap_data w
                JOIN eu_countries c ON w.country_code = c.country_code
                ORDER BY w.year DESC, c.country_name
            """
            df = pd.read_sql_query(query, conn)
            logger.info(f"✅ Loaded {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Complete data query failed: {e}")
            return sample_data.get_sample_all_data()


# ============================================================
# HEALTH CHECK UTILITIES
# ============================================================

def check_database_health() -> dict:
    """
    Check database connection health
    Returns status dict for monitoring
    """
    status = {
        'available': db_pool.is_available(),
        'using_sample_data': not db_pool.is_available(),
        'connection_type': None
    }

    if status['available']:
        # Determine connection type
        if os.getenv('POSTGRES_HOST'):
            status['connection_type'] = 'docker'
        elif os.path.exists('/var/run/postgresql'):
            status['connection_type'] = 'unix_socket'
        else:
            status['connection_type'] = 'tcp_ip'

    return status


def clear_all_caches():
    """Clear all Streamlit caches - useful for admin operations"""
    st.cache_data.clear()
    logger.info("All caches cleared")
