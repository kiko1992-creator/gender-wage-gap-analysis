"""
Database Connection Helper for EU Wage Gap Research
Makes it easy to connect to PostgreSQL from anywhere in the app
Falls back to sample data when database is unavailable (for Streamlit Cloud)

Supports:
- Docker environment (reads from env vars)
- Local Unix socket (Linux/Mac)
- Local TCP/IP (Windows)
- Cloud deployment (fallback to sample data)
"""

import psycopg2
import pandas as pd
from typing import Optional
import sample_data
import os

def get_connection():
    """
    Connect to the EU wage gap research database

    Priority order:
    1. Docker environment (POSTGRES_HOST env var)
    2. Unix socket (Linux/Mac development)
    3. TCP/IP localhost (Windows development)
    4. Returns None (triggers sample data fallback)
    """
    # Try Docker environment first (if running in container)
    if os.getenv('POSTGRES_HOST'):
        try:
            conn = psycopg2.connect(
                dbname=os.getenv('POSTGRES_DB', 'practice_db'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('POSTGRES_PORT', '5432')
            )
            return conn
        except Exception as e:
            print(f"❌ Docker database connection failed: {e}")

    # Try Unix socket (Linux/Mac - development environment)
    try:
        conn = psycopg2.connect(
            dbname="practice_db",  # Updated to match Docker
            user="postgres",
            host="/var/run/postgresql"
        )
        return conn
    except:
        pass

    # Fallback to TCP/IP (Windows - your laptop)
    try:
        conn = psycopg2.connect(
            dbname="practice_db",
            user="postgres",
            password="",  # Add your password if needed
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def get_all_countries_2023():
    """Get all 27 EU countries with 2023 wage gap data"""
    conn = get_connection()
    if not conn:
        # Fallback to sample data when database is unavailable
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
        conn.close()
        return df
    except:
        # If query fails, use sample data
        if conn:
            conn.close()
        return sample_data.get_sample_countries_2023()

def get_country_trend(country_name: str):
    """Get 4-year trend for a specific country"""
    conn = get_connection()
    if not conn:
        # Fallback to sample data
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
        conn.close()
        return df
    except:
        if conn:
            conn.close()
        return sample_data.get_sample_country_trend(country_name)

def get_regional_comparison():
    """Get regional averages for 2023"""
    conn = get_connection()
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
        conn.close()
        return df
    except:
        if conn:
            conn.close()
        return sample_data.get_sample_regional_comparison()

def get_improvement_rankings():
    """Get countries ranked by improvement (2020 → 2023)"""
    conn = get_connection()
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
        conn.close()
        return df
    except:
        if conn:
            conn.close()
        return sample_data.get_sample_improvement_rankings()
