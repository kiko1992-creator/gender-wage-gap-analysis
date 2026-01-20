"""
Gender Wage Gap Analysis - Production-Optimized Version
Enhanced with connection pooling, caching, error handling, and monitoring

Author: Kiril Mickovski
Version: 2.0.0 (Production-Ready)

Features:
- Database connection pooling
- Enhanced caching strategy
- Production error handling
- Performance monitoring
- Environment-aware configuration
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Production imports
from config import config, configure_streamlit_page
from database_pool import get_all_countries_2023, get_country_trend
from utils.error_handler import handle_streamlit_errors, ErrorContext, show_error
from utils.performance import enable_performance_monitoring, PerformanceTimer
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PRODUCTION SETUP
# ============================================================

# Install global error handler
handle_streamlit_errors()

# Configure page with production settings
configure_streamlit_page("Gender Wage Gap Analysis")

# Enable performance monitoring if configured
if config.get('ENABLE_PROFILING'):
    profiler = enable_performance_monitoring()
else:
    profiler = None

# ============================================================
# THEME MANAGEMENT
# ============================================================

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if config.get('ENABLE_DARK_MODE'):
    # Dark mode toggle in sidebar
    with st.sidebar:
        if st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode):
            st.session_state.dark_mode = True
        else:
            st.session_state.dark_mode = False

# Theme colors
if st.session_state.dark_mode:
    bg_color = "#1e1e1e"
    text_color = "#ffffff"
    card_bg = "#2d2d2d"
    accent_color = "#4da6ff"
else:
    bg_color = "#ffffff"
    text_color = "#333333"
    card_bg = "#f0f2f6"
    accent_color = "#1f77b4"

# Apply custom CSS
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {accent_color};
        text-align: center;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
    .country-card {{
        background: linear-gradient(135deg, {card_bg} 0%, #e8e8e8 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">📊 Gender Wage Gap Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">European Union - Interactive Research Dashboard</div>', unsafe_allow_html=True)

# Environment indicator (dev mode only)
if config.get('ENABLE_DEBUG_INFO'):
    st.info(f"🔧 Running in **{config.environment.upper()}** mode | Version {config.get('APP_VERSION')}")

# ============================================================
# LOAD DATA WITH OPTIMIZED CACHING
# ============================================================

with PerformanceTimer("Load Countries Data", show_in_ui=config.get('ENABLE_DEBUG_INFO')):
    with ErrorContext("Loading country data"):
        df_countries = get_all_countries_2023()

if df_countries is None or df_countries.empty:
    show_error("Failed to load data. Please check database connection or contact support.")
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📋 Navigation")
st.sidebar.info(
    """
    **Quick Links:**
    - Overview (this page)
    - Advanced pages (use menu above)
    - Health Check (page 99)
    """
)

# Data source indicator
with st.sidebar:
    st.divider()
    if config.get('ENABLE_DATABASE'):
        st.caption("📊 Data: PostgreSQL")
    else:
        st.caption("📊 Data: Sample Dataset")

# ============================================================
# PAGE 1: OVERVIEW & KEY METRICS
# ============================================================

st.header("🎯 Key Metrics - 2023")

# Calculate KPIs
eu_avg = df_countries['wage_gap_percent'].mean()
highest_gap = df_countries['wage_gap_percent'].max()
lowest_gap = df_countries['wage_gap_percent'].min()
median_gap = df_countries['wage_gap_percent'].median()

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "EU Average Gap",
        f"{eu_avg:.1f}%",
        help="Mean wage gap across all 27 EU countries"
    )

with col2:
    st.metric(
        "Highest Gap",
        f"{highest_gap:.1f}%",
        help=f"Largest wage gap in EU"
    )

with col3:
    st.metric(
        "Lowest Gap",
        f"{lowest_gap:.1f}%",
        help=f"Smallest wage gap in EU"
    )

with col4:
    st.metric(
        "Median Gap",
        f"{median_gap:.1f}%",
        help="Median wage gap across EU"
    )

# ============================================================
# RANKING TABLE
# ============================================================

st.header("🏆 Country Rankings - 2023")

# Create ranking
df_ranked = df_countries.sort_values('wage_gap_percent', ascending=False).reset_index(drop=True)
df_ranked['rank'] = df_ranked.index + 1

# Display table
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Complete Rankings")
    st.dataframe(
        df_ranked[['rank', 'country_name', 'wage_gap_percent', 'region']],
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("🥇 Top/Bottom 5")

    # Best performers
    st.markdown("**✅ Smallest Gaps:**")
    best_5 = df_ranked.tail(5)[['country_name', 'wage_gap_percent']]
    for _, row in best_5.iterrows():
        st.write(f"- {row['country_name']}: {row['wage_gap_percent']:.1f}%")

    st.divider()

    # Worst performers
    st.markdown("**⚠️ Largest Gaps:**")
    worst_5 = df_ranked.head(5)[['country_name', 'wage_gap_percent']]
    for _, row in worst_5.iterrows():
        st.write(f"- {row['country_name']}: {row['wage_gap_percent']:.1f}%")

# ============================================================
# VISUALIZATIONS
# ============================================================

st.header("📈 Visualizations")

tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🗺️ Map", "📉 Distribution"])

with tab1:
    # Bar chart
    fig_bar = px.bar(
        df_ranked,
        x='wage_gap_percent',
        y='country_name',
        orientation='h',
        title='Wage Gap by Country (2023)',
        labels={'wage_gap_percent': 'Wage Gap (%)', 'country_name': 'Country'},
        color='wage_gap_percent',
        color_continuous_scale='RdYlGn_r'
    )
    fig_bar.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # Map (if region data available)
    if 'region' in df_countries.columns:
        fig_map = px.scatter_geo(
            df_countries,
            locations='country_name',
            locationmode='country names',
            size='wage_gap_percent',
            color='wage_gap_percent',
            hover_name='country_name',
            title='Geographic Distribution of Wage Gap',
            color_continuous_scale='RdYlGn_r'
        )
        fig_map.update_geos(scope='europe')
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Map view requires regional data")

with tab3:
    # Distribution
    fig_hist = px.histogram(
        df_countries,
        x='wage_gap_percent',
        nbins=20,
        title='Distribution of Wage Gaps Across EU',
        labels={'wage_gap_percent': 'Wage Gap (%)', 'count': 'Number of Countries'}
    )
    fig_hist.add_vline(x=eu_avg, line_dash="dash", line_color="red",
                       annotation_text=f"EU Avg: {eu_avg:.1f}%")
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# REGIONAL ANALYSIS
# ============================================================

if 'region' in df_countries.columns:
    st.header("🌍 Regional Analysis")

    # Regional averages
    regional_avg = df_countries.groupby('region')['wage_gap_percent'].agg(['mean', 'min', 'max', 'count']).reset_index()
    regional_avg.columns = ['Region', 'Average Gap', 'Min Gap', 'Max Gap', 'Countries']

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(regional_avg, use_container_width=True, hide_index=True)

    with col2:
        fig_regional = px.bar(
            regional_avg,
            x='Region',
            y='Average Gap',
            title='Average Wage Gap by Region',
            color='Average Gap',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_regional, use_container_width=True)

# ============================================================
# DATA EXPORT
# ============================================================

if config.get('ENABLE_DATA_EXPORT'):
    st.header("💾 Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df_ranked.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            csv_data,
            "wage_gap_rankings_2023.csv",
            "text/csv"
        )

    with col2:
        json_data = df_ranked.to_json(orient='records', indent=2)
        st.download_button(
            "📥 Download as JSON",
            json_data,
            "wage_gap_rankings_2023.json",
            "application/json"
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"📊 Data: Eurostat, World Bank")

with col2:
    st.caption(f"👤 Author: {config.get('APP_AUTHOR')}")

with col3:
    st.caption(f"🔧 Version: {config.get('APP_VERSION')}")

# Performance metrics (if enabled)
if config.get('ENABLE_DEBUG_INFO') and profiler:
    with st.expander("⚡ Performance Info"):
        st.write("Page rendered successfully")
        profiler.snapshot_memory()
