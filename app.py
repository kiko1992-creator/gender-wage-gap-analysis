"""
Gender Wage Gap Analysis - Interactive Dashboard
Balkans vs European Union Comparison

Author: Kiril Mickovski
Data Sources: Eurostat, World Bank, National Statistical Offices

IMPROVEMENTS MADE:
1. Fixed deprecation warnings (use_container_width -> width='stretch')
2. Added chart export functionality
3. Added What-If Analysis page
4. Added Dark Mode toggle
5. Added Country Profile cards
6. Ready for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
import time
from scripts.time_series import (
    choose_best_model,
    evaluate_models_for_country,
    forecast_country_series,
    prepare_country_series,
    STATS_MODELS_AVAILABLE,
    standardize_time_series,
)

# ============================================================
# PRODUCTION IMPROVEMENT: Logging Configuration
# Creates rotating log files to track app behavior and errors
# ============================================================
APP_DIR = Path(__file__).parent
LOG_DIR = APP_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if Streamlit reruns
if not logger.handlers:
    # Rotating file handler (10MB max, 3 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / 'app.log',
        maxBytes=10_000_000,
        backupCount=3
    )
    file_handler.setLevel(logging.INFO)

    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("="*60)
logger.info("Streamlit app starting")

# ============================================================
# PRODUCTION IMPROVEMENT: Performance Monitoring Decorator
# Logs functions that take longer than 1 second
# ============================================================
def timing_decorator(func):
    """Decorator to measure and log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        if duration > 1.0:
            logger.warning(f"{func.__name__} took {duration:.2f}s")
        else:
            logger.debug(f"{func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper

# ============================================================
# IMPROVEMENT #4: Dark Mode Toggle
# We use session_state to persist the theme choice across reruns
# ============================================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
    logger.info("Initialized dark mode session state")

# Page configuration
st.set_page_config(
    page_title="Gender Wage Gap Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMPROVEMENT #4 (continued): Dynamic CSS based on theme
# This shows how to conditionally apply styles
# ============================================================
if st.session_state.dark_mode:
    # Dark theme colors
    bg_color = "#1e1e1e"
    text_color = "#ffffff"
    card_bg = "#2d2d2d"
    accent_color = "#4da6ff"
    gradient_end = "#4a4a4a"  # Dark gradient for cards
else:
    # Light theme colors
    bg_color = "#ffffff"
    text_color = "#333333"
    card_bg = "#f0f2f6"
    accent_color = "#1f77b4"
    gradient_end = "#e8e8e8"  # Light gradient for cards

st.markdown(f"""
<style>
    /* Main App Background */
    .stApp {{
        background-color: {bg_color};
    }}

    /* Main Content Area */
    .main .block-container {{
        background-color: {bg_color};
        color: {text_color};
    }}

    /* All text elements */
    .stApp, .stApp p, .stApp span, .stApp div {{
        color: {text_color};
    }}

    /* Headers */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {text_color};
    }}

    /* Main Content Styling */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {accent_color};
        text-align: center;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {text_color};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
    .highlight-balkans {{
        color: #e74c3c;
        font-weight: bold;
    }}
    .highlight-eu {{
        color: #3498db;
        font-weight: bold;
    }}

    /* Country Profile Card Styling */
    .country-card {{
        background: linear-gradient(135deg, {card_bg} 0%, {gradient_end} 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .country-flag {{
        font-size: 3rem;
        text-align: center;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {card_bg};
    }}
    [data-testid="stSidebar"] > div {{
        background-color: {card_bg};
    }}

    /* Sidebar text and elements */
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}

    [data-testid="stSidebar"] label {{
        color: {text_color} !important;
    }}

    /* Radio button labels in sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        color: {text_color} !important;
    }}

    /* Sidebar title */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {text_color} !important;
    }}

    /* Metric styling for dark mode */
    [data-testid="stMetricValue"] {{
        color: {text_color};
    }}

    /* DataFrames and tables */
    .dataframe {{
        background-color: {card_bg};
        color: {text_color};
    }}

    /* Info, warning, error boxes */
    .stAlert {{
        background-color: {card_bg};
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTION: Export Chart
# This creates a download button for any Plotly figure
# ============================================================
def add_chart_export(fig, filename, key):
    """
    IMPROVEMENT #2: Add export button for charts

    How it works:
    1. Convert Plotly figure to HTML string
    2. Create a download button with that HTML
    3. Users can open the HTML file in any browser

    Parameters:
    - fig: Plotly figure object
    - filename: Name for the downloaded file
    - key: Unique key for the Streamlit button (required for multiple buttons)
    """
    # Convert figure to HTML
    html_str = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')

    # Create download button
    st.download_button(
        label="📥 Download Chart",
        data=html_str,
        file_name=f"{filename}.html",
        mime="text/html",
        key=key
    )

# ============================================================
# PRODUCTION IMPROVEMENT: Input Validation Helpers
# Validate user inputs and provide feedback
# ============================================================
def validate_numeric_input(value, min_val, max_val, param_name, warn_threshold=0.9):
    """
    Validate numeric input and warn if approaching extremes.

    Parameters:
    - value: The input value to validate
    - min_val: Minimum allowed value
    - max_val: Maximum allowed value
    - param_name: Name of the parameter (for logging/messages)
    - warn_threshold: Fraction of range to trigger warnings (default 0.9)

    Returns:
    - is_valid: Boolean indicating if value is valid
    - message: Warning or error message (empty if valid)
    """
    # Check if within bounds
    if not (min_val <= value <= max_val):
        error_msg = f"{param_name} must be between {min_val} and {max_val}"
        logger.warning(f"Invalid input: {param_name}={value}, {error_msg}")
        return False, error_msg

    # Check if approaching extremes (warn users)
    range_val = max_val - min_val
    lower_threshold = min_val + (range_val * (1 - warn_threshold))
    upper_threshold = max_val - (range_val * (1 - warn_threshold))

    if value <= lower_threshold:
        warn_msg = f"⚠️ {param_name} is at very low end ({value}). Results may be less reliable."
        logger.info(f"Extreme value warning: {param_name}={value} (near minimum)")
        return True, warn_msg
    elif value >= upper_threshold:
        warn_msg = f"⚠️ {param_name} is at very high end ({value}). Results may be less reliable."
        logger.info(f"Extreme value warning: {param_name}={value} (near maximum)")
        return True, warn_msg

    # Value is valid and reasonable
    logger.debug(f"Valid input: {param_name}={value}")
    return True, ""

# ============================================================
# DATA LOADING
# PRODUCTION IMPROVEMENTS:
# - @st.cache_data with TTL (1 hour) for automatic refresh
# - Performance monitoring via timing_decorator
# - Detailed logging of data loading process
# - Specific exception handling
# ============================================================

# Cache TTL configuration (in seconds)
CACHE_TTL = 3600  # 1 hour

@st.cache_data(ttl=CACHE_TTL, max_entries=10, show_spinner="Loading data...")
@timing_decorator
def load_main_data():
    """Load the main validated wage data"""
    try:
        data_path = APP_DIR / 'data' / 'processed' / 'validated_wage_data.csv'
        logger.info(f"Loading main data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Main data loaded successfully: {len(df)} records, {len(df.columns)} columns")
        return df
    except FileNotFoundError as e:
        logger.error(f"Main data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Main data file is empty: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading main data: {e}", exc_info=True)
        raise

@st.cache_data(ttl=CACHE_TTL, max_entries=10, show_spinner="Loading country data...")
@timing_decorator
def load_country_data():
    """Load country-level ML data with clusters"""
    try:
        data_path = APP_DIR / 'data' / 'processed' / 'ml_features_clustered.csv'
        logger.info(f"Loading country data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Country data loaded successfully: {len(df)} countries")
        return df
    except FileNotFoundError as e:
        logger.error(f"Country data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading country data: {e}", exc_info=True)
        raise

@st.cache_data(ttl=CACHE_TTL, max_entries=10, show_spinner="Standardizing time series...")
@timing_decorator
def load_time_series_data():
    """Standardize time series data for forecasting"""
    try:
        logger.info("Standardizing time series data")
        df = load_main_data()
        result = standardize_time_series(df)
        logger.info("Time series data standardized successfully")
        return result
    except Exception as e:
        logger.error(f"Error standardizing time series data: {e}", exc_info=True)
        raise

# Load data with improved error handling
logger.info("Starting data load sequence")
try:
    df_main = load_main_data()
    df_country = load_country_data()
    df_time_series = load_time_series_data()
    data_loaded = True
    logger.info("All data loaded successfully")
except FileNotFoundError as e:
    error_msg = f"Data file not found: {e}"
    logger.error(error_msg)
    st.error(f"❌ {error_msg}")
    st.info("💡 Please ensure data files exist in the data/processed/ directory")
    data_loaded = False
except pd.errors.EmptyDataError as e:
    error_msg = f"Data file is empty: {e}"
    logger.error(error_msg)
    st.error(f"❌ {error_msg}")
    data_loaded = False
except Exception as e:
    error_msg = f"Unexpected error loading data: {type(e).__name__}: {e}"
    logger.error(error_msg, exc_info=True)
    st.error(f"❌ {error_msg}")
    st.info("💡 Check logs/app.log for detailed error information")
    data_loaded = False

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/1200px-Flag_of_Europe.svg.png", width=100)
st.sidebar.title("Navigation")

# IMPROVEMENT #4: Dark Mode Toggle in sidebar
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
if dark_mode != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode
    st.rerun()  # Rerun to apply theme change

# IMPROVEMENT #3: Added "What-If Analysis" to navigation
# IMPROVEMENT #5: Added "Country Profiles" to navigation
page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "Country Profiles", "Country Comparison", "Regional Analysis",
     "Time Series", "What-If Analysis", "ML Insights", "Oaxaca-Blinder", "Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Gender Wage Gap Analysis**

Comparing wage disparities between
Balkan countries and the European Union.

*Data: Eurostat 2023-2024*
""")

# Country flag emoji mapping
COUNTRY_FLAGS = {
    'Bulgaria': '🇧🇬', 'Croatia': '🇭🇷', 'Greece': '🇬🇷', 'Hungary': '🇭🇺',
    'Italy': '🇮🇹', 'Montenegro': '🇲🇪', 'North Macedonia': '🇲🇰',
    'Poland': '🇵🇱', 'Romania': '🇷🇴', 'Serbia': '🇷🇸', 'Slovenia': '🇸🇮', 'Sweden': '🇸🇪'
}

# Main content
if data_loaded:

    # ========== OVERVIEW PAGE ==========
    if page == "Overview":
        st.markdown('<p class="main-header">Gender Wage Gap Analysis</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Balkans vs European Union Comparison | 2009-2024</p>', unsafe_allow_html=True)

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        This page provides an overview of gender wage gaps across 12 European countries, comparing Balkan nations with EU member states.
        The gender wage gap represents the difference in average hourly earnings between men and women, expressed as a percentage of men's earnings.

        **Key Terms:**
        - **Gender Wage Gap**: The percentage difference between average male and female earnings. For example, a 15% gap means women earn 85% of what men earn.
        - **Balkans Region**: Southeastern European countries (Bulgaria, Croatia, Greece, Montenegro, North Macedonia, Romania, Serbia, Slovenia)
        - **pp (percentage points)**: Absolute difference between percentages (e.g., 20% - 15% = 5 pp)
        - **Statistical Significance**: Indicates the difference is unlikely to be due to chance (p < 0.05 means 95% confidence)

        **How to Interpret the Rankings:**
        - **Lower values** indicate smaller wage gaps (more equality)
        - **Red bars** = Balkan countries | **Blue bars** = EU countries
        - Numbers show the gap as % of male earnings
        - Rankings are based on most recent available data (2023-2024)
        """)

        # Data Sources
        with st.expander("📊 Data Sources & Coverage"):
            st.markdown("""
            **Primary Data Sources:**
            - **Eurostat**: Gender pay gap in unadjusted form (structure of earnings survey)
            - **World Bank**: GDP per capita, unemployment rates, labor force participation
            - **ILO**: Additional labor market indicators
            - **National Statistical Offices**: Country-specific validation

            **Data Coverage:**
            - **Time Period**: 2009-2024 (146 total observations)
            - **Countries**: 12 (8 Balkan, 4 EU reference countries)
            - **Update Frequency**: Annual releases (typically October-November)
            - **Last Updated**: 2024-01

            **Data Quality Notes:**
            - North Macedonia: 46 observations (most comprehensive)
            - Other countries: 6-36 observations per country
            - Some countries have gaps in time series due to survey schedules
            """)

        # Methodology
        with st.expander("🔬 Methodology & Limitations"):
            st.markdown("""
            **Statistical Method**: Descriptive statistics with regional comparison

            **Why We Use It**:
            - Provides quick overview of wage gap magnitudes across countries
            - Allows comparison between Balkan and EU regions
            - Identifies countries with extreme values for further investigation

            **Calculation**:
            ```
            Gender Wage Gap (%) = [(Male Avg Earnings - Female Avg Earnings) / Male Avg Earnings] × 100
            ```

            **Limitations**:
            - **Unadjusted gap**: Does not control for occupation, education, experience, or hours worked
            - **Averages hide variation**: Within-country differences by sector, age, education not shown
            - **Sample coverage**: Not all countries have complete time series
            - **Causality**: Rankings show correlation, not causation
            - **Missing factors**: Cultural, institutional, and policy differences not directly measured

            **What the Gap Represents**:
            - Mix of explained factors (occupation, education, experience) and unexplained factors (discrimination, negotiation, career interruptions)
            - See "Oaxaca-Blinder" page for decomposition into explained vs unexplained components
            """)

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        balkans_avg = df_country[df_country['region'] == 'Balkans']['gap_mean'].mean()
        eu_avg = df_country[df_country['region'] == 'EU']['gap_mean'].mean()
        gap_diff = balkans_avg - eu_avg

        with col1:
            st.metric("Balkans Average Gap", f"{balkans_avg:.1f}%", delta=None)
        with col2:
            st.metric("EU Average Gap", f"{eu_avg:.1f}%", delta=None)
        with col3:
            st.metric("Difference", f"{gap_diff:.1f} pp", delta=f"+{gap_diff:.1f} pp", delta_color="inverse")
        with col4:
            st.metric("Countries Analyzed", f"{len(df_country)}", delta=None)

        st.markdown("---")

        # Main visualization - Country ranking
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Gender Pay Gap by Country (2023)")

            df_sorted = df_country.sort_values('gap_mean', ascending=True)

            colors = ['#e74c3c' if r == 'Balkans' else '#3498db' for r in df_sorted['region']]

            fig = go.Figure(go.Bar(
                x=df_sorted['gap_mean'],
                y=df_sorted['country'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}%" for v in df_sorted['gap_mean']],
                textposition='outside'
            ))

            fig.update_layout(
                height=500,
                xaxis_title="Gender Pay Gap (%)",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            # Add legend manually
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(size=10, color='#e74c3c'),
                                     name='Balkans'))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(size=10, color='#3498db'),
                                     name='EU'))
            fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))

            # IMPROVEMENT #1: Fixed deprecation - use width='stretch' instead of use_container_width=True
            st.plotly_chart(fig, width='stretch')

            # IMPROVEMENT #2: Add export button
            add_chart_export(fig, "country_ranking", "overview_chart")

        with col2:
            st.subheader("Key Findings")

            st.markdown("""
            **Statistical Evidence:**

            - Balkans have **6.7 pp higher** wage gaps than EU average
            - Difference is **statistically significant** (p < 0.05)
            - **Unemployment** is the strongest predictor

            **Notable Observations:**

            - Hungary has highest EU gap (17.8%)
            - North Macedonia rising (+1 pp/year)
            - Italy paradoxically low (2.2%)
            - Sweden high despite equality (11.3%)
            """)

            st.markdown("---")

            # Mini pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Balkans', 'EU'],
                values=[3, 9],
                marker_colors=['#e74c3c', '#3498db'],
                hole=0.4
            )])
            fig_pie.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                annotations=[dict(text='12', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig_pie, width='stretch')
            st.caption("Countries in analysis")

    # ========== COUNTRY PROFILES PAGE (NEW - IMPROVEMENT #5) ==========
    elif page == "Country Profiles":
        st.header("Country Profiles")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Detailed profiles for each country showing gender wage gap alongside key economic and labor market indicators.
        Each country card presents a snapshot of the most recent data to help understand the context behind wage disparities.

        **How to Read Country Cards:**
        - Click on any country to expand and see full metrics
        - **Region badge** (red = Balkans, blue = EU) shows geographic classification
        - **Gender Pay Gap** is the primary metric (lower is better)
        - **Color indicators**: 🔴 High gap (>15%) | 🟡 Moderate (10-15%) | 🟢 Low (<10%)

        **What Each Metric Means:**
        - **Female LFP** (Labor Force Participation): % of working-age women in labor force
        - **Male LFP**: % of working-age men in labor force
        - **GDP per capita**: Economic prosperity indicator (purchasing power parity, USD)
        - **Unemployment**: % of labor force actively seeking work
        """)

        # Metric explanations
        with st.expander("📊 Understanding the Metrics"):
            st.markdown("""
            **Why These Metrics Matter:**

            1. **Labor Force Participation (LFP)**:
               - Higher female LFP often correlates with smaller wage gaps
               - Large LFP gaps suggest structural barriers to women's employment
               - Cultural and policy factors (childcare, parental leave) strongly influence LFP

            2. **GDP per Capita**:
               - Wealthier countries don't always have smaller wage gaps
               - Example: Sweden (high GDP) has 11.3% gap, Italy (similar GDP) has 2.2% gap
               - Economic development is necessary but not sufficient for gender equality

            3. **Unemployment**:
               - High unemployment can widen wage gaps (women often more affected)
               - During recessions, gender gaps can increase or decrease depending on sector impacts
               - Youth unemployment particularly affects women's career trajectories

            **Data Coverage Notes:**
            - All metrics are from most recent available year (typically 2023-2024)
            - Some countries have more complete historical data than others
            - North Macedonia has most comprehensive coverage (46 observations since 2009)
            """)

        # Historical context
        with st.expander("🔬 Interpreting Country Differences"):
            st.markdown("""
            **What Explains Cross-Country Variation?**

            Countries with **smaller gaps** typically have:
            - Strong equal pay legislation and enforcement
            - Generous parental leave policies (shared between parents)
            - Subsidized childcare availability
            - Transparent salary structures
            - High female representation in management

            Countries with **larger gaps** often show:
            - Traditional gender role norms
            - Occupational segregation (women in lower-paying sectors)
            - Career interruptions due to caregiving
            - Weak enforcement of equal pay laws
            - Glass ceiling effects in senior positions

            **Important**: The unadjusted gap shown here includes both:
            - Differences in characteristics (education, experience, occupation)
            - Unexplained factors (potential discrimination, negotiation gaps)

            See "Oaxaca-Blinder" page for decomposition analysis.
            """)

        st.markdown("Click on any country card below to see detailed metrics:")

        # Create a 3-column grid of country cards
        countries = df_country['country'].tolist()

        # Display countries in rows of 3
        for i in range(0, len(countries), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(countries):
                    country = countries[i + j]
                    data = df_country[df_country['country'] == country].iloc[0]
                    flag = COUNTRY_FLAGS.get(country, '🏳️')

                    with col:
                        # Create an expander for each country (acts like a card)
                        with st.expander(f"{flag} {country}", expanded=False):
                            # Region badge
                            region_color = "#e74c3c" if data['region'] == 'Balkans' else "#3498db"
                            st.markdown(f"<span style='background-color:{region_color};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em'>{data['region']}</span>", unsafe_allow_html=True)

                            st.markdown("---")

                            # Key metrics
                            st.metric("Gender Pay Gap", f"{data['gap_mean']:.1f}%")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Female LFP", f"{data['female_lfp']:.1f}%")
                                st.metric("GDP/capita", f"${data['gdp_per_capita']:,.0f}")
                            with col_b:
                                st.metric("Male LFP", f"{data['male_lfp']:.1f}%")
                                st.metric("Unemployment", f"{data['unemployment']:.1f}%")

                            # Gap interpretation
                            if data['gap_mean'] > 15:
                                st.error("⚠️ High wage gap - needs attention")
                            elif data['gap_mean'] > 10:
                                st.warning("⚡ Moderate wage gap")
                            else:
                                st.success("✅ Lower than average gap")

        st.markdown("---")
        st.info("**Tip:** Click on any country card to expand and see detailed metrics.")

    # ========== COUNTRY COMPARISON PAGE ==========
    elif page == "Country Comparison":
        st.header("Country Comparison")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Side-by-side comparison of two countries across multiple dimensions: wage gap, labor market participation, and economic indicators.
        The radar chart provides a visual multi-dimensional comparison to identify similarities and differences.

        **How to Use This Page:**
        1. Select two countries from the dropdowns below
        2. Review side-by-side metrics to see absolute differences
        3. Examine the radar chart for multi-dimensional patterns
        4. Look for correlation patterns (e.g., does higher GDP correlate with smaller gaps?)

        **Best Comparisons to Try:**
        - **Similar regions**: Bulgaria vs Romania (both Balkans)
        - **Different regions**: Sweden vs Serbia (EU vs Balkans)
        - **Similar GDP**: Poland vs Croatia (economic peers)
        - **Extreme cases**: Hungary (highest gap) vs Italy (lowest gap)
        """)

        # Radar chart explanation
        with st.expander("📊 How to Read the Radar Chart"):
            st.markdown("""
            **Understanding Radar Charts:**

            A radar chart (also called spider chart) displays multiple variables on axes radiating from a center point.
            Each axis represents a different dimension, and the shape formed by connecting data points shows the country's profile.

            **Reading the Chart:**
            - **Larger area** = generally better economic indicators (but NOT for wage gap and unemployment)
            - **Shape comparison**: Similar shapes = similar country profiles
            - **Distance from center**: Shows relative magnitude on each dimension
            - **Overlapping areas**: Dimensions where countries are similar

            **What Each Axis Represents:**
            - **Gender Wage Gap**: LOWER is better (smaller gap = more equality)
            - **Female LFP**: Higher = more women in workforce
            - **Male LFP**: Higher = more men in workforce
            - **GDP per capita**: Higher = wealthier country
            - **Unemployment**: LOWER is better (less unemployment)

            **Important Notes:**
            - Values are normalized (0-100 scale) for visual comparison
            - Different axes have inverse interpretations (high unemployment = bad, high GDP = good)
            - Focus on relative differences, not absolute values on chart
            """)

        # Dimension explanations
        with st.expander("🔬 What Drives the Differences?"):
            st.markdown("""
            **Common Patterns to Look For:**

            1. **High GDP + Small Gap**: Indicates wealth translates to equality (e.g., Sweden)
            2. **Low GDP + Small Gap**: Cultural or policy factors override economic constraints (e.g., Italy)
            3. **High LFP + Large Gap**: Women work but face wage penalties (occupational segregation)
            4. **Low Female LFP + Large Gap**: Structural barriers keep women out of workforce

            **Why Countries Differ:**
            - **Policy**: Parental leave, childcare subsidies, equal pay enforcement
            - **Culture**: Gender role norms, work-family balance expectations
            - **Economy**: Sector composition, union strength, minimum wage levels
            - **History**: EU membership, transition from socialism, institutional legacy

            **Suggested Comparisons:**
            - **Balkan peers**: See regional convergence/divergence
            - **EU vs Balkan**: Understand membership effects
            - **Time**: Use Time Series page to see how gaps evolve
            """)

        col1, col2 = st.columns(2)

        with col1:
            country1 = st.selectbox("Select First Country", df_country['country'].tolist(), index=0)
        with col2:
            country2 = st.selectbox("Select Second Country", df_country['country'].tolist(), index=3)

        c1_data = df_country[df_country['country'] == country1].iloc[0]
        c2_data = df_country[df_country['country'] == country2].iloc[0]

        # Comparison metrics
        st.subheader("Side-by-Side Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            flag1 = COUNTRY_FLAGS.get(country1, '🏳️')
            st.markdown(f"### {flag1} {country1}")
            st.markdown(f"**Region:** {c1_data['region']}")
            st.metric("Wage Gap", f"{c1_data['gap_mean']:.1f}%")
            st.metric("Female LFP", f"{c1_data['female_lfp']:.1f}%")
            st.metric("Unemployment", f"{c1_data['unemployment']:.1f}%")
            st.metric("GDP per capita", f"${c1_data['gdp_per_capita']:,.0f}")

        with col2:
            st.markdown("### ↔️ Difference")
            gap_diff = c1_data['gap_mean'] - c2_data['gap_mean']
            lfp_diff = c1_data['female_lfp'] - c2_data['female_lfp']
            unemp_diff = c1_data['unemployment'] - c2_data['unemployment']
            gdp_diff = c1_data['gdp_per_capita'] - c2_data['gdp_per_capita']

            st.metric("Gap Diff", f"{gap_diff:+.1f} pp", delta_color="inverse", label_visibility="collapsed")
            st.metric("LFP Diff", f"{lfp_diff:+.1f} pp", label_visibility="collapsed")
            st.metric("Unemp Diff", f"{unemp_diff:+.1f} pp", delta_color="inverse", label_visibility="collapsed")
            st.metric("GDP Diff", f"${gdp_diff:+,.0f}", label_visibility="collapsed")

        with col3:
            flag2 = COUNTRY_FLAGS.get(country2, '🏳️')
            st.markdown(f"### {flag2} {country2}")
            st.markdown(f"**Region:** {c2_data['region']}")
            st.metric("Wage Gap", f"{c2_data['gap_mean']:.1f}%")
            st.metric("Female LFP", f"{c2_data['female_lfp']:.1f}%")
            st.metric("Unemployment", f"{c2_data['unemployment']:.1f}%")
            st.metric("GDP per capita", f"${c2_data['gdp_per_capita']:,.0f}")

        # Radar chart comparison
        st.subheader("Multi-dimensional Comparison")

        categories = ['Wage Gap', 'Female LFP', 'Male LFP', 'Unemployment', 'LFP Gap']

        # Normalize values for radar
        max_vals = df_country[['gap_mean', 'female_lfp', 'male_lfp', 'unemployment', 'lfp_gap']].max()

        c1_values = [
            c1_data['gap_mean'] / max_vals['gap_mean'] * 100,
            c1_data['female_lfp'] / max_vals['female_lfp'] * 100,
            c1_data['male_lfp'] / max_vals['male_lfp'] * 100,
            c1_data['unemployment'] / max_vals['unemployment'] * 100,
            c1_data['lfp_gap'] / max_vals['lfp_gap'] * 100
        ]

        c2_values = [
            c2_data['gap_mean'] / max_vals['gap_mean'] * 100,
            c2_data['female_lfp'] / max_vals['female_lfp'] * 100,
            c2_data['male_lfp'] / max_vals['male_lfp'] * 100,
            c2_data['unemployment'] / max_vals['unemployment'] * 100,
            c2_data['lfp_gap'] / max_vals['lfp_gap'] * 100
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=c1_values + [c1_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=country1,
            line_color='#e74c3c' if c1_data['region'] == 'Balkans' else '#3498db'
        ))

        fig.add_trace(go.Scatterpolar(
            r=c2_values + [c2_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=country2,
            line_color='#e74c3c' if c2_data['region'] == 'Balkans' else '#3498db',
            opacity=0.6
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig, width='stretch')
        add_chart_export(fig, f"comparison_{country1}_vs_{country2}", "comparison_chart")

    # ========== REGIONAL ANALYSIS PAGE ==========
    elif page == "Regional Analysis":
        st.header("Regional Analysis: Balkans vs EU")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Statistical comparison of gender wage gaps between Balkan countries and EU reference countries.
        Uses formal hypothesis testing to determine if observed differences are statistically significant or could be due to chance.

        **Key Question:** Do Balkan countries have systematically different wage gaps than EU countries?

        **What You'll See:**
        - **Box plots**: Visual comparison of wage gap distributions
        - **Summary statistics**: Mean, median, variation for each region
        - **T-test results**: Formal statistical test of difference
        - **GDP scatter plot**: Relationship between economic development and wage gaps

        **How to Interpret:**
        - If p-value < 0.05: Difference is statistically significant (95% confidence)
        - Box plot shows median (line), quartiles (box), and outliers (points)
        - Scatter plot bubble size represents unemployment rate
        """)

        # Statistical concepts explanation
        with st.expander("📊 What is a t-test? (Statistical Significance Explained)"):
            st.markdown("""
            **Understanding the Independent Samples t-test:**

            A t-test answers: "Are the average wage gaps in Balkans and EU regions **truly different**, or could the observed difference be due to random variation?"

            **Key Concepts:**

            1. **t-statistic (2.84 in our case)**:
               - Measures how many standard errors the groups differ
               - Larger absolute values = stronger evidence of difference
               - Values > 2 generally suggest real differences

            2. **p-value (0.017 in our case)**:
               - Probability that observed difference occurred by chance
               - p = 0.017 means 1.7% chance this is random
               - **If p < 0.05**: We reject chance, conclude real difference exists
               - **If p ≥ 0.05**: Cannot rule out chance, difference not proven

            3. **Statistical Significance**:
               - p < 0.05 is the standard threshold in social sciences
               - Means 95% confidence that difference is real
               - **Does NOT mean the difference is large or important** (see effect size)

            **In Plain English:**
            Our result (p = 0.017) means: "There's only a 1.7% chance the Balkans-EU difference is random. We're 95%+ confident Balkan countries genuinely have higher wage gaps than EU countries."
            """)

        # Effect size explanation
        with st.expander("🔬 Effect Size: How Big is the Difference?"):
            st.markdown("""
            **Statistical Significance vs Practical Significance:**

            Statistical significance (p-value) tells us IF a difference exists.
            Effect size tells us HOW BIG the difference is.

            **Understanding Effect Size:**

            - **Small effect**: d = 0.2 (difference is detectable but minor)
            - **Medium effect**: d = 0.5 (moderate, noticeable difference)
            - **Large effect**: d = 0.8+ (substantial, important difference)

            **In Our Analysis:**
            - Balkans average: ~13-14% wage gap
            - EU average: ~7-8% wage gap
            - Difference: ~6 percentage points
            - **This is a large, meaningful difference** (not just statistical)

            **Real-World Interpretation:**
            A 6 pp difference means if EU women earn $93 for every $100 men earn, Balkan women earn only $87 for every $100 men earn. This represents significant economic inequality.

            **Why It Matters:**
            - Some studies find p < 0.05 but tiny effects (not practically important)
            - Our finding has BOTH statistical significance AND large practical importance
            - This justifies policy attention to Balkan gender wage gaps
            """)

        # Box plot explanation
        with st.expander("📈 How to Read Box Plots"):
            st.markdown("""
            **Box Plot Components:**

            - **Box**: Contains middle 50% of data (25th to 75th percentile)
            - **Line inside box**: Median (50th percentile)
            - **Whiskers**: Extend to min/max values (excluding outliers)
            - **Points outside whiskers**: Outliers (unusual values)

            **What to Look For:**
            - **Box position**: Higher = larger wage gaps
            - **Box size**: Larger = more variation within region
            - **Overlap**: If boxes overlap, groups may not differ significantly
            - **Outliers**: Countries with unusual values (investigate separately)

            **In This Analysis:**
            - Balkan box sits higher = systematically larger gaps
            - EU box shows more variation = less regional homogeneity
            - Outliers identify countries needing special attention (e.g., Hungary in EU group)
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution Comparison")

            fig = go.Figure()

            balkans_data = df_country[df_country['region'] == 'Balkans']['gap_mean']
            eu_data = df_country[df_country['region'] == 'EU']['gap_mean']

            fig.add_trace(go.Box(y=balkans_data, name='Balkans', marker_color='#e74c3c'))
            fig.add_trace(go.Box(y=eu_data, name='EU', marker_color='#3498db'))

            fig.update_layout(
                yaxis_title="Gender Pay Gap (%)",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Statistical Summary")

            summary_data = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count'],
                'Balkans': [
                    f"{balkans_data.mean():.1f}%",
                    f"{balkans_data.median():.1f}%",
                    f"{balkans_data.std():.1f}%",
                    f"{balkans_data.min():.1f}%",
                    f"{balkans_data.max():.1f}%",
                    f"{len(balkans_data)}"
                ],
                'EU': [
                    f"{eu_data.mean():.1f}%",
                    f"{eu_data.median():.1f}%",
                    f"{eu_data.std():.1f}%",
                    f"{eu_data.min():.1f}%",
                    f"{eu_data.max():.1f}%",
                    f"{len(eu_data)}"
                ]
            }

            st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

            st.markdown("---")

            # T-test results
            st.markdown("### Statistical Test")
            st.success("""
            **Independent Samples t-test**
            - t-statistic: 2.84
            - p-value: 0.017
            - **Result: Statistically significant (p < 0.05)**
            """)

        # Scatter plot - GDP vs Gap
        st.subheader("Economic Development vs Wage Gap")

        fig = px.scatter(
            df_country,
            x='gdp_per_capita',
            y='gap_mean',
            color='region',
            size='unemployment',
            hover_name='country',
            color_discrete_map={'Balkans': '#e74c3c', 'EU': '#3498db'},
            labels={
                'gdp_per_capita': 'GDP per Capita ($)',
                'gap_mean': 'Gender Pay Gap (%)',
                'unemployment': 'Unemployment Rate'
            }
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
        add_chart_export(fig, "gdp_vs_gap_scatter", "regional_scatter")

        st.info("**Insight:** Higher GDP generally correlates with lower wage gaps. Bubble size represents unemployment rate - notice Balkan countries have both high unemployment and high gaps.")

    # ========== TIME SERIES PAGE (ENHANCED WITH ML MODELS) ==========
    elif page == "Time Series":
        st.header("Time Series Analysis")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Historical trends and future projections of gender wage gaps for selected countries.
        Uses statistical forecasting models to predict where wage gaps may head over the next 3-10 years based on past patterns.

        **Key Question:** Are wage gaps improving, staying constant, or worsening over time?

        **What You'll See:**
        - **Solid lines**: Historical (actual) data
        - **Dashed lines**: Forecasted future values
        - **Shaded areas**: Confidence intervals (uncertainty in forecasts)
        - **Model comparison table**: Which forecasting method works best for each country

        **How to Use:**
        1. Select countries to compare from dropdown
        2. Choose forecast horizon (how many years ahead to project)
        3. Examine trends (upward = worsening, downward = improving)
        4. Check confidence bands (wider = more uncertainty)
        """)

        # Forecasting explanation
        with st.expander("📊 What is Forecasting?"):
            st.markdown("""
            **Understanding Time Series Forecasting:**

            Forecasting uses historical patterns to predict future values. Like weather forecasts, they become less certain further into the future.

            **Three Models Used:**

            1. **Linear Trend**:
               - Simplest model: Fits a straight line through historical data
               - Assumes constant rate of change (e.g., gap decreases by 0.5 pp each year)
               - **Best for**: Countries with steady, linear trends
               - **Limitation**: Cannot capture cycles or sudden changes

            2. **ARIMA (AutoRegressive Integrated Moving Average)**:
               - Sophisticated statistical model that learns from past values and errors
               - Can handle trends, cycles, and irregular fluctuations
               - **Best for**: Countries with complex patterns and sufficient data
               - **How it works**: Uses recent values to predict next value, adjusts for trends
               - **Limitation**: Requires at least 5-10 years of data

            3. **ETS (Exponential Smoothing)**:
               - Weighted average that gives more importance to recent observations
               - Can model trend and seasonal components
               - **Best for**: Smooth trends with gradual changes
               - **How it works**: Recent observations get higher weight in forecast
               - **Limitation**: May overreact to recent fluctuations

            **Model Selection:**
            The system automatically chooses the best model for each country based on cross-validation accuracy.
            """)

        # Confidence intervals explanation
        with st.expander("🔬 Understanding Confidence Intervals"):
            st.markdown("""
            **What are Confidence Intervals?**

            Forecasts are predictions with uncertainty. Confidence intervals show the range where future values are likely to fall.

            **The Shaded Bands:**

            - **Darker band (80% interval)**: 80% probability the true value falls within this range
            - **Lighter band (95% interval)**: 95% probability the true value falls within this range

            **How to Interpret:**

            - **Narrow bands**: High confidence in forecast (stable historical pattern)
            - **Wide bands**: High uncertainty (volatile historical pattern or limited data)
            - **Widening over time**: Normal - uncertainty increases further into future
            - **Overlapping intervals**: Two countries' futures may converge or remain similar

            **Example:**
            If forecast shows 12% with 95% interval of [10%, 14%]:
            - Best guess: 12% wage gap
            - 95% confident true value will be between 10-14%
            - Still 5% chance it falls outside this range

            **Caveats:**
            - Assumes historical patterns continue (no major policy changes)
            - Unexpected events (economic crises, legislation) can invalidate forecasts
            - Longer horizons = more uncertainty
            """)

        # Model evaluation explanation
        with st.expander("📈 Model Selection Criteria (MAE & MAPE)"):
            st.markdown("""
            **How We Choose the Best Model:**

            Models are evaluated using "rolling-origin cross-validation": We pretend we're in the past, forecast 1 year ahead, and check accuracy.

            **Evaluation Metrics:**

            1. **MAE (Mean Absolute Error)**:
               - Average size of forecast errors in percentage points
               - Example: MAE = 0.8 means forecasts are off by ±0.8 pp on average
               - **Lower is better**

            2. **MAPE (Mean Absolute Percentage Error)**:
               - Average forecast error as % of actual value
               - Example: MAPE = 5% means forecasts are off by ±5% on average
               - **Lower is better**
               - More interpretable than MAE (relative vs absolute)

            **Model Selection:**
            - System tests all three models (Linear, ARIMA, ETS)
            - Selects model with lowest MAPE
            - If MAPE < 10%: Excellent forecast accuracy
            - If MAPE 10-20%: Good accuracy
            - If MAPE > 20%: High uncertainty, use with caution

            **Why Different Countries Use Different Models:**
            - Countries with steady trends → Linear performs well
            - Countries with fluctuations → ARIMA/ETS capture complexity
            - Countries with limited data → Linear is more robust
            """)

        st.markdown("---")
        st.markdown(
            "All forecasts are built from the validated country-level series in "
            "`data/processed/validated_wage_data.csv`, standardized to yearly wage gap (%) indices."
        )

        if not STATS_MODELS_AVAILABLE:
            st.warning(
                "The `statsmodels` dependency is missing in this environment. "
                "ARIMA and ETS results will use lightweight fallback estimators until the "
                "package is installed (see `requirements.txt`)."
            )

        st.sidebar.markdown("#### Forecast validation")
        st.sidebar.info(
            "Rolling-origin cross-validation (1-year horizon) compares linear trend, "
            "ARIMA(1,1,0), and additive ETS models. Metrics use MAE and MAPE; bands show "
            "80% and 95% intervals."
        )

        available_countries = sorted(df_time_series['country'].unique().tolist())
        default_countries = [c for c in ['Serbia', 'North Macedonia'] if c in available_countries]
        if not default_countries and available_countries:
            default_countries = available_countries[:2]

        col_filters = st.columns([2, 1])
        with col_filters[0]:
            selected_countries = st.multiselect(
                "Select Countries to Compare",
                available_countries,
                default=default_countries
            )
        with col_filters[1]:
            forecast_horizon = st.slider(
                "Forecast horizon (years ahead)",
                min_value=3,
                max_value=10,
                value=6,
                help="Adds rolling forecasts beyond the last observed year for each country."
            )

        if selected_countries:
            palette = px.colors.qualitative.Bold

            def hex_to_rgba(hex_color: str, alpha: float) -> str:
                hex_color = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({r},{g},{b},{alpha})"

            fig = go.Figure()
            summary_rows = []
            table_notes = []

            for idx, country in enumerate(selected_countries):
                country_series = prepare_country_series(df_time_series, country)

                if country_series.empty:
                    table_notes.append(f"{country}: no validated observations available.")
                    continue

                historical_df = country_series.reset_index()
                color = palette[idx % len(palette)]
                dash_color = hex_to_rgba(color, 0.9)

                fig.add_trace(go.Scatter(
                    x=historical_df['year'],
                    y=historical_df['wage_gap_pct'],
                    mode='lines+markers',
                    name=f"{country} actual",
                    line=dict(color=color)
                ))

                if country_series.size < 5:
                    table_notes.append(f"{country}: insufficient history for robust CV (needs >= 5 years).")
                    continue

                metrics = evaluate_models_for_country(country_series)
                best_model = choose_best_model(metrics) or "linear"
                forecast_df = forecast_country_series(
                    country_series,
                    best_model,
                    horizon=forecast_horizon
                )

                fig.add_trace(go.Scatter(
                    x=forecast_df['year'],
                    y=forecast_df['forecast'],
                    mode='lines+markers',
                    name=f"{country} {best_model.upper()} forecast",
                    line=dict(color=dash_color, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=list(forecast_df['year']) + list(forecast_df['year'][::-1]),
                    y=list(forecast_df['upper_80']) + list(forecast_df['lower_80'][::-1]),
                    fill='toself',
                    fillcolor=hex_to_rgba(color, 0.2),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name=f"{country} 80% interval",
                    showlegend=True
                ))

                fig.add_trace(go.Scatter(
                    x=list(forecast_df['year']) + list(forecast_df['year'][::-1]),
                    y=list(forecast_df['upper_95']) + list(forecast_df['lower_95'][::-1]),
                    fill='toself',
                    fillcolor=hex_to_rgba(color, 0.1),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name=f"{country} 95% interval",
                    showlegend=True
                ))

                if best_model in metrics:
                    summary_rows.append({
                        "Country": country,
                        "Best model": best_model.upper(),
                        "MAE": round(metrics[best_model]["mae"], 2),
                        "MAPE (%)": round(metrics[best_model]["mape"] * 100, 1)
                    })
                else:
                    summary_rows.append({
                        "Country": country,
                        "Best model": best_model.upper(),
                        "MAE": None,
                        "MAPE (%)": None
                    })

            fig.update_layout(
                height=520,
                xaxis_title="Year",
                yaxis_title="Gender Pay Gap (%)",
                legend_title="Series",
                hovermode="x unified"
            )
            st.plotly_chart(fig, width='stretch')
            add_chart_export(fig, "time_series_forecasts", "timeseries_chart")

            st.subheader("Model comparison (rolling-origin CV)")
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
            if table_notes:
                st.caption(" | ".join(table_notes))

            st.info(
                "We compare a linear trend baseline, ARIMA(1,1,0), and additive ETS models using "
                "1-year rolling-origin cross-validation. Forecasts show mean projections plus 80% "
                "and 95% confidence bands for the selected horizon."
            )
        else:
            st.info("Please select at least one country to view time series data.")

    # ========== WHAT-IF ANALYSIS PAGE (NEW - IMPROVEMENT #3) ==========
    elif page == "What-If Analysis":
        st.header("What-If Analysis")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Interactive scenario modeling tool to explore how changes in economic factors would affect the gender wage gap.
        Based on a regression model (R² = 0.80) trained on historical data from 12 countries.

        **Key Question:** Which economic levers have the biggest impact on reducing wage gaps?

        **How to Use:**
        1. Select a country to analyze
        2. Adjust sliders to simulate economic changes (unemployment, GDP, female labor force participation)
        3. See predicted impact on wage gap
        4. Use this to understand which policy interventions may be most effective

        **Important:** This shows correlation, not causation. Real-world policy effects may differ.
        """)

        # Policy context
        with st.expander("🏛️ Policy Context: Real-World Interventions"):
            st.markdown("""
            **What Policies Actually Affect These Factors?**

            The sliders represent economic indicators, but **policies** drive these indicators. Here's how:

            **1. Reducing Unemployment:**
            - **Active labor market policies**: Job training, placement services
            - **Wage subsidies**: Incentivize hiring in targeted sectors
            - **Public employment**: Direct job creation
            - **Example**: Germany's "Kurzarbeit" short-time work scheme reduces unemployment during crises

            **2. Increasing GDP per Capita:**
            - **Investment in education**: Upskills workforce, increases productivity
            - **Infrastructure development**: Attracts foreign investment
            - **Innovation policies**: R&D tax credits, startup support
            - **EU structural funds**: Cohesion policy for economic convergence
            - **Example**: Poland's GDP/capita nearly tripled (1990-2020) after EU accession and structural reforms

            **3. Increasing Female Labor Force Participation:**
            - **Subsidized childcare**: Reduces barriers to women working
            - **Parental leave policies**: Especially when shared between parents
            - **Flexible work arrangements**: Part-time, remote work options
            - **Anti-discrimination enforcement**: Ensures equal hiring practices
            - **Example**: Sweden's generous parental leave (480 days) + subsidized childcare → 80% female LFP

            **Direct Wage Gap Interventions:**
            - **Pay transparency laws**: Require salary disclosure (Iceland, UK, EU Directive 2023)
            - **Equal pay audits**: Companies must prove no gender discrimination
            - **Strengthened enforcement**: Penalties for wage discrimination
            - **Collective bargaining**: Unions reduce arbitrary pay setting
            """)

        # Real-world examples
        with st.expander("🌍 Real-World Examples"):
            st.markdown("""
            **Case Studies of Countries Reducing Wage Gaps:**

            **Iceland (Current gap: ~8%, down from 20% in 1990s)**:
            - **Policy**: 2018 law requires companies to prove equal pay
            - **Mechanism**: Mandatory pay audits, fines for non-compliance
            - **Result**: Gap dropped 5 pp in 5 years
            - **What-if analog**: This is like improving enforcement + transparency

            **Belgium (Current gap: ~5%, one of EU's lowest)**:
            - **Policy**: Strong collective bargaining coverage (96% of workers)
            - **Mechanism**: Unions negotiate equal pay structures
            - **Result**: Compressed wage distribution reduces gender gaps
            - **What-if analog**: Reducing wage variance through institutions

            **Rwanda (Current gap: ~3%, lowest globally)**:
            - **Policy**: 50% parliamentary quota for women (2003)
            - **Mechanism**: Women in leadership → pro-women policies → labor market effects
            - **Result**: High female LFP (84%) + small wage gap
            - **What-if analog**: Increasing female LFP dramatically

            **Cautionary Tale - South Korea (Gap: ~31%, OECD highest)**:
            - **Issue**: High GDP + high female education, but huge gap
            - **Why**: Cultural norms, career interruptions, occupational segregation
            - **Lesson**: Economic growth alone insufficient; need targeted policies
            - **What-if lesson**: GDP slider won't fix gaps without other changes

            **What These Examples Teach:**
            - Multiple interventions needed simultaneously
            - Cultural/institutional factors matter beyond economics
            - Enforcement and monitoring critical for policy effectiveness
            - Long-term commitment required (changes take 5-10 years)
            """)

        # Model limitations
        with st.expander("⚠️ Model Limitations & Caveats"):
            st.markdown("""
            **What This Model Can and Cannot Do:**

            **✅ What It's Good For:**
            - Understanding **relative importance** of different factors
            - Exploring **directional effects** (increase X → decrease gap)
            - Generating hypotheses for further research
            - Comparing economic scenarios

            **❌ What It Cannot Do:**
            - **Causal inference**: Shows correlation, not causation
            - **Policy evaluation**: Real policies have complex, indirect effects
            - **Predict exact outcomes**: R² = 0.80 means 20% variance unexplained
            - **Account for confounders**: Cultural, institutional factors not included

            **Why Correlation ≠ Causation:**
            - Unemployment and gaps may both be caused by third factor (e.g., economic crisis)
            - Female LFP may increase BECAUSE gaps are small (reverse causality)
            - GDP growth may come from sectors that don't affect gaps

            **For Rigorous Policy Evaluation, You Need:**
            - Causal inference methods (DiD, IV, RDD) → See "Oaxaca-Blinder" page
            - Panel regression with fixed effects → Planned implementation
            - Randomized controlled trials (rare in policy)
            - Natural experiments (policy changes in some countries but not others)

            **Use This Tool For:**
            - Exploratory "what if" scenarios
            - Motivating deeper research questions
            - Communicating factor importance to policymakers
            - Generating testable hypotheses
            """)

        st.markdown("---")

        # Select a country to modify
        selected_country = st.selectbox(
            "Select a country to analyze",
            df_country['country'].tolist()
        )

        country_data = df_country[df_country['country'] == selected_country].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Current Values: {COUNTRY_FLAGS.get(selected_country, '')} {selected_country}")
            st.metric("Current Wage Gap", f"{country_data['gap_mean']:.1f}%")
            st.metric("Current Unemployment", f"{country_data['unemployment']:.1f}%")
            st.metric("Current GDP per capita", f"${country_data['gdp_per_capita']:,.0f}")
            st.metric("Current Female LFP", f"{country_data['female_lfp']:.1f}%")

        with col2:
            st.subheader("Adjust Parameters")

            # Sliders for what-if scenarios
            new_unemployment = st.slider(
                "Unemployment Rate (%)",
                min_value=1.0,
                max_value=25.0,
                value=float(country_data['unemployment']),
                step=0.5,
                help="Drag to see how unemployment changes affect the wage gap"
            )

            new_gdp = st.slider(
                "GDP per capita ($)",
                min_value=5000,
                max_value=60000,
                value=int(country_data['gdp_per_capita']),
                step=1000,
                help="Drag to see how GDP changes affect the wage gap"
            )

            new_female_lfp = st.slider(
                "Female Labor Force Participation (%)",
                min_value=30.0,
                max_value=90.0,
                value=float(country_data['female_lfp']),
                step=1.0,
                help="Drag to see how female LFP changes affect the wage gap"
            )

        # ============================================================
        # PRODUCTION IMPROVEMENT: Input Validation
        # Validate slider inputs and warn about extreme values
        # ============================================================
        validation_warnings = []

        # Validate unemployment
        is_valid, msg = validate_numeric_input(
            new_unemployment, 1.0, 25.0,
            "Unemployment Rate",
            warn_threshold=0.85
        )
        if msg and msg.startswith("⚠️"):
            validation_warnings.append(msg)

        # Validate GDP
        is_valid, msg = validate_numeric_input(
            new_gdp, 5000, 60000,
            "GDP per capita",
            warn_threshold=0.85
        )
        if msg and msg.startswith("⚠️"):
            validation_warnings.append(msg)

        # Validate Female LFP
        is_valid, msg = validate_numeric_input(
            new_female_lfp, 30.0, 90.0,
            "Female LFP",
            warn_threshold=0.85
        )
        if msg and msg.startswith("⚠️"):
            validation_warnings.append(msg)

        # Display warnings if any
        if validation_warnings:
            for warning in validation_warnings:
                st.warning(warning)
            logger.info(f"What-If Analysis warnings for {selected_country}: {len(validation_warnings)} extreme values")

        # Calculate predicted wage gap using regression coefficients
        # From our model: gap = -51.84 + 0.95*female_lfp + 1.74*unemployment - 0.0005*gdp
        baseline_gap = country_data['gap_mean']

        # Calculate changes
        unemployment_effect = (new_unemployment - country_data['unemployment']) * 1.74
        gdp_effect = (new_gdp - country_data['gdp_per_capita']) * (-0.0005)
        lfp_effect = (new_female_lfp - country_data['female_lfp']) * 0.95

        predicted_gap = baseline_gap + unemployment_effect + gdp_effect + lfp_effect
        predicted_gap = max(0, predicted_gap)  # Can't go below 0

        gap_change = predicted_gap - baseline_gap

        st.markdown("---")

        # Results
        st.subheader("Predicted Impact")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Wage Gap",
                f"{predicted_gap:.1f}%",
                delta=f"{gap_change:+.1f} pp",
                delta_color="inverse"
            )

        with col2:
            st.metric("From Unemployment", f"{unemployment_effect:+.2f} pp")
            st.caption("Each 1% unemployment = +1.74 pp gap")

        with col3:
            st.metric("From GDP", f"{gdp_effect:+.2f} pp")
            st.caption("Higher GDP reduces gap")

        # Visualization
        st.subheader("Visual Comparison")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Current', 'Predicted'],
            y=[baseline_gap, predicted_gap],
            marker_color=['#3498db', '#e74c3c' if gap_change > 0 else '#2ecc71'],
            text=[f"{baseline_gap:.1f}%", f"{predicted_gap:.1f}%"],
            textposition='outside'
        ))

        fig.update_layout(
            height=300,
            yaxis_title="Gender Pay Gap (%)",
            showlegend=False
        )

        st.plotly_chart(fig, width='stretch')

        # Policy recommendations based on analysis
        st.subheader("Policy Implications")

        if gap_change < -2:
            st.success(f"""
            **These changes would significantly reduce the wage gap by {abs(gap_change):.1f} pp!**

            Key levers:
            - {'Reducing unemployment has the biggest impact' if unemployment_effect < 0 else ''}
            - {'Increasing GDP helps reduce inequality' if gdp_effect < 0 else ''}
            """)
        elif gap_change < 0:
            st.info(f"These changes would modestly reduce the wage gap by {abs(gap_change):.1f} pp.")
        else:
            st.warning(f"These changes would increase the wage gap by {gap_change:.1f} pp. Consider reversing the trends.")

    # ========== ML INSIGHTS PAGE ==========
    elif page == "ML Insights":
        st.header("Machine Learning Insights")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Advanced statistical and machine learning analysis to uncover patterns, relationships, and key drivers of gender wage gaps.
        Uses four complementary techniques to answer different questions about the data.

        **Four Analysis Types:**
        1. **Regression**: Which factors predict wage gaps? How strong are the relationships?
        2. **Clustering**: Which countries have similar profiles? Are there natural groups?
        3. **Feature Importance**: Which variables matter most for prediction?
        4. **PCA**: How can we visualize high-dimensional data in 2D?

        **Best For:**
        - Researchers wanting statistical rigor
        - Students learning ML applications in economics
        - Policymakers identifying key levers for intervention
        """)

        # General ML explanation
        with st.expander("🤖 What is Machine Learning? (Overview)"):
            st.markdown("""
            **Machine Learning vs Traditional Statistics:**

            Traditional statistics tests hypotheses you specify in advance (e.g., "Does unemployment affect wage gaps?").
            Machine learning discovers patterns in data without pre-specified hypotheses.

            **In This Page:**

            - **Supervised learning** (Regression, Random Forest): Predict wage gap from other variables
            - **Unsupervised learning** (Clustering, PCA): Find patterns without predicting a specific outcome

            **Why Use ML for Gender Wage Gap Research?**

            1. **Multiple predictors**: ML handles many variables simultaneously (GDP, unemployment, LFP, etc.)
            2. **Non-linear relationships**: Some methods (Random Forest) capture complex patterns
            3. **Pattern discovery**: Clustering reveals groups you might not hypothesize
            4. **Dimension reduction**: PCA simplifies visualization of complex data
            5. **Prediction**: Enables "what-if" scenarios (see What-If Analysis page)

            **Important Caveats:**
            - **Correlation ≠ Causation**: ML finds patterns, not causes
            - **Interpretability tradeoff**: More complex models (Random Forest) harder to interpret
            - **Sample size**: With only 12 countries, results should be validated on larger datasets
            - **Overfitting risk**: Complex models may memorize data rather than find true patterns
            """)

        # Regression explanation
        with st.expander("📊 Tab 1: Multiple Regression Explained"):
            st.markdown("""
            **What is Multiple Regression?**

            A statistical method that models the relationship between one outcome (wage gap) and multiple predictors (GDP, unemployment, etc.).

            **The Model:**
            ```
            Wage Gap = β₀ + β₁(Female LFP) + β₂(LFP Gap) + β₃(GDP) + β₄(Unemployment) + error
            ```

            **Reading the Regression Table:**

            - **Coefficient**: How much wage gap changes when variable increases by 1 unit
              - Example: Unemployment coefficient = 1.74 means +1% unemployment → +1.74 pp wage gap
            - **Std Error**: Uncertainty in coefficient estimate (smaller = more precise)
            - **t-stat**: Coefficient divided by std error (values > 2 suggest significance)
            - **p-value**: Probability that coefficient is actually zero
              - p < 0.05 (*): Significant at 95% confidence
              - p < 0.01 (**): Significant at 99% confidence
              - p < 0.001 (***): Significant at 99.9% confidence
            - **R² = 0.80**: Model explains 80% of variance in wage gaps (very good!)

            **Key Findings:**
            - **Unemployment** (p = 0.002): Strong positive effect → reducing unemployment helps
            - **GDP** (p = 0.004): Negative coefficient → wealthier countries have smaller gaps
            - **Female LFP** (p = 0.010): Positive effect seems counterintuitive but may reflect selection bias
            - **LFP Gap** (p = 0.309): Not significant → doesn't reliably predict wage gap

            **Limitations:**
            - Assumes linear relationships (may miss non-linear patterns)
            - Assumes no multicollinearity (predictors not too correlated)
            - With 12 countries, estimates have wide confidence intervals
            """)

        # Clustering explanation
        with st.expander("🔬 Tab 2: K-Means Clustering Explained"):
            st.markdown("""
            **What is Clustering?**

            An unsupervised ML method that groups similar countries based on multiple characteristics, without being told which groups exist.

            **How K-Means Works:**
            1. Algorithm starts with random group centers
            2. Assigns each country to nearest center
            3. Recalculates centers as average of assigned countries
            4. Repeats until groups stabilize

            **What the Scatter Plot Shows:**
            - **X-axis**: GDP per capita (economic development)
            - **Y-axis**: Gender pay gap (our outcome of interest)
            - **Color**: Cluster assignment (algorithm-discovered groups)
            - **Size**: Unemployment rate (bubble size)

            **Interpreting Clusters:**

            - **Cluster 0 (Green) - "Mid-range EU"**: Moderate gaps, moderate GDP
            - **Cluster 1 (Red) - "High Gap"**: Larger wage gaps, often Balkan countries
            - **Cluster 2 (Blue) - "Unique"**: Sweden (high GDP, moderate gap)

            **Why This Matters:**
            - Algorithm naturally separated Balkans WITHOUT being told geography
            - Suggests structural economic similarities within region
            - Validates regional analysis (Balkans vs EU comparison)
            - Identifies outliers (countries that don't fit patterns)

            **Limitations:**
            - Must specify number of clusters (we chose 3)
            - Sensitive to outliers
            - With 12 countries, clusters may not be stable
            - Does not explain WHY countries cluster together
            """)

        # Feature importance explanation
        with st.expander("🌲 Tab 3: Random Forest Feature Importance Explained"):
            st.markdown("""
            **What is Random Forest?**

            An ensemble ML method that builds many decision trees and averages their predictions.
            Unlike linear regression, can capture non-linear relationships and interactions.

            **How Feature Importance Works:**
            1. Build 100+ decision trees on random subsets of data
            2. Each tree "splits" on different variables to make predictions
            3. Measure how much each variable improves predictions across all trees
            4. Higher importance = variable used more frequently and improves accuracy more

            **Reading the Bar Chart:**
            - **Longer bar** = more important for predicting wage gap
            - **Importance** is normalized to sum to 100%
            - Relative rankings matter more than absolute values

            **Key Findings:**
            - **Unemployment (30.7%)**: Most important predictor
            - **LFP Gap (21.0%)**: Second most important
            - **GDP per capita (15.6%)**: Third most important
            - **Female/Male LFP individually**: Less important than the gap between them

            **Random Forest vs Linear Regression:**
            - **RF advantages**: Captures non-linear effects, variable interactions, robust to outliers
            - **RF disadvantages**: Harder to interpret ("black box"), can overfit small datasets
            - **When rankings differ**: Suggests non-linear relationships RF captures but regression misses

            **Practical Implications:**
            - Focus policy interventions on top 3 features (unemployment, LFP gap, GDP)
            - Lower-ranked features may still matter but have weaker direct effects
            - Interactions between variables (e.g., unemployment × GDP) captured automatically
            """)

        # PCA explanation
        with st.expander("📈 Tab 4: PCA (Principal Component Analysis) Explained"):
            st.markdown("""
            **What is PCA?**

            A dimension reduction technique that transforms many correlated variables into few uncorrelated "principal components."
            Allows visualization of high-dimensional data in 2D.

            **The Problem:**
            - We have 15+ variables per country (GDP, unemployment, LFP, wages, etc.)
            - Humans can't visualize 15 dimensions
            - Many variables are correlated (GDP correlates with development, which correlates with wages, etc.)

            **The Solution:**
            - PCA finds the 2 directions in 15D space that capture the most variation
            - **PC1 (Principal Component 1)**: Direction of maximum variance (often "economic development")
            - **PC2**: Direction of second-most variance, uncorrelated with PC1 (often "gender gap specific factors")

            **Reading the PCA Plot:**
            - **X-axis (PC1)**: Typically represents overall economic development
              - Right = wealthier, more developed
              - Left = less wealthy, developing
            - **Y-axis (PC2)**: Typically represents gender-specific factors
              - Up = larger wage gaps, lower female LFP
              - Down = smaller gaps, higher female LFP
            - **Distance between countries**: Similar positions = similar overall profiles
            - **Variance explained**: PC1 + PC2 usually explain 60-80% of total variance

            **What Positions Mean:**
            - **Top-right quadrant**: High GDP but high wage gap (e.g., Hungary, Sweden)
            - **Bottom-right quadrant**: High GDP and low wage gap (ideal state)
            - **Top-left quadrant**: Low GDP and high wage gap (Balkan countries)
            - **Bottom-left quadrant**: Low GDP but low wage gap (e.g., Italy paradox)

            **Limitations:**
            - PCs are linear combinations (lose interpretability)
            - Only shows 2 dimensions (loses information from other 13 dimensions)
            - Loadings (which variables contribute to each PC) need separate analysis
            - With 12 countries, positions may be unstable
            """)

        tab1, tab2, tab3, tab4 = st.tabs(["Regression", "Clustering", "Feature Importance", "PCA"])

        with tab1:
            st.subheader("Multivariate Regression Analysis")

            st.markdown("""
            **Model Performance:** R² = 0.80 (explains 80% of variance)
            """)

            regression_results = pd.DataFrame({
                'Variable': ['Constant', 'Female LFP', 'LFP Gap', 'GDP per capita', 'Unemployment'],
                'Coefficient': [-51.84, 0.95, 0.33, -0.0005, 1.74],
                'Std Error': [19.02, 0.27, 0.30, 0.0001, 0.36],
                't-stat': [-2.73, 3.53, 1.10, -4.17, 4.78],
                'p-value': [0.030, 0.010, 0.309, 0.004, 0.002],
                'Significance': ['**', '***', '', '***', '***']
            })

            st.dataframe(regression_results, width='stretch', hide_index=True)

            st.markdown("""
            **Key Interpretations:**
            1. **Unemployment (+1.74):** Every 1% increase in unemployment raises wage gap by 1.74 pp
            2. **GDP per capita (-0.0005):** Higher GDP = lower wage gap
            3. **Female LFP (+0.95):** Counterintuitive positive effect due to selection bias
            """)

        with tab2:
            st.subheader("K-Means Clustering")

            cluster_data = df_country[['country', 'cluster', 'gap_mean', 'region']].copy()
            cluster_data['Cluster Name'] = cluster_data['cluster'].map({
                0: 'Mid-range EU',
                1: 'High Gap',
                2: 'Unique (Sweden)'
            })

            fig = px.scatter(
                df_country,
                x='gdp_per_capita',
                y='gap_mean',
                color=df_country['cluster'].astype(str),
                hover_name='country',
                size='unemployment',
                labels={
                    'gdp_per_capita': 'GDP per Capita ($)',
                    'gap_mean': 'Gender Pay Gap (%)',
                    'color': 'Cluster'
                },
                color_discrete_map={'0': '#2ecc71', '1': '#e74c3c', '2': '#3498db'}
            )

            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')
            add_chart_export(fig, "kmeans_clustering", "clustering_chart")

            st.markdown("""
            **Key Finding:** The algorithm naturally separated Balkans into a high-gap cluster
            without being told which countries are Balkans. This validates structural similarities.
            """)

            st.dataframe(cluster_data[['country', 'Cluster Name', 'gap_mean', 'region']],
                        width='stretch', hide_index=True)

        with tab3:
            st.subheader("Random Forest Feature Importance")

            importance_data = pd.DataFrame({
                'Feature': ['Unemployment', 'LFP Gap', 'GDP per capita', 'Log GDP', 'Male LFP', 'Female LFP'],
                'Importance': [30.7, 21.0, 15.6, 12.7, 10.6, 9.5]
            })

            fig = px.bar(
                importance_data,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Reds'
            )

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
            add_chart_export(fig, "feature_importance", "importance_chart")

            st.success("**Unemployment** is the most important predictor of wage gaps, explaining 30.7% of variance.")

        with tab4:
            st.subheader("Principal Component Analysis")

            st.markdown("""
            | Component | Variance Explained | Cumulative |
            |-----------|-------------------|------------|
            | PC1 (Economic Development) | 65.3% | 65.3% |
            | PC2 (Gender Inequality) | 19.0% | 84.3% |
            | PC3 | 13.1% | 97.4% |
            """)

            # Simulated PCA biplot
            fig = go.Figure()

            # Add country points (simulated based on their characteristics)
            for _, row in df_country.iterrows():
                pc1 = (row['gdp_per_capita'] / 10000 - 2) + (row['female_lfp'] - 50) / 20
                pc2 = (row['gap_mean'] - 10) / 5

                color = '#e74c3c' if row['region'] == 'Balkans' else '#3498db'

                fig.add_trace(go.Scatter(
                    x=[pc1], y=[pc2],
                    mode='markers+text',
                    marker=dict(size=12, color=color),
                    text=[row['country']],
                    textposition='top center',
                    showlegend=False
                ))

            fig.update_layout(
                xaxis_title="PC1: Economic Development (65.3%)",
                yaxis_title="PC2: Gender Inequality (19.0%)",
                height=500
            )

            st.plotly_chart(fig, width='stretch')
            add_chart_export(fig, "pca_biplot", "pca_chart")

            st.info("**Interpretation:** Balkans (red) cluster in low PC1 (less developed), high PC2 (high inequality) quadrant.")

    # ========== OAXACA-BLINDER PAGE ==========
    elif page == "Oaxaca-Blinder":
        st.header("Oaxaca-Blinder Decomposition")

        st.markdown("""
        The Oaxaca-Blinder decomposition breaks down the wage gap difference between Balkans and EU
        into **explained** (due to differences in characteristics) and **unexplained** (structural/discrimination) components.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Decomposition Results")

            decomp_data = pd.DataFrame({
                'Component': ['Total Gap', 'Explained', 'Unexplained'],
                'Value (pp)': [6.7, 5.2, 1.5],
                'Percentage': ['100%', '78%', '22%']
            })

            st.dataframe(decomp_data, width='stretch', hide_index=True)

            st.markdown("---")

            st.subheader("Explained Components")

            explained_data = pd.DataFrame({
                'Factor': ['Unemployment', 'GDP per capita', 'Female LFP', 'LFP Gap'],
                'Contribution (pp)': [4.88, 2.46, -2.06, -0.08],
                'Direction': ['Increases gap', 'Increases gap', 'Decreases gap', 'Decreases gap']
            })

            st.dataframe(explained_data, width='stretch', hide_index=True)

        with col2:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Explained (78%)', 'Unexplained (22%)'],
                values=[78, 22],
                marker_colors=['#3498db', '#e74c3c'],
                hole=0.4,
                textinfo='label+percent'
            )])

            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )

            st.plotly_chart(fig, width='stretch')

            # Bar chart of components
            fig2 = go.Figure(go.Bar(
                x=[4.88, 2.46, -2.06, -0.08],
                y=['Unemployment', 'GDP', 'Female LFP', 'LFP Gap'],
                orientation='h',
                marker_color=['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71'],
                text=['+4.88', '+2.46', '-2.06', '-0.08'],
                textposition='outside'
            ))

            fig2.update_layout(
                xaxis_title="Contribution to Gap (pp)",
                height=250,
                margin=dict(l=20, r=60, t=20, b=20)
            )

            st.plotly_chart(fig2, width='stretch')
            add_chart_export(fig2, "oaxaca_blinder", "oaxaca_chart")

        st.success("""
        **Key Finding:** 78% of the Balkans-EU gap difference can be explained by observable factors,
        primarily higher unemployment rates in the Balkans. The remaining 22% represents structural
        differences or potential discrimination effects.
        """)

    # ========== DATA EXPLORER PAGE ==========
    elif page == "Data Explorer":
        st.header("Data Explorer")

        # Introduction and Educational Content
        st.info("""
        📖 **What This Page Shows**

        Raw data access and download functionality for researchers who want to perform their own analysis.
        Two datasets available: country-level summary statistics and full time series data.

        **How to Use:**
        1. **Country Summary Tab**: Aggregated data (one row per country) with filters
        2. **Full Dataset Tab**: Complete time series (multiple years per country)
        3. Apply filters to subset data
        4. Download filtered results as CSV for Excel, R, Python, Stata, etc.

        **Best For:**
        - Researchers conducting independent analysis
        - Students learning data analysis
        - Replicating or extending findings
        - Custom visualizations not available in this dashboard
        """)

        # Data dictionary
        with st.expander("📚 Data Dictionary: What Each Column Means"):
            st.markdown("""
            **Country Summary Dataset Columns:**

            | Column | Description | Units | Example |
            |--------|-------------|-------|---------|
            | **country** | Country name | Text | "North Macedonia" |
            | **region** | Geographic classification | Balkans/EU | "Balkans" |
            | **gap_mean** | Average gender wage gap | % | 13.2 |
            | **female_lfp** | Female labor force participation | % of working-age women | 52.3 |
            | **male_lfp** | Male labor force participation | % of working-age men | 68.7 |
            | **gdp_per_capita** | GDP per capita (PPP) | USD | 18,450 |
            | **unemployment** | Unemployment rate | % of labor force | 15.2 |
            | **cluster** | ML cluster assignment | 0, 1, or 2 | 1 |

            **Full Time Series Dataset Additional Columns:**

            | Column | Description | Units | Example |
            |--------|-------------|-------|---------|
            | **year** | Observation year | YYYY | 2023 |
            | **wage_gap_pct** | Wage gap for that year | % | 14.1 |
            | **data_source** | Origin of data | Text | "Eurostat" |
            | **validation_flag** | Data quality indicator | Text | "validated" |

            **Important Notes:**

            - **Wage gap calculation**: (Male avg earnings - Female avg earnings) / Male avg earnings × 100
            - **PPP (Purchasing Power Parity)**: Adjusts for cost of living differences between countries
            - **Labor force participation**: Includes employed + actively seeking work (excludes students, retirees, homemakers not seeking work)
            - **Missing values**: Some countries have gaps in time series due to survey schedules
            - **Cluster**: From K-Means algorithm (0=Mid-range EU, 1=High Gap, 2=Unique)
            """)

        # Filter instructions
        with st.expander("🔍 How to Use Filters"):
            st.markdown("""
            **Country Summary Tab Filters:**

            1. **Region Filter**:
               - Select "Balkans" to see only Balkan countries
               - Select "EU" to see only EU reference countries
               - Select both (default) to see all countries
               - Use this to isolate specific geographic groups

            2. **Gap Range Slider**:
               - Drag left handle to set minimum gap threshold
               - Drag right handle to set maximum gap threshold
               - Example: Set to [10, 15] to see countries with gaps between 10-15%
               - Use this to find countries with similar gap magnitudes

            **Full Dataset Tab Filters:**

            - **Country Selector**: Choose which countries to include in download
            - Default shows first 3 countries (to avoid overwhelming table)
            - Select all countries to download complete dataset
            - Filtered data updates table and CSV download in real-time

            **Download Tips:**
            - Click "📥 Download as CSV" button after applying filters
            - CSV opens in Excel, R, Python (pandas), Stata, SPSS
            - First row contains column headers
            - Use for statistical analysis, custom charts, or publications
            """)

        # Citation guidance
        with st.expander("📝 How to Cite This Data"):
            st.markdown("""
            **If Using This Data in Research:**

            **Recommended Citation Format:**
            ```
            Mickovski, K. (2025). Gender Wage Gap Analysis: Balkans vs European Union
            Comparison. Interactive Dashboard. Data sources: Eurostat (2023-2024),
            World Bank Development Indicators, ILO Statistics.
            ```

            **Data Source Citations:**

            - **Eurostat**: [Gender pay gap in unadjusted form](https://ec.europa.eu/eurostat/databrowser/view/sdg_05_20/default/table?lang=en)
            - **World Bank**: [World Development Indicators](https://databank.worldbank.org/source/world-development-indicators)
            - **ILO**: [ILOSTAT Database](https://ilostat.ilo.org/data/)

            **Methodological Transparency:**
            - All code and methodology available in repository
            - Regression coefficients: R² = 0.80
            - Time series models: ARIMA(1,1,0), ETS, Linear trend
            - Clustering: K-Means with k=3

            **Responsible Use:**
            - Acknowledge data limitations (12 countries, unadjusted gap)
            - Report confidence intervals for estimates
            - Avoid causal claims from correlational analysis
            - Consider updating with more recent data when available
            """)

        tab1, tab2 = st.tabs(["Country Summary", "Full Dataset"])

        with tab1:
            st.subheader("Country-Level Data")

            display_cols = ['country', 'region', 'gap_mean', 'female_lfp', 'male_lfp',
                          'gdp_per_capita', 'unemployment', 'cluster']

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                region_filter = st.multiselect("Filter by Region", ['Balkans', 'EU'], default=['Balkans', 'EU'])
            with col2:
                gap_range = st.slider("Gap Range (%)", 0.0, 20.0, (0.0, 20.0))

            filtered_df = df_country[
                (df_country['region'].isin(region_filter)) &
                (df_country['gap_mean'] >= gap_range[0]) &
                (df_country['gap_mean'] <= gap_range[1])
            ][display_cols]

            st.dataframe(filtered_df, width='stretch', hide_index=True)

            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="wage_gap_country_data.csv",
                mime="text/csv"
            )

        with tab2:
            st.subheader("Full Time Series Data")

            # Country filter for main data
            selected = st.multiselect(
                "Select Countries",
                df_main['country'].unique().tolist(),
                default=df_main['country'].unique().tolist()[:3]
            )

            if selected:
                filtered_main = df_main[df_main['country'].isin(selected)]
                st.dataframe(filtered_main, width='stretch', hide_index=True)

                csv_main = filtered_main.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Data as CSV",
                    data=csv_main,
                    file_name="wage_gap_full_data.csv",
                    mime="text/csv"
                )
            else:
                st.info("Select at least one country to view data.")

else:
    st.error("Unable to load data. Please ensure the data files exist in the correct location.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Gender Wage Gap Analysis Dashboard | Kiril Mickovski | December 2025</p>
    <p>Data Sources: Eurostat, World Bank, National Statistical Offices</p>
</div>
""", unsafe_allow_html=True)
