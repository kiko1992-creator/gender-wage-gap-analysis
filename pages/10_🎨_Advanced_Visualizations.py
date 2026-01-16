"""
Advanced Visualizations - Interactive EU Wage Gap Analysis
Features: Interactive Map, Country Comparison, Correlation Analysis
Perfect for PhD presentations and defense!
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from database_connection import (
    get_all_countries_2023,
    get_country_trend,
    get_regional_comparison,
    get_improvement_rankings
)

# Page config
st.set_page_config(page_title="Advanced Visualizations", page_icon="🎨", layout="wide")

st.title("🎨 Advanced Visualizations & Analysis")
st.markdown("**Interactive visualizations for PhD research presentation**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Visualization Settings")

    color_scheme = st.selectbox(
        "Color Scheme",
        ["Red-Green (Diverging)", "Viridis", "Plasma", "Blues", "Reds"]
    )

    show_labels = st.checkbox("Show Data Labels", value=True)
    animation_speed = st.slider("Animation Speed (ms)", 500, 2000, 1000, 100)

# Load data
df_2023 = get_all_countries_2023()
df_improvement = get_improvement_rankings()

if df_2023 is None or len(df_2023) == 0:
    st.error("Could not load data. Please check database connection.")
    st.stop()

# Map country names to ISO Alpha-3 codes for choropleth map
country_codes = {
    'Austria': 'AUT', 'Belgium': 'BEL', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
    'Cyprus': 'CYP', 'Czechia': 'CZE', 'Denmark': 'DNK', 'Estonia': 'EST',
    'Finland': 'FIN', 'France': 'FRA', 'Germany': 'DEU', 'Greece': 'GRC',
    'Hungary': 'HUN', 'Ireland': 'IRL', 'Italy': 'ITA', 'Latvia': 'LVA',
    'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Malta': 'MLT', 'Netherlands': 'NLD',
    'Poland': 'POL', 'Portugal': 'PRT', 'Romania': 'ROU', 'Slovakia': 'SVK',
    'Slovenia': 'SVN', 'Spain': 'ESP', 'Sweden': 'SWE'
}

df_2023['iso_alpha'] = df_2023['country_name'].map(country_codes)

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Interactive Map",
    "🔄 Country Comparison",
    "📊 Correlation Analysis",
    "🎬 Animated Trends",
    "📈 Statistical Dashboard"
])

# ============================================================================
# TAB 1: INTERACTIVE EU MAP
# ============================================================================
with tab1:
    st.header("🗺️ Interactive European Wage Gap Map")

    st.markdown("""
    **Instructions:**
    - Hover over countries to see exact values
    - Click and drag to pan
    - Scroll to zoom
    - Red = Higher wage gaps (worse), Green = Lower gaps (better)
    """)

    # Color scale based on user selection
    if color_scheme == "Red-Green (Diverging)":
        color_continuous_scale = "RdYlGn_r"
    elif color_scheme == "Viridis":
        color_continuous_scale = "Viridis"
    elif color_scheme == "Plasma":
        color_continuous_scale = "Plasma"
    elif color_scheme == "Blues":
        color_continuous_scale = "Blues"
    else:
        color_continuous_scale = "Reds"

    # Create choropleth map
    fig_map = px.choropleth(
        df_2023,
        locations='iso_alpha',
        color='wage_gap_percent',
        hover_name='country_name',
        hover_data={
            'iso_alpha': False,
            'wage_gap_percent': ':.1f',
            'region': True,
            'population': ':,',
            'gdp_billions': ':.1f'
        },
        color_continuous_scale=color_continuous_scale,
        scope='europe',
        title='Gender Wage Gap Across Europe (2023)',
        labels={'wage_gap_percent': 'Wage Gap (%)'}
    )

    fig_map.update_geos(
        fitbounds="locations",
        visible=True,
        showcountries=True,
        countrycolor="lightgray"
    )

    fig_map.update_layout(
        height=600,
        coloraxis_colorbar=dict(
            title="Wage Gap %",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Key insights below map
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🔴 Highest Gap",
            f"{df_2023['wage_gap_percent'].max():.1f}%",
            df_2023.loc[df_2023['wage_gap_percent'].idxmax(), 'country_name']
        )

    with col2:
        st.metric(
            "🟢 Lowest Gap",
            f"{df_2023['wage_gap_percent'].min():.1f}%",
            df_2023.loc[df_2023['wage_gap_percent'].idxmin(), 'country_name']
        )

    with col3:
        st.metric(
            "📊 EU Average",
            f"{df_2023['wage_gap_percent'].mean():.1f}%",
            "27 countries"
        )

# ============================================================================
# TAB 2: COUNTRY COMPARISON TOOL
# ============================================================================
with tab2:
    st.header("🔄 Multi-Country Comparison Tool")

    # Country selector
    selected_countries = st.multiselect(
        "Select countries to compare (2-6 recommended):",
        options=sorted(df_2023['country_name'].tolist()),
        default=['Croatia', 'Serbia', 'Poland', 'Germany']
    )

    if len(selected_countries) == 0:
        st.warning("Please select at least one country to compare.")
    else:
        # Get trend data for selected countries
        comparison_data = []

        for country in selected_countries:
            trend_df = get_country_trend(country)
            if trend_df is not None and len(trend_df) > 0:
                trend_df['country'] = country
                comparison_data.append(trend_df)

        if comparison_data:
            df_comparison = pd.concat(comparison_data, ignore_index=True)

            # Side-by-side metrics
            cols = st.columns(len(selected_countries))

            for idx, country in enumerate(selected_countries):
                country_data = df_comparison[df_comparison['country'] == country]
                if len(country_data) > 0:
                    latest = country_data.iloc[-1]['wage_gap_percent']
                    earliest = country_data.iloc[0]['wage_gap_percent']
                    change = latest - earliest

                    with cols[idx]:
                        st.metric(
                            country,
                            f"{latest:.1f}%",
                            f"{change:+.1f}pp",
                            delta_color="inverse"
                        )

            st.markdown("---")

            # Multi-line comparison chart
            st.subheader("📈 Trend Comparison (2020-2023)")

            fig_comparison = px.line(
                df_comparison,
                x='year',
                y='wage_gap_percent',
                color='country',
                markers=True,
                title='Wage Gap Trends: Selected Countries',
                labels={
                    'year': 'Year',
                    'wage_gap_percent': 'Wage Gap (%)',
                    'country': 'Country'
                },
                height=500
            )

            fig_comparison.update_traces(marker_size=10, line_width=3)
            fig_comparison.update_layout(
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_comparison, use_container_width=True)

            # Comparison table
            st.subheader("📋 Detailed Comparison")

            # Pivot table for easy comparison
            pivot_df = df_comparison.pivot(index='year', columns='country', values='wage_gap_percent')
            pivot_df = pivot_df.round(1)

            # Add change row
            if len(pivot_df) > 1:
                change_row = pivot_df.iloc[-1] - pivot_df.iloc[0]
                change_row.name = 'Change (2020→2023)'
                display_df = pd.concat([pivot_df, pd.DataFrame([change_row])])

                st.dataframe(display_df.style.format("{:.1f}%").background_gradient(
                    cmap='RdYlGn_r',
                    axis=None
                ), use_container_width=True)
            else:
                st.dataframe(pivot_df)
        else:
            st.warning("No trend data available for selected countries.")

# ============================================================================
# TAB 3: CORRELATION ANALYSIS
# ============================================================================
with tab3:
    st.header("📊 Correlation & Scatter Plot Analysis")

    st.markdown("**Explore relationships between wage gap and economic indicators**")

    # GDP vs Wage Gap
    st.subheader("💰 Economic Development vs Gender Equality")

    # Calculate GDP per capita
    df_2023['gdp_per_capita'] = (df_2023['gdp_billions'] * 1000000000 / df_2023['population']).round(0)

    # Scatter plot: GDP per capita vs Wage Gap
    fig_scatter1 = px.scatter(
        df_2023,
        x='gdp_per_capita',
        y='wage_gap_percent',
        size='population',
        color='region',
        hover_name='country_name',
        title='GDP per Capita vs Wage Gap',
        labels={
            'gdp_per_capita': 'GDP per Capita (€)',
            'wage_gap_percent': 'Wage Gap (%)',
            'population': 'Population',
            'region': 'Region'
        },
        height=500,
        trendline="ols"  # Add trend line
    )

    fig_scatter1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    st.plotly_chart(fig_scatter1, use_container_width=True)

    # Calculate correlation
    corr_gdp = df_2023['gdp_per_capita'].corr(df_2023['wage_gap_percent'])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Correlation Coefficient",
            f"{corr_gdp:.3f}",
            "GDP per Capita vs Wage Gap"
        )

    with col2:
        if abs(corr_gdp) > 0.5:
            strength = "STRONG"
        elif abs(corr_gdp) > 0.3:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        direction = "negative" if corr_gdp < 0 else "positive"
        st.info(f"**{strength} {direction} correlation**")

    st.markdown("---")

    # Population vs Wage Gap
    st.subheader("👥 Population Size vs Wage Gap")

    fig_scatter2 = px.scatter(
        df_2023,
        x='population',
        y='wage_gap_percent',
        size='gdp_billions',
        color='region',
        hover_name='country_name',
        title='Population vs Wage Gap',
        labels={
            'population': 'Population',
            'wage_gap_percent': 'Wage Gap (%)',
            'gdp_billions': 'GDP (billions)',
            'region': 'Region'
        },
        height=500,
        log_x=True  # Log scale for population
    )

    st.plotly_chart(fig_scatter2, use_container_width=True)

    # Regional box plots
    st.subheader("📦 Distribution by Region")

    fig_box = px.box(
        df_2023,
        x='region',
        y='wage_gap_percent',
        color='region',
        title='Wage Gap Distribution by Region',
        labels={
            'wage_gap_percent': 'Wage Gap (%)',
            'region': 'Region'
        },
        height=500
    )

    st.plotly_chart(fig_box, use_container_width=True)

# ============================================================================
# TAB 4: ANIMATED TRENDS
# ============================================================================
with tab4:
    st.header("🎬 Animated Wage Gap Evolution (2020-2023)")

    st.markdown("**Watch how wage gaps changed over time across Europe**")

    # Collect all trend data
    all_trends = []
    for country in df_2023['country_name'].unique():
        trend = get_country_trend(country)
        if trend is not None and len(trend) > 0:
            trend['country'] = country
            # Add region info
            region = df_2023[df_2023['country_name'] == country]['region'].values[0]
            trend['region'] = region
            all_trends.append(trend)

    if all_trends:
        df_animation = pd.concat(all_trends, ignore_index=True)

        # Animated bar chart
        fig_anim = px.bar(
            df_animation,
            x='wage_gap_percent',
            y='country',
            animation_frame='year',
            color='region',
            orientation='h',
            title='Wage Gap Evolution by Country',
            labels={
                'wage_gap_percent': 'Wage Gap (%)',
                'country': 'Country',
                'region': 'Region'
            },
            height=800,
            range_x=[0, df_animation['wage_gap_percent'].max() + 2]
        )

        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = animation_speed
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = animation_speed // 2

        st.plotly_chart(fig_anim, use_container_width=True)

        st.info("💡 **Tip:** Press the Play button to see the animation, or drag the year slider manually!")
    else:
        st.warning("Animation data not available.")

# ============================================================================
# TAB 5: STATISTICAL DASHBOARD
# ============================================================================
with tab5:
    st.header("📈 Statistical Dashboard")

    st.markdown("**Comprehensive statistical analysis for PhD research**")

    # Summary statistics
    st.subheader("📊 Descriptive Statistics")

    stats_df = df_2023['wage_gap_percent'].describe().round(2)
    stats_df = pd.DataFrame(stats_df)
    stats_df.columns = ['Wage Gap (%)']

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        # Distribution histogram
        fig_hist = px.histogram(
            df_2023,
            x='wage_gap_percent',
            nbins=15,
            title='Distribution of Wage Gaps Across EU27',
            labels={'wage_gap_percent': 'Wage Gap (%)'},
            color_discrete_sequence=['#FF6B6B']
        )

        # Add mean line
        mean_gap = df_2023['wage_gap_percent'].mean()
        fig_hist.add_vline(
            x=mean_gap,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: {mean_gap:.1f}%"
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Improvement analysis
    if df_improvement is not None and len(df_improvement) > 0:
        st.subheader("📉 Improvement Analysis (2020→2023)")

        # Split into improved and worsened
        df_improved = df_improvement[df_improvement['change'] < 0].head(10)
        df_worsened = df_improvement[df_improvement['change'] > 0].tail(10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Most Improved**")
            fig_improved = px.bar(
                df_improved,
                x='change',
                y='country_name',
                orientation='h',
                title='Top 10 Improvers',
                color='change',
                color_continuous_scale='Greens',
                labels={'change': 'Change (pp)', 'country_name': ''}
            )
            fig_improved.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_improved, use_container_width=True)

        with col2:
            st.markdown("**⚠️ Most Worsened**")
            if len(df_worsened) > 0:
                fig_worsened = px.bar(
                    df_worsened,
                    x='change',
                    y='country_name',
                    orientation='h',
                    title='Top 10 Worseners',
                    color='change',
                    color_continuous_scale='Reds',
                    labels={'change': 'Change (pp)', 'country_name': ''}
                )
                fig_worsened.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_worsened, use_container_width=True)

    # Export section
    st.markdown("---")
    st.subheader("📥 Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_2023 = df_2023.to_csv(index=False)
        st.download_button(
            "Download 2023 Data (CSV)",
            csv_2023,
            "eu27_2023_data.csv",
            "text/csv"
        )

    with col2:
        if df_improvement is not None:
            csv_improvement = df_improvement.to_csv(index=False)
            st.download_button(
                "Download Trends (CSV)",
                csv_improvement,
                "eu27_improvement.csv",
                "text/csv"
            )

    with col3:
        # Summary report
        summary_text = f"""
EU27 GENDER WAGE GAP ANALYSIS SUMMARY
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

OVERALL STATISTICS:
- Countries: {len(df_2023)}
- Average Gap: {df_2023['wage_gap_percent'].mean():.2f}%
- Median Gap: {df_2023['wage_gap_percent'].median():.2f}%
- Std Dev: {df_2023['wage_gap_percent'].std():.2f}%
- Range: {df_2023['wage_gap_percent'].min():.2f}% - {df_2023['wage_gap_percent'].max():.2f}%

WORST PERFORMERS:
{df_2023.nlargest(5, 'wage_gap_percent')[['country_name', 'wage_gap_percent']].to_string(index=False)}

BEST PERFORMERS:
{df_2023.nsmallest(5, 'wage_gap_percent')[['country_name', 'wage_gap_percent']].to_string(index=False)}
"""
        st.download_button(
            "Download Summary Report (TXT)",
            summary_text,
            "eu27_summary.txt",
            "text/plain"
        )

# Footer
st.markdown("---")
st.caption("🎨 Advanced Visualizations | Data: Eurostat | Analysis: PhD Research Project")
