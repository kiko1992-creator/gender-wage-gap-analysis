"""
EU27 Gender Wage Gap Database - Interactive Dashboard
Displays real Eurostat data for all 27 EU member states (2020-2023)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database_connection import (
    get_all_countries_2023,
    get_country_trend,
    get_regional_comparison,
    get_improvement_rankings
)

# Page config
st.set_page_config(page_title="EU27 Database", page_icon="🇪🇺", layout="wide")

st.title("🇪🇺 EU27 Gender Wage Gap Database")
st.markdown("**Real Eurostat data for all 27 EU member states (2020-2023)**")

# Sidebar
with st.sidebar:
    st.header("📊 Data Source")
    st.info("""
    **Database:** PostgreSQL
    **Records:** 108 observations
    **Countries:** 27 EU members
    **Years:** 2020-2023
    **Source:** Eurostat
    """)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 All Countries",
    "📈 Country Details",
    "🗺️ Regional Analysis",
    "🏆 Improvement Rankings"
])

# ============================================================================
# TAB 1: ALL COUNTRIES
# ============================================================================
with tab1:
    st.header("All 27 EU Countries - 2023 Wage Gap")

    # Load data
    df_2023 = get_all_countries_2023()

    if df_2023 is not None and len(df_2023) > 0:
        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "EU Average",
                f"{df_2023['wage_gap_percent'].mean():.1f}%"
            )

        with col2:
            worst = df_2023.iloc[0]
            st.metric(
                "Worst Gap",
                f"{worst['wage_gap_percent']:.1f}%",
                f"{worst['country_name']}"
            )

        with col3:
            best = df_2023.iloc[-1]
            st.metric(
                "Best Gap",
                f"{best['wage_gap_percent']:.1f}%",
                f"{best['country_name']}"
            )

        with col4:
            st.metric(
                "Gap Range",
                f"{df_2023['wage_gap_percent'].max() - df_2023['wage_gap_percent'].min():.1f}pp"
            )

        st.markdown("---")

        # Interactive bar chart
        st.subheader("📊 Country Rankings")

        fig = px.bar(
            df_2023,
            x='wage_gap_percent',
            y='country_name',
            orientation='h',
            color='region',
            title='Gender Wage Gap by Country (2023)',
            labels={
                'wage_gap_percent': 'Wage Gap (%)',
                'country_name': 'Country',
                'region': 'Region'
            },
            height=800
        )

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("📋 Complete Dataset")

        # Format the dataframe for display
        df_display = df_2023.copy()
        df_display['population'] = df_display['population'].apply(lambda x: f"{x:,}")
        df_display['gdp_billions'] = df_display['gdp_billions'].apply(lambda x: f"${x:.1f}B")
        df_display['wage_gap_percent'] = df_display['wage_gap_percent'].apply(lambda x: f"{x:.1f}%")

        df_display.columns = ['Country', 'Region', 'Population', 'GDP', 'Wage Gap']

        st.dataframe(df_display, use_container_width=True, height=600)

        # Download button
        csv = df_2023.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="eu27_wage_gap_2023.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ Could not load data. Make sure PostgreSQL is running and the database exists.")

# ============================================================================
# TAB 2: COUNTRY DETAILS
# ============================================================================
with tab2:
    st.header("Country Deep Dive")

    # Load country list
    df_countries = get_all_countries_2023()

    if df_countries is not None and len(df_countries) > 0:
        # Country selector
        country = st.selectbox(
            "Select a country:",
            options=sorted(df_countries['country_name'].tolist())
        )

        if country:
            # Get country trend
            df_trend = get_country_trend(country)

            if df_trend is not None and len(df_trend) > 0:
                # Show metrics
                col1, col2, col3 = st.columns(3)

                latest_gap = df_trend.iloc[-1]['wage_gap_percent']
                earliest_gap = df_trend.iloc[0]['wage_gap_percent']
                change = latest_gap - earliest_gap

                with col1:
                    st.metric(
                        "2023 Wage Gap",
                        f"{latest_gap:.1f}%"
                    )

                with col2:
                    st.metric(
                        "2020 Wage Gap",
                        f"{earliest_gap:.1f}%"
                    )

                with col3:
                    st.metric(
                        "Change (2020→2023)",
                        f"{change:+.1f}pp",
                        delta=f"{change:.1f}pp",
                        delta_color="inverse"  # Lower is better
                    )

                st.markdown("---")

                # Trend chart
                fig = px.line(
                    df_trend,
                    x='year',
                    y='wage_gap_percent',
                    markers=True,
                    title=f'{country} - Wage Gap Trend (2020-2023)',
                    labels={
                        'year': 'Year',
                        'wage_gap_percent': 'Wage Gap (%)'
                    }
                )

                fig.update_traces(line_color='#FF6B6B', marker_size=10)
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                if change < -1:
                    st.success(f"✅ **Good news!** {country} has **improved** by {abs(change):.1f} percentage points since 2020.")
                elif change > 1:
                    st.error(f"⚠️ {country} has **worsened** by {change:.1f} percentage points since 2020.")
                else:
                    st.info(f"📊 {country}'s wage gap has remained relatively **stable** (±{abs(change):.1f}pp).")

                # Comparison to EU average
                eu_avg = df_countries['wage_gap_percent'].mean()
                diff_from_avg = latest_gap - eu_avg

                st.markdown("### Comparison to EU Average")
                if diff_from_avg > 0:
                    st.write(f"{country} is **{diff_from_avg:.1f}pp above** the EU average ({eu_avg:.1f}%).")
                else:
                    st.write(f"{country} is **{abs(diff_from_avg):.1f}pp below** the EU average ({eu_avg:.1f}%).")
            else:
                st.warning("No trend data available for this country.")
    else:
        st.error("Could not load country list.")

# ============================================================================
# TAB 3: REGIONAL ANALYSIS
# ============================================================================
with tab3:
    st.header("Regional Comparison")

    df_regional = get_regional_comparison()

    if df_regional is not None and len(df_regional) > 0:
        # Regional bar chart
        fig = px.bar(
            df_regional,
            x='region',
            y='avg_gap',
            title='Average Wage Gap by Region (2023)',
            labels={
                'region': 'Region',
                'avg_gap': 'Average Wage Gap (%)'
            },
            color='avg_gap',
            color_continuous_scale='RdYlGn_r'  # Red for high, green for low
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Regional stats table
        st.subheader("📊 Regional Statistics")

        df_regional_display = df_regional.copy()
        df_regional_display.columns = ['Region', 'Countries', 'Avg Gap', 'Min Gap', 'Max Gap']
        df_regional_display['Avg Gap'] = df_regional_display['Avg Gap'].apply(lambda x: f"{x:.1f}%")
        df_regional_display['Min Gap'] = df_regional_display['Min Gap'].apply(lambda x: f"{x:.1f}%")
        df_regional_display['Max Gap'] = df_regional_display['Max Gap'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(df_regional_display, use_container_width=True)

        # Key findings
        st.markdown("### 🔍 Key Findings")

        worst_region = df_regional.iloc[0]
        best_region = df_regional.iloc[-1]

        st.write(f"""
        - **Worst Region:** {worst_region['region']} (avg: {worst_region['avg_gap']:.1f}%)
        - **Best Region:** {best_region['region']} (avg: {best_region['avg_gap']:.1f}%)
        - **Regional Gap:** {worst_region['avg_gap'] - best_region['avg_gap']:.1f} percentage points
        """)
    else:
        st.error("Could not load regional data.")

# ============================================================================
# TAB 4: IMPROVEMENT RANKINGS
# ============================================================================
with tab4:
    st.header("🏆 Who Improved Most? (2020 → 2023)")

    df_improvement = get_improvement_rankings()

    if df_improvement is not None and len(df_improvement) > 0:
        col1, col2 = st.columns(2)

        # Most improved
        with col1:
            st.subheader("✅ Most Improved")
            df_improved = df_improvement.head(10)

            for idx, row in df_improved.iterrows():
                st.success(f"""
                **{row['country_name']}**
                {row['gap_2020']:.1f}% → {row['gap_2023']:.1f}% ({row['change']:+.1f}pp)
                """)

        # Most worsened
        with col2:
            st.subheader("⚠️ Most Worsened")
            df_worsened = df_improvement.tail(10)

            for idx, row in df_worsened.iterrows():
                st.error(f"""
                **{row['country_name']}**
                {row['gap_2020']:.1f}% → {row['gap_2023']:.1f}% ({row['change']:+.1f}pp)
                """)

        # Full comparison chart
        st.markdown("---")
        st.subheader("📊 Complete Change Comparison")

        fig = px.bar(
            df_improvement,
            x='change',
            y='country_name',
            orientation='h',
            title='Change in Wage Gap (2020-2023)',
            labels={
                'change': 'Change (percentage points)',
                'country_name': 'Country'
            },
            color='change',
            color_continuous_scale='RdYlGn_r',
            height=800
        )

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not load improvement data.")

# Footer
st.markdown("---")
st.caption("Data source: Eurostat | Database: PostgreSQL | Analysis: PhD Research Project")
