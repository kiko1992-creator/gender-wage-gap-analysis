"""
Advanced Statistical Analysis & Econometric Models
PhD-level analysis for EU Gender Wage Gap Research
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="Advanced Statistics", page_icon="📊", layout="wide")

# ============================================================================
# PAGE HEADER
# ============================================================================
st.title("📊 Advanced Statistical Analysis")
st.markdown("""
**Econometric models and statistical tests for PhD research**
Panel regression • Convergence analysis • Clustering • Forecasting • Hypothesis testing
""")

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_panel_data():
    """Load complete panel dataset (all countries, all years)"""
    df_2023 = get_all_countries_2023()

    panel_data = []
    for country in df_2023['country_name'].tolist():
        trend = get_country_trend(country)
        if trend is not None and len(trend) > 0:
            for _, row in trend.iterrows():
                panel_data.append({
                    'country': country,
                    'year': row['year'],
                    'wage_gap': row['wage_gap_percent']
                })

    df_panel = pd.DataFrame(panel_data)

    # Merge with country characteristics
    country_info = df_2023[['country_name', 'region', 'population', 'gdp_billions']].copy()
    country_info['gdp_per_capita'] = (country_info['gdp_billions'] * 1000000000) / country_info['population']
    country_info = country_info.rename(columns={'country_name': 'country'})

    df_panel = df_panel.merge(country_info, on='country', how='left')

    return df_panel

df_panel = load_panel_data()
df_2023 = get_all_countries_2023()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Panel Regression",
    "🎯 Convergence Analysis",
    "🔍 Clustering",
    "🔮 Forecasting",
    "📋 Statistical Tests"
])

# ============================================================================
# TAB 1: PANEL REGRESSION ANALYSIS
# ============================================================================
with tab1:
    st.header("📈 Panel Data Regression Analysis")

    st.markdown("""
    **Fixed Effects Model:** Controls for unobserved country-specific characteristics
    **Random Effects Model:** Assumes country effects are random and uncorrelated with predictors
    """)

    # Prepare panel data
    df_regression = df_panel.dropna(subset=['wage_gap', 'gdp_per_capita'])
    df_regression['log_gdp_pc'] = np.log(df_regression['gdp_per_capita'])
    df_regression['year_numeric'] = df_regression['year'].astype(int)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Pooled OLS Model")
        st.markdown("**Model:** `wage_gap = β₀ + β₁·log(GDP_per_capita) + β₂·year + ε`")

        # Pooled OLS
        X = df_regression[['log_gdp_pc', 'year_numeric']].copy()
        X = sm.add_constant(X)
        y = df_regression['wage_gap']

        model_pooled = sm.OLS(y, X).fit()

        st.write("**Regression Results:**")
        results_df = pd.DataFrame({
            'Variable': ['Constant', 'log(GDP per capita)', 'Year'],
            'Coefficient': [model_pooled.params[0], model_pooled.params[1], model_pooled.params[2]],
            'Std Error': [model_pooled.bse[0], model_pooled.bse[1], model_pooled.bse[2]],
            'p-value': [model_pooled.pvalues[0], model_pooled.pvalues[1], model_pooled.pvalues[2]]
        })
        st.dataframe(results_df.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        st.metric("R-squared", f"{model_pooled.rsquared:.4f}")
        st.metric("Adj. R-squared", f"{model_pooled.rsquared_adj:.4f}")
        st.metric("F-statistic p-value", f"{model_pooled.f_pvalue:.6f}")

    with col2:
        st.subheader("🔹 Fixed Effects (Within) Model")
        st.markdown("**Controls for country-specific unobserved heterogeneity**")

        # Fixed Effects: Demean variables by country
        df_fe = df_regression.copy()
        for var in ['wage_gap', 'log_gdp_pc', 'year_numeric']:
            country_means = df_fe.groupby('country')[var].transform('mean')
            df_fe[f'{var}_demeaned'] = df_fe[var] - country_means

        X_fe = df_fe[['log_gdp_pc_demeaned', 'year_numeric_demeaned']]
        X_fe = sm.add_constant(X_fe)
        y_fe = df_fe['wage_gap_demeaned']

        model_fe = sm.OLS(y_fe, X_fe).fit()

        st.write("**Regression Results (Demeaned):**")
        results_fe_df = pd.DataFrame({
            'Variable': ['Constant', 'log(GDP per capita)', 'Year'],
            'Coefficient': [model_fe.params[0], model_fe.params[1], model_fe.params[2]],
            'Std Error': [model_fe.bse[0], model_fe.bse[1], model_fe.bse[2]],
            'p-value': [model_fe.pvalues[0], model_fe.pvalues[1], model_fe.pvalues[2]]
        })
        st.dataframe(results_fe_df.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        st.metric("Within R-squared", f"{model_fe.rsquared:.4f}")

    # Visualization
    st.subheader("📊 Residual Analysis")

    fig_residuals = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuals vs Fitted", "Q-Q Plot")
    )

    # Residuals vs Fitted
    fitted = model_pooled.fittedvalues
    residuals = model_pooled.resid

    fig_residuals.add_trace(
        go.Scatter(x=fitted, y=residuals, mode='markers',
                  marker=dict(color='steelblue', size=8),
                  name='Residuals'),
        row=1, col=1
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Q-Q Plot
    qq = stats.probplot(residuals, dist="norm")
    fig_residuals.add_trace(
        go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                  marker=dict(color='steelblue', size=8),
                  name='Q-Q'),
        row=1, col=2
    )
    fig_residuals.add_trace(
        go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0],
                  mode='lines', line=dict(color='red', dash='dash'),
                  name='Normal'),
        row=1, col=2
    )

    fig_residuals.update_xaxes(title_text="Fitted values", row=1, col=1)
    fig_residuals.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig_residuals.update_yaxes(title_text="Residuals", row=1, col=1)
    fig_residuals.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig_residuals.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_residuals, use_container_width=True)

# ============================================================================
# TAB 2: CONVERGENCE ANALYSIS
# ============================================================================
with tab2:
    st.header("🎯 Convergence Analysis")

    st.markdown("""
    **Beta Convergence:** Do countries with higher initial wage gaps reduce them faster?
    **Sigma Convergence:** Is the dispersion of wage gaps decreasing over time?
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("β-Convergence (Beta)")
        st.markdown("**Test:** Countries with higher gaps in 2020 → larger reductions by 2023")

        # Calculate changes
        convergence_data = []
        for country in df_panel['country'].unique():
            country_data = df_panel[df_panel['country'] == country].sort_values('year')
            if len(country_data) >= 2:
                initial = country_data[country_data['year'] == 2020]['wage_gap'].values
                final = country_data[country_data['year'] == 2023]['wage_gap'].values

                if len(initial) > 0 and len(final) > 0:
                    convergence_data.append({
                        'country': country,
                        'initial_gap_2020': initial[0],
                        'final_gap_2023': final[0],
                        'change': final[0] - initial[0],
                        'growth_rate': (final[0] - initial[0]) / initial[0] * 100
                    })

        df_conv = pd.DataFrame(convergence_data)

        # Beta convergence regression
        X_conv = sm.add_constant(df_conv['initial_gap_2020'])
        y_conv = df_conv['growth_rate']

        model_conv = sm.OLS(y_conv, X_conv).fit()

        st.write(f"**Regression:** growth_rate = β₀ + β₁·initial_gap")
        st.write(f"- **β₁ coefficient:** {model_conv.params[1]:.4f}")
        st.write(f"- **p-value:** {model_conv.pvalues[1]:.4f}")
        st.write(f"- **R²:** {model_conv.rsquared:.4f}")

        if model_conv.params[1] < 0 and model_conv.pvalues[1] < 0.05:
            st.success("✅ **Beta convergence detected!** Countries with higher initial gaps are reducing them faster.")
        elif model_conv.params[1] < 0:
            st.info("⚠️ **Weak convergence** (β < 0 but not significant)")
        else:
            st.warning("❌ **No convergence** (β ≥ 0)")

        # Plot
        fig_beta = px.scatter(
            df_conv, x='initial_gap_2020', y='growth_rate',
            hover_name='country',
            labels={'initial_gap_2020': '2020 Wage Gap (%)',
                   'growth_rate': 'Growth Rate 2020-2023 (%)'},
            title='Beta Convergence Test',
            trendline='ols'
        )
        fig_beta.update_traces(marker=dict(size=12, color='steelblue'))
        st.plotly_chart(fig_beta, use_container_width=True)

    with col2:
        st.subheader("σ-Convergence (Sigma)")
        st.markdown("**Test:** Is the cross-country dispersion decreasing?")

        # Calculate sigma (standard deviation) for each year
        sigma_by_year = df_panel.groupby('year')['wage_gap'].agg(['std', 'var', 'mean']).reset_index()
        sigma_by_year.columns = ['year', 'std_dev', 'variance', 'mean']
        sigma_by_year['cv'] = sigma_by_year['std_dev'] / sigma_by_year['mean'] * 100  # Coefficient of variation

        st.dataframe(sigma_by_year.style.format({
            'std_dev': '{:.2f}',
            'variance': '{:.2f}',
            'mean': '{:.2f}',
            'cv': '{:.2f}%'
        }), hide_index=True)

        # Test for decreasing trend
        X_sigma = sm.add_constant(sigma_by_year['year'])
        y_sigma = sigma_by_year['std_dev']
        model_sigma = sm.OLS(y_sigma, X_sigma).fit()

        st.write(f"**Trend test:** std_dev = β₀ + β₁·year")
        st.write(f"- **β₁ coefficient:** {model_sigma.params[1]:.4f}")
        st.write(f"- **p-value:** {model_sigma.pvalues[1]:.4f}")

        if model_sigma.params[1] < 0 and model_sigma.pvalues[1] < 0.05:
            st.success("✅ **Sigma convergence detected!** Dispersion is decreasing significantly.")
        elif model_sigma.params[1] < 0:
            st.info("⚠️ **Weak sigma convergence** (trend negative but not significant)")
        else:
            st.warning("❌ **No sigma convergence** (dispersion stable/increasing)")

        # Plot
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(
            x=sigma_by_year['year'],
            y=sigma_by_year['std_dev'],
            mode='lines+markers',
            name='Standard Deviation',
            line=dict(color='steelblue', width=3),
            marker=dict(size=10)
        ))
        fig_sigma.update_layout(
            title='Sigma Convergence: Dispersion Over Time',
            xaxis_title='Year',
            yaxis_title='Standard Deviation (percentage points)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_sigma, use_container_width=True)

# ============================================================================
# TAB 3: CLUSTERING ANALYSIS
# ============================================================================
with tab3:
    st.header("🔍 Clustering Analysis")
    st.markdown("**Group countries with similar wage gap patterns using K-Means clustering**")

    # Prepare data for clustering
    clustering_data = []
    for country in df_panel['country'].unique():
        country_data = df_panel[df_panel['country'] == country].sort_values('year')

        if len(country_data) >= 4:
            gaps = country_data['wage_gap'].values
            clustering_data.append({
                'country': country,
                'gap_2020': gaps[0] if len(gaps) > 0 else np.nan,
                'gap_2021': gaps[1] if len(gaps) > 1 else np.nan,
                'gap_2022': gaps[2] if len(gaps) > 2 else np.nan,
                'gap_2023': gaps[3] if len(gaps) > 3 else np.nan,
                'mean_gap': np.mean(gaps),
                'trend': gaps[-1] - gaps[0],
                'volatility': np.std(gaps)
            })

    df_cluster = pd.DataFrame(clustering_data)

    # Number of clusters
    n_clusters = st.slider("Number of clusters:", min_value=2, max_value=6, value=3)

    # Features for clustering
    features = ['mean_gap', 'trend', 'volatility']
    X_cluster = df_cluster[features].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

    col1, col2 = st.columns([2, 1])

    with col1:
        # 3D scatter plot
        fig_3d = px.scatter_3d(
            df_cluster,
            x='mean_gap',
            y='trend',
            z='volatility',
            color='cluster',
            hover_name='country',
            labels={
                'mean_gap': 'Average Wage Gap (%)',
                'trend': 'Change 2020-2023 (pp)',
                'volatility': 'Volatility (Std Dev)'
            },
            title=f'Country Clusters (K={n_clusters})',
            color_continuous_scale='Viridis'
        )
        fig_3d.update_traces(marker=dict(size=8))
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("Cluster Profiles")

        cluster_profiles = df_cluster.groupby('cluster').agg({
            'mean_gap': 'mean',
            'trend': 'mean',
            'volatility': 'mean',
            'country': 'count'
        }).reset_index()
        cluster_profiles.columns = ['Cluster', 'Avg Gap', 'Avg Trend', 'Avg Volatility', 'N Countries']

        st.dataframe(cluster_profiles.style.format({
            'Avg Gap': '{:.1f}%',
            'Avg Trend': '{:.2f}pp',
            'Avg Volatility': '{:.2f}'
        }), hide_index=True)

    # Show countries by cluster
    st.subheader("Countries by Cluster")

    for cluster_id in sorted(df_cluster['cluster'].unique()):
        countries_in_cluster = df_cluster[df_cluster['cluster'] == cluster_id]['country'].tolist()
        profile = cluster_profiles[cluster_profiles['Cluster'] == cluster_id].iloc[0]

        with st.expander(f"**Cluster {cluster_id}** ({len(countries_in_cluster)} countries) - Avg Gap: {profile['Avg Gap']:.1f}%, Trend: {profile['Avg Trend']:.2f}pp"):
            st.write(", ".join(sorted(countries_in_cluster)))

# ============================================================================
# TAB 4: FORECASTING
# ============================================================================
with tab4:
    st.header("🔮 Wage Gap Forecasting")
    st.markdown("**Project future wage gaps using linear trend extrapolation**")

    # Country selector
    forecast_country = st.selectbox(
        "Select country for forecast:",
        options=sorted(df_panel['country'].unique())
    )

    # Get historical data
    country_data = df_panel[df_panel['country'] == forecast_country].sort_values('year')

    if len(country_data) >= 3:
        # Fit linear model
        X_hist = country_data['year'].values.reshape(-1, 1)
        y_hist = country_data['wage_gap'].values

        model_forecast = LinearRegression()
        model_forecast.fit(X_hist, y_hist)

        # Forecast 2024-2030
        future_years = np.array(range(2024, 2031)).reshape(-1, 1)
        forecast_values = model_forecast.predict(future_years)

        # Combine historical and forecast
        df_forecast = pd.DataFrame({
            'year': list(country_data['year']) + list(future_years.flatten()),
            'wage_gap': list(y_hist) + list(forecast_values),
            'type': ['Historical']*len(y_hist) + ['Forecast']*len(forecast_values)
        })

        col1, col2 = st.columns([2, 1])

        with col1:
            # Plot
            fig_forecast = px.line(
                df_forecast,
                x='year',
                y='wage_gap',
                color='type',
                markers=True,
                labels={'wage_gap': 'Wage Gap (%)', 'year': 'Year'},
                title=f'Wage Gap Forecast: {forecast_country}',
                color_discrete_map={'Historical': 'steelblue', 'Forecast': 'coral'}
            )
            fig_forecast.update_traces(line=dict(width=3), marker=dict(size=10))
            st.plotly_chart(fig_forecast, use_container_width=True)

        with col2:
            st.subheader("Forecast Summary")
            st.metric("2024 Forecast", f"{forecast_values[0]:.1f}%")
            st.metric("2030 Forecast", f"{forecast_values[-1]:.1f}%")
            st.metric("Annual Change", f"{model_forecast.coef_[0]:.2f}pp/year")

            r2 = model_forecast.score(X_hist, y_hist)
            st.metric("Model R²", f"{r2:.3f}")

            if forecast_values[-1] < 5:
                st.success(f"✅ Projected to reach <5% by 2030")
            elif forecast_values[-1] < 10:
                st.info(f"⚠️ Moderate gap expected in 2030")
            else:
                st.warning(f"❌ High gap persists in projections")

# ============================================================================
# TAB 5: STATISTICAL TESTS
# ============================================================================
with tab5:
    st.header("📋 Statistical Hypothesis Tests")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Regional Differences (ANOVA)")
        st.markdown("**H₀:** All regions have equal mean wage gaps")

        # One-way ANOVA
        regions = df_2023['region'].unique()
        groups = [df_2023[df_2023['region'] == r]['wage_gap_percent'].values for r in regions]

        f_stat, p_value = stats.f_oneway(*groups)

        st.write(f"**F-statistic:** {f_stat:.4f}")
        st.write(f"**p-value:** {p_value:.6f}")

        if p_value < 0.05:
            st.success(f"✅ **Reject H₀** (p < 0.05): Significant regional differences exist")
        else:
            st.info(f"⚠️ **Fail to reject H₀**: No significant regional differences")

        # Box plot by region
        fig_region = px.box(
            df_2023,
            x='region',
            y='wage_gap_percent',
            color='region',
            title='Wage Gap Distribution by Region (2023)',
            labels={'wage_gap_percent': 'Wage Gap (%)'}
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        st.subheader("📏 Normality Tests")
        st.markdown("**Test if wage gaps follow normal distribution**")

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(df_2023['wage_gap_percent'])

        st.write("**Shapiro-Wilk Test:**")
        st.write(f"- **Statistic:** {shapiro_stat:.4f}")
        st.write(f"- **p-value:** {shapiro_p:.6f}")

        if shapiro_p > 0.05:
            st.success("✅ Data is normally distributed (p > 0.05)")
        else:
            st.warning("⚠️ Data deviates from normality (p < 0.05)")

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(
            (df_2023['wage_gap_percent'] - df_2023['wage_gap_percent'].mean()) / df_2023['wage_gap_percent'].std(),
            'norm'
        )

        st.write("**Kolmogorov-Smirnov Test:**")
        st.write(f"- **Statistic:** {ks_stat:.4f}")
        st.write(f"- **p-value:** {ks_p:.6f}")

        # Histogram with normal curve
        fig_normal = go.Figure()

        # Histogram
        fig_normal.add_trace(go.Histogram(
            x=df_2023['wage_gap_percent'],
            name='Observed',
            nbinsx=15,
            histnorm='probability density',
            marker_color='steelblue',
            opacity=0.7
        ))

        # Normal curve
        x_range = np.linspace(
            df_2023['wage_gap_percent'].min(),
            df_2023['wage_gap_percent'].max(),
            100
        )
        normal_curve = stats.norm.pdf(
            x_range,
            df_2023['wage_gap_percent'].mean(),
            df_2023['wage_gap_percent'].std()
        )

        fig_normal.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            name='Normal Distribution',
            line=dict(color='red', width=3)
        ))

        fig_normal.update_layout(
            title='Distribution vs Normal Curve',
            xaxis_title='Wage Gap (%)',
            yaxis_title='Density',
            showlegend=True
        )

        st.plotly_chart(fig_normal, use_container_width=True)

    # Correlation tests
    st.subheader("🔗 Correlation Tests")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pearson Correlation: Wage Gap vs GDP per capita**")
        df_corr = df_2023.dropna(subset=['wage_gap_percent', 'gdp_billions', 'population'])
        df_corr['gdp_per_capita'] = (df_corr['gdp_billions'] * 1000000000) / df_corr['population']

        pearson_r, pearson_p = stats.pearsonr(df_corr['wage_gap_percent'], df_corr['gdp_per_capita'])

        st.write(f"- **Correlation (r):** {pearson_r:.4f}")
        st.write(f"- **p-value:** {pearson_p:.6f}")

        if abs(pearson_r) > 0.5 and pearson_p < 0.05:
            st.success(f"✅ Strong {'positive' if pearson_r > 0 else 'negative'} correlation")
        elif pearson_p < 0.05:
            st.info(f"⚠️ Weak but significant correlation")
        else:
            st.warning("❌ No significant correlation")

    with col2:
        st.write("**Spearman Rank Correlation: Wage Gap vs GDP**")

        spearman_r, spearman_p = stats.spearmanr(df_corr['wage_gap_percent'], df_corr['gdp_per_capita'])

        st.write(f"- **Correlation (ρ):** {spearman_r:.4f}")
        st.write(f"- **p-value:** {spearman_p:.6f}")

        if abs(spearman_r) > 0.5 and spearman_p < 0.05:
            st.success(f"✅ Strong monotonic relationship")
        elif spearman_p < 0.05:
            st.info(f"⚠️ Weak but significant relationship")
        else:
            st.warning("❌ No significant relationship")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Advanced Statistical Analysis for PhD Research</strong></p>
    <p>Panel regression • Convergence tests • Machine learning • Hypothesis testing</p>
</div>
""", unsafe_allow_html=True)
