"""
Time Series Analysis
Structural breaks, ARIMA models, unit roots, and forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="Time Series Analysis", page_icon="📉", layout="wide")

# ============================================================================
# PAGE HEADER
# ============================================================================
st.title("📉 Time Series Econometrics")
st.markdown("""
**Advanced time series methods for economic data**

Time series data has temporal dependence - observations are correlated over time.
This requires specialized methods beyond standard regression.
""")

# ============================================================================
# THEORY SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📚 Time Series Theory")

    topic = st.selectbox(
        "Select topic:",
        ["Stationarity", "Unit Roots", "ACF & PACF", "ARIMA Models",
         "Structural Breaks", "Forecasting"]
    )

    if topic == "Stationarity":
        st.markdown("""
        ### Stationarity

        **Definition:** Process is stationary if statistical properties don't change over time

        **Weak (covariance) stationarity:**
        1. E[Y_t] = μ (constant mean)
        2. Var(Y_t) = σ² (constant variance)
        3. Cov(Y_t, Y_t-k) depends only on k, not t

        **Why it matters:**
        - Non-stationary → spurious regression
        - Most econometric theory assumes stationarity
        - Need to test before modeling

        **Common violations:**
        - Trend (deterministic or stochastic)
        - Structural breaks
        - Heteroskedasticity
        """)

    elif topic == "Unit Roots":
        st.markdown("""
        ### Unit Root Tests

        **Null hypothesis:** Series has a unit root (non-stationary)

        **Test equation:**
        ```
        ΔY_t = α + βt + ρY_t-1 + Σγ_iΔY_t-i + ε_t
        ```

        **Tests:**
        1. **Augmented Dickey-Fuller (ADF)**
           - H₀: ρ = 0 (unit root)
           - H₁: ρ < 0 (stationary)

        2. **KPSS**
           - H₀: Stationary
           - H₁: Unit root
           - Complement to ADF!

        3. **Phillips-Perron**
           - Robust to heteroskedasticity

        **If unit root exists:**
        - Take first difference
        - Or model as I(1) process
        """)

    elif topic == "ACF & PACF":
        st.markdown("""
        ### Autocorrelation Functions

        **ACF (Autocorrelation Function):**
        ```
        ρ_k = Corr(Y_t, Y_t-k)
        ```
        Correlation at lag k

        **PACF (Partial Autocorrelation):**
        ```
        α_k = Corr(Y_t, Y_t-k | Y_t-1, ..., Y_t-k+1)
        ```
        Correlation after removing effect of intermediate lags

        **Patterns:**
        - **AR(p):** PACF cuts off at lag p, ACF decays
        - **MA(q):** ACF cuts off at lag q, PACF decays
        - **ARMA:** Both decay
        """)

    elif topic == "ARIMA Models":
        st.markdown("""
        ### ARIMA(p,d,q) Models

        **Components:**
        - **AR(p):** Autoregressive of order p
        - **I(d):** Integrated of order d
        - **MA(q):** Moving average of order q

        **ARIMA(p,d,q):**
        ```
        (1 - φ₁L - ... - φ_pL^p)(1-L)^d Y_t
          = (1 + θ₁L + ... + θ_qL^q)ε_t
        ```

        **Model selection:**
        1. Check ACF/PACF
        2. Try different (p,d,q)
        3. Compare AIC/BIC
        4. Check residuals (Ljung-Box test)

        **Common models:**
        - ARIMA(1,0,0) = AR(1)
        - ARIMA(0,1,1) = Random walk + drift
        - ARIMA(1,1,1) = ARIMA
        """)

    elif topic == "Structural Breaks":
        st.markdown("""
        ### Structural Break Tests

        **Problem:** Parameters change at some point t*

        **Chow Test:**
        - Known break point
        - Test equality of coefficients before/after

        **Quandt-Andrews:**
        - Unknown break point
        - Test all possible breaks

        **Bai-Perron:**
        - Multiple unknown breaks
        - Information criteria for # of breaks

        **CUSUM Test:**
        - Cumulative sum of recursive residuals
        - Detects parameter instability
        """)

    elif topic == "Forecasting":
        st.markdown("""
        ### Time Series Forecasting

        **1-step ahead:**
        ```
        Ŷ_t+1|t = E[Y_t+1 | Y_t, Y_t-1, ...]
        ```

        **h-step ahead:**
        ```
        Ŷ_t+h|t = E[Y_t+h | Y_t, Y_t-1, ...]
        ```

        **Forecast evaluation:**
        - **RMSE:** Root Mean Squared Error
        - **MAE:** Mean Absolute Error
        - **MAPE:** Mean Absolute Percentage Error

        **Out-of-sample validation:**
        1. Split: training vs test
        2. Fit on training
        3. Forecast on test
        4. Compare forecast to actual
        """)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_panel_data():
    """Load complete panel dataset"""
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

    return df_panel

df_panel = load_panel_data()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Unit Root Tests",
    "📊 ACF & PACF",
    "📈 ARIMA Modeling",
    "⚡ Structural Breaks",
    "🔮 Forecasting"
])

# ============================================================================
# TAB 1: UNIT ROOT TESTS
# ============================================================================
with tab1:
    st.header("🔍 Unit Root Tests")

    st.markdown("""
    ### Testing for Stationarity

    **Question:** Is the wage gap stationary or does it have a unit root?

    **Why it matters:** Non-stationary series can lead to spurious regression!
    """)

    # Select country for time series analysis
    country_ts = st.selectbox(
        "Select country for time series analysis:",
        options=sorted(df_panel['country'].unique()),
        index=sorted(df_panel['country'].unique()).tolist().index('France') if 'France' in df_panel['country'].unique() else 0
    )

    df_country_ts = df_panel[df_panel['country'] == country_ts].sort_values('year')
    ts_data = df_country_ts['wage_gap'].values

    st.write(f"**Time series length:** {len(ts_data)} observations (2020-2023)")

    if len(ts_data) < 10:
        st.warning("""
        ⚠️ **Short time series!**

        With only 4 years, unit root tests have low power.
        Results are illustrative only.

        **For your PhD:** Obtain at least 20-30 observations for reliable inference.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Augmented Dickey-Fuller Test")

        # ADF test
        adf_result = adfuller(ts_data, autolag='AIC')

        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_lags = adf_result[2]
        adf_crit_values = adf_result[4]

        st.write("**H₀:** Series has a unit root (non-stationary)")
        st.write("**H₁:** Series is stationary")

        st.metric("ADF Statistic", f"{adf_stat:.4f}")
        st.metric("p-value", f"{adf_pvalue:.4f}")
        st.metric("Lags used", f"{adf_lags}")

        st.write("**Critical values:**")
        for key, value in adf_crit_values.items():
            st.write(f"- {key}: {value:.4f}")

        if adf_pvalue < 0.05:
            st.success(f"""
            ✅ **Reject H₀** (p = {adf_pvalue:.4f})

            **Conclusion:** Series is **stationary** (no unit root)

            Safe to use in regression without differencing.
            """)
        else:
            st.warning(f"""
            ⚠️ **Fail to reject H₀** (p = {adf_pvalue:.4f})

            **Conclusion:** Cannot reject unit root.

            Consider first-differencing or check KPSS test.
            """)

    with col2:
        st.subheader("KPSS Test")

        # KPSS test
        kpss_result = kpss(ts_data, regression='c', nlags='auto')

        kpss_stat = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        kpss_lags = kpss_result[2]
        kpss_crit_values = kpss_result[3]

        st.write("**H₀:** Series is stationary")
        st.write("**H₁:** Series has a unit root")

        st.metric("KPSS Statistic", f"{kpss_stat:.4f}")
        st.metric("p-value", f"{kpss_pvalue:.4f}")
        st.metric("Lags used", f"{kpss_lags}")

        st.write("**Critical values:**")
        for key, value in kpss_crit_values.items():
            st.write(f"- {key}: {value:.4f}")

        if kpss_pvalue > 0.05:
            st.success(f"""
            ✅ **Fail to reject H₀** (p > 0.05)

            **Conclusion:** Series is **stationary**

            Consistent with ADF (if both agree).
            """)
        else:
            st.warning(f"""
            ⚠️ **Reject H₀** (p ≤ 0.05)

            **Conclusion:** Evidence of unit root.

            Consider differencing.
            """)

    # Decision matrix
    st.markdown("---")
    st.subheader("📊 Decision Matrix")

    if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
        st.success("""
        ✅ **Both tests agree: Series is STATIONARY**

        - ADF rejects H₀ (no unit root)
        - KPSS fails to reject H₀ (stationary)

        **Action:** Use levels (no differencing needed)
        """)
    elif adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
        st.error("""
        ❌ **Both tests agree: Series has UNIT ROOT**

        - ADF fails to reject H₀ (unit root)
        - KPSS rejects H₀ (not stationary)

        **Action:** Take first difference: ΔY_t = Y_t - Y_t-1
        """)
    else:
        st.info("""
        ⚠️ **Tests disagree**

        - Inconclusive results
        - May indicate trend-stationary process
        - Need more data or visual inspection
        """)

    # Plot time series
    st.markdown("---")
    st.subheader("Time Series Plot")

    fig_ts = go.Figure()

    fig_ts.add_trace(go.Scatter(
        x=df_country_ts['year'],
        y=df_country_ts['wage_gap'],
        mode='lines+markers',
        name='Wage Gap',
        line=dict(color='steelblue', width=2),
        marker=dict(size=10)
    ))

    fig_ts.update_layout(
        title=f'Wage Gap Time Series: {country_ts}',
        xaxis_title='Year',
        yaxis_title='Wage Gap (%)',
        hovermode='x unified'
    )

    st.plotly_chart(fig_ts, use_container_width=True)

# ============================================================================
# TAB 2: ACF & PACF
# ============================================================================
with tab2:
    st.header("📊 Autocorrelation Analysis")

    st.markdown("""
    ### ACF and PACF Plots

    **Use:** Identify appropriate ARIMA(p, d, q) model

    **Rules of thumb:**
    - **AR(p):** PACF cuts off at lag p
    - **MA(q):** ACF cuts off at lag q
    - **ARMA:** Both decay gradually
    """)

    # Compute ACF and PACF
    max_lags = min(10, len(ts_data) // 2 - 1)

    acf_values = acf(ts_data, nlags=max_lags)
    pacf_values = pacf(ts_data, nlags=max_lags)

    # Confidence intervals (95%)
    conf_interval = 1.96 / np.sqrt(len(ts_data))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ACF (Autocorrelation Function)")

        fig_acf = go.Figure()

        # Bars
        fig_acf.add_trace(go.Bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            marker_color='steelblue',
            name='ACF'
        ))

        # Confidence bands
        fig_acf.add_hline(y=conf_interval, line_dash="dash", line_color="red",
                         annotation_text=f"95% CI")
        fig_acf.add_hline(y=-conf_interval, line_dash="dash", line_color="red")

        fig_acf.update_layout(
            title='Autocorrelation Function',
            xaxis_title='Lag',
            yaxis_title='ACF',
            showlegend=False
        )

        st.plotly_chart(fig_acf, use_container_width=True)

        # Ljung-Box test
        lb_test = acorr_ljungbox(ts_data, lags=min(10, len(ts_data) - 1), return_df=True)

        st.write("**Ljung-Box Test (H₀: No autocorrelation):**")
        st.write(f"- Lag 1 p-value: {lb_test['lb_pvalue'].iloc[0]:.4f}")

        if lb_test['lb_pvalue'].iloc[0] < 0.05:
            st.success("✅ Significant autocorrelation detected (p < 0.05)")
        else:
            st.info("ℹ️ No significant autocorrelation (p ≥ 0.05)")

    with col2:
        st.subheader("PACF (Partial Autocorrelation)")

        fig_pacf = go.Figure()

        # Bars
        fig_pacf.add_trace(go.Bar(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            marker_color='coral',
            name='PACF'
        ))

        # Confidence bands
        fig_pacf.add_hline(y=conf_interval, line_dash="dash", line_color="red",
                          annotation_text=f"95% CI")
        fig_pacf.add_hline(y=-conf_interval, line_dash="dash", line_color="red")

        fig_pacf.update_layout(
            title='Partial Autocorrelation Function',
            xaxis_title='Lag',
            yaxis_title='PACF',
            showlegend=False
        )

        st.plotly_chart(fig_pacf, use_container_width=True)

    # Model suggestions
    st.markdown("---")
    st.subheader("📋 Model Suggestions")

    # Count significant lags
    acf_significant = np.sum(np.abs(acf_values[1:]) > conf_interval)
    pacf_significant = np.sum(np.abs(pacf_values[1:]) > conf_interval)

    if pacf_significant > 0 and acf_significant == 0:
        st.info(f"""
        **Pattern suggests AR({pacf_significant}) model**

        - PACF cuts off at lag {pacf_significant}
        - ACF decays

        Try ARIMA({pacf_significant}, 0, 0)
        """)
    elif acf_significant > 0 and pacf_significant == 0:
        st.info(f"""
        **Pattern suggests MA({acf_significant}) model**

        - ACF cuts off at lag {acf_significant}
        - PACF decays

        Try ARIMA(0, 0, {acf_significant})
        """)
    elif acf_significant > 0 and pacf_significant > 0:
        st.info(f"""
        **Pattern suggests ARMA model**

        Both ACF and PACF decay gradually.

        Try ARIMA({min(pacf_significant, 2)}, 0, {min(acf_significant, 2)})
        """)
    else:
        st.info("""
        **No clear pattern**

        - May be white noise
        - Or insufficient data
        - Try ARIMA(1,0,0) as baseline
        """)

# ============================================================================
# TAB 3: ARIMA MODELING
# ============================================================================
with tab3:
    st.header("📈 ARIMA Model Estimation")

    st.markdown("""
    ### Autoregressive Integrated Moving Average

    **ARIMA(p, d, q):**
    - p = AR order
    - d = differencing order
    - q = MA order
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Selection")

        p_order = st.slider("AR order (p):", min_value=0, max_value=3, value=1)
        d_order = st.slider("Differencing (d):", min_value=0, max_value=2, value=0)
        q_order = st.slider("MA order (q):", min_value=0, max_value=3, value=0)

        st.write(f"**Model:** ARIMA({p_order}, {d_order}, {q_order})")

    with col2:
        st.subheader("Estimation Results")

        try:
            # Fit ARIMA model
            model_arima = ARIMA(ts_data, order=(p_order, d_order, q_order))
            results_arima = model_arima.fit()

            st.write("**Model Summary:**")

            # Extract key statistics
            st.write(f"- **AIC:** {results_arima.aic:.2f}")
            st.write(f"- **BIC:** {results_arima.bic:.2f}")
            st.write(f"- **Log-Likelihood:** {results_arima.llf:.2f}")

            # Parameters
            st.write("**Parameters:**")
            params_df = pd.DataFrame({
                'Parameter': results_arima.params.index,
                'Coefficient': results_arima.params.values,
                'Std Error': results_arima.bse.values,
                'p-value': results_arima.pvalues.values
            })

            st.dataframe(params_df.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                'p-value': '{:.4f}'
            }), hide_index=True)

        except Exception as e:
            st.error(f"Error fitting ARIMA model: {str(e)}")
            st.info("Try different (p,d,q) values or check if series has enough variation.")

    # Residual diagnostics
    if 'results_arima' in locals():
        st.markdown("---")
        st.subheader("📊 Residual Diagnostics")

        residuals_arima = results_arima.resid

        col1, col2 = st.columns(2)

        with col1:
            # Residual plot
            fig_resid = go.Figure()

            fig_resid.add_trace(go.Scatter(
                x=df_country_ts['year'],
                y=residuals_arima,
                mode='lines+markers',
                name='Residuals',
                line=dict(color='gray'),
                marker=dict(size=8)
            ))

            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")

            fig_resid.update_layout(
                title='Residuals Over Time',
                xaxis_title='Year',
                yaxis_title='Residual'
            )

            st.plotly_chart(fig_resid, use_container_width=True)

        with col2:
            # Histogram of residuals
            fig_hist_resid = go.Figure()

            fig_hist_resid.add_trace(go.Histogram(
                x=residuals_arima,
                nbinsx=10,
                marker_color='steelblue',
                name='Residuals'
            ))

            # Normal curve overlay
            x_range_resid = np.linspace(residuals_arima.min(), residuals_arima.max(), 100)
            normal_resid = stats.norm.pdf(x_range_resid, residuals_arima.mean(), residuals_arima.std())
            normal_resid_scaled = normal_resid * len(residuals_arima) * (residuals_arima.max() - residuals_arima.min()) / 10

            fig_hist_resid.add_trace(go.Scatter(
                x=x_range_resid,
                y=normal_resid_scaled,
                mode='lines',
                name='Normal',
                line=dict(color='red', width=2)
            ))

            fig_hist_resid.update_layout(
                title='Residual Distribution',
                xaxis_title='Residual',
                yaxis_title='Frequency'
            )

            st.plotly_chart(fig_hist_resid, use_container_width=True)

        # Ljung-Box test on residuals
        lb_resid = acorr_ljungbox(residuals_arima, lags=min(5, len(residuals_arima) - 1), return_df=True)

        st.write("**Ljung-Box Test on Residuals:**")
        st.write("H₀: Residuals are white noise (no autocorrelation)")

        if lb_resid['lb_pvalue'].iloc[-1] > 0.05:
            st.success(f"""
            ✅ **Model is adequate** (p = {lb_resid['lb_pvalue'].iloc[-1]:.4f})

            Residuals show no significant autocorrelation.
            """)
        else:
            st.warning(f"""
            ⚠️ **Model may be inadequate** (p = {lb_resid['lb_pvalue'].iloc[-1]:.4f})

            Residuals still show autocorrelation. Try increasing p or q.
            """)

    # Model comparison
    st.markdown("---")
    st.subheader("🔍 Model Comparison")

    st.markdown("""
    **Try multiple models and compare AIC/BIC:**

    - Lower AIC/BIC = better fit (adjusted for complexity)
    - AIC penalizes complexity less than BIC
    - BIC preferred for small samples
    """)

    # Fit several candidate models
    candidate_models = [
        (1, 0, 0),
        (2, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 1)
    ]

    model_comparison = []

    for (p, d, q) in candidate_models:
        try:
            model_temp = ARIMA(ts_data, order=(p, d, q))
            results_temp = model_temp.fit()

            model_comparison.append({
                'Model': f'ARIMA({p},{d},{q})',
                'AIC': results_temp.aic,
                'BIC': results_temp.bic,
                'Log-Likelihood': results_temp.llf
            })
        except:
            pass

    if model_comparison:
        df_model_comp = pd.DataFrame(model_comparison).sort_values('AIC')

        st.dataframe(df_model_comp.style.format({
            'AIC': '{:.2f}',
            'BIC': '{:.2f}',
            'Log-Likelihood': '{:.2f}'
        }).background_gradient(subset=['AIC', 'BIC'], cmap='RdYlGn_r'), hide_index=True)

        best_model = df_model_comp.iloc[0]['Model']
        st.success(f"""
        ✅ **Best model by AIC:** {best_model}

        This model balances fit and complexity.
        """)

# ============================================================================
# TAB 4: STRUCTURAL BREAKS
# ============================================================================
with tab4:
    st.header("⚡ Structural Break Analysis")

    st.markdown("""
    ### Testing for Parameter Instability

    **Question:** Did the relationship change over time?

    **Examples:**
    - Policy changes (e.g., EU directive)
    - Economic shocks (e.g., COVID-19)
    - Institutional changes
    """)

    st.info("""
    **Note:** With only 4 observations, formal structural break tests have no power.

    **Illustration:** Chow test with hypothetical break in 2021
    """)

    # Hypothetical Chow test
    if len(ts_data) >= 4:
        # Split at 2021
        split_year = 2021
        split_idx = df_country_ts[df_country_ts['year'] == split_year].index[0] - df_country_ts.index[0] if split_year in df_country_ts['year'].values else 2

        ts_before = ts_data[:split_idx]
        ts_after = ts_data[split_idx:]

        st.subheader(f"Chow Test: Break at {split_year}")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Period 1:** {df_country_ts['year'].iloc[0]} - {df_country_ts['year'].iloc[split_idx-1]}")
            st.write(f"- Observations: {len(ts_before)}")
            st.write(f"- Mean: {np.mean(ts_before):.2f}")
            st.write(f"- Std Dev: {np.std(ts_before):.2f}")

        with col2:
            st.write(f"**Period 2:** {df_country_ts['year'].iloc[split_idx]} - {df_country_ts['year'].iloc[-1]}")
            st.write(f"- Observations: {len(ts_after)}")
            st.write(f"- Mean: {np.mean(ts_after):.2f}")
            st.write(f"- Std Dev: {np.std(ts_after):.2f}")

        # Simple t-test for mean difference
        if len(ts_before) >= 2 and len(ts_after) >= 2:
            t_stat, p_val_chow = stats.ttest_ind(ts_before, ts_after)

            st.metric("t-statistic", f"{t_stat:.4f}")
            st.metric("p-value", f"{p_val_chow:.4f}")

            if p_val_chow < 0.05:
                st.warning(f"""
                ⚠️ **Possible structural break** (p = {p_val_chow:.4f})

                Mean wage gap differs significantly between periods.
                """)
            else:
                st.success(f"""
                ✅ **No evidence of break** (p = {p_val_chow:.4f})

                Mean wage gap similar across periods.
                """)

        # Visualization
        fig_break = go.Figure()

        fig_break.add_trace(go.Scatter(
            x=df_country_ts['year'],
            y=df_country_ts['wage_gap'],
            mode='lines+markers',
            name='Wage Gap',
            line=dict(color='steelblue', width=2),
            marker=dict(size=10)
        ))

        fig_break.add_vline(x=split_year, line_dash="dash", line_color="red",
                           annotation_text="Hypothetical Break")

        fig_break.update_layout(
            title=f'Structural Break Analysis: {country_ts}',
            xaxis_title='Year',
            yaxis_title='Wage Gap (%)'
        )

        st.plotly_chart(fig_break, use_container_width=True)

    st.markdown("---")
    st.subheader("📚 Advanced Break Tests (Require More Data)")

    break_tests = pd.DataFrame({
        'Test': ['Chow Test', 'Quandt-Andrews', 'Bai-Perron', 'CUSUM', 'CUSUM-SQ'],
        'Break Point': ['Known', 'Unknown (1)', 'Unknown (Multiple)', 'Any', 'Any'],
        'Min Obs Required': ['~20', '~30', '~50', '~30', '~30'],
        'Python Package': ['statsmodels', 'statsmodels', 'ruptures', 'statsmodels', 'statsmodels']
    })

    st.table(break_tests)

    st.info("""
    **For your PhD:**

    1. Obtain longer time series (15-30 years)
    2. Use Bai-Perron to detect unknown multiple breaks
    3. Test robustness with CUSUM
    4. Relate breaks to historical events/policies
    """)

# ============================================================================
# TAB 5: FORECASTING
# ============================================================================
with tab5:
    st.header("🔮 Time Series Forecasting")

    st.markdown("""
    ### Out-of-Sample Prediction

    **Goal:** Predict future wage gaps

    **Method:** Use ARIMA model fitted on historical data
    """)

    # Select ARIMA model for forecasting
    st.subheader("Forecast Settings")

    col1, col2 = st.columns(2)

    with col1:
        p_forecast = st.number_input("AR order (p):", min_value=0, max_value=3, value=1, key='p_forecast')
        d_forecast = st.number_input("Differencing (d):", min_value=0, max_value=2, value=0, key='d_forecast')
        q_forecast = st.number_input("MA order (q):", min_value=0, max_value=3, value=0, key='q_forecast')

    with col2:
        n_periods = st.slider("Forecast horizon (years):", min_value=1, max_value=10, value=5)

    # Fit model and forecast
    try:
        model_forecast = ARIMA(ts_data, order=(p_forecast, d_forecast, q_forecast))
        results_forecast = model_forecast.fit()

        # Forecast
        forecast = results_forecast.forecast(steps=n_periods)
        forecast_ci = results_forecast.get_forecast(steps=n_periods).conf_int()

        # Create forecast dataframe
        last_year = int(df_country_ts['year'].iloc[-1])
        forecast_years = list(range(last_year + 1, last_year + n_periods + 1))

        df_forecast = pd.DataFrame({
            'year': forecast_years,
            'forecast': forecast,
            'lower': forecast_ci.iloc[:, 0],
            'upper': forecast_ci.iloc[:, 1]
        })

        # Plot
        fig_forecast = go.Figure()

        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=df_country_ts['year'],
            y=df_country_ts['wage_gap'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='steelblue', width=2),
            marker=dict(size=10)
        ))

        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=df_forecast['year'],
            y=df_forecast['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='coral', width=2, dash='dash'),
            marker=dict(size=10)
        ))

        # Confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=list(df_forecast['year']) + list(df_forecast['year'][::-1]),
            y=list(df_forecast['upper']) + list(df_forecast['lower'][::-1]),
            fill='toself',
            fillcolor='rgba(255,127,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% CI'
        ))

        fig_forecast.update_layout(
            title=f'Wage Gap Forecast: {country_ts} (ARIMA{(p_forecast, d_forecast, q_forecast)})',
            xaxis_title='Year',
            yaxis_title='Wage Gap (%)',
            hovermode='x unified'
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Values")

        st.dataframe(df_forecast.style.format({
            'forecast': '{:.2f}%',
            'lower': '{:.2f}%',
            'upper': '{:.2f}%'
        }), hide_index=True)

        # Interpretation
        final_forecast = df_forecast['forecast'].iloc[-1]
        current_value = ts_data[-1]
        change = final_forecast - current_value

        st.markdown("---")
        st.subheader("📊 Forecast Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current (2023)", f"{current_value:.2f}%")

        with col2:
            st.metric(f"Forecast ({forecast_years[-1]})", f"{final_forecast:.2f}%",
                     delta=f"{change:.2f}pp")

        with col3:
            if change < -1:
                st.success("📉 Decreasing trend")
            elif change > 1:
                st.warning("📈 Increasing trend")
            else:
                st.info("→ Stable trend")

    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        st.info("Try different ARIMA parameters or check data quality.")

    st.markdown("---")
    st.subheader("⚠️ Forecast Limitations")

    st.warning("""
    **With only 4 historical observations:**
    - Forecasts have **very wide confidence intervals**
    - Limited ability to capture trends
    - Sensitive to outliers
    - Should NOT be used for policy decisions

    **For reliable forecasting:**
    - Need at least 20-30 observations
    - Consider exogenous variables (ARIMAX)
    - Compare multiple models
    - Validate on hold-out sample
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Time Series Econometrics for PhD Research</strong></p>
    <p>Unit Roots • ACF/PACF • ARIMA • Structural Breaks • Forecasting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📚 Essential References:

1. **Hamilton (1994)** - "Time Series Analysis" - THE comprehensive textbook
2. **Enders (2014)** - "Applied Econometric Time Series" (4th ed) - Very accessible
3. **Tsay (2010)** - "Analysis of Financial Time Series" (3rd ed)
4. **Box, Jenkins & Reinsel (2015)** - "Time Series Analysis: Forecasting and Control"
5. **Lütkepohl (2005)** - "New Introduction to Multiple Time Series Analysis"
6. **Stock & Watson (2019)** - "Introduction to Econometrics" (4th ed) - Ch. 14-15

### 🐍 Python Packages:

- **statsmodels.tsa** - ARIMA, SARIMA, VAR, state space models
- **pmdarima** - Auto-ARIMA (automatic model selection)
- **prophet** - Facebook's forecasting tool (additive models)
- **ruptures** - Structural break detection
- **arch** - GARCH models, unit root tests
""")
