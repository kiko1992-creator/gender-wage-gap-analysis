"""
Advanced Panel Data Econometrics
Dynamic panels, GMM estimation, and panel diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS
from linearmodels.iv import IV2SLS
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="Panel Econometrics", page_icon="📈", layout="wide")

# ============================================================================
# PAGE HEADER
# ============================================================================
st.title("📈 Advanced Panel Data Econometrics")
st.markdown("""
**State-of-the-art panel data methods for PhD research**

Panel data combines cross-sectional (countries) and time-series dimensions.
This allows controlling for unobserved heterogeneity and analyzing dynamics.
""")

# ============================================================================
# THEORY SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📚 Panel Data Theory")

    method = st.selectbox(
        "Select topic:",
        ["Overview", "Fixed vs Random Effects", "Dynamic Panels",
         "GMM Estimation", "Hausman Test", "Panel Unit Roots"]
    )

    if method == "Overview":
        st.markdown("""
        ### Panel Data Structure

        **Data:** Y_it where i = country, t = time

        **Advantages:**
        1. Control for unobserved heterogeneity
        2. More degrees of freedom
        3. Study dynamics (lagged effects)
        4. Reduce collinearity

        **Model:**
        ```
        Y_it = α_i + X_it'β + ε_it
        ```

        α_i = individual fixed effects
        """)

    elif method == "Fixed vs Random Effects":
        st.markdown("""
        ### Fixed Effects (FE)

        **Assumes:** α_i correlated with X_it
        **Estimation:** Within transformation
        **Removes:** Time-invariant variables

        ### Random Effects (RE)

        **Assumes:** α_i uncorrelated with X_it
        **Estimation:** GLS
        **Allows:** Time-invariant variables

        ### Hausman Test

        **H₀:** RE is consistent
        **H₁:** Only FE is consistent

        If p < 0.05: Use FE
        """)

    elif method == "Dynamic Panels":
        st.markdown("""
        ### Dynamic Panel Model

        ```
        Y_it = γ·Y_i,t-1 + X_it'β + α_i + ε_it
        ```

        **Problem:** Y_i,t-1 correlated with α_i
        → OLS/FE biased!

        **Solution:** Arellano-Bond GMM
        - Use Y_i,t-2 as instrument for ΔY_i,t-1
        - Difference out fixed effects
        - Moment conditions for identification

        **Tests:**
        - AR(2) test: No 2nd-order autocorrelation
        - Sargan: Overidentification
        """)

    elif method == "GMM Estimation":
        st.markdown("""
        ### Generalized Method of Moments

        **Idea:** Choose β̂ to make sample moments
        close to population moments

        **Moment conditions:**
        ```
        E[Z_i'ε_i] = 0
        ```

        Z = instruments

        **Difference GMM (Arellano-Bond):**
        - First-difference to remove α_i
        - Use lags as instruments

        **System GMM (Blundell-Bond):**
        - Combines levels and differences
        - More efficient with persistent data
        """)

    elif method == "Hausman Test":
        st.markdown("""
        ### Hausman Specification Test

        **Tests:** Consistency of RE vs FE

        **Statistic:**
        ```
        H = (β̂_FE - β̂_RE)'[Var(β̂_FE) - Var(β̂_RE)]⁻¹(β̂_FE - β̂_RE)
        ```

        **Distribution:** χ² with K degrees of freedom

        **Decision:**
        - p < 0.05: Use FE (RE inconsistent)
        - p ≥ 0.05: Use RE (more efficient)
        """)

    elif method == "Panel Unit Roots":
        st.markdown("""
        ### Panel Unit Root Tests

        **Question:** Is Y_it non-stationary?

        **Tests:**
        1. **Levin-Lin-Chu:** Assumes common ρ
        2. **Im-Pesaran-Shin:** Heterogeneous ρ
        3. **Fisher-ADF:** Combines p-values

        **H₀:** Unit root (non-stationary)
        **H₁:** Stationary

        **Why it matters:**
        - Spurious regression if non-stationary
        - Need cointegration or first-difference
        """)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_panel_data():
    """Load complete panel dataset"""
    df_2023 = get_all_countries_2023()

    # Ensure df_2023 is a DataFrame
    if not isinstance(df_2023, pd.DataFrame):
        st.error("Unable to load country data. Please check database connection.")
        return pd.DataFrame()

    if df_2023.empty or 'country_name' not in df_2023.columns:
        st.warning("No country data available.")
        return pd.DataFrame()

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

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔄 FE vs RE Comparison",
    "⚡ Hausman Test",
    "🚀 Dynamic Panels",
    "🎯 GMM Estimation",
    "📊 Panel Diagnostics"
])

# ============================================================================
# TAB 1: FIXED EFFECTS VS RANDOM EFFECTS
# ============================================================================
with tab1:
    st.header("🔄 Fixed Effects vs Random Effects")

    st.markdown("""
    ### Model Specification

    **Equation:** `wage_gap_it = β₀ + β₁·log(GDP_pc)_it + β₂·year_t + α_i + ε_it`

    - **Fixed Effects:** Treats α_i as country-specific constants (dummy variables)
    - **Random Effects:** Treats α_i as random draws from a distribution
    """)

    # Prepare panel data for estimation
    df_est = df_panel.dropna(subset=['wage_gap', 'gdp_per_capita']).copy()
    df_est['log_gdp_pc'] = np.log(df_est['gdp_per_capita'])
    df_est['year_numeric'] = df_est['year'].astype(int)

    # Create multi-index for linearmodels
    df_est = df_est.set_index(['country', 'year'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Fixed Effects (Within) Estimator")

        try:
            # Fixed Effects model using linearmodels
            exog = sm.add_constant(df_est[['log_gdp_pc', 'year_numeric']])
            mod_fe = PanelOLS(df_est['wage_gap'], exog, entity_effects=True)
            res_fe = mod_fe.fit(cov_type='clustered', cluster_entity=True)

            st.write("**Fixed Effects Results:**")

            # Extract results
            fe_summary = pd.DataFrame({
                'Variable': ['const', 'log(GDP pc)', 'year'],
                'Coefficient': [res_fe.params['const'], res_fe.params['log_gdp_pc'], res_fe.params['year_numeric']],
                'Std Error': [res_fe.std_errors['const'], res_fe.std_errors['log_gdp_pc'], res_fe.std_errors['year_numeric']],
                't-statistic': [res_fe.tstats['const'], res_fe.tstats['log_gdp_pc'], res_fe.tstats['year_numeric']],
                'p-value': [res_fe.pvalues['const'], res_fe.pvalues['log_gdp_pc'], res_fe.pvalues['year_numeric']]
            })

            st.dataframe(fe_summary.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                't-statistic': '{:.4f}',
                'p-value': '{:.4f}'
            }), hide_index=True)

            st.metric("R² (within)", f"{res_fe.rsquared:.4f}")
            st.metric("R² (overall)", f"{res_fe.rsquared_overall:.4f}")
            st.metric("F-statistic", f"{res_fe.f_statistic.stat:.2f}")

            st.info("""
            **Interpretation:**
            - Within R²: Explains variation within countries over time
            - Standard errors are clustered by country
            - F-test tests joint significance
            """)

        except Exception as e:
            st.error(f"Error in FE estimation: {str(e)}")
            st.info("Using simpler manual FE estimation...")

            # Manual FE via demeaning
            df_manual = df_panel.dropna(subset=['wage_gap', 'gdp_per_capita']).copy()
            df_manual['log_gdp_pc'] = np.log(df_manual['gdp_per_capita'])
            df_manual['year_numeric'] = df_manual['year'].astype(int)

            for var in ['wage_gap', 'log_gdp_pc', 'year_numeric']:
                country_means = df_manual.groupby('country')[var].transform('mean')
                df_manual[f'{var}_demeaned'] = df_manual[var] - country_means

            X_fe = sm.add_constant(df_manual[['log_gdp_pc_demeaned', 'year_numeric_demeaned']])
            y_fe = df_manual['wage_gap_demeaned']

            model_fe_manual = sm.OLS(y_fe, X_fe).fit()

            st.write("**Fixed Effects Results (Manual):**")
            fe_manual_summary = pd.DataFrame({
                'Variable': ['const', 'log(GDP pc)', 'year'],
                'Coefficient': model_fe_manual.params.values,
                'Std Error': model_fe_manual.bse.values,
                'p-value': model_fe_manual.pvalues.values
            })

            st.dataframe(fe_manual_summary.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                'p-value': '{:.4f}'
            }), hide_index=True)

            st.metric("R²", f"{model_fe_manual.rsquared:.4f}")

    with col2:
        st.subheader("🔹 Random Effects (GLS) Estimator")

        try:
            # Random Effects model
            mod_re = RandomEffects(df_est['wage_gap'], exog)
            res_re = mod_re.fit()

            st.write("**Random Effects Results:**")

            re_summary = pd.DataFrame({
                'Variable': ['const', 'log(GDP pc)', 'year'],
                'Coefficient': [res_re.params['const'], res_re.params['log_gdp_pc'], res_re.params['year_numeric']],
                'Std Error': [res_re.std_errors['const'], res_re.std_errors['log_gdp_pc'], res_re.std_errors['year_numeric']],
                't-statistic': [res_re.tstats['const'], res_re.tstats['log_gdp_pc'], res_re.tstats['year_numeric']],
                'p-value': [res_re.pvalues['const'], res_re.pvalues['log_gdp_pc'], res_re.pvalues['year_numeric']]
            })

            st.dataframe(re_summary.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                't-statistic': '{:.4f}',
                'p-value': '{:.4f}'
            }), hide_index=True)

            st.metric("R² (overall)", f"{res_re.rsquared_overall:.4f}")
            st.metric("Theta (RE weight)", f"{res_re.theta:.4f}")

            st.info("""
            **Theta interpretation:**
            - θ = 0: Pooled OLS
            - θ = 1: Fixed Effects
            - 0 < θ < 1: Compromise between pooled and FE
            """)

        except Exception as e:
            st.error(f"Error in RE estimation: {str(e)}")
            st.info("Random Effects requires more time periods or entities for reliable estimation.")

    # Comparison table
    st.markdown("---")
    st.subheader("📊 Model Comparison")

    try:
        comparison = pd.DataFrame({
            'Coefficient': ['const', 'log(GDP pc)', 'year'],
            'Fixed Effects': [res_fe.params['const'], res_fe.params['log_gdp_pc'], res_fe.params['year_numeric']],
            'Random Effects': [res_re.params['const'], res_re.params['log_gdp_pc'], res_re.params['year_numeric']],
            'Difference': [
                res_fe.params['const'] - res_re.params['const'],
                res_fe.params['log_gdp_pc'] - res_re.params['log_gdp_pc'],
                res_fe.params['year_numeric'] - res_re.params['year_numeric']
            ]
        })

        st.dataframe(comparison.style.format({
            'Fixed Effects': '{:.4f}',
            'Random Effects': '{:.4f}',
            'Difference': '{:.4f}'
        }), hide_index=True)

        st.markdown("""
        **If coefficients differ substantially:**
        - Suggests correlation between α_i and X_it
        - Hausman test will likely reject H₀
        - Use Fixed Effects
        """)
    except:
        st.info("Model comparison requires both FE and RE to be estimated successfully.")

# ============================================================================
# TAB 2: HAUSMAN TEST
# ============================================================================
with tab2:
    st.header("⚡ Hausman Specification Test")

    st.markdown("""
    ### Testing Fixed vs Random Effects

    **Null Hypothesis (H₀):** Random Effects is consistent (preferred for efficiency)

    **Alternative (H₁):** Only Fixed Effects is consistent

    **Test Statistic:**
    ```
    H = (β̂_FE - β̂_RE)'[Var(β̂_FE) - Var(β̂_RE)]⁻¹(β̂_FE - β̂_RE) ~ χ²(K)
    ```

    **Decision Rule:**
    - p < 0.05: Reject H₀ → Use **Fixed Effects**
    - p ≥ 0.05: Fail to reject → Use **Random Effects** (more efficient)
    """)

    try:
        # Compute Hausman test manually
        # H = (b_fe - b_re)' * [Var(b_fe) - Var(b_re)]^(-1) * (b_fe - b_re)

        # Get coefficients (excluding constant for comparison)
        b_fe = np.array([res_fe.params['log_gdp_pc'], res_fe.params['year_numeric']])
        b_re = np.array([res_re.params['log_gdp_pc'], res_re.params['year_numeric']])

        # Get covariance matrices
        cov_fe = res_fe.cov[['log_gdp_pc', 'year_numeric']].loc[['log_gdp_pc', 'year_numeric']].values
        cov_re = res_re.cov[['log_gdp_pc', 'year_numeric']].loc[['log_gdp_pc', 'year_numeric']].values

        # Difference in coefficients
        diff = b_fe - b_re

        # Difference in covariance matrices
        cov_diff = cov_fe - cov_re

        # Hausman statistic
        try:
            H_stat = diff.T @ np.linalg.inv(cov_diff) @ diff
            df_hausman = len(diff)
            p_value_hausman = 1 - stats.chi2.cdf(H_stat, df_hausman)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Hausman Statistic", f"{H_stat:.4f}")

            with col2:
                st.metric("Degrees of Freedom", f"{df_hausman}")

            with col3:
                st.metric("p-value", f"{p_value_hausman:.4f}")

            st.markdown("---")

            if p_value_hausman < 0.05:
                st.error(f"""
                ❌ **Reject H₀ (p = {p_value_hausman:.4f} < 0.05)**

                **Conclusion:** Random Effects is **inconsistent**. Use **Fixed Effects**.

                **Reason:** Unobserved country effects (α_i) are correlated with regressors.
                Examples: culture, institutions, historical factors.
                """)
            else:
                st.success(f"""
                ✅ **Fail to Reject H₀ (p = {p_value_hausman:.4f} ≥ 0.05)**

                **Conclusion:** Random Effects is consistent and **more efficient**. Use **Random Effects**.

                **Reason:** No evidence of correlation between α_i and X_it.
                RE uses both within and between variation → lower standard errors.
                """)

            # Show coefficient differences
            st.subheader("Coefficient Differences")

            diff_df = pd.DataFrame({
                'Variable': ['log(GDP pc)', 'year'],
                'FE Estimate': b_fe,
                'RE Estimate': b_re,
                'Difference': diff,
                'Abs Difference': np.abs(diff)
            })

            st.dataframe(diff_df.style.format({
                'FE Estimate': '{:.4f}',
                'RE Estimate': '{:.4f}',
                'Difference': '{:.4f}',
                'Abs Difference': '{:.4f}'
            }).background_gradient(subset=['Abs Difference'], cmap='Reds'), hide_index=True)

        except np.linalg.LinAlgError:
            st.error("""
            ⚠️ **Hausman test computation failed** (singular covariance matrix difference)

            This can happen with:
            - Small sample size
            - High collinearity
            - Too few time periods

            **Recommendation:** Use Fixed Effects (conservative choice for panel data)
            """)

    except Exception as e:
        st.error(f"Error computing Hausman test: {str(e)}")
        st.info("""
        **Alternative:** Examine coefficient differences manually.
        If FE and RE give very different results → use FE.
        """)

    st.markdown("---")
    st.subheader("📚 Additional Specification Tests")

    st.markdown("""
    ### Other Panel Data Tests:

    1. **F-test for Fixed Effects**
       - H₀: α₁ = α₂ = ... = α_N (pooled OLS is adequate)
       - H₁: Individual effects exist
       - If p < 0.05: Need FE or RE

    2. **Breusch-Pagan LM Test**
       - H₀: Var(α_i) = 0 (pooled OLS is adequate)
       - H₁: Random effects exist
       - If p < 0.05: Need RE

    3. **Wooldridge Test (Autocorrelation)**
       - H₀: No first-order autocorrelation
       - Important for dynamic panels

    **For your PhD:** Implement all three tests for robustness
    """)

# ============================================================================
# TAB 3: DYNAMIC PANELS
# ============================================================================
with tab3:
    st.header("🚀 Dynamic Panel Models")

    st.markdown("""
    ### Model with Lagged Dependent Variable

    **Equation:** `Y_it = γ·Y_i,t-1 + X_it'β + α_i + ε_it`

    **Problem:** Y_i,t-1 correlated with (α_i + ε_i,t-1) → **OLS/FE biased!**

    **Nickell Bias:** FE estimator is biased when T is small

    **Solution:** Arellano-Bond GMM estimator
    """)

    # Create lagged variable
    df_dynamic = df_panel.sort_values(['country', 'year']).copy()
    df_dynamic['wage_gap_lag1'] = df_dynamic.groupby('country')['wage_gap'].shift(1)
    df_dynamic = df_dynamic.dropna(subset=['wage_gap_lag1', 'gdp_per_capita'])
    df_dynamic['log_gdp_pc'] = np.log(df_dynamic['gdp_per_capita'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pooled OLS (Biased)")

        X_pooled = sm.add_constant(df_dynamic[['wage_gap_lag1', 'log_gdp_pc']])
        y_pooled = df_dynamic['wage_gap']

        model_pooled = sm.OLS(y_pooled, X_pooled).fit()

        st.write("**Pooled OLS Results:**")
        pooled_summary = pd.DataFrame({
            'Variable': ['const', 'wage_gap(t-1)', 'log(GDP pc)'],
            'Coefficient': model_pooled.params.values,
            'Std Error': model_pooled.bse.values,
            'p-value': model_pooled.pvalues.values
        })

        st.dataframe(pooled_summary.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        st.metric("R²", f"{model_pooled.rsquared:.4f}")

        gamma_pooled = model_pooled.params['wage_gap_lag1']

        st.warning(f"""
        ⚠️ **Biased upward** due to correlation with α_i

        **γ̂ = {gamma_pooled:.4f}** (likely overestimated)
        """)

    with col2:
        st.subheader("Fixed Effects (Also Biased)")

        # FE with lagged dependent variable
        df_fe_dyn = df_dynamic.copy()
        for var in ['wage_gap', 'wage_gap_lag1', 'log_gdp_pc']:
            country_means = df_fe_dyn.groupby('country')[var].transform('mean')
            df_fe_dyn[f'{var}_demeaned'] = df_fe_dyn[var] - country_means

        X_fe_dyn = sm.add_constant(df_fe_dyn[['wage_gap_lag1_demeaned', 'log_gdp_pc_demeaned']])
        y_fe_dyn = df_fe_dyn['wage_gap_demeaned']

        model_fe_dyn = sm.OLS(y_fe_dyn, X_fe_dyn).fit()

        st.write("**Fixed Effects Results:**")
        fe_dyn_summary = pd.DataFrame({
            'Variable': ['const', 'wage_gap(t-1)', 'log(GDP pc)'],
            'Coefficient': model_fe_dyn.params.values,
            'Std Error': model_fe_dyn.bse.values,
            'p-value': model_fe_dyn.pvalues.values
        })

        st.dataframe(fe_dyn_summary.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        st.metric("R²", f"{model_fe_dyn.rsquared:.4f}")

        gamma_fe = model_fe_dyn.params['wage_gap_lag1_demeaned']

        st.warning(f"""
        ⚠️ **Nickell bias** (downward bias when T small)

        **γ̂ = {gamma_fe:.4f}** (likely underestimated)
        """)

    st.markdown("---")
    st.subheader("🎯 Arellano-Bond GMM Estimator")

    st.markdown("""
    **Methodology:**
    1. **First-difference** to remove α_i:
       ```
       ΔY_it = γ·ΔY_i,t-1 + ΔX_it'β + Δε_it
       ```

    2. **Use instruments:** Y_i,t-2, Y_i,t-3, ... (in levels) for ΔY_i,t-1

    3. **Moment conditions:** E[Y_i,t-s · Δε_it] = 0 for s ≥ 2

    4. **GMM minimizes:** ||g_N(β)||² where g_N are sample moments

    **Tests:**
    - **AR(1) test:** Should reject (expect negative autocorrelation in differences)
    - **AR(2) test:** Should NOT reject (no 2nd-order autocorrelation)
    - **Sargan/Hansen test:** Overidentification (H₀: instruments valid)
    """)

    st.info("""
    **Implementation Note:**

    Full Arellano-Bond GMM requires specialized package:
    - Python: `pydynpd` or `linearmodels`
    - Stata: `xtabond`, `xtdpdsys`
    - R: `plm::pgmm()`

    With only 4 years of data (2020-2023), GMM is not well-identified.
    **For your PhD:** Obtain longer time series (10+ years recommended)
    """)

    # Bounds
    st.subheader("📏 Bounds for γ")

    st.markdown(f"""
    **True γ should lie between pooled OLS and FE estimates:**

    - **Lower bound (FE):** {gamma_fe:.4f}
    - **Upper bound (Pooled):** {gamma_pooled:.4f}

    If GMM estimate falls outside these bounds → specification problem!
    """)

# ============================================================================
# TAB 4: GMM ESTIMATION
# ============================================================================
with tab4:
    st.header("🎯 Generalized Method of Moments")

    st.markdown("""
    ### GMM Theory

    **Core Idea:** Minimize distance between sample moments and population moments

    **Population moment condition:**
    ```
    E[g(Y, X, θ)] = 0
    ```

    **Sample analog:**
    ```
    ĝ_N(θ) = (1/N) Σ g(Y_i, X_i, θ) ≈ 0
    ```

    **GMM estimator:**
    ```
    θ̂_GMM = argmin ĝ_N(θ)' W ĝ_N(θ)
    ```

    W = weighting matrix (optimal: W = Var(ĝ)⁻¹)
    """)

    st.subheader("📊 GMM Variants for Panel Data")

    variants = pd.DataFrame({
        'Method': ['Difference GMM', 'System GMM', 'Two-Step GMM'],
        'Reference': ['Arellano-Bond (1991)', 'Blundell-Bond (1998)', 'Hansen (1982)'],
        'Advantages': [
            'Removes fixed effects via differencing',
            'More efficient with persistent data',
            'Asymptotically efficient weighting'
        ],
        'Disadvantages': [
            'Weak instruments if series persistent',
            'Requires stationarity assumption',
            'Downward biased SE (use robust SE)'
        ]
    })

    st.table(variants)

    st.markdown("---")
    st.subheader("🧮 Illustrative GMM: IV Estimation")

    st.markdown("""
    **Recall:** 2SLS is a special case of GMM!

    **Moment condition:** E[Z'(Y - Xβ)] = 0

    **Example:** Instrument GDP with Population
    """)

    # Prepare IV data
    df_gmm = df_panel[df_panel['year'] == 2023].dropna(subset=['wage_gap', 'gdp_per_capita', 'population']).copy()
    df_gmm['log_gdp_pc'] = np.log(df_gmm['gdp_per_capita'])
    df_gmm['log_pop'] = np.log(df_gmm['population'])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**GMM/2SLS Estimation:**")

        # Manual 2SLS as GMM
        # First stage
        X_fs_gmm = sm.add_constant(df_gmm['log_pop'])
        y_fs_gmm = df_gmm['log_gdp_pc']
        model_fs_gmm = sm.OLS(y_fs_gmm, X_fs_gmm).fit()

        # Second stage
        df_gmm['log_gdp_pc_fitted'] = model_fs_gmm.fittedvalues
        X_ss_gmm = sm.add_constant(df_gmm['log_gdp_pc_fitted'])
        y_ss_gmm = df_gmm['wage_gap']
        model_gmm = sm.OLS(y_ss_gmm, X_ss_gmm).fit()

        gmm_results = pd.DataFrame({
            'Variable': ['const', 'log(GDP pc) [IV]'],
            'Coefficient': model_gmm.params.values,
            'Std Error': model_gmm.bse.values,
            'p-value': model_gmm.pvalues.values
        })

        st.dataframe(gmm_results.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

    with col2:
        st.write("**Moment Conditions:**")

        # Calculate residuals
        residuals = y_ss_gmm - model_gmm.fittedvalues

        # Moment condition: E[Z'ε] = 0
        Z = df_gmm['log_pop'].values
        moment = np.mean(Z * residuals)

        st.metric("E[Z·ε]", f"{moment:.6f}")

        if abs(moment) < 0.01:
            st.success("✅ Moment condition approximately satisfied")
        else:
            st.warning("⚠️ Moment condition not well-satisfied")

        # J-statistic (with just-identified model, J=0)
        st.info("""
        **J-statistic (Hansen):**

        With K=1 instrument and K=1 endogenous variable:
        Model is **just-identified** → J = 0

        **Need overidentification:**
        Use multiple instruments (Z₁, Z₂, ...) for testing
        """)

    st.markdown("---")
    st.subheader("📚 Advanced GMM Topics")

    st.markdown("""
    ### For PhD Research:

    1. **Optimal Weighting Matrix**
       ```
       W_opt = [Σ(Z_i'ε_i ε_i'Z_i)]⁻¹
       ```
       Achieves asymptotic efficiency

    2. **Two-Step GMM**
       - Step 1: Use identity matrix W = I
       - Step 2: Use estimated Ω̂ from step 1

    3. **Continuously Updating GMM (CUE)**
       - Update W at each iteration
       - Better finite-sample properties

    4. **Tests**
       - **Hansen J-test:** H₀: instruments valid (p > 0.05 is good!)
       - **C-test (difference-in-Sargan):** Test subset of instruments

    **Reading:** Hansen (1982), Newey-West (1987), Hall (2005)
    """)

# ============================================================================
# TAB 5: PANEL DIAGNOSTICS
# ============================================================================
with tab5:
    st.header("📊 Panel Data Diagnostics")

    st.markdown("""
    ### Diagnostic Tests for Panel Models

    Before trusting your panel regression, check these:
    """)

    diagnostic_tests = pd.DataFrame({
        'Test': [
            'Serial Correlation',
            'Cross-Sectional Dependence',
            'Heteroskedasticity',
            'Unit Roots',
            'Cointegration'
        ],
        'Null Hypothesis': [
            'No autocorrelation',
            'Errors independent across units',
            'Constant variance',
            'Non-stationary (unit root exists)',
            'No long-run relationship'
        ],
        'Test Statistic': [
            'Wooldridge (F-test)',
            'Pesaran CD test',
            'Breusch-Pagan LM',
            'Im-Pesaran-Shin, Levin-Lin-Chu',
            'Pedroni, Kao'
        ],
        'Python Package': [
            'linearmodels',
            'statsmodels',
            'statsmodels',
            'arch.unitroot',
            'statsmodels.tsa'
        ]
    })

    st.table(diagnostic_tests)

    st.markdown("---")
    st.subheader("🔍 Visual Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Individual Time Series**")

        # Plot a few countries
        sample_countries = df_panel['country'].unique()[:6]
        df_sample = df_panel[df_panel['country'].isin(sample_countries)]

        fig_ts = px.line(
            df_sample,
            x='year',
            y='wage_gap',
            color='country',
            markers=True,
            title='Wage Gap Trends by Country'
        )

        st.plotly_chart(fig_ts, use_container_width=True)

        st.info("""
        **Check for:**
        - Common trends (suggests cross-sectional dependence)
        - Unit roots (non-stationary series)
        - Structural breaks
        """)

    with col2:
        st.write("**Cross-Sectional Correlation**")

        # Pivot to wide format
        df_wide = df_panel.pivot(index='year', columns='country', values='wage_gap')

        # Compute correlation matrix
        corr_matrix = df_wide.corr()

        # Plot heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 6}
        ))

        fig_corr.update_layout(
            title='Cross-Country Correlation Matrix',
            height=600
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.info("""
        **High correlations suggest:**
        - Common shocks (e.g., EU-wide policies)
        - Need for robust SEs clustered by time
        - Driscoll-Kraay SEs for spatial dependence
        """)

    st.markdown("---")
    st.subheader("⚠️ Common Panel Data Pitfalls")

    pitfalls = pd.DataFrame({
        'Problem': [
            'Too few time periods (T < 10)',
            'Unbalanced panel',
            'Attrition bias',
            'Time-invariant regressors with FE',
            'Persistent series + weak instruments'
        ],
        'Consequence': [
            'Nickell bias in dynamic panels',
            'Inconsistent estimators if non-random',
            'Selection bias if exit non-random',
            'Cannot estimate (demeaned out)',
            'Weak IV problem in GMM'
        ],
        'Solution': [
            'Use GMM with T ≥ 10',
            'Test for randomness, use weighted estimators',
            'Heckman selection correction',
            'Use random effects or between estimator',
            'System GMM or find better instruments'
        ]
    })

    st.table(pitfalls)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Advanced Panel Econometrics for PhD Research</strong></p>
    <p>FE • RE • Hausman • Dynamic Panels • GMM • Diagnostics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📚 Essential References:

1. **Wooldridge (2010)** - "Econometric Analysis of Cross Section and Panel Data"
2. **Baltagi (2021)** - "Econometric Analysis of Panel Data" (6th ed)
3. **Arellano (2003)** - "Panel Data Econometrics"
4. **Hsiao (2014)** - "Analysis of Panel Data" (3rd ed)
5. **Cameron & Trivedi (2005)** - "Microeconometrics: Methods and Applications"
6. **Roodman (2009)** - "How to do xtabond2: An introduction to difference and system GMM in Stata"
""")
