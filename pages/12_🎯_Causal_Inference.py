"""
Causal Inference Methods for Gender Wage Gap Analysis
Advanced econometric methods for identifying causal relationships
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="Causal Inference", page_icon="🎯", layout="wide")

# ============================================================================
# PAGE HEADER
# ============================================================================
st.title("🎯 Causal Inference Methods")
st.markdown("""
**Advanced econometric techniques for identifying causal relationships**

This page implements state-of-the-art methods from the causal inference literature.
Each method addresses the fundamental problem: **correlation ≠ causation**
""")

# ============================================================================
# THEORY SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📚 Methodology Guide")

    method = st.selectbox(
        "Select method to learn about:",
        ["Overview", "Difference-in-Differences", "Instrumental Variables",
         "Synthetic Control", "Propensity Score Matching"]
    )

    if method == "Overview":
        st.markdown("""
        ### The Causal Inference Problem

        **Question:** Does policy X reduce the wage gap?

        **Problem:** Countries that adopt X may differ in unobservable ways

        **Solution:** Use methods that create valid counterfactuals
        """)

    elif method == "Difference-in-Differences":
        st.markdown("""
        ### DiD Methodology

        **Idea:** Compare changes over time between treated and control groups

        **Equation:**
        ```
        Y_it = β₀ + β₁·Treated_i + β₂·Post_t
               + β₃·(Treated × Post) + ε_it
        ```

        **Key Assumption:** Parallel trends (absent treatment, both groups would follow same trend)

        **β₃ = ATT** (Average Treatment Effect on Treated)
        """)

    elif method == "Instrumental Variables":
        st.markdown("""
        ### IV/2SLS Methodology

        **Problem:** X is endogenous (correlated with error term)

        **Solution:** Find instrument Z that:
        1. **Relevance:** Corr(Z, X) ≠ 0
        2. **Exogeneity:** Corr(Z, ε) = 0

        **Two stages:**
        ```
        Stage 1: X = π₀ + π₁·Z + v
        Stage 2: Y = β₀ + β₁·X̂ + u
        ```

        **Tests:**
        - F-stat > 10 (weak instrument)
        - Sargan/Hansen (overidentification)
        """)

    elif method == "Synthetic Control":
        st.markdown("""
        ### Synthetic Control Method

        **Idea:** Create synthetic counterfactual from weighted combination of control units

        **Weights chosen to minimize:**
        ```
        ||X₁ - X₀W||
        ```

        Where X₁ = treated pre-treatment characteristics

        **Advantages:**
        - Transparent data-driven procedure
        - No functional form assumptions
        - Allows heterogeneous effects

        **Reference:** Abadie et al. (2010)
        """)

    elif method == "Propensity Score Matching":
        st.markdown("""
        ### PSM Methodology

        **Idea:** Match treated units to similar control units based on observables

        **Propensity score:**
        ```
        e(X) = P(Treatment = 1 | X)
        ```

        **Matching methods:**
        - Nearest neighbor
        - Kernel matching
        - Caliper matching

        **Assumption:** Conditional independence
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Difference-in-Differences",
    "🎸 Instrumental Variables",
    "🧬 Synthetic Control",
    "🎯 Propensity Score Matching"
])

# ============================================================================
# TAB 1: DIFFERENCE-IN-DIFFERENCES
# ============================================================================
with tab1:
    st.header("📊 Difference-in-Differences (DiD) Analysis")

    st.markdown("""
    ### Research Question: Do EU membership effects reduce wage gaps?

    **Hypothetical scenario:** Suppose countries that joined EU after 2004 (Eastern enlargement)
    experienced policy interventions promoting gender equality.

    **Treatment group:** Countries joining 2004+ (Bulgaria, Romania, Croatia)
    **Control group:** Old EU members (pre-2004)
    **Time period:** Compare 2020 vs 2023
    """)

    # Prepare DiD data
    post_2004_countries = ['Bulgaria', 'Romania', 'Croatia', 'Estonia', 'Latvia',
                          'Lithuania', 'Poland', 'Czech Republic', 'Slovakia',
                          'Slovenia', 'Hungary', 'Cyprus', 'Malta']

    df_did = df_panel[df_panel['year'].isin([2020, 2023])].copy()
    df_did['treated'] = df_did['country'].isin(post_2004_countries).astype(int)
    df_did['post'] = (df_did['year'] == 2023).astype(int)
    df_did['treated_post'] = df_did['treated'] * df_did['post']

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("DiD Estimation")

        # DiD regression
        formula = 'wage_gap ~ treated + post + treated_post'
        model_did = ols(formula, data=df_did).fit()

        # Results table
        results_df = pd.DataFrame({
            'Variable': ['Intercept', 'Treated', 'Post', 'Treated × Post (DiD)'],
            'Coefficient': model_did.params.values,
            'Std Error': model_did.bse.values,
            't-statistic': model_did.tvalues.values,
            'p-value': model_did.pvalues.values,
            '95% CI Lower': model_did.conf_int()[0].values,
            '95% CI Upper': model_did.conf_int()[1].values
        })

        st.dataframe(results_df.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            't-statistic': '{:.4f}',
            'p-value': '{:.4f}',
            '95% CI Lower': '{:.4f}',
            '95% CI Upper': '{:.4f}'
        }).apply(lambda x: ['background-color: #90EE90' if x.name == 3 else '' for i in x], axis=1),
        hide_index=True)

        st.metric("R-squared", f"{model_did.rsquared:.4f}")
        st.metric("Adj. R-squared", f"{model_did.rsquared_adj:.4f}")
        st.metric("F-statistic p-value", f"{model_did.f_pvalue:.6f}")

        # Interpretation
        did_effect = model_did.params['treated_post']
        did_pvalue = model_did.pvalues['treated_post']

        st.markdown("---")
        st.subheader("📖 Interpretation")

        if did_pvalue < 0.05:
            if did_effect < 0:
                st.success(f"""
                ✅ **Significant negative DiD effect: {did_effect:.2f} percentage points (p={did_pvalue:.4f})**

                Post-2004 EU members reduced their wage gap by {abs(did_effect):.2f}pp more than old members
                between 2020 and 2023.

                **Causal interpretation (if parallel trends holds):** EU membership policies
                causally reduced wage gaps in newer member states.
                """)
            else:
                st.warning(f"""
                ⚠️ **Significant positive DiD effect: {did_effect:.2f}pp (p={did_pvalue:.4f})**

                Wage gaps increased more in post-2004 countries.
                """)
        else:
            st.info(f"""
            ℹ️ **No significant DiD effect (p={did_pvalue:.4f})**

            We cannot reject the null hypothesis that treatment had no differential effect.
            """)

    with col2:
        st.subheader("DiD Visualization")

        # Calculate group means
        did_means = df_did.groupby(['treated', 'year'])['wage_gap'].mean().reset_index()
        did_means['group'] = did_means['treated'].map({0: 'Control (Pre-2004)', 1: 'Treated (Post-2004)'})

        # Parallel trends plot
        fig_did = px.line(
            did_means,
            x='year',
            y='wage_gap',
            color='group',
            markers=True,
            labels={'wage_gap': 'Average Wage Gap (%)', 'year': 'Year'},
            title='Parallel Trends Plot',
            color_discrete_map={'Control (Pre-2004)': 'steelblue', 'Treated (Post-2004)': 'coral'}
        )

        fig_did.update_traces(line=dict(width=3), marker=dict(size=12))
        fig_did.add_vline(x=2021.5, line_dash="dash", line_color="gray",
                         annotation_text="Treatment Period")

        st.plotly_chart(fig_did, use_container_width=True)

        # 2x2 table
        st.subheader("2×2 DiD Table")

        pivot_table = did_means.pivot(index='group', columns='year', values='wage_gap')
        pivot_table['Difference'] = pivot_table[2023] - pivot_table[2020]

        st.dataframe(pivot_table.style.format("{:.2f}"))

        # Calculate DiD manually
        control_diff = pivot_table.loc['Control (Pre-2004)', 'Difference']
        treated_diff = pivot_table.loc['Treated (Post-2004)', 'Difference']
        did_manual = treated_diff - control_diff

        st.metric("DiD Estimate (Manual)", f"{did_manual:.2f}pp")

    # Parallel trends test
    st.markdown("---")
    st.subheader("🧪 Parallel Trends Assumption Test")

    st.markdown("""
    **Critical assumption:** Absent treatment, both groups would have followed parallel trends.

    **Test:** We need pre-treatment data (before 2020) to test this. With only 2020-2023, we cannot formally test.

    **What you should do for your PhD:**
    1. Obtain data from 2015-2019 to test pre-trends
    2. Run placebo tests (pretend treatment happened earlier)
    3. Event study specification with year-specific effects
    """)

    # If we had more years, show event study
    st.info("""
    **Event Study Model (for future implementation):**
    ```
    Y_it = α_i + λ_t + Σ β_k·D_ik + ε_it
    ```
    Where D_ik are treatment indicators for each year k relative to treatment.
    """)

# ============================================================================
# TAB 2: INSTRUMENTAL VARIABLES
# ============================================================================
with tab2:
    st.header("🎸 Instrumental Variables (2SLS) Analysis")

    st.markdown("""
    ### The Endogeneity Problem

    **Question:** Does higher GDP per capita reduce wage gaps?

    **Problem:** GDP might be endogenous:
    - Reverse causality: Lower wage gaps → higher GDP (more women working)
    - Omitted variables: Innovation, institutions affect both
    - Measurement error in GDP

    **Solution:** Find an instrument Z that affects GDP but not wage gaps directly
    """)

    # Prepare IV data
    df_iv = df_panel[df_panel['year'] == 2023].dropna(subset=['wage_gap', 'gdp_per_capita', 'population']).copy()
    df_iv['log_gdp_pc'] = np.log(df_iv['gdp_per_capita'])
    df_iv['log_population'] = np.log(df_iv['population'])

    # Create instrument: population (arguably exogenous to current wage gaps)
    # This is for illustration - not necessarily a valid instrument!

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stage 1: First Stage Regression")
        st.markdown("**Equation:** `log(GDP_pc) = π₀ + π₁·log(Population) + v`")

        # First stage
        X_fs = sm.add_constant(df_iv['log_population'])
        y_fs = df_iv['log_gdp_pc']

        model_fs = sm.OLS(y_fs, X_fs).fit()

        st.write("**First Stage Results:**")
        fs_results = pd.DataFrame({
            'Variable': ['Constant', 'log(Population)'],
            'Coefficient': model_fs.params.values,
            'Std Error': model_fs.bse.values,
            't-statistic': model_fs.tvalues.values,
            'p-value': model_fs.pvalues.values
        })

        st.dataframe(fs_results.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            't-statistic': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        # F-statistic for weak instrument test
        f_stat = model_fs.fvalue
        st.metric("First-Stage F-statistic", f"{f_stat:.2f}")

        if f_stat > 10:
            st.success("✅ **Strong instrument** (F > 10)")
        else:
            st.error("❌ **Weak instrument** (F < 10) - IV estimates unreliable!")

        st.metric("First-Stage R²", f"{model_fs.rsquared:.4f}")

    with col2:
        st.subheader("First Stage Scatter Plot")

        fig_fs = px.scatter(
            df_iv,
            x='log_population',
            y='log_gdp_pc',
            hover_name='country',
            trendline='ols',
            labels={'log_population': 'log(Population)', 'log_gdp_pc': 'log(GDP per capita)'},
            title='Instrument Relevance'
        )

        st.plotly_chart(fig_fs, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stage 2: Second Stage (IV)")
        st.markdown("**Equation:** `wage_gap = β₀ + β₁·log(GDP_pc)̂ + u`")

        # Second stage: use fitted values from first stage
        df_iv['log_gdp_pc_fitted'] = model_fs.fittedvalues

        X_ss = sm.add_constant(df_iv['log_gdp_pc_fitted'])
        y_ss = df_iv['wage_gap']

        model_2sls = sm.OLS(y_ss, X_ss).fit()

        st.write("**Second Stage (IV) Results:**")
        ss_results = pd.DataFrame({
            'Variable': ['Constant', 'log(GDP pc) [instrumented]'],
            'Coefficient': model_2sls.params.values,
            'Std Error': model_2sls.bse.values,
            't-statistic': model_2sls.tvalues.values,
            'p-value': model_2sls.pvalues.values
        })

        st.dataframe(ss_results.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            't-statistic': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        st.metric("IV R²", f"{model_2sls.rsquared:.4f}")

    with col2:
        st.subheader("Compare OLS vs IV")

        # OLS for comparison
        X_ols = sm.add_constant(df_iv['log_gdp_pc'])
        model_ols = sm.OLS(y_ss, X_ols).fit()

        comparison = pd.DataFrame({
            'Method': ['OLS (biased)', 'IV (2SLS)'],
            'β₁ Coefficient': [model_ols.params[1], model_2sls.params[1]],
            'Std Error': [model_ols.bse[1], model_2sls.bse[1]],
            'p-value': [model_ols.pvalues[1], model_2sls.pvalues[1]]
        })

        st.dataframe(comparison.style.format({
            'β₁ Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'p-value': '{:.4f}'
        }), hide_index=True)

        st.markdown("""
        **Interpretation:**
        - If IV ≠ OLS, suggests endogeneity bias
        - IV typically has larger standard errors
        - Trust IV if instrument is valid
        """)

    st.markdown("---")
    st.subheader("⚠️ IV Validity Concerns")

    st.warning("""
    **This is an illustrative example. For your PhD, you need:**

    1. **Better instruments:**
       - Historical variables (e.g., 1800s literacy rates)
       - Geographic features (e.g., distance to trading hubs)
       - Policy shocks in neighboring countries

    2. **Formal tests:**
       - Sargan/Hansen test (overidentification - need multiple instruments)
       - Durbin-Wu-Hausman test (endogeneity test)
       - Anderson-Rubin weak instrument robust test

    3. **Theoretical justification:**
       - Why does Z affect X? (relevance)
       - Why doesn't Z affect Y except through X? (exclusion restriction)

    **Reading:** Angrist & Pischke (2009) "Mostly Harmless Econometrics"
    """)

# ============================================================================
# TAB 3: SYNTHETIC CONTROL
# ============================================================================
with tab3:
    st.header("🧬 Synthetic Control Method")

    st.markdown("""
    ### Scenario: Policy Intervention Analysis

    **Research question:** What if France implemented a major gender pay equity law in 2021?

    **Method:** Create a "synthetic France" from weighted average of other EU countries
    that resembles France's pre-treatment characteristics.

    **Reference:** Abadie, Diamond & Hainmueller (2010, 2015)
    """)

    # Select treated unit
    treated_country = st.selectbox(
        "Select treated country (hypothetical intervention in 2021):",
        options=sorted(df_panel['country'].unique()),
        index=sorted(df_panel['country'].unique()).tolist().index('France')
    )

    # Get data
    df_treated = df_panel[df_panel['country'] == treated_country].sort_values('year')
    df_donors = df_panel[df_panel['country'] != treated_country]

    st.subheader(f"Synthetic Control for {treated_country}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Algorithm:**
        1. Pre-treatment period: 2020
        2. Post-treatment period: 2021-2023
        3. Find weights W such that Synthetic ≈ Treated (pre-treatment)
        4. Compare post-treatment outcomes
        """)

        # For simplicity, use equal weights (proper SC uses optimization)
        # In real implementation, you'd use cvxpy or similar to solve for optimal weights

        st.info("""
        **Note:** This is a simplified implementation. A rigorous synthetic control requires:
        - Nested optimization (Abadie & Gardeazabal 2003)
        - Multiple pre-treatment covariates
        - Cross-validation for weight selection
        - Placebo tests for inference

        **Python package:** Use `SparseSC` or implement with `cvxpy`
        """)

        # Calculate simple synthetic control (average of similar countries by region)
        treated_region = df_2023[df_2023['country_name'] == treated_country]['region'].values[0]
        similar_countries = df_2023[
            (df_2023['region'] == treated_region) &
            (df_2023['country_name'] != treated_country)
        ]['country_name'].head(5).tolist()

        st.write(f"**Donor pool (similar countries):** {', '.join(similar_countries)}")

        # Create synthetic control as average
        df_synthetic = df_donors[df_donors['country'].isin(similar_countries)].groupby('year')['wage_gap'].mean().reset_index()
        df_synthetic['type'] = 'Synthetic Control'

        df_treated_plot = df_treated[['year', 'wage_gap']].copy()
        df_treated_plot['type'] = treated_country

        df_comparison = pd.concat([df_treated_plot, df_synthetic.rename(columns={'wage_gap': 'wage_gap'})])

        # Plot
        fig_sc = px.line(
            df_comparison,
            x='year',
            y='wage_gap',
            color='type',
            markers=True,
            labels={'wage_gap': 'Wage Gap (%)', 'year': 'Year'},
            title=f'{treated_country} vs Synthetic Control',
            color_discrete_map={treated_country: 'red', 'Synthetic Control': 'blue'}
        )

        fig_sc.add_vline(x=2020.5, line_dash="dash", line_color="gray",
                        annotation_text="Hypothetical Treatment")
        fig_sc.update_traces(line=dict(width=3), marker=dict(size=10))

        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        st.subheader("Treatment Effects")

        # Calculate gap for each post-treatment year
        effects = []
        for year in [2021, 2022, 2023]:
            treated_val = df_treated[df_treated['year'] == year]['wage_gap'].values
            synthetic_val = df_synthetic[df_synthetic['year'] == year]['wage_gap'].values

            if len(treated_val) > 0 and len(synthetic_val) > 0:
                effect = treated_val[0] - synthetic_val[0]
                effects.append({
                    'Year': year,
                    'Treated': treated_val[0],
                    'Synthetic': synthetic_val[0],
                    'Effect': effect
                })

        df_effects = pd.DataFrame(effects)

        st.dataframe(df_effects.style.format({
            'Treated': '{:.2f}',
            'Synthetic': '{:.2f}',
            'Effect': '{:.2f}'
        }), hide_index=True)

        avg_effect = df_effects['Effect'].mean()
        st.metric("Average Treatment Effect", f"{avg_effect:.2f}pp")

        st.markdown("---")
        st.subheader("📊 Interpretation")

        if avg_effect < -0.5:
            st.success(f"""
            ✅ **Negative effect:** {treated_country} reduced wage gap by {abs(avg_effect):.2f}pp more than synthetic control
            """)
        elif avg_effect > 0.5:
            st.warning(f"""
            ⚠️ **Positive effect:** {treated_country} increased wage gap by {avg_effect:.2f}pp vs synthetic control
            """)
        else:
            st.info("ℹ️ **Small/no effect** detected")

    st.markdown("---")
    st.subheader("🧪 Inference: Placebo Tests")

    st.markdown("""
    **Problem:** How do we know if the effect is statistically significant?

    **Solution:** Run placebo tests (in-space and in-time)

    **In-space placebo:**
    1. Pretend each donor country was treated
    2. Calculate placebo effects
    3. Compare actual effect to distribution of placebo effects
    4. p-value = rank / (N+1)

    **In-time placebo:**
    1. Pretend treatment happened in 2019 (before actual treatment)
    2. Should find no effect if method is valid

    **For your PhD:** Implement full placebo distribution and calculate exact p-values
    """)

# ============================================================================
# TAB 4: PROPENSITY SCORE MATCHING
# ============================================================================
with tab4:
    st.header("🎯 Propensity Score Matching (PSM)")

    st.markdown("""
    ### Matching for Causal Inference

    **Scenario:** Countries with high female labor force participation vs low

    **Problem:** High-participation countries differ in many ways (GDP, education, culture)

    **Solution:** Match each treated country to similar control country based on observables
    """)

    # Create treatment variable (high female labor force participation)
    # For illustration, use wage gap median as cutoff
    median_gap = df_2023['wage_gap_percent'].median()

    df_psm = df_2023.copy()
    df_psm['treatment'] = (df_psm['wage_gap_percent'] > median_gap).astype(int)
    df_psm['gdp_per_capita'] = (df_psm['gdp_billions'] * 1e9) / df_psm['population']
    df_psm['log_gdp_pc'] = np.log(df_psm['gdp_per_capita'])
    df_psm['log_pop'] = np.log(df_psm['population'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step 1: Estimate Propensity Scores")

        st.markdown("""
        **Logit model:** `P(Treatment=1) = Λ(β₀ + β₁·log(GDP) + β₂·log(Pop))`

        Where Λ is the logistic function
        """)

        # Estimate propensity scores with logit
        X_psm = df_psm[['log_gdp_pc', 'log_pop']].dropna()
        y_psm = df_psm.loc[X_psm.index, 'treatment']

        from sklearn.linear_model import LogisticRegression

        logit = LogisticRegression()
        logit.fit(X_psm, y_psm)

        df_psm.loc[X_psm.index, 'propensity_score'] = logit.predict_proba(X_psm)[:, 1]

        # Show distribution
        fig_pscore = go.Figure()

        for treat in [0, 1]:
            fig_pscore.add_trace(go.Histogram(
                x=df_psm[df_psm['treatment'] == treat]['propensity_score'],
                name=f'Treatment={treat}',
                opacity=0.7,
                nbinsx=10
            ))

        fig_pscore.update_layout(
            title='Propensity Score Distribution',
            xaxis_title='Propensity Score',
            yaxis_title='Count',
            barmode='overlay'
        )

        st.plotly_chart(fig_pscore, use_container_width=True)

        st.info("""
        **Common support:** Both groups should have overlapping propensity score distributions.
        If no overlap, matching won't work well.
        """)

    with col2:
        st.subheader("Step 2: Nearest Neighbor Matching")

        # Perform 1:1 nearest neighbor matching
        treated = df_psm[df_psm['treatment'] == 1].dropna(subset=['propensity_score'])
        control = df_psm[df_psm['treatment'] == 0].dropna(subset=['propensity_score'])

        matches = []

        for idx, treated_unit in treated.iterrows():
            # Find nearest control unit
            distances = np.abs(control['propensity_score'] - treated_unit['propensity_score'])
            nearest_idx = distances.idxmin()

            matches.append({
                'Treated Country': treated_unit['country_name'],
                'Treated PS': treated_unit['propensity_score'],
                'Control Country': control.loc[nearest_idx, 'country_name'],
                'Control PS': control.loc[nearest_idx, 'propensity_score'],
                'Distance': distances[nearest_idx]
            })

        df_matches = pd.DataFrame(matches)

        st.write("**Matched Pairs:**")
        st.dataframe(df_matches.style.format({
            'Treated PS': '{:.3f}',
            'Control PS': '{:.3f}',
            'Distance': '{:.4f}'
        }), hide_index=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step 3: Estimate ATT")

        st.markdown("**Average Treatment Effect on the Treated (ATT)**")

        # Calculate outcomes for matched pairs
        att_data = []

        for _, match in df_matches.iterrows():
            treated_outcome = df_psm[df_psm['country_name'] == match['Treated Country']]['wage_gap_percent'].values[0]
            control_outcome = df_psm[df_psm['country_name'] == match['Control Country']]['wage_gap_percent'].values[0]

            att_data.append({
                'Pair': _ + 1,
                'Treated Outcome': treated_outcome,
                'Control Outcome': control_outcome,
                'Difference': treated_outcome - control_outcome
            })

        df_att = pd.DataFrame(att_data)

        st.dataframe(df_att.style.format({
            'Treated Outcome': '{:.2f}',
            'Control Outcome': '{:.2f}',
            'Difference': '{:.2f}'
        }), hide_index=True)

        att = df_att['Difference'].mean()
        att_se = df_att['Difference'].std() / np.sqrt(len(df_att))
        t_stat = att / att_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(df_att) - 1))

        st.metric("ATT Estimate", f"{att:.2f}pp")
        st.metric("Standard Error", f"{att_se:.2f}")
        st.metric("t-statistic", f"{t_stat:.2f}")
        st.metric("p-value", f"{p_value:.4f}")

        if p_value < 0.05:
            st.success("✅ **Statistically significant** (p < 0.05)")
        else:
            st.info("ℹ️ **Not statistically significant** (p ≥ 0.05)")

    with col2:
        st.subheader("Balance Check")

        st.markdown("""
        **Before matching:** Treated and control differ on observables

        **After matching:** Should be balanced (similar covariates)
        """)

        # Calculate standardized differences
        treated_matched = df_psm[df_psm['country_name'].isin(df_matches['Treated Country'])]
        control_matched = df_psm[df_psm['country_name'].isin(df_matches['Control Country'])]

        balance = []

        for var in ['log_gdp_pc', 'log_pop']:
            mean_treat = treated_matched[var].mean()
            mean_control = control_matched[var].mean()
            std_pooled = np.sqrt((treated_matched[var].var() + control_matched[var].var()) / 2)
            std_diff = (mean_treat - mean_control) / std_pooled * 100

            balance.append({
                'Variable': var,
                'Treated Mean': mean_treat,
                'Control Mean': mean_control,
                'Std Diff (%)': std_diff
            })

        df_balance = pd.DataFrame(balance)

        st.dataframe(df_balance.style.format({
            'Treated Mean': '{:.3f}',
            'Control Mean': '{:.3f}',
            'Std Diff (%)': '{:.2f}'
        }), hide_index=True)

        st.markdown("""
        **Rule of thumb:** Standardized difference < 10% indicates good balance
        """)

        if df_balance['Std Diff (%)'].abs().max() < 10:
            st.success("✅ **Good balance achieved**")
        else:
            st.warning("⚠️ **Poor balance** - consider adding more covariates or different matching method")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Causal Inference for PhD Research</strong></p>
    <p>DiD • IV/2SLS • Synthetic Control • Propensity Score Matching</p>
    <p><em>These are illustrative implementations. For publication-quality analysis, consult econometric textbooks and use specialized packages.</em></p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📚 Recommended Reading:

1. **Angrist & Pischke (2009)** - "Mostly Harmless Econometrics"
2. **Cunningham (2021)** - "Causal Inference: The Mixtape"
3. **Huntington-Klein (2021)** - "The Effect: An Introduction to Research Design and Causality"
4. **Abadie (2021)** - "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects"
5. **Imbens & Rubin (2015)** - "Causal Inference for Statistics, Social, and Biomedical Sciences"
""")
