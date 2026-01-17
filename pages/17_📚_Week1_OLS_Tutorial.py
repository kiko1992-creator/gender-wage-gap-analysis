"""
Week 1: OLS Regression Fundamentals - Interactive Tutorial
Learn regression step-by-step using EU wage gap database
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="OLS Tutorial", page_icon="📚", layout="wide")

# ============================================================================
# HEADER
# ============================================================================
st.title("📚 Week 1: OLS Regression Fundamentals")
st.markdown("""
**Interactive Tutorial: Learn by Doing**

This is your first step in the 18-month PhD preparation journey.
Today you'll learn OLS regression using REAL EU wage gap data from your PostgreSQL database.

**Learning Objectives:**
- ✅ Understand the 5 OLS assumptions
- ✅ Run your first regression
- ✅ Interpret coefficients, p-values, and R²
- ✅ Test assumptions visually and statistically
- ✅ Understand when OLS works and when it fails
""")

# Progress tracking
if 'week1_progress' not in st.session_state:
    st.session_state.week1_progress = {
        'theory_complete': False,
        'first_regression': False,
        'assumptions_tested': False,
        'interpretation_done': False
    }

# ============================================================================
# SIDEBAR: LEARNING PATH
# ============================================================================
with st.sidebar:
    st.header("📖 Learning Path")

    st.markdown("### Your Progress:")
    progress_items = {
        'theory_complete': '1. Theory Review',
        'first_regression': '2. First Regression',
        'assumptions_tested': '3. Test Assumptions',
        'interpretation_done': '4. Interpretation'
    }

    for key, label in progress_items.items():
        if st.session_state.week1_progress[key]:
            st.success(f"✅ {label}")
        else:
            st.info(f"⬜ {label}")

    progress_pct = sum(st.session_state.week1_progress.values()) / len(st.session_state.week1_progress) * 100
    st.metric("Completion", f"{progress_pct:.0f}%")

    st.markdown("---")
    st.markdown("""
    ### 📚 Today's Resources:

    **Required Reading:**
    - Wooldridge Ch. 2 (Simple Regression)
    - Stock & Watson Ch. 4 (OLS Assumptions)

    **Videos:**
    - Ben Lambert: OLS Intuition
    - Marginal Revolution: Regression Basics

    **Time Estimate:** 3-4 hours
    """)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_tutorial_data():
    """Load data from PostgreSQL database"""
    df_2023 = get_all_countries_2023()

    # Calculate additional variables
    df_2023['gdp_per_capita'] = (df_2023['gdp_billions'] * 1e9) / df_2023['population']
    df_2023['log_gdp_pc'] = np.log(df_2023['gdp_per_capita'])
    df_2023['log_population'] = np.log(df_2023['population'])
    df_2023['gdp_per_capita_1000'] = df_2023['gdp_per_capita'] / 1000

    return df_2023

df = load_tutorial_data()

st.success(f"""
✅ **Data loaded from PostgreSQL database!**
- {len(df)} EU countries
- Data year: 2023
- Variables available: wage_gap, GDP per capita, population, region
""")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Theory Review",
    "🔍 Explore Data",
    "📊 First Regression",
    "🧪 Test Assumptions",
    "📝 Interpretation",
    "🎯 Quiz & Practice"
])

# ============================================================================
# TAB 1: THEORY REVIEW
# ============================================================================
with tab1:
    st.header("📖 The 5 Classical OLS Assumptions")

    st.markdown("""
    Before running ANY regression, you must understand these assumptions.
    Every advanced method you'll learn is either building on these or fixing violations.
    """)

    # Assumption 1
    with st.expander("**Assumption 1: Linearity in Parameters** (Click to expand)"):
        st.markdown("""
        ### Y = β₀ + β₁X₁ + β₂X₂ + ... + ε

        **What it means:**
        - The model is LINEAR in the β coefficients
        - Y can be nonlinear in X (e.g., log(X), X²) ✅
        - But Y MUST be linear in β (no β², β in exponent, etc.) ❌

        **Examples:**
        ```python
        # ✅ LINEAR in parameters (β₀, β₁)
        wage_gap = β₀ + β₁·log(GDP) + ε

        # ✅ Still LINEAR in parameters (β₀, β₁, β₂)
        wage_gap = β₀ + β₁·GDP + β₂·GDP² + ε

        # ❌ NONLINEAR in parameters
        wage_gap = β₀·GDP^β₁ + ε  # β₁ in exponent!
        ```

        **Why it matters:**
        - OLS only works for linear-in-parameters models
        - If nonlinear → need Maximum Likelihood Estimation (MLE) or Nonlinear Least Squares (NLS)
        """)

        st.info("""
        **💡 Key Insight:**
        You can model curved relationships (log, polynomials) while staying linear in parameters!
        """)

    # Assumption 2
    with st.expander("**Assumption 2: Random Sampling (i.i.d.)** (Click to expand)"):
        st.markdown("""
        ### Each observation is independently and identically distributed

        **What it means:**
        - Observation i doesn't affect observation j
        - All observations come from the same distribution
        - No systematic selection bias

        **Violations:**
        - ❌ Time series data (Y_t correlated with Y_t-1)
        - ❌ Panel data (same country over time)
        - ❌ Clustered data (students within schools)
        - ❌ Selection bias (only rich countries in sample)

        **Fixes:**
        - Time series → AR models, ARIMA (Week 6-8)
        - Panel data → Fixed Effects (Week 2-4)
        - Clustered data → Cluster standard errors
        - Selection bias → Heckman correction

        **For EU wage gap:**
        - 27 countries in 2023 ✅ (independent)
        - BUT if we use multiple years → Panel data ⚠️
        """)

    # Assumption 3
    with st.expander("**Assumption 3: No Perfect Multicollinearity** (Click to expand)"):
        st.markdown("""
        ### No exact linear relationship between X variables

        **What it means:**
        - Can't have X₁ = c·X₂ exactly
        - Can't include all dummy variables + intercept
        - Matrix X'X must be invertible

        **Perfect multicollinearity (fatal):**
        ```python
        # ❌ Include GDP_total and GDP_per_capita when pop is constant
        # GDP_per_capita = GDP_total / population
        # If population doesn't vary, these are perfectly correlated!

        # ❌ Include dummies for ALL 27 countries + intercept
        # Sum of all dummies = 1 = intercept (dummy variable trap)
        ```

        **High multicollinearity (not fatal, but problematic):**
        - Correlation between X's is high (e.g., 0.95) but not perfect
        - OLS still works, but standard errors are HUGE
        - Can't tell which X is actually important

        **Detection:**
        - Variance Inflation Factor (VIF) > 10 is concerning
        - Correlation matrix shows r > 0.9

        **Fixes:**
        - Drop one of the correlated variables
        - Use Ridge/LASSO regression (Week 17)
        - Principal Component Analysis
        """)

        st.warning("""
        ⚠️ **Perfect multicollinearity:** Python will give error "singular matrix"
        ⚠️ **High multicollinearity:** Results will have huge standard errors
        """)

    # Assumption 4
    with st.expander("**Assumption 4: Zero Conditional Mean E[ε|X] = 0** ⭐ MOST IMPORTANT"):
        st.markdown("""
        ### Error term is uncorrelated with X

        **This is THE crucial assumption for causality!**

        **What it means:**
        - Knowing X tells you nothing about ε
        - No omitted variables correlated with X
        - No reverse causality (Y causing X)
        - No measurement error in X

        **Why it matters:**
        - ✅ If TRUE: OLS is UNBIASED and CONSISTENT
        - ❌ If FALSE: OLS is BIASED (even with infinite data!)

        **Common violations:**

        **1. Omitted Variable Bias (OVB):**
        ```
        True model:    wage_gap = β₀ + β₁·GDP + β₂·culture + ε
        You estimate:  wage_gap = β₀ + β₁·GDP + ε*

        If culture is omitted and Corr(GDP, culture) ≠ 0:
        → β̂₁ is BIASED
        ```

        **2. Reverse Causality:**
        ```
        Model: wage_gap → GDP_growth
        But also: GDP_growth → wage_gap

        Which causes which? Both! → Endogeneity
        ```

        **3. Measurement Error:**
        ```
        True:     wage_gap = β₁·GDP_true + ε
        Observed: wage_gap = β₁·GDP_measured + ε*

        If GDP_measured ≠ GDP_true → bias
        ```

        **Detection:**
        - Think hard about theory
        - Can't test statistically (ε is unobserved!)
        - Placebo tests, robustness checks

        **Fixes:**
        - Instrumental Variables (Week 11-12)
        - Fixed Effects (Week 3-4)
        - Difference-in-Differences (Week 10)
        - Randomized experiments
        """)

        st.error("""
        🚨 **CRITICAL:**
        Assumption 4 violation = BIAS
        All other assumption violations = just efficiency problems (larger SEs)

        This is why causal inference is hard!
        """)

    # Assumption 5
    with st.expander("**Assumption 5: Homoskedasticity Var(ε|X) = σ²** (Click to expand)"):
        st.markdown("""
        ### Constant error variance

        **What it means:**
        - Variance of ε doesn't depend on X
        - Errors spread out equally for all X values
        - σ² is constant

        **Heteroskedasticity (violation):**
        - Var(ε|X) varies with X
        - Common in cross-sectional data
        - Errors fan out as X increases

        **Consequences:**
        - ✅ OLS estimates still UNBIASED
        - ✅ OLS estimates still CONSISTENT
        - ❌ Standard errors are WRONG
        - ❌ t-tests, F-tests are invalid
        - ❌ Confidence intervals are wrong

        **Detection:**
        - Visual: plot residuals vs fitted values
        - Breusch-Pagan test (H₀: homoskedasticity)
        - White test

        **Fix (EASY!):**
        ```python
        # Use robust standard errors (heteroskedasticity-consistent)
        model = sm.OLS(Y, X).fit(cov_type='HC3')  # White standard errors
        ```

        **Why it's not as bad as Assumption 4:**
        - Doesn't bias estimates (just SEs)
        - Easy to fix with robust SEs
        - Very common in practice
        """)

        st.success("""
        ✅ **Good news:** Heteroskedasticity is easy to fix!
        Just use robust standard errors (always recommended anyway).
        """)

    # Summary
    st.markdown("---")
    st.subheader("📋 Quick Summary")

    summary_df = pd.DataFrame({
        'Assumption': [
            '1. Linearity',
            '2. Random Sampling',
            '3. No Perfect Multicollinearity',
            '4. Zero Conditional Mean',
            '5. Homoskedasticity'
        ],
        'Violation Consequence': [
            'OLS doesn\'t apply',
            'SEs wrong (need clustering)',
            'Can\'t estimate (or huge SEs)',
            '🚨 BIAS 🚨',
            'SEs wrong (use robust SEs)'
        ],
        'Severity': [
            'Fatal',
            'Moderate',
            'Fatal (perfect) / Moderate (high)',
            '🔥 CRITICAL 🔥',
            'Minor (fixable)'
        ]
    })

    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    if st.button("✅ I understand the 5 assumptions", key='theory_complete'):
        st.session_state.week1_progress['theory_complete'] = True
        st.success("Great! Move to the next tab: Explore Data")
        st.balloons()

# ============================================================================
# TAB 2: EXPLORE DATA
# ============================================================================
with tab2:
    st.header("🔍 Exploratory Data Analysis")

    st.markdown("""
    **Before running regression, ALWAYS explore your data!**

    **The Three Questions:**
    1. What do the distributions look like?
    2. Are there outliers?
    3. What's the relationship between Y and X?
    """)

    # Show data
    st.subheader("📊 Your Database (from PostgreSQL)")

    st.dataframe(df[['country_name', 'wage_gap_percent', 'gdp_per_capita', 'population', 'region']].style.format({
        'wage_gap_percent': '{:.1f}%',
        'gdp_per_capita': '${:,.0f}',
        'population': '{:,.0f}'
    }), hide_index=True, use_container_width=True)

    st.markdown("---")

    # Summary statistics
    st.subheader("📈 Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Wage Gap (Dependent Variable):**")
        st.dataframe(df['wage_gap_percent'].describe().to_frame().T.style.format("{:.2f}"),
                    use_container_width=True)

        # Histogram
        fig_y = px.histogram(df, x='wage_gap_percent', nbins=15,
                            title='Distribution of Wage Gap',
                            labels={'wage_gap_percent': 'Wage Gap (%)'},
                            color_discrete_sequence=['steelblue'])
        fig_y.add_vline(x=df['wage_gap_percent'].mean(), line_dash="dash",
                       line_color="red", annotation_text="Mean")
        st.plotly_chart(fig_y, use_container_width=True)

    with col2:
        st.write("**GDP per Capita (Independent Variable):**")
        st.dataframe(df['gdp_per_capita'].describe().to_frame().T.style.format("{:,.0f}"),
                    use_container_width=True)

        # Histogram
        fig_x = px.histogram(df, x='gdp_per_capita', nbins=15,
                            title='Distribution of GDP per Capita',
                            labels={'gdp_per_capita': 'GDP per Capita ($)'},
                            color_discrete_sequence=['coral'])
        fig_x.add_vline(x=df['gdp_per_capita'].mean(), line_dash="dash",
                       line_color="red", annotation_text="Mean")
        st.plotly_chart(fig_x, use_container_width=True)

    # Scatter plot
    st.markdown("---")
    st.subheader("🔍 Relationship: Wage Gap vs GDP per Capita")

    fig_scatter = px.scatter(df, x='gdp_per_capita', y='wage_gap_percent',
                            hover_name='country_name',
                            labels={'gdp_per_capita': 'GDP per Capita ($)',
                                   'wage_gap_percent': 'Wage Gap (%)'},
                            title='Does GDP per Capita Affect Wage Gap?',
                            trendline='ols',
                            trendline_color_override='red')

    fig_scatter.update_traces(marker=dict(size=12, color='steelblue'))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation
    correlation = df['wage_gap_percent'].corr(df['gdp_per_capita'])
    st.metric("Correlation Coefficient (r)", f"{correlation:.4f}")

    if abs(correlation) > 0.5:
        st.info(f"**Strong {'positive' if correlation > 0 else 'negative'} relationship** (|r| > 0.5)")
    elif abs(correlation) > 0.3:
        st.info(f"**Moderate relationship** (0.3 < |r| < 0.5)")
    else:
        st.info(f"**Weak relationship** (|r| < 0.3)")

    # Outliers
    st.markdown("---")
    st.subheader("🎯 Identify Potential Outliers")

    # Z-score method
    df['z_score_wage_gap'] = np.abs(stats.zscore(df['wage_gap_percent']))
    df['z_score_gdp'] = np.abs(stats.zscore(df['gdp_per_capita']))

    outliers = df[(df['z_score_wage_gap'] > 2) | (df['z_score_gdp'] > 2)]

    if len(outliers) > 0:
        st.warning(f"**Found {len(outliers)} potential outliers** (Z-score > 2):")
        st.dataframe(outliers[['country_name', 'wage_gap_percent', 'gdp_per_capita', 'z_score_wage_gap', 'z_score_gdp']].style.format({
            'wage_gap_percent': '{:.1f}%',
            'gdp_per_capita': '${:,.0f}',
            'z_score_wage_gap': '{:.2f}',
            'z_score_gdp': '{:.2f}'
        }), hide_index=True)

        st.info("""
        **What to do about outliers:**
        - Investigate: Are they data errors or real?
        - Report results with AND without outliers
        - Use robust regression methods if many outliers
        """)
    else:
        st.success("✅ No major outliers detected")

    st.markdown("---")
    st.success("""
    **✅ Data Exploration Complete!**

    **What we learned:**
    - 27 EU countries in dataset
    - Wage gap ranges from 0.7% to 21.2%
    - GDP per capita varies widely
    - {'Negative' if correlation < 0 else 'Positive'} correlation between wage gap and GDP

    **Ready for regression!**
    """)

# ============================================================================
# TAB 3: FIRST REGRESSION
# ============================================================================
with tab3:
    st.header("📊 Your First OLS Regression")

    st.markdown("""
    ### Research Question:
    > **Does GDP per capita affect the gender wage gap in EU countries?**

    **Model Specification:**
    ```
    wage_gap_i = β₀ + β₁·GDP_per_capita_i + ε_i
    ```

    Where:
    - i = country index (1 to 27)
    - β₀ = intercept (wage gap when GDP = 0)
    - β₁ = slope (change in wage gap per $1 increase in GDP)
    - ε_i = error term
    """)

    # Variable selection
    st.markdown("---")
    st.subheader("Step 1: Choose Your Variables")

    col1, col2 = st.columns(2)

    with col1:
        y_var = st.selectbox(
            "Dependent Variable (Y):",
            ['wage_gap_percent'],
            help="What are you trying to explain?"
        )
        st.success("✅ wage_gap_percent selected")

    with col2:
        x_var = st.selectbox(
            "Independent Variable (X):",
            ['gdp_per_capita', 'gdp_per_capita_1000', 'log_gdp_pc', 'log_population'],
            help="What explains Y?"
        )

        if x_var == 'gdp_per_capita':
            st.info("GDP in dollars. β₁ will be very small (per $1 change)")
        elif x_var == 'gdp_per_capita_1000':
            st.info("GDP in thousands. β₁ easier to interpret (per $1000 change)")
        elif x_var == 'log_gdp_pc':
            st.info("Log transformation. β₁ = % change interpretation")
        else:
            st.info("Log population. Testing different hypothesis")

    # Run regression
    st.markdown("---")
    st.subheader("Step 2: Run OLS Regression")

    if st.button("▶️ RUN REGRESSION", type="primary", use_container_width=True):

        # Prepare data
        X = df[[x_var]].copy()
        X = sm.add_constant(X)  # Add intercept
        Y = df[y_var]

        # Fit OLS model
        model = sm.OLS(Y, X).fit()

        # Store in session state
        st.session_state.week1_progress['first_regression'] = True
        st.session_state['model'] = model
        st.session_state['X'] = X
        st.session_state['Y'] = Y
        st.session_state['x_var'] = x_var
        st.session_state['y_var'] = y_var

        st.success("✅ Regression complete! Results below:")
        st.balloons()

    # Show results if model exists
    if 'model' in st.session_state:
        model = st.session_state['model']
        x_var = st.session_state['x_var']

        st.markdown("---")
        st.subheader("📊 Regression Results")

        # Coefficients table
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Estimated Coefficients:**")

            results_df = pd.DataFrame({
                'Variable': ['Intercept (β₀)', f'{x_var} (β₁)'],
                'Coefficient': [model.params[0], model.params[1]],
                'Std Error': [model.bse[0], model.bse[1]],
                't-statistic': [model.tvalues[0], model.tvalues[1]],
                'p-value': [model.pvalues[0], model.pvalues[1]],
                '95% CI Lower': [model.conf_int().iloc[0, 0], model.conf_int().iloc[1, 0]],
                '95% CI Upper': [model.conf_int().iloc[0, 1], model.conf_int().iloc[1, 1]]
            })

            st.dataframe(results_df.style.format({
                'Coefficient': '{:.6f}',
                'Std Error': '{:.6f}',
                't-statistic': '{:.4f}',
                'p-value': '{:.4f}',
                '95% CI Lower': '{:.6f}',
                '95% CI Upper': '{:.6f}'
            }), hide_index=True, use_container_width=True)

        with col2:
            st.write("**Model Fit:**")
            st.metric("R-squared", f"{model.rsquared:.4f}")
            st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
            st.metric("F-statistic", f"{model.fvalue:.2f}")
            st.metric("Prob (F-stat)", f"{model.f_pvalue:.4f}")
            st.metric("Observations", f"{int(model.nobs)}")

        # Interpretation helper
        st.markdown("---")
        st.subheader("🎓 How to Interpret These Results")

        beta_0 = model.params[0]
        beta_1 = model.params[1]
        p_val = model.pvalues[1]
        r2 = model.rsquared

        with st.expander("**1. Intercept (β₀) Interpretation**", expanded=True):
            st.write(f"""
            **β₀ = {beta_0:.4f}**

            **Meaning:** When {x_var} = 0, the predicted wage gap is {beta_0:.2f}%.

            **Is this meaningful?**
            - Often not! (GDP = 0 is not realistic)
            - Intercept is mainly for prediction, not interpretation
            - We care more about β₁ (the slope)
            """)

        with st.expander("**2. Slope (β₁) Interpretation** ⭐ MOST IMPORTANT", expanded=True):
            st.write(f"""
            **β₁ = {beta_1:.6f}**

            **Meaning:**
            """)

            if 'log' in x_var and 'log' not in y_var:
                st.write(f"""
                - A 1% increase in {x_var.replace('log_', '')} is associated with a **{beta_1/100:.4f} percentage point** change in wage gap
                - (Semi-elasticity interpretation)
                """)
            elif 'log' not in x_var and 'log' not in y_var:
                if '1000' in x_var:
                    st.write(f"""
                    - A $1,000 increase in GDP per capita is associated with a **{beta_1:.4f} percentage point** {'increase' if beta_1 > 0 else 'decrease'} in wage gap
                    """)
                else:
                    st.write(f"""
                    - A $1 increase in GDP per capita is associated with a **{beta_1:.6f} percentage point** {'increase' if beta_1 > 0 else 'decrease'} in wage gap
                    - (Very small because GDP is in dollars!)
                    """)

            if beta_1 < 0:
                st.success("📉 Negative relationship: Higher GDP → Lower wage gap")
            else:
                st.warning("📈 Positive relationship: Higher GDP → Higher wage gap")

        with st.expander("**3. Statistical Significance (p-value)**", expanded=True):
            st.write(f"""
            **p-value = {p_val:.4f}**

            **Null Hypothesis (H₀):** β₁ = 0 (no relationship)
            **Alternative (H₁):** β₁ ≠ 0 (there is a relationship)

            **Decision rule:**
            - If p < 0.05: Reject H₀ → **Statistically significant** ✅
            - If p ≥ 0.05: Fail to reject H₀ → **Not significant** ❌

            **Your result:**
            """)

            if p_val < 0.01:
                st.success(f"""
                ✅ **Highly significant** (p = {p_val:.4f} < 0.01)

                Very strong evidence that {x_var} affects wage gap.
                We can be very confident this relationship is not due to chance.
                """)
            elif p_val < 0.05:
                st.success(f"""
                ✅ **Significant** (p = {p_val:.4f} < 0.05)

                Evidence that {x_var} affects wage gap.
                We reject the null hypothesis.
                """)
            elif p_val < 0.10:
                st.info(f"""
                ⚠️ **Marginally significant** (p = {p_val:.4f} < 0.10)

                Weak evidence of a relationship.
                Not significant at conventional 5% level.
                """)
            else:
                st.warning(f"""
                ❌ **Not significant** (p = {p_val:.4f} ≥ 0.10)

                No statistical evidence that {x_var} affects wage gap.
                Cannot reject null hypothesis.
                """)

        with st.expander("**4. R-squared (Model Fit)**", expanded=True):
            st.write(f"""
            **R² = {r2:.4f}**

            **Meaning:** {r2*100:.1f}% of the variation in wage gap is explained by {x_var}.

            **Interpretation:**
            """)

            if r2 > 0.7:
                st.success(f"✅ **Excellent fit** (R² > 0.7): Model explains most variation")
            elif r2 > 0.5:
                st.success(f"✅ **Good fit** (0.5 < R² < 0.7): Model explains substantial variation")
            elif r2 > 0.3:
                st.info(f"⚠️ **Moderate fit** (0.3 < R² < 0.5): Some explanatory power")
            elif r2 > 0.1:
                st.warning(f"❌ **Weak fit** (0.1 < R² < 0.3): Limited explanatory power")
            else:
                st.error(f"❌ **Very weak fit** (R² < 0.1): Almost no explanatory power")

            st.write(f"""
            **What this means:**
            - {r2*100:.1f}% explained by model
            - {(1-r2)*100:.1f}% explained by other factors (in error term)

            **Should you worry if R² is low?**
            - For prediction: Yes, low R² = poor predictions
            - For causal inference: Not necessarily!
              - Even low R² can give unbiased β₁
              - R² measures fit, not causality
            """)

        # Regression equation
        st.markdown("---")
        st.subheader("📝 Final Regression Equation")

        st.latex(f"\\text{{wage\\_gap}} = {beta_0:.4f} + {beta_1:.6f} \\cdot \\text{{{x_var}}} + \\epsilon")

        # Visualization
        st.markdown("---")
        st.subheader("📊 Visual Representation")

        # Create prediction line
        X_plot = df[[x_var]].copy()
        X_plot = sm.add_constant(X_plot)
        df['predicted'] = model.predict(X_plot)
        df['residual'] = df[y_var] - df['predicted']

        fig_reg = go.Figure()

        # Scatter points
        fig_reg.add_trace(go.Scatter(
            x=df[x_var],
            y=df[y_var],
            mode='markers',
            name='Actual Data',
            text=df['country_name'],
            marker=dict(size=12, color='steelblue'),
            hovertemplate='<b>%{text}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>'
        ))

        # Regression line
        df_sorted = df.sort_values(x_var)
        fig_reg.add_trace(go.Scatter(
            x=df_sorted[x_var],
            y=df_sorted['predicted'],
            mode='lines',
            name='OLS Regression Line',
            line=dict(color='red', width=3)
        ))

        fig_reg.update_layout(
            title='OLS Regression: Fitted vs Actual',
            xaxis_title=x_var,
            yaxis_title='Wage Gap (%)',
            hovermode='closest',
            showlegend=True
        )

        st.plotly_chart(fig_reg, use_container_width=True)

        st.info("""
        **How OLS works:**
        - OLS finds the line that minimizes the sum of squared residuals
        - Residual = vertical distance from point to line
        - Red line = "best fit" line through the data
        """)

# ============================================================================
# TAB 4: TEST ASSUMPTIONS
# ============================================================================
with tab4:
    st.header("🧪 Testing OLS Assumptions")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please run a regression first (go to 'First Regression' tab)")
    else:
        model = st.session_state['model']
        X = st.session_state['X']
        Y = st.session_state['Y']
        x_var = st.session_state['x_var']

        st.markdown("""
        Now we check if our regression is VALID by testing the 5 assumptions.
        """)

        # Get residuals and fitted values
        residuals = model.resid
        fitted = model.fittedvalues

        # Assumption 1: Linearity
        with st.expander("**✅ Assumption 1: Linearity** (Already satisfied)", expanded=False):
            st.write("""
            **Our model:**
            ```
            wage_gap = β₀ + β₁·X + ε
            ```

            This is linear in β₀ and β₁ ✅

            No test needed - we specified a linear model.
            """)

        # Assumption 2: Random sampling
        with st.expander("**✅ Assumption 2: Random Sampling** (Assumed true)", expanded=False):
            st.write(f"""
            **Our data:** {int(model.nobs)} EU countries in 2023

            **Is it random?**
            - All 27 EU member states included ✅
            - No selection bias (all countries represented)
            - Each country independent (not panel data)

            **Verdict:** Assumption satisfied ✅
            """)

        # Assumption 3: No perfect multicollinearity
        with st.expander("**✅ Assumption 3: No Multicollinearity** (Only 1 X variable)", expanded=False):
            st.write("""
            **Our model:** Only ONE independent variable

            Can't have multicollinearity with only one X! ✅

            **Note:** This becomes important when you add more variables
            (Week 2: multiple regression)
            """)

        # Assumption 4: Zero conditional mean
        with st.expander("**⚠️ Assumption 4: Zero Conditional Mean E[ε|X] = 0** (Think hard!)", expanded=True):
            st.write(f"""
            **This is THE critical assumption for causal inference!**

            **Question:** Is E[ε|{x_var}] = 0?

            **What could violate this?**
            """)

            st.markdown("""
            **1. Omitted Variables:**
            - Culture (gender norms, traditions)
            - Legal system (enforcement of equal pay laws)
            - Union strength
            - Education systems
            - Childcare policies

            **Are these correlated with GDP?** YES! ⚠️
            → Possible OVB (omitted variable bias)

            **2. Reverse Causality:**
            - Does GDP → wage gap? (what we want)
            - Or does wage gap → GDP? (women's participation affects GDP)
            - Probably BOTH! ⚠️

            **3. Measurement Error:**
            - Is wage gap measured correctly? (survey data quality)
            - Is GDP measured correctly? (informal economy)

            **Verdict:** ⚠️ **Likely VIOLATED**

            **What this means:**
            - Our β₁ estimate may be BIASED
            - Can't interpret as CAUSAL effect
            - This is CORRELATION, not causation

            **How to fix:** (You'll learn these!)
            - Week 3-4: Fixed Effects (control for time-invariant omitted variables)
            - Week 11-12: Instrumental Variables (find instrument for GDP)
            - Week 10: Difference-in-Differences (exploit policy changes)
            """)

            st.error("""
            🚨 **IMPORTANT LESSON:**

            Just because you CAN run OLS doesn't mean the results are CAUSAL!

            Always think critically about E[ε|X] = 0.
            This is what separates good econometrics from bad.
            """)

        # Assumption 5: Homoskedasticity
        with st.expander("**🧪 Assumption 5: Homoskedasticity** (Test visually + statistically)", expanded=True):
            st.write("""
            **Test:** Is Var(ε|X) constant?
            """)

            # Visual test: Residuals vs Fitted
            fig_resid = go.Figure()

            fig_resid.add_trace(go.Scatter(
                x=fitted,
                y=residuals,
                mode='markers',
                text=df['country_name'],
                marker=dict(size=10, color='steelblue'),
                hovertemplate='<b>%{text}</b><br>Fitted=%{x:.2f}<br>Residual=%{y:.2f}<extra></extra>'
            ))

            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")

            fig_resid.update_layout(
                title='Residuals vs Fitted Values (Test for Heteroskedasticity)',
                xaxis_title='Fitted Values',
                yaxis_title='Residuals',
                hovermode='closest'
            )

            st.plotly_chart(fig_resid, use_container_width=True)

            st.write("""
            **What to look for:**
            - ✅ Random scatter around zero → Homoskedastic
            - ❌ Funnel shape (fan out) → Heteroskedastic
            - ❌ Curved pattern → Nonlinearity problem
            """)

            # Statistical test: Breusch-Pagan
            st.markdown("---")
            st.subheader("Breusch-Pagan Test for Heteroskedasticity")

            # Run BP test
            bp_test = het_breuschpagan(residuals, X)
            bp_statistic = bp_test[0]
            bp_pvalue = bp_test[1]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("LM Statistic", f"{bp_statistic:.4f}")
                st.metric("p-value", f"{bp_pvalue:.4f}")

            with col2:
                st.write("""
                **H₀:** Homoskedasticity (constant variance)
                **H₁:** Heteroskedasticity
                """)

                if bp_pvalue < 0.05:
                    st.warning(f"""
                    ❌ **Reject H₀** (p = {bp_pvalue:.4f} < 0.05)

                    Evidence of heteroskedasticity!
                    Standard errors are likely wrong.
                    """)
                else:
                    st.success(f"""
                    ✅ **Fail to reject H₀** (p = {bp_pvalue:.4f} ≥ 0.05)

                    No evidence of heteroskedasticity.
                    Standard errors are likely correct.
                    """)

            # Fix with robust SEs
            st.markdown("---")
            st.subheader("💡 Solution: Robust Standard Errors")

            # Refit with robust SEs
            model_robust = sm.OLS(Y, X).fit(cov_type='HC3')

            comparison = pd.DataFrame({
                'Variable': ['Intercept', x_var],
                'OLS SE': [model.bse[0], model.bse[1]],
                'Robust SE': [model_robust.bse[0], model_robust.bse[1]],
                'OLS p-value': [model.pvalues[0], model.pvalues[1]],
                'Robust p-value': [model_robust.pvalues[0], model_robust.pvalues[1]]
            })

            st.dataframe(comparison.style.format({
                'OLS SE': '{:.6f}',
                'Robust SE': '{:.6f}',
                'OLS p-value': '{:.4f}',
                'Robust p-value': '{:.4f}'
            }), hide_index=True, use_container_width=True)

            st.success("""
            ✅ **Always use robust standard errors in practice!**

            They're valid whether or not you have heteroskedasticity.
            """)

        # Normality of residuals (bonus)
        with st.expander("**📊 Bonus: Normality of Residuals** (Not an assumption, but good to check)", expanded=False):
            st.write("""
            **Note:** OLS doesn't require normal residuals for unbiasedness!

            But normality helps with:
            - Small sample inference
            - Confidence intervals
            """)

            # Q-Q plot
            fig_qq = go.Figure()

            qq = stats.probplot(residuals, dist="norm")

            fig_qq.add_trace(go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name='Residuals',
                marker=dict(size=10, color='steelblue')
            ))

            fig_qq.add_trace(go.Scatter(
                x=qq[0][0],
                y=qq[1][1] + qq[1][0]*qq[0][0],
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash', width=2)
            ))

            fig_qq.update_layout(
                title='Q-Q Plot (Test for Normality)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles'
            )

            st.plotly_chart(fig_qq, use_container_width=True)

            st.write("""
            **What to look for:**
            - Points close to diagonal line → Normal
            - Points deviate → Non-normal

            **Our data:** With only 27 observations, some deviation is expected.
            """)

        # Mark as complete
        if st.button("✅ I've checked all assumptions"):
            st.session_state.week1_progress['assumptions_tested'] = True
            st.success("Great! Move to interpretation.")

# ============================================================================
# TAB 5: INTERPRETATION
# ============================================================================
with tab5:
    st.header("📝 How to Write Up Results")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please run a regression first")
    else:
        model = st.session_state['model']
        x_var = st.session_state['x_var']

        beta_1 = model.params[1]
        se = model.bse[1]
        t_stat = model.tvalues[1]
        p_val = model.pvalues[1]
        r2 = model.rsquared
        n = int(model.nobs)

        st.markdown("""
        **This is how you'd write it in an academic paper:**
        """)

        # Example paragraph
        st.markdown("---")
        st.subheader("📄 Results Section (Example)")

        st.info(f"""
        We estimate the relationship between GDP per capita and the gender wage gap using OLS regression on a cross-section of {n} EU countries in 2023. The estimated equation is:

        **wage_gap = {model.params[0]:.4f} + {beta_1:.6f}·{x_var}**

        The coefficient on {x_var} is {beta_1:.6f} (SE = {se:.6f}, t = {t_stat:.4f}, p = {p_val:.4f}). This indicates that a {'$1,000' if '1000' in x_var else '1%' if 'log' in x_var else '$1'} increase in GDP per capita is associated with a {abs(beta_1):.4f} percentage point {'decrease' if beta_1 < 0 else 'increase'} in the gender wage gap. This relationship is {'statistically significant at the 5% level' if p_val < 0.05 else 'not statistically significant'}. The model explains {r2*100:.1f}% of the variation in wage gaps across EU countries (R² = {r2:.4f}).

        **However, we cannot interpret this as a causal effect.** The regression likely suffers from omitted variable bias, as cultural norms, legal institutions, and labor market policies are not controlled for. These factors are correlated with both GDP and wage gaps, violating the zero conditional mean assumption (E[ε|GDP] = 0). Therefore, our estimate reflects correlation rather than causation.
        """)

        st.markdown("---")
        st.subheader("📊 Table for Paper")

        # Create publication-style table
        pub_table = pd.DataFrame({
            '': ['GDP per capita', '', 'Constant', '', 'Observations', 'R-squared', 'Adj. R-squared'],
            'Coefficient (SE)': [
                f"{beta_1:.6f}",
                f"({se:.6f})",
                f"{model.params[0]:.4f}",
                f"({model.bse[0]:.4f})",
                f"{n}",
                f"{r2:.4f}",
                f"{model.rsquared_adj:.4f}"
            ],
            'Significance': [
                '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else '',
                '',
                '***' if model.pvalues[0] < 0.01 else '**' if model.pvalues[0] < 0.05 else '*' if model.pvalues[0] < 0.10 else '',
                '',
                '',
                '',
                ''
            ]
        })

        st.dataframe(pub_table, hide_index=True, use_container_width=True)

        st.caption("Note: *** p<0.01, ** p<0.05, * p<0.10. Robust standard errors in parentheses.")

        st.markdown("---")
        st.subheader("🎓 Key Takeaways")

        st.success("""
        **What you learned:**

        1. ✅ How to run OLS regression
        2. ✅ How to interpret coefficients
        3. ✅ Statistical significance (p-values)
        4. ✅ Model fit (R²)
        5. ✅ Testing assumptions
        6. ⚠️ **Correlation ≠ Causation**

        **Next steps:**
        - Week 2: Multiple regression (control for more variables)
        - Week 3: Fixed Effects (control for unobserved country characteristics)
        - Week 11: Instrumental Variables (find causal effects)
        """)

        if st.button("✅ I understand how to interpret results"):
            st.session_state.week1_progress['interpretation_done'] = True
            st.balloons()
            st.success("Week 1 Complete! 🎉")

# ============================================================================
# TAB 6: QUIZ
# ============================================================================
with tab6:
    st.header("🎯 Week 1 Quiz & Practice")

    st.markdown("""
    Test your understanding before moving to Week 2!
    """)

    # Quiz questions
    st.subheader("📝 Quiz")

    with st.form("quiz_form"):
        st.markdown("**Question 1:** Which OLS assumption, if violated, causes BIAS in estimates?")
        q1 = st.radio("", [
            "A. Homoskedasticity",
            "B. No perfect multicollinearity",
            "C. Zero conditional mean E[ε|X] = 0",
            "D. Random sampling"
        ], key='q1')

        st.markdown("**Question 2:** If R² = 0.20, what does this mean?")
        q2 = st.radio("", [
            "A. The model explains 20% of variation in Y",
            "B. β₁ is biased",
            "C. The model is useless",
            "D. Standard errors are wrong"
        ], key='q2')

        st.markdown("**Question 3:** You find heteroskedasticity. What should you do?")
        q3 = st.radio("", [
            "A. Give up - the regression is invalid",
            "B. Use robust standard errors",
            "C. Drop outliers",
            "D. Transform Y with logarithm"
        ], key='q3')

        st.markdown("**Question 4:** The coefficient on log(GDP) is -2.5 with p-value = 0.03. What can you conclude?")
        q4 = st.radio("", [
            "A. Higher GDP causes lower wage gap (causal!)",
            "B. Higher GDP is associated with lower wage gap (correlation)",
            "C. The model is wrong",
            "D. Need more data"
        ], key='q4')

        st.markdown("**Question 5:** You include dummies for all 27 countries + intercept. What happens?")
        q5 = st.radio("", [
            "A. Everything is fine",
            "B. Perfect multicollinearity - can't estimate",
            "C. Heteroskedasticity",
            "D. Bias in β estimates"
        ], key='q5')

        submitted = st.form_submit_button("Submit Quiz")

        if submitted:
            score = 0

            if 'C' in q1:
                score += 1
                st.success("Q1: ✅ Correct! Zero conditional mean violation → BIAS")
            else:
                st.error("Q1: ❌ Wrong. Answer: C. Only Assumption 4 violation causes bias.")

            if 'A' in q2:
                score += 1
                st.success("Q2: ✅ Correct!")
            else:
                st.error("Q2: ❌ Wrong. Answer: A. R² is % of variation explained.")

            if 'B' in q3:
                score += 1
                st.success("Q3: ✅ Correct! Robust SEs are the solution.")
            else:
                st.error("Q3: ❌ Wrong. Answer: B. Use robust standard errors (easy fix!).")

            if 'B' in q4:
                score += 1
                st.success("Q4: ✅ Correct! Correlation, not causation (unless we have exogeneity).")
            else:
                st.error("Q4: ❌ Wrong. Answer: B. Can't claim causation without addressing E[ε|X]=0.")

            if 'B' in q5:
                score += 1
                st.success("Q5: ✅ Correct! Dummy variable trap!")
            else:
                st.error("Q5: ❌ Wrong. Answer: B. Perfect multicollinearity (sum of dummies = intercept).")

            st.markdown("---")
            st.metric("Your Score", f"{score}/5")

            if score == 5:
                st.balloons()
                st.success("🎉 Perfect score! You're ready for Week 2!")
            elif score >= 3:
                st.info("Good! Review the questions you missed and move forward.")
            else:
                st.warning("Review the theory tab and try again!")

    # Practice exercises
    st.markdown("---")
    st.subheader("💪 Practice Exercises")

    st.markdown("""
    **Exercise 1:** Run regression with log(population) as X instead of GDP. How do results change?

    **Exercise 2:** What happens if you add population AND GDP as X variables? (Week 2 preview!)

    **Exercise 3:** Download data for 2020-2023 and run panel regression (Week 3 preview!)

    **Exercise 4:** Read this paper and identify:
    - What's the dependent variable?
    - What's the main independent variable?
    - What assumptions might be violated?
    - How do authors address endogeneity?

    **Suggested reading:** Card & Krueger (1994) Minimum Wage paper
    """)

    # Week 1 completion
    st.markdown("---")
    if all(st.session_state.week1_progress.values()):
        st.success("""
        🎉 **WEEK 1 COMPLETE!**

        **You learned:**
        - ✅ 5 OLS assumptions
        - ✅ Running regression in Python
        - ✅ Interpreting coefficients, p-values, R²
        - ✅ Testing assumptions
        - ✅ Correlation vs causation

        **Next week:** Multiple regression + dummy variables

        **Deliverable:** Write a 2-page analysis of EU wage gap determinants using OLS
        """)
    else:
        incomplete = [v for k,v in progress_items.items() if not st.session_state.week1_progress[k]]
        st.info(f"""
        **Still to complete:**
        {chr(10).join(['- ' + item for item in incomplete])}
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Week 1 of 48: OLS Regression Fundamentals</strong></p>
    <p>Part of your 18-month PhD preparation journey</p>
    <p><em>Next: Week 2 - Multiple Regression & Dummy Variables</em></p>
</div>
""", unsafe_allow_html=True)
