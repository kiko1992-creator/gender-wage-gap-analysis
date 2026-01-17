"""
Bayesian Methods for Econometrics
Bayesian regression, hierarchical models, and posterior inference
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="Bayesian Methods", page_icon="🎲", layout="wide")

# ============================================================================
# PAGE HEADER
# ============================================================================
st.title("🎲 Bayesian Econometrics")
st.markdown("""
**Bayesian approach to statistical inference in economics**

Unlike frequentist methods (OLS, MLE), Bayesian methods:
- Treat parameters as random variables with probability distributions
- Update prior beliefs with data to get posterior distributions
- Provide probabilistic statements about parameters
- Handle uncertainty naturally through full posterior distributions
""")

# ============================================================================
# THEORY SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📚 Bayesian Theory")

    topic = st.selectbox(
        "Select topic:",
        ["Bayes' Theorem", "Prior vs Posterior", "Credible Intervals",
         "Hierarchical Models", "MCMC Sampling", "Bayesian vs Frequentist"]
    )

    if topic == "Bayes' Theorem":
        st.markdown("""
        ### Bayes' Theorem

        **Fundamental equation:**
        ```
        P(θ|Data) = P(Data|θ) × P(θ) / P(Data)
        ```

        Or more intuitively:
        ```
        Posterior ∝ Likelihood × Prior
        ```

        **Components:**
        - **Prior:** P(θ) - beliefs before seeing data
        - **Likelihood:** P(Data|θ) - model
        - **Posterior:** P(θ|Data) - updated beliefs

        **Example:**
        - Prior: β ~ N(0, 10²)
        - Likelihood: Y|β ~ N(Xβ, σ²)
        - Posterior: β|Y ~ N(μ_post, Σ_post)
        """)

    elif topic == "Prior vs Posterior":
        st.markdown("""
        ### Choosing Priors

        **Non-informative (vague):**
        - β ~ N(0, 1000²)
        - Lets data dominate
        - Similar to frequentist

        **Informative:**
        - β ~ N(0.5, 0.1²)
        - Incorporates expert knowledge
        - Regularizes estimates

        **Conjugate:**
        - Prior and posterior same family
        - Normal-Normal, Beta-Binomial
        - Allows analytical solutions

        **Empirical Bayes:**
        - Estimate prior from data
        - Hybrid approach
        """)

    elif topic == "Credible Intervals":
        st.markdown("""
        ### Credible vs Confidence Intervals

        **Credible Interval (Bayesian):**
        ```
        P(β ∈ [a,b] | Data) = 0.95
        ```
        "There's 95% probability β is in [a,b]"

        **Confidence Interval (Frequentist):**
        ```
        P([a,b] contains β) = 0.95
        ```
        "If we repeat, 95% of intervals contain β"

        **Bayesian advantage:**
        - Direct probability statements!
        - More intuitive interpretation
        """)

    elif topic == "Hierarchical Models":
        st.markdown("""
        ### Hierarchical (Multilevel) Models

        **Idea:** Parameters themselves have distributions

        **Example (panel data):**
        ```
        Level 1: Y_it ~ N(α_i + X_it'β, σ²)
        Level 2: α_i ~ N(μ_α, τ²)
        ```

        **Advantages:**
        - Partial pooling (borrow strength)
        - Handle clustered data
        - Shrinkage toward group mean

        **vs Fixed Effects:**
        - FE: Treat α_i as fixed constants
        - Hierarchical: α_i drawn from distribution
        """)

    elif topic == "MCMC Sampling":
        st.markdown("""
        ### Markov Chain Monte Carlo

        **Problem:** Posterior too complex for analytical solution

        **Solution:** Draw samples from posterior

        **Algorithms:**
        1. **Metropolis-Hastings**
           - Propose new value
           - Accept/reject based on ratio

        2. **Gibbs Sampling**
           - Sample each parameter conditional on others
           - Requires conjugate priors

        3. **Hamiltonian Monte Carlo (HMC)**
           - Uses gradients (faster mixing)
           - Default in Stan/PyMC

        **Diagnostics:**
        - Trace plots (visual inspection)
        - R-hat < 1.1 (convergence)
        - Effective sample size > 1000
        """)

    elif topic == "Bayesian vs Frequentist":
        st.markdown("""
        ### Paradigm Comparison

        | Aspect | Frequentist | Bayesian |
        |--------|-------------|----------|
        | Parameters | Fixed (unknown) | Random |
        | Inference | Long-run frequency | Probability |
        | Prior | Not used | Required |
        | Uncertainty | SE, CI | Posterior distribution |
        | Regularization | Ad-hoc (LASSO) | Natural (prior) |

        **When to use Bayesian:**
        - Small sample sizes
        - Need to incorporate prior info
        - Want probability statements
        - Hierarchical/complex models

        **When to use Frequentist:**
        - Large samples
        - No prior knowledge
        - Want objective analysis
        - Computational constraints
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
    "📊 Bayesian Regression",
    "🏗️ Hierarchical Model",
    "🎯 Prior Sensitivity",
    "📈 Posterior Inference"
])

# ============================================================================
# TAB 1: BAYESIAN REGRESSION
# ============================================================================
with tab1:
    st.header("📊 Bayesian Linear Regression")

    st.markdown("""
    ### Analytical Bayesian Regression (Conjugate Priors)

    **Model:**
    ```
    Y = Xβ + ε, where ε ~ N(0, σ²I)
    ```

    **Prior:**
    ```
    β ~ N(β₀, Σ₀)  (multivariate normal)
    ```

    **Posterior (analytical):**
    ```
    β|Y ~ N(β_post, Σ_post)

    where:
    Σ_post = (Σ₀⁻¹ + X'X/σ²)⁻¹
    β_post = Σ_post(Σ₀⁻¹β₀ + X'Y/σ²)
    ```

    **Special case:** If Σ₀ = ∞·I (vague prior), then β_post → OLS estimate
    """)

    # Prepare data
    df_bayes = df_panel[df_panel['year'] == 2023].dropna(subset=['wage_gap', 'gdp_per_capita']).copy()
    df_bayes['log_gdp_pc'] = np.log(df_bayes['gdp_per_capita'])

    X_bayes = df_bayes[['log_gdp_pc']].values
    X_bayes = np.column_stack([np.ones(len(X_bayes)), X_bayes])  # Add intercept
    Y_bayes = df_bayes['wage_gap'].values

    n, p = X_bayes.shape

    # OLS for comparison
    model_ols = sm.OLS(Y_bayes, X_bayes).fit()
    beta_ols = model_ols.params
    sigma2_ols = model_ols.mse_resid

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prior Specification")

        st.write("**Choose prior for β:**")

        prior_mean_intercept = st.number_input("Prior mean (intercept):", value=10.0, step=1.0)
        prior_mean_slope = st.number_input("Prior mean (slope):", value=0.0, step=0.5)
        prior_sd = st.number_input("Prior std dev (both):", value=10.0, step=1.0, min_value=0.1)

        # Prior parameters
        beta_0 = np.array([prior_mean_intercept, prior_mean_slope])
        Sigma_0 = np.eye(p) * (prior_sd ** 2)

        st.write("**Prior distribution:**")
        st.latex(r"\beta \sim N\left(\begin{bmatrix}" + f"{prior_mean_intercept:.1f} \\\\ {prior_mean_slope:.1f}" + r"\end{bmatrix}, " +
                 f"{prior_sd:.1f}^2 I" + r"\right)")

    with col2:
        st.subheader("Posterior Distribution")

        # Compute posterior
        Sigma_0_inv = np.linalg.inv(Sigma_0)
        XtX = X_bayes.T @ X_bayes
        XtY = X_bayes.T @ Y_bayes

        Sigma_post = np.linalg.inv(Sigma_0_inv + XtX / sigma2_ols)
        beta_post = Sigma_post @ (Sigma_0_inv @ beta_0 + XtY / sigma2_ols)

        st.write("**Posterior mean:**")
        st.write(f"- Intercept: {beta_post[0]:.4f}")
        st.write(f"- Slope: {beta_post[1]:.4f}")

        st.write("**Posterior std dev:**")
        post_sd = np.sqrt(np.diag(Sigma_post))
        st.write(f"- Intercept: {post_sd[0]:.4f}")
        st.write(f"- Slope: {post_sd[1]:.4f}")

        st.metric("Posterior correlation", f"{Sigma_post[0,1] / (post_sd[0] * post_sd[1]):.4f}")

    # Visualization
    st.markdown("---")
    st.subheader("📊 Prior vs Posterior vs OLS")

    comparison_bayes = pd.DataFrame({
        'Parameter': ['Intercept', 'Slope'],
        'Prior Mean': beta_0,
        'Posterior Mean': beta_post,
        'OLS Estimate': beta_ols,
        'Posterior SD': post_sd
    })

    st.dataframe(comparison_bayes.style.format({
        'Prior Mean': '{:.4f}',
        'Posterior Mean': '{:.4f}',
        'OLS Estimate': '{:.4f}',
        'Posterior SD': '{:.4f}'
    }), hide_index=True)

    # Plot posterior distribution for slope
    st.markdown("---")
    st.subheader("Posterior Distribution of Slope")

    # Generate samples from posterior
    n_samples = 10000
    beta_samples = np.random.multivariate_normal(beta_post, Sigma_post, n_samples)

    fig_post = go.Figure()

    # Posterior histogram
    fig_post.add_trace(go.Histogram(
        x=beta_samples[:, 1],
        name='Posterior',
        nbinsx=50,
        histnorm='probability density',
        marker_color='steelblue',
        opacity=0.7
    ))

    # Prior distribution
    x_range = np.linspace(beta_samples[:, 1].min(), beta_samples[:, 1].max(), 100)
    prior_density = stats.norm.pdf(x_range, prior_mean_slope, prior_sd)

    fig_post.add_trace(go.Scatter(
        x=x_range,
        y=prior_density,
        name='Prior',
        line=dict(color='red', width=2, dash='dash')
    ))

    # OLS point estimate
    fig_post.add_vline(x=beta_ols[1], line_dash="dot", line_color="green",
                      annotation_text="OLS", annotation_position="top")

    # Posterior mean
    fig_post.add_vline(x=beta_post[1], line_dash="solid", line_color="blue",
                      annotation_text="Posterior Mean", annotation_position="top right")

    fig_post.update_layout(
        title='Prior vs Posterior Distribution (Slope Parameter)',
        xaxis_title='β₁ (effect of log(GDP pc))',
        yaxis_title='Density',
        showlegend=True
    )

    st.plotly_chart(fig_post, use_container_width=True)

    # Credible interval
    credible_interval = np.percentile(beta_samples[:, 1], [2.5, 97.5])

    st.write(f"**95% Credible Interval:** [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]")

    st.success(f"""
    **Interpretation:**

    There is a 95% probability that the true slope parameter lies between
    {credible_interval[0]:.4f} and {credible_interval[1]:.4f}.

    This is a **direct probability statement** about the parameter!
    (Unlike frequentist confidence intervals)
    """)

# ============================================================================
# TAB 2: HIERARCHICAL MODEL
# ============================================================================
with tab2:
    st.header("🏗️ Hierarchical (Multilevel) Model")

    st.markdown("""
    ### Two-Level Hierarchical Model for Panel Data

    **Level 1 (within-country):**
    ```
    Y_it = α_i + β·X_it + ε_it
    ε_it ~ N(0, σ²)
    ```

    **Level 2 (between-country):**
    ```
    α_i ~ N(μ_α, τ²)
    ```

    **Hyperpriors:**
    ```
    μ_α ~ N(0, 100)
    τ² ~ InvGamma(1, 1)
    ```

    **Partial pooling:** Each α_i is shrunk toward μ_α based on data quality
    """)

    st.info("""
    **Note:** Full hierarchical Bayesian models require MCMC sampling.

    For illustration, we use **Empirical Bayes** (estimate hyperparameters from data):
    1. Estimate α_i for each country separately (no pooling)
    2. Estimate μ_α, τ² from the α_i's
    3. Shrink each α_i toward μ_α
    """)

    # Estimate country-specific intercepts
    country_intercepts = []

    for country in df_panel['country'].unique():
        df_country = df_panel[(df_panel['country'] == country) &
                              (df_panel['gdp_per_capita'].notna())].copy()

        if len(df_country) >= 3:
            df_country['log_gdp_pc'] = np.log(df_country['gdp_per_capita'])
            X_country = sm.add_constant(df_country['log_gdp_pc'])
            y_country = df_country['wage_gap']

            try:
                model_country = sm.OLS(y_country, X_country).fit()
                alpha_i = model_country.params[0]
                se_i = model_country.bse[0]

                country_intercepts.append({
                    'country': country,
                    'alpha_unpooled': alpha_i,
                    'se': se_i,
                    'n_obs': len(df_country)
                })
            except:
                pass

    df_intercepts = pd.DataFrame(country_intercepts)

    # Estimate hyperparameters
    mu_alpha = df_intercepts['alpha_unpooled'].mean()
    tau_sq = df_intercepts['alpha_unpooled'].var()

    # Shrinkage (James-Stein estimator)
    df_intercepts['shrinkage_factor'] = 1 - (df_intercepts['se'] ** 2) / \
                                        (df_intercepts['se'] ** 2 + tau_sq)
    df_intercepts['alpha_pooled'] = mu_alpha + df_intercepts['shrinkage_factor'] * \
                                   (df_intercepts['alpha_unpooled'] - mu_alpha)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Shrinkage Plot")

        fig_shrink = go.Figure()

        fig_shrink.add_trace(go.Scatter(
            x=df_intercepts['alpha_unpooled'],
            y=df_intercepts['alpha_pooled'],
            mode='markers+text',
            text=df_intercepts['country'],
            textposition='top center',
            marker=dict(size=10, color='steelblue'),
            name='Countries'
        ))

        # 45-degree line
        min_val = min(df_intercepts['alpha_unpooled'].min(), df_intercepts['alpha_pooled'].min())
        max_val = max(df_intercepts['alpha_unpooled'].max(), df_intercepts['alpha_pooled'].max())

        fig_shrink.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='No Shrinkage'
        ))

        # Grand mean
        fig_shrink.add_hline(y=mu_alpha, line_dash="dot", line_color="green",
                            annotation_text=f"Grand Mean (μ_α={mu_alpha:.2f})")

        fig_shrink.update_layout(
            title='Hierarchical Shrinkage',
            xaxis_title='Unpooled Estimate (α_i)',
            yaxis_title='Pooled Estimate (Shrunk toward μ_α)',
            hovermode='closest'
        )

        st.plotly_chart(fig_shrink, use_container_width=True)

        st.info("""
        **Interpretation:**
        - Points below red line: Shrunk down toward grand mean
        - Points above red line: Shrunk up toward grand mean
        - More shrinkage for countries with less data (larger SE)
        """)

    with col2:
        st.subheader("Hyperparameters")

        st.metric("Grand Mean (μ_α)", f"{mu_alpha:.2f}")
        st.metric("Between-Country Var (τ²)", f"{tau_sq:.2f}")
        st.metric("Within-Country Var (σ²)", f"{sigma2_ols:.2f}")

        # Intraclass correlation
        icc = tau_sq / (tau_sq + sigma2_ols)
        st.metric("Intraclass Correlation", f"{icc:.4f}")

        st.markdown("""
        **ICC Interpretation:**
        - How much variation is between-country vs within-country
        - ICC ≈ 0: All variation within countries
        - ICC ≈ 1: All variation between countries
        """)

    # Show table
    st.markdown("---")
    st.subheader("Country-Specific Estimates")

    display_cols = ['country', 'alpha_unpooled', 'alpha_pooled', 'shrinkage_factor', 'n_obs']
    st.dataframe(df_intercepts[display_cols].sort_values('shrinkage_factor').style.format({
        'alpha_unpooled': '{:.2f}',
        'alpha_pooled': '{:.2f}',
        'shrinkage_factor': '{:.4f}'
    }), hide_index=True)

# ============================================================================
# TAB 3: PRIOR SENSITIVITY
# ============================================================================
with tab3:
    st.header("🎯 Prior Sensitivity Analysis")

    st.markdown("""
    ### Robustness to Prior Choice

    **Question:** How much do results depend on our prior?

    **Good practice:**
    1. Try multiple priors
    2. Check if posterior is similar
    3. If priors disagree, data is weak → need more data

    **This tab:** Compare vague vs informative vs skeptical priors
    """)

    # Define three different priors
    priors = {
        'Vague (Non-informative)': {
            'beta_0': np.array([10.0, 0.0]),
            'Sigma_0': np.eye(2) * 1000,
            'color': 'blue'
        },
        'Informative': {
            'beta_0': np.array([12.0, -1.0]),  # Expect negative relationship
            'Sigma_0': np.eye(2) * 4,
            'color': 'green'
        },
        'Skeptical': {
            'beta_0': np.array([10.0, 0.5]),  # Expect positive relationship
            'Sigma_0': np.eye(2) * 4,
            'color': 'red'
        }
    }

    # Compute posteriors for each prior
    posteriors = {}

    for prior_name, prior_params in priors.items():
        beta_0 = prior_params['beta_0']
        Sigma_0 = prior_params['Sigma_0']

        Sigma_0_inv = np.linalg.inv(Sigma_0)
        Sigma_post = np.linalg.inv(Sigma_0_inv + XtX / sigma2_ols)
        beta_post = Sigma_post @ (Sigma_0_inv @ beta_0 + XtY / sigma2_ols)

        posteriors[prior_name] = {
            'mean': beta_post,
            'cov': Sigma_post,
            'color': prior_params['color']
        }

    # Comparison table
    comparison_prior_sens = []

    for prior_name, post_params in posteriors.items():
        post_sd = np.sqrt(np.diag(post_params['cov']))
        comparison_prior_sens.append({
            'Prior': prior_name,
            'Intercept (Post)': post_params['mean'][0],
            'Slope (Post)': post_params['mean'][1],
            'Slope SD': post_sd[1]
        })

    df_prior_comp = pd.DataFrame(comparison_prior_sens)

    st.subheader("Posterior Estimates Under Different Priors")

    st.dataframe(df_prior_comp.style.format({
        'Intercept (Post)': '{:.4f}',
        'Slope (Post)': '{:.4f}',
        'Slope SD': '{:.4f}'
    }), hide_index=True)

    # Plot all posteriors
    st.markdown("---")
    st.subheader("Posterior Distributions (Slope)")

    fig_sens = go.Figure()

    for prior_name, post_params in posteriors.items():
        # Sample from posterior
        samples = np.random.multivariate_normal(post_params['mean'], post_params['cov'], 5000)

        fig_sens.add_trace(go.Histogram(
            x=samples[:, 1],
            name=prior_name,
            nbinsx=40,
            histnorm='probability density',
            opacity=0.6,
            marker_color=post_params['color']
        ))

    # OLS estimate
    fig_sens.add_vline(x=beta_ols[1], line_dash="dash", line_color="black",
                      annotation_text="OLS")

    fig_sens.update_layout(
        title='Posterior Sensitivity to Prior Choice',
        xaxis_title='β₁ (Slope)',
        yaxis_title='Density',
        barmode='overlay'
    )

    st.plotly_chart(fig_sens, use_container_width=True)

    # Analysis
    slope_range = df_prior_comp['Slope (Post)'].max() - df_prior_comp['Slope (Post)'].min()

    if slope_range < 0.5:
        st.success(f"""
        ✅ **Robust to prior choice** (range = {slope_range:.4f})

        All three priors lead to similar posteriors.
        → Data is informative enough to dominate the prior.
        """)
    else:
        st.warning(f"""
        ⚠️ **Sensitive to prior choice** (range = {slope_range:.4f})

        Different priors lead to different posteriors.
        → Data is weak, need more observations or better instruments.
        """)

# ============================================================================
# TAB 4: POSTERIOR INFERENCE
# ============================================================================
with tab4:
    st.header("📈 Posterior Inference & Hypothesis Testing")

    st.markdown("""
    ### Bayesian Hypothesis Testing

    **Frequentist:** p-value, reject/fail to reject H₀

    **Bayesian:** Compute probability that hypothesis is true

    **Example:** What's the probability that β₁ < 0 (GDP reduces wage gap)?
    """)

    # Use vague prior posterior
    post_mean = posteriors['Vague (Non-informative)']['mean']
    post_cov = posteriors['Vague (Non-informative)']['cov']

    # Sample from posterior
    n_samples_inference = 100000
    beta_samples_inf = np.random.multivariate_normal(post_mean, post_cov, n_samples_inference)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hypothesis: β₁ < 0")

        # Probability that slope is negative
        prob_negative = np.mean(beta_samples_inf[:, 1] < 0)

        st.metric("P(β₁ < 0 | Data)", f"{prob_negative:.4f}")

        if prob_negative > 0.95:
            st.success(f"""
            ✅ **Strong evidence** (prob = {prob_negative:.4f})

            Very high probability that GDP per capita reduces wage gap.
            """)
        elif prob_negative > 0.5:
            st.info(f"""
            ⚠️ **Weak evidence** (prob = {prob_negative:.4f})

            More likely negative than positive, but not conclusive.
            """)
        else:
            st.warning(f"""
            ❌ **No evidence** (prob = {prob_negative:.4f})

            More likely positive than negative.
            """)

    with col2:
        st.subheader("Hypothesis: |β₁| > 1")

        # Probability that effect is large in magnitude
        prob_large = np.mean(np.abs(beta_samples_inf[:, 1]) > 1)

        st.metric("P(|β₁| > 1 | Data)", f"{prob_large:.4f}")

        st.markdown("""
        **Interpretation:**

        Probability that a 1-unit increase in log(GDP) changes wage gap by more than 1 percentage point.
        """)

    # Posterior predictive distribution
    st.markdown("---")
    st.subheader("📊 Posterior Predictive Distribution")

    st.markdown("""
    **Question:** What wage gap would we predict for a new country with log(GDP pc) = 11?

    **Posterior predictive:**
    ```
    P(Y_new | Data) = ∫ P(Y_new | β, σ²) P(β, σ² | Data) dβ dσ²
    ```

    Integrates out parameter uncertainty.
    """)

    # New X value
    X_new = np.array([1, 11])  # Intercept + log(GDP) = 11

    # Predictive samples
    y_pred_samples = []

    for beta_sample in beta_samples_inf[:1000]:  # Use 1000 samples
        y_pred_mean = X_new @ beta_sample
        y_pred = np.random.normal(y_pred_mean, np.sqrt(sigma2_ols))
        y_pred_samples.append(y_pred)

    y_pred_samples = np.array(y_pred_samples)

    # Predictive interval
    pred_interval = np.percentile(y_pred_samples, [2.5, 97.5])
    pred_mean = np.mean(y_pred_samples)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Mean", f"{pred_mean:.2f}%")

    with col2:
        st.metric("95% Pred. Interval Lower", f"{pred_interval[0]:.2f}%")

    with col3:
        st.metric("95% Pred. Interval Upper", f"{pred_interval[1]:.2f}%")

    # Plot predictive distribution
    fig_pred = go.Figure()

    fig_pred.add_trace(go.Histogram(
        x=y_pred_samples,
        nbinsx=50,
        histnorm='probability density',
        marker_color='purple',
        opacity=0.7,
        name='Posterior Predictive'
    ))

    fig_pred.add_vline(x=pred_mean, line_color='red', line_dash='dash',
                      annotation_text='Mean')

    fig_pred.update_layout(
        title=f'Posterior Predictive Distribution (log(GDP pc) = 11)',
        xaxis_title='Predicted Wage Gap (%)',
        yaxis_title='Density'
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    st.success(f"""
    **Interpretation:**

    For a country with log(GDP per capita) = 11, we predict:
    - **Expected wage gap:** {pred_mean:.2f}%
    - **95% prediction interval:** [{pred_interval[0]:.2f}%, {pred_interval[1]:.2f}%]

    This interval accounts for both **parameter uncertainty** (β) and **residual uncertainty** (σ²).
    Wider than credible interval because it includes σ²!
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Bayesian Econometrics for PhD Research</strong></p>
    <p>Prior • Posterior • Hierarchical Models • Sensitivity Analysis • Inference</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📚 Essential References:

1. **Gelman et al. (2013)** - "Bayesian Data Analysis" (3rd ed) - THE textbook
2. **Kruschke (2014)** - "Doing Bayesian Data Analysis" (Puppies book - very accessible)
3. **McElreath (2020)** - "Statistical Rethinking" (2nd ed) - Modern Bayesian thinking
4. **Koop (2003)** - "Bayesian Econometrics"
5. **Lancaster (2004)** - "An Introduction to Modern Bayesian Econometrics"
6. **Greenberg (2012)** - "Introduction to Bayesian Econometrics" (2nd ed)

### 🐍 Python Packages for MCMC:

- **PyMC** - Probabilistic programming (HMC, NUTS)
- **Stan** (via PyStan) - Industry standard for Bayesian inference
- **emcee** - Ensemble MCMC sampler (simpler interface)
- **ArviZ** - Exploratory analysis of Bayesian models
""")
