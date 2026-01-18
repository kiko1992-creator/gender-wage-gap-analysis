"""
Machine Learning for Economics
Double ML, Causal Forests, LASSO, and high-dimensional methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from database_connection import get_all_countries_2023, get_country_trend

st.set_page_config(page_title="ML for Economics", page_icon="🤖", layout="wide")

# ============================================================================
# PAGE HEADER
# ============================================================================
st.title("🤖 Machine Learning for Economics")
st.markdown("""
**Modern ML methods for causal inference and prediction**

This page implements cutting-edge ML techniques adapted for economic research,
particularly focused on causal inference and heterogeneous treatment effects.
""")

# ============================================================================
# THEORY SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📚 ML Methods Guide")

    method = st.selectbox(
        "Select method:",
        ["Overview", "LASSO/Ridge", "Double ML", "Causal Forests",
         "SHAP Values", "Cross-Validation"]
    )

    if method == "Overview":
        st.markdown("""
        ### ML for Causal Inference

        **Traditional ML Problem:**
        - Maximize prediction accuracy
        - All variables treated equally

        **Econometric Problem:**
        - Identify causal effects
        - Some variables are "special" (treatment)

        **Solution:** Hybrid methods
        - Use ML for nuisance parameters
        - Preserve causal interpretation
        """)

    elif method == "LASSO/Ridge":
        st.markdown("""
        ### Regularized Regression

        **Ridge (L2):**
        ```
        min ||Y - Xβ||² + λ||β||²
        ```
        Shrinks coefficients toward zero

        **LASSO (L1):**
        ```
        min ||Y - Xβ||² + λ||β||₁
        ```
        Sets some coefficients exactly to zero
        → Automatic variable selection

        **Elastic Net:**
        ```
        min ||Y - Xβ||² + λ₁||β||₁ + λ₂||β||²
        ```
        Combines both penalties

        **Use when:** p >> N (many predictors)
        """)

    elif method == "Double ML":
        st.markdown("""
        ### Double/Debiased Machine Learning

        **Reference:** Chernozhukov et al. (2018)

        **Problem:** Using ML directly for causal inference
        → biased estimates (regularization bias)

        **Solution:** Orthogonalization
        1. Predict Y using controls (ML)
        2. Predict D (treatment) using controls (ML)
        3. Regress residuals: Ỹ ~ D̃

        **Result:** √N-consistent, asymptotically normal

        **Allows:** Any ML method (Random Forest, GBM, Neural Nets)
        """)

    elif method == "Causal Forests":
        st.markdown("""
        ### Causal Forest (Athey & Imbens 2016)

        **Idea:** Random Forest adapted for treatment effects

        **Each tree:**
        1. Split on covariates X
        2. Estimate τ(x) = E[Y(1) - Y(0) | X=x]

        **Advantages:**
        - Heterogeneous treatment effects
        - Non-parametric
        - Honest inference (sample splitting)

        **Output:** CATE = Conditional Average Treatment Effect

        **Python:** `econml.dml.CausalForestDML`
        """)

    elif method == "SHAP Values":
        st.markdown("""
        ### SHAP (SHapley Additive exPlanations)

        **Problem:** ML models are black boxes

        **Solution:** Game-theoretic feature importance

        **SHAP value for feature i:**
        ```
        φ_i = Σ [weight × (f(S ∪ {i}) - f(S))]
        ```

        Over all possible coalitions S

        **Properties:**
        - Local interpretability
        - Global feature importance
        - Fair allocation (Shapley values)

        **Use:** Explain which factors drive wage gaps
        """)

    elif method == "Cross-Validation":
        st.markdown("""
        ### Cross-Validation for Hyperparameter Tuning

        **K-Fold CV:**
        1. Split data into K folds
        2. Train on K-1, test on 1
        3. Repeat K times
        4. Average performance

        **Metrics:**
        - MSE (regression)
        - R² (explained variance)
        - MAE (robust to outliers)

        **For λ selection:**
        - Plot validation error vs λ
        - Choose λ minimizing CV error
        - 1-SE rule (simpler model)
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
    "📉 LASSO/Ridge Regression",
    "🎯 Double Machine Learning",
    "🌲 Random Forest & Feature Importance",
    "📊 Model Comparison"
])

# ============================================================================
# TAB 1: LASSO/RIDGE
# ============================================================================
with tab1:
    st.header("📉 Regularized Regression: LASSO & Ridge")

    st.markdown("""
    ### High-Dimensional Regression

    **Problem:** When you have many potential predictors, OLS can overfit

    **Solution:** Add penalty term to shrink coefficients

    **This example:** Predict wage gap using GDP, population, and polynomial/interaction terms
    """)

    # Prepare data with polynomial features
    df_ml = df_2023.dropna(subset=['wage_gap_percent', 'gdp_billions', 'population']).copy()
    df_ml['gdp_per_capita'] = (df_ml['gdp_billions'] * 1e9) / df_ml['population']
    df_ml['log_gdp_pc'] = np.log(df_ml['gdp_per_capita'])
    df_ml['log_pop'] = np.log(df_ml['population'])

    # Create polynomial and interaction features
    df_ml['log_gdp_pc_sq'] = df_ml['log_gdp_pc'] ** 2
    df_ml['log_pop_sq'] = df_ml['log_pop'] ** 2
    df_ml['gdp_pop_interact'] = df_ml['log_gdp_pc'] * df_ml['log_pop']

    # Feature matrix
    feature_cols = ['log_gdp_pc', 'log_pop', 'log_gdp_pc_sq', 'log_pop_sq', 'gdp_pop_interact']
    X = df_ml[feature_cols].values
    y = df_ml['wage_gap_percent'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 LASSO Regression (L1)")

        # LASSO with cross-validation
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso_cv.fit(X_scaled, y)

        st.write(f"**Optimal λ (alpha):** {lasso_cv.alpha_:.4f}")
        st.write(f"**Cross-validated R²:** {lasso_cv.score(X_scaled, y):.4f}")

        # Coefficients
        lasso_coefs = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': lasso_cv.coef_,
            'Abs Coefficient': np.abs(lasso_cv.coef_)
        }).sort_values('Abs Coefficient', ascending=False)

        st.write("**LASSO Coefficients:**")
        st.dataframe(lasso_coefs.style.format({
            'Coefficient': '{:.4f}',
            'Abs Coefficient': '{:.4f}'
        }).background_gradient(subset=['Abs Coefficient'], cmap='Blues'), hide_index=True)

        # Count non-zero
        n_nonzero = np.sum(lasso_coefs['Coefficient'] != 0)
        st.metric("Non-zero coefficients", f"{n_nonzero}/{len(feature_cols)}")

        st.success(f"""
        ✅ **LASSO performed variable selection**

        Set {len(feature_cols) - n_nonzero} coefficients to exactly zero.
        This simplifies the model and prevents overfitting.
        """)

    with col2:
        st.subheader("🔹 Ridge Regression (L2)")

        # Ridge with different alphas
        alphas = np.logspace(-2, 2, 20)
        ridge_scores = []

        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
            ridge_scores.append(scores.mean())

        # Best alpha
        best_idx = np.argmax(ridge_scores)
        best_alpha = alphas[best_idx]

        st.write(f"**Optimal λ (alpha):** {best_alpha:.4f}")
        st.write(f"**Cross-validated R²:** {ridge_scores[best_idx]:.4f}")

        # Fit final model
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_scaled, y)

        # Coefficients
        ridge_coefs = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': ridge.coef_,
            'Abs Coefficient': np.abs(ridge.coef_)
        }).sort_values('Abs Coefficient', ascending=False)

        st.write("**Ridge Coefficients:**")
        st.dataframe(ridge_coefs.style.format({
            'Coefficient': '{:.4f}',
            'Abs Coefficient': '{:.4f}'
        }).background_gradient(subset=['Abs Coefficient'], cmap='Greens'), hide_index=True)

        st.info("""
        ℹ️ **Ridge shrinks all coefficients**

        But doesn't set any exactly to zero.
        Better when all features are relevant.
        """)

    # Regularization path
    st.markdown("---")
    st.subheader("📈 Regularization Path")

    col1, col2 = st.columns(2)

    with col1:
        # LASSO path
        alphas_path = np.logspace(-2, 1, 50)
        coefs_lasso = []

        for alpha in alphas_path:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_scaled, y)
            coefs_lasso.append(lasso.coef_)

        coefs_lasso = np.array(coefs_lasso)

        fig_lasso_path = go.Figure()

        for i, feat in enumerate(feature_cols):
            fig_lasso_path.add_trace(go.Scatter(
                x=alphas_path,
                y=coefs_lasso[:, i],
                mode='lines',
                name=feat,
                line=dict(width=2)
            ))

        fig_lasso_path.update_layout(
            title='LASSO Regularization Path',
            xaxis_title='λ (alpha)',
            yaxis_title='Coefficient',
            xaxis_type='log',
            hovermode='x unified'
        )

        st.plotly_chart(fig_lasso_path, use_container_width=True)

        st.info("As λ increases, coefficients shrink to zero. Some reach zero faster → variable selection.")

    with col2:
        # Ridge path
        coefs_ridge = []

        for alpha in alphas_path:
            ridge_temp = Ridge(alpha=alpha)
            ridge_temp.fit(X_scaled, y)
            coefs_ridge.append(ridge_temp.coef_)

        coefs_ridge = np.array(coefs_ridge)

        fig_ridge_path = go.Figure()

        for i, feat in enumerate(feature_cols):
            fig_ridge_path.add_trace(go.Scatter(
                x=alphas_path,
                y=coefs_ridge[:, i],
                mode='lines',
                name=feat,
                line=dict(width=2)
            ))

        fig_ridge_path.update_layout(
            title='Ridge Regularization Path',
            xaxis_title='λ (alpha)',
            yaxis_title='Coefficient',
            xaxis_type='log',
            hovermode='x unified'
        )

        st.plotly_chart(fig_ridge_path, use_container_width=True)

        st.info("Ridge shrinks coefficients smoothly but never exactly to zero.")

# ============================================================================
# TAB 2: DOUBLE MACHINE LEARNING
# ============================================================================
with tab2:
    st.header("🎯 Double Machine Learning (DML)")

    st.markdown("""
    ### Chernozhukov, Hansen, Spindler (2018)

    **Goal:** Estimate causal effect of D on Y, controlling for high-dimensional X

    **Problem with naive ML:**
    - Regularization bias: λ shrinks ALL coefficients (including treatment effect!)
    - Not √N-consistent

    **DML Solution:**
    1. **Partial out** confounders using ML
    2. **Orthogonal scores** → debiased estimates
    3. **Cross-fitting** → avoid overfitting

    **Algorithm:**
    ```
    1. Split sample: S₁, S₂
    2. Train Ŷ(X) on S₁, predict on S₂ → get Ỹ = Y - Ŷ(X)
    3. Train D̂(X) on S₁, predict on S₂ → get D̃ = D - D̂(X)
    4. Regress Ỹ ~ D̃ → get θ̂
    5. Repeat with S₁↔S₂, average θ̂
    ```
    """)

    # Prepare data
    df_dml = df_2023.dropna(subset=['wage_gap_percent', 'gdp_billions', 'population']).copy()

    # Create treatment variable (high vs low GDP)
    median_gdp_pc = ((df_dml['gdp_billions'] * 1e9) / df_dml['population']).median()
    df_dml['treatment'] = ((df_dml['gdp_billions'] * 1e9) / df_dml['population'] > median_gdp_pc).astype(int)

    # Covariates
    df_dml['log_pop'] = np.log(df_dml['population'])
    df_dml['gdp_total'] = df_dml['gdp_billions']

    # Encode region
    region_dummies = pd.get_dummies(df_dml['region'], prefix='region', drop_first=True)
    df_dml = pd.concat([df_dml, region_dummies], axis=1)

    # Features
    X_cols = ['log_pop', 'gdp_total'] + [col for col in df_dml.columns if col.startswith('region_')]
    X_dml = df_dml[X_cols].values
    D = df_dml['treatment'].values
    Y = df_dml['wage_gap_percent'].values

    st.subheader("Step-by-Step DML")

    # Sample splitting
    n = len(Y)
    idx_1 = np.random.choice(n, size=n//2, replace=False)
    idx_2 = np.array([i for i in range(n) if i not in idx_1])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Step 1: Partial out Y using ML**")

        # Train on S1, predict on S2
        rf_y = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf_y.fit(X_dml[idx_1], Y[idx_1])
        Y_pred_2 = rf_y.predict(X_dml[idx_2])
        Y_resid_2 = Y[idx_2] - Y_pred_2

        # Train on S2, predict on S1
        rf_y_2 = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf_y_2.fit(X_dml[idx_2], Y[idx_2])
        Y_pred_1 = rf_y_2.predict(X_dml[idx_1])
        Y_resid_1 = Y[idx_1] - Y_pred_1

        st.metric("Mean Residual (S1)", f"{np.mean(Y_resid_1):.4f}")
        st.metric("Mean Residual (S2)", f"{np.mean(Y_resid_2):.4f}")

        st.info("Residuals = part of Y not explained by X")

    with col2:
        st.write("**Step 2: Partial out D using ML**")

        # Train on S1, predict on S2
        rf_d = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf_d.fit(X_dml[idx_1], D[idx_1])
        D_pred_2 = rf_d.predict(X_dml[idx_2])
        D_resid_2 = D[idx_2] - D_pred_2

        # Train on S2, predict on S1
        rf_d_2 = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        rf_d_2.fit(X_dml[idx_2], D[idx_2])
        D_pred_1 = rf_d_2.predict(X_dml[idx_1])
        D_resid_1 = D[idx_1] - D_pred_1

        st.metric("Mean Residual (S1)", f"{np.mean(D_resid_1):.4f}")
        st.metric("Mean Residual (S2)", f"{np.mean(D_resid_2):.4f}")

        st.info("Residuals = part of D not explained by X")

    st.markdown("---")
    st.subheader("Step 3: Final Regression")

    # Combine residuals from both samples
    Y_resid_all = np.concatenate([Y_resid_1, Y_resid_2])
    D_resid_all = np.concatenate([D_resid_1, D_resid_2])

    # Final regression: Ỹ ~ D̃
    theta_dml = np.sum(D_resid_all * Y_resid_all) / np.sum(D_resid_all ** 2)

    # Standard error
    epsilon = Y_resid_all - theta_dml * D_resid_all
    se_dml = np.sqrt(np.mean(epsilon ** 2) / np.mean(D_resid_all ** 2)) / np.sqrt(n)

    # t-statistic
    t_stat_dml = theta_dml / se_dml
    p_value_dml = 2 * (1 - stats.t.cdf(np.abs(t_stat_dml), n - 1))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("DML Estimate (θ̂)", f"{theta_dml:.4f}")

    with col2:
        st.metric("Standard Error", f"{se_dml:.4f}")

    with col3:
        st.metric("p-value", f"{p_value_dml:.4f}")

    # Confidence interval
    ci_lower = theta_dml - 1.96 * se_dml
    ci_upper = theta_dml + 1.96 * se_dml

    st.write(f"**95% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]")

    if p_value_dml < 0.05:
        st.success(f"""
        ✅ **Statistically significant** (p = {p_value_dml:.4f})

        **Interpretation:** High-GDP countries have wage gaps that differ by {theta_dml:.2f} percentage points
        from low-GDP countries, after controlling for population and region using ML.

        **DML advantage:** This estimate is robust to model misspecification and high-dimensional confounders.
        """)
    else:
        st.info(f"""
        ℹ️ **Not statistically significant** (p = {p_value_dml:.4f})

        Cannot reject null hypothesis of no effect.
        """)

    st.markdown("---")
    st.subheader("📚 DML vs Naive Approaches")

    # Compare with naive regression
    X_naive = sm.add_constant(np.column_stack([D, X_dml]))
    model_naive = sm.OLS(Y, X_naive).fit()
    theta_naive = model_naive.params[1]

    comparison_dml = pd.DataFrame({
        'Method': ['Naive OLS', 'Double Machine Learning'],
        'Estimate': [theta_naive, theta_dml],
        'Std Error': [model_naive.bse[1], se_dml],
        'p-value': [model_naive.pvalues[1], p_value_dml]
    })

    st.dataframe(comparison_dml.style.format({
        'Estimate': '{:.4f}',
        'Std Error': '{:.4f}',
        'p-value': '{:.4f}'
    }), hide_index=True)

    st.info("""
    **Why differences exist:**
    - DML uses flexible ML for nuisance parameters (better control for confounding)
    - Cross-fitting prevents overfitting bias
    - Orthogonalization removes regularization bias
    """)

# ============================================================================
# TAB 3: RANDOM FOREST & FEATURE IMPORTANCE
# ============================================================================
with tab3:
    st.header("🌲 Random Forest & Feature Importance")

    st.markdown("""
    ### Random Forest for Prediction

    **Ensemble method:** Average predictions from many decision trees

    **Advantages:**
    - Non-parametric (no functional form assumptions)
    - Handles interactions automatically
    - Robust to outliers
    - Built-in feature importance

    **For economics:** Use for complex relationships, then interpret with SHAP
    """)

    # Prepare features
    df_rf = df_2023.dropna(subset=['wage_gap_percent', 'gdp_billions', 'population']).copy()
    df_rf['gdp_per_capita'] = (df_rf['gdp_billions'] * 1e9) / df_rf['population']
    df_rf['log_gdp_pc'] = np.log(df_rf['gdp_per_capita'])
    df_rf['log_pop'] = np.log(df_rf['population'])

    # One-hot encode region
    region_dummies_rf = pd.get_dummies(df_rf['region'], prefix='region')
    df_rf = pd.concat([df_rf, region_dummies_rf], axis=1)

    feature_cols_rf = ['log_gdp_pc', 'log_pop'] + [col for col in df_rf.columns if col.startswith('region_')]
    X_rf = df_rf[feature_cols_rf].values
    y_rf = df_rf['wage_gap_percent'].values

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Random Forest Training")

        # Train Random Forest
        n_trees = st.slider("Number of trees:", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.slider("Max depth:", min_value=2, max_value=10, value=5)

        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)

        # Cross-validation
        cv_scores = cross_val_score(rf, X_rf, y_rf, cv=5, scoring='r2')

        st.write(f"**Cross-validated R²:** {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Fit full model
        rf.fit(X_rf, y_rf)

        # Train R²
        train_r2 = rf.score(X_rf, y_rf)
        st.metric("Training R²", f"{train_r2:.4f}")

        if train_r2 - cv_scores.mean() > 0.2:
            st.warning("⚠️ Large gap between training and CV R² suggests overfitting. Reduce max_depth or n_estimators.")
        else:
            st.success("✅ Model generalizes well (minimal overfitting)")

    with col2:
        st.subheader("Predictions")

        # Show predictions for a few countries
        df_rf['prediction'] = rf.predict(X_rf)
        df_rf['error'] = df_rf['wage_gap_percent'] - df_rf['prediction']

        sample_preds = df_rf[['country_name', 'wage_gap_percent', 'prediction', 'error']].head(10)

        st.dataframe(sample_preds.style.format({
            'wage_gap_percent': '{:.2f}',
            'prediction': '{:.2f}',
            'error': '{:.2f}'
        }), hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Feature Importance")

    col1, col2 = st.columns(2)

    with col1:
        # Gini importance (built-in)
        importances = rf.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols_rf,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig_imp = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Gini)',
            labels={'Importance': 'Mean Decrease in Impurity'}
        )

        st.plotly_chart(fig_imp, use_container_width=True)

        st.info("""
        **Gini Importance:**
        - How much each feature reduces impurity (variance) in trees
        - Higher = more important for prediction
        """)

    with col2:
        st.subheader("Interpretation")

        top_feature = importance_df.iloc[0]['Feature']
        top_importance = importance_df.iloc[0]['Importance']

        st.write(f"**Most important feature:** {top_feature} ({top_importance:.4f})")

        st.markdown("""
        **Economic Interpretation:**

        If `log_gdp_pc` has highest importance:
        → Economic development is key predictor of wage gaps

        If regional dummies are important:
        → Geography/culture matters beyond economics

        **Caution:** Importance ≠ causality!
        - Correlation, not causation
        - Use DML or causal forests for causal effects
        """)

        # Partial dependence (simplified)
        st.subheader("Partial Dependence")

        st.info("""
        **Partial Dependence Plot** shows how predictions change with one feature

        **For your PhD:** Use `from sklearn.inspection import PartialDependenceDisplay`

        Shows non-linear relationships ML captured
        """)

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================
with tab4:
    st.header("📊 Model Comparison & Selection")

    st.markdown("""
    ### Comparing ML Methods

    **Goal:** Find best predictor for wage gap

    **Candidates:**
    1. OLS (baseline)
    2. Ridge Regression
    3. LASSO Regression
    4. Random Forest
    5. Gradient Boosting
    """)

    # Prepare data
    X_comp = X_rf
    y_comp = y_rf

    # Models to compare
    models = {
        'OLS': sm.OLS(y_comp, sm.add_constant(X_comp)).fit(),
        'Ridge': Ridge(alpha=1.0),
        'LASSO': Lasso(alpha=0.1, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }

    # Cross-validation comparison
    cv_results = []

    for name, model in models.items():
        if name == 'OLS':
            # OLS already fitted
            train_score = model.rsquared
            # Approximate CV with adjusted R²
            cv_score = model.rsquared_adj
        else:
            # Fit and CV
            model.fit(X_comp, y_comp)
            train_score = model.score(X_comp, y_comp)
            cv_scores_model = cross_val_score(model, X_comp, y_comp, cv=5, scoring='r2')
            cv_score = cv_scores_model.mean()

        cv_results.append({
            'Model': name,
            'Training R²': train_score,
            'CV R²': cv_score,
            'Overfit Gap': train_score - cv_score
        })

    df_comparison = pd.DataFrame(cv_results).sort_values('CV R²', ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Performance")

        st.dataframe(df_comparison.style.format({
            'Training R²': '{:.4f}',
            'CV R²': '{:.4f}',
            'Overfit Gap': '{:.4f}'
        }).background_gradient(subset=['CV R²'], cmap='Greens'), hide_index=True)

        # Bar chart
        fig_comp = go.Figure()

        fig_comp.add_trace(go.Bar(
            name='Training R²',
            x=df_comparison['Model'],
            y=df_comparison['Training R²'],
            marker_color='lightblue'
        ))

        fig_comp.add_trace(go.Bar(
            name='CV R²',
            x=df_comparison['Model'],
            y=df_comparison['CV R²'],
            marker_color='darkblue'
        ))

        fig_comp.update_layout(
            title='Model Comparison',
            xaxis_title='Model',
            yaxis_title='R²',
            barmode='group'
        )

        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        st.subheader("Model Selection")

        best_model = df_comparison.iloc[0]['Model']
        best_cv_r2 = df_comparison.iloc[0]['CV R²']

        st.metric("Best Model", best_model)
        st.metric("CV R²", f"{best_cv_r2:.4f}")

        st.success(f"""
        ✅ **{best_model}** wins!

        Use this for prediction tasks.
        """)

        st.markdown("---")
        st.subheader("Trade-offs")

        st.markdown("""
        **Interpretability vs Performance:**

        - **OLS:** Most interpretable, weakest predictions
        - **LASSO:** Variable selection, interpretable
        - **Ridge:** Better than OLS, still interpretable
        - **Random Forest:** Black box, strong predictions
        - **GBM:** Black box, often best performance

        **For PhD:**
        - Use OLS/LASSO for causal inference
        - Use RF/GBM for pure prediction
        - Use SHAP to interpret complex models
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Machine Learning for Economic Research</strong></p>
    <p>LASSO • Ridge • Double ML • Random Forest • Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📚 Essential References:

1. **Chernozhukov et al. (2018)** - "Double/debiased machine learning for treatment and structural parameters"
2. **Athey & Imbens (2019)** - "Machine Learning Methods Economists Should Know About"
3. **Mullainathan & Spiess (2017)** - "Machine Learning: An Applied Econometric Approach"
4. **Athey et al. (2019)** - "Generalized random forests"
5. **Hastie, Tibshirani & Friedman (2009)** - "The Elements of Statistical Learning"
6. **James et al. (2021)** - "An Introduction to Statistical Learning" (2nd ed)
7. **Lundberg & Lee (2017)** - "A Unified Approach to Interpreting Model Predictions" (SHAP)
""")
