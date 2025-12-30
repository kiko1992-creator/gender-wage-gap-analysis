# Gender Wage Gap Research Platform
## PhD-Grade Methodological Database System

**Purpose**: Systematic, automated, extensible research infrastructure for wage gap determinants analysis

---

## 🎯 RESEARCH REQUIREMENTS

### Core Objectives:
1. **Automated data collection** from multiple sources
2. **Systematic factor analysis** framework
3. **Reproducible methodology** with full lineage
4. **Extensible architecture** for new variables/countries
5. **Academic rigor** - proper citations, validation, statistical tests
6. **Database-backed** - not CSV files
7. **No tutorials** - professional research interface only

---

## 🏗️ PROPOSED ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                   RESEARCH PLATFORM STACK                    │
└─────────────────────────────────────────────────────────────┘

Layer 1: DATA COLLECTION (Automated)
├── Schedulers (Apache Airflow or cron)
├── API Connectors
│   ├── Eurostat
│   ├── World Bank
│   ├── ILO
│   ├── OECD
│   ├── National Statistics Offices
│   └── Custom scrapers
├── Data Validation Pipeline
└── Database Writer

Layer 2: DATABASE (PostgreSQL or SQLite)
├── raw_data (as received from APIs)
├── processed_data (cleaned, validated)
├── factors (determinant variables)
├── metadata (source, timestamp, version)
├── analysis_results (cached computations)
└── research_log (audit trail)

Layer 3: ANALYTICAL ENGINE
├── Factor Analysis Module
│   ├── Economic factors (GDP, unemployment, etc.)
│   ├── Institutional factors (policies, laws)
│   ├── Educational factors (STEM enrollment, etc.)
│   ├── Demographic factors (fertility, age distribution)
│   └── Cultural factors (gender norms indices)
├── Statistical Methods
│   ├── Panel data regression (fixed effects, random effects)
│   ├── Difference-in-differences
│   ├── Instrumental variables
│   ├── Synthetic control
│   ├── Machine learning (XGBoost, Neural Networks)
│   └── Causal inference (propensity score matching)
├── Model Validation
│   ├── Cross-validation
│   ├── Robustness checks
│   └── Sensitivity analysis

Layer 4: RESEARCH INTERFACE (Streamlit)
├── Data Explorer (SQL query builder)
├── Factor Analysis Dashboard
├── Model Comparison
├── Export to R/Stata/SPSS
└── Academic Report Generator

Layer 5: OUTPUTS
├── Peer-reviewed paper drafts (LaTeX)
├── Datasets (Zenodo, Dataverse)
├── Replication packages
└── API for other researchers
```

---

## 📊 DATABASE SCHEMA

### Table: `raw_wage_data`
```sql
CREATE TABLE raw_wage_data (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,           -- 'eurostat', 'worldbank', etc.
    country_code VARCHAR(3) NOT NULL,
    year INTEGER NOT NULL,
    gender VARCHAR(10),
    sector VARCHAR(50),
    education_level VARCHAR(50),
    age_group VARCHAR(20),
    wage_value DECIMAL(10,2),
    wage_type VARCHAR(30),                 -- 'monthly', 'hourly', 'annual'
    currency VARCHAR(3),
    collection_date TIMESTAMP DEFAULT NOW(),
    api_version VARCHAR(20),
    data_quality_flag INTEGER,             -- 1=official, 2=estimated, 3=imputed
    notes TEXT,
    UNIQUE(source, country_code, year, gender, sector, education_level, age_group)
);
```

### Table: `determinant_factors`
```sql
CREATE TABLE determinant_factors (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) NOT NULL,
    year INTEGER NOT NULL,

    -- Economic Factors
    gdp_per_capita DECIMAL(12,2),
    gdp_growth_rate DECIMAL(5,2),
    unemployment_rate DECIMAL(5,2),
    unemployment_female DECIMAL(5,2),
    unemployment_male DECIMAL(5,2),
    inflation_rate DECIMAL(5,2),
    minimum_wage DECIMAL(10,2),
    gini_coefficient DECIMAL(5,4),

    -- Labor Market Factors
    female_labor_force_participation DECIMAL(5,2),
    male_labor_force_participation DECIMAL(5,2),
    part_time_employment_female DECIMAL(5,2),
    part_time_employment_male DECIMAL(5,2),
    union_density DECIMAL(5,2),
    employment_protection_index DECIMAL(5,2),

    -- Educational Factors
    tertiary_education_female DECIMAL(5,2),
    tertiary_education_male DECIMAL(5,2),
    stem_graduates_female DECIMAL(5,2),
    stem_graduates_male DECIMAL(5,2),
    education_spending_gdp_pct DECIMAL(5,2),

    -- Institutional/Policy Factors
    maternity_leave_weeks INTEGER,
    paternity_leave_weeks INTEGER,
    childcare_spending_gdp_pct DECIMAL(5,2),
    gender_quota_policy BOOLEAN,
    equal_pay_legislation_year INTEGER,

    -- Demographic Factors
    fertility_rate DECIMAL(5,3),
    median_age DECIMAL(5,2),
    population_density DECIMAL(10,2),
    urbanization_rate DECIMAL(5,2),

    -- Cultural/Social Factors
    gender_development_index DECIMAL(5,4),
    gender_inequality_index DECIMAL(5,4),
    women_in_parliament_pct DECIMAL(5,2),

    -- Sector Structure
    agriculture_employment_pct DECIMAL(5,2),
    industry_employment_pct DECIMAL(5,2),
    services_employment_pct DECIMAL(5,2),
    public_sector_employment_pct DECIMAL(5,2),

    -- Data Provenance
    last_updated TIMESTAMP DEFAULT NOW(),
    data_completeness_score DECIMAL(3,2),  -- 0.0 to 1.0

    UNIQUE(country_code, year)
);
```

### Table: `analysis_metadata`
```sql
CREATE TABLE analysis_metadata (
    id SERIAL PRIMARY KEY,
    analysis_type VARCHAR(50),             -- 'regression', 'decomposition', etc.
    run_timestamp TIMESTAMP DEFAULT NOW(),
    parameters JSONB,                      -- Store analysis parameters
    model_specification TEXT,
    r_squared DECIMAL(5,4),
    adj_r_squared DECIMAL(5,4),
    n_observations INTEGER,
    statistical_significance JSONB,
    diagnostics JSONB,                     -- Heteroskedasticity, autocorrelation tests
    code_version VARCHAR(20),
    researcher_notes TEXT
);
```

### Table: `data_sources`
```sql
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100),
    api_endpoint TEXT,
    last_fetch TIMESTAMP,
    next_scheduled_fetch TIMESTAMP,
    fetch_frequency VARCHAR(20),           -- 'daily', 'weekly', 'monthly', 'annual'
    data_license VARCHAR(200),
    citation TEXT,
    reliability_score INTEGER,             -- 1-5 scale
    contact_email VARCHAR(100),
    documentation_url TEXT
);
```

---

## 🤖 AUTOMATED DATA PIPELINE

### Pipeline Architecture:

```python
# pipelines/wage_gap_etl.py

class WageGapETL:
    """
    Automated Extract-Transform-Load pipeline for wage gap research
    """

    def __init__(self, db_connection):
        self.db = db_connection
        self.sources = self.load_data_sources()
        self.logger = self.setup_logger()

    def extract(self, source, start_date, end_date):
        """
        Extract data from external API

        Returns:
            pd.DataFrame with raw data + metadata
        """
        # API-specific extraction logic
        # Error handling with retry logic
        # Rate limiting compliance
        # Data quality checks
        pass

    def transform(self, raw_data):
        """
        Clean, validate, standardize data

        Quality checks:
        - Missing value analysis
        - Outlier detection
        - Currency conversion
        - Wage type standardization
        - Temporal consistency
        """
        pass

    def load(self, processed_data, table_name):
        """
        Load to database with full lineage tracking
        """
        pass

    def run_scheduled_update(self):
        """
        Daily/weekly/monthly automated run
        """
        for source in self.sources:
            if self.is_update_due(source):
                raw = self.extract(source)
                clean = self.transform(raw)
                self.load(clean)
                self.log_pipeline_run(source, len(clean))
```

### Scheduling (Airflow DAG):

```python
# dags/wage_gap_daily.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'phd_researcher',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'wage_gap_data_pipeline',
    default_args=default_args,
    description='Automated wage gap data collection',
    schedule_interval='@daily',
)

fetch_eurostat = PythonOperator(
    task_id='fetch_eurostat_data',
    python_callable=fetch_eurostat_api,
    dag=dag,
)

fetch_worldbank = PythonOperator(
    task_id='fetch_worldbank_data',
    python_callable=fetch_worldbank_api,
    dag=dag,
)

validate_data = PythonOperator(
    task_id='validate_and_clean',
    python_callable=run_validation_pipeline,
    dag=dag,
)

update_factors = PythonOperator(
    task_id='compute_determinant_factors',
    python_callable=compute_factors,
    dag=dag,
)

fetch_eurostat >> validate_data
fetch_worldbank >> validate_data
validate_data >> update_factors
```

---

## 📈 FACTOR ANALYSIS FRAMEWORK

### Systematic Factor Testing:

```python
# research/factor_analysis.py

class DeterminantAnalysis:
    """
    Systematic analysis of wage gap determinants
    """

    FACTOR_CATEGORIES = {
        'economic': [
            'gdp_per_capita', 'unemployment_rate', 'inflation_rate',
            'gini_coefficient', 'minimum_wage'
        ],
        'labor_market': [
            'female_lfp', 'union_density', 'part_time_employment',
            'employment_protection_index'
        ],
        'educational': [
            'tertiary_education_female', 'stem_graduates_female',
            'education_spending_gdp_pct'
        ],
        'institutional': [
            'maternity_leave_weeks', 'childcare_spending',
            'gender_quota_policy', 'equal_pay_legislation'
        ],
        'demographic': [
            'fertility_rate', 'median_age', 'urbanization_rate'
        ],
        'cultural': [
            'gender_development_index', 'women_in_parliament_pct'
        ]
    }

    def run_comprehensive_analysis(self):
        """
        Test all factors systematically
        """
        results = {}

        # 1. Univariate analysis
        for category, factors in self.FACTOR_CATEGORIES.items():
            results[category] = self.univariate_regression(factors)

        # 2. Multivariate analysis (category-wise)
        for category, factors in self.FACTOR_CATEGORIES.items():
            results[f'{category}_multivariate'] = self.multivariate_regression(factors)

        # 3. Full model with all factors
        all_factors = [f for factors in self.FACTOR_CATEGORIES.values() for f in factors]
        results['full_model'] = self.panel_regression(
            factors=all_factors,
            fixed_effects='country',
            time_effects=True
        )

        # 4. Stepwise selection
        results['optimal_model'] = self.stepwise_selection(all_factors)

        # 5. Interaction effects
        results['interactions'] = self.test_interactions([
            ('tertiary_education_female', 'gdp_per_capita'),
            ('maternity_leave_weeks', 'female_lfp'),
            ('union_density', 'employment_protection_index')
        ])

        # 6. Non-linear relationships
        results['non_linear'] = self.test_polynomial_effects(
            ['gdp_per_capita', 'education_spending_gdp_pct']
        )

        # 7. Causal inference
        results['causal'] = {
            'did': self.difference_in_differences(
                treatment='equal_pay_legislation',
                outcome='wage_gap'
            ),
            'synthetic_control': self.synthetic_control_method(
                treated_unit='Country_X',
                treatment_year=2015
            ),
            'iv': self.instrumental_variables(
                endogenous='female_lfp',
                instrument='childcare_availability'
            )
        }

        return results

    def panel_regression(self, factors, fixed_effects=None, time_effects=False):
        """
        Panel data regression with fixed/random effects
        """
        from linearmodels import PanelOLS

        # Prepare panel data
        data = self.prepare_panel_data()

        # Specify model
        formula = f'wage_gap ~ {" + ".join(factors)}'

        if fixed_effects:
            formula += f' + EntityEffects'
        if time_effects:
            formula += f' + TimeEffects'

        # Fit model
        model = PanelOLS.from_formula(formula, data)
        results = model.fit(cov_type='clustered', cluster_entity=True)

        # Diagnostics
        diagnostics = {
            'r_squared': results.rsquared,
            'f_statistic': results.f_statistic,
            'hausman_test': self.hausman_test(results),
            'heteroskedasticity': self.breusch_pagan_test(results),
            'autocorrelation': self.durbin_watson_test(results)
        }

        return {
            'results': results,
            'diagnostics': diagnostics,
            'specification': formula
        }
```

---

## 🔬 RESEARCH INTERFACE (NO TUTORIALS)

### Professional Research Dashboard:

```python
# app_research.py

import streamlit as st
from research.database import ResearchDB
from research.factor_analysis import DeterminantAnalysis

st.set_page_config(
    page_title="Wage Gap Research Platform",
    layout="wide",
    initial_sidebar_state="collapsed"  # Professional, minimal UI
)

# Clean header - no fluff
st.title("Gender Wage Gap Determinants Analysis")

db = ResearchDB()
analysis = DeterminantAnalysis(db)

# Sidebar: Research Tools Only
with st.sidebar:
    st.header("Research Tools")

    mode = st.radio("", [
        "Data Query",
        "Factor Analysis",
        "Model Comparison",
        "Export Results"
    ], label_visibility="collapsed")

# Main content based on mode
if mode == "Data Query":
    # SQL query interface
    query = st.text_area("SQL Query:", height=150)
    if st.button("Execute"):
        results = db.execute(query)
        st.dataframe(results)
        st.download_button("Export CSV", results.to_csv())

elif mode == "Factor Analysis":
    # Factor selection and analysis
    category = st.selectbox("Factor Category:",
        list(DeterminantAnalysis.FACTOR_CATEGORIES.keys()))

    factors = st.multiselect(
        "Select Factors:",
        DeterminantAnalysis.FACTOR_CATEGORIES[category]
    )

    method = st.selectbox("Method:", [
        "Panel OLS (Fixed Effects)",
        "Panel OLS (Random Effects)",
        "Difference-in-Differences",
        "Instrumental Variables",
        "Synthetic Control"
    ])

    if st.button("Run Analysis"):
        results = analysis.run_analysis(factors, method)

        # Results display (academic format)
        st.subheader("Regression Results")
        st.text(results.summary)  # Statsmodels/linearmodels output

        st.subheader("Diagnostics")
        st.json(results.diagnostics)

        st.download_button("Export LaTeX Table", results.to_latex())

elif mode == "Model Comparison":
    # Compare multiple specifications
    st.write("Model specifications comparison")

elif mode == "Export Results":
    # Export to academic formats
    format = st.selectbox("Format:", [
        "Stata (.dta)",
        "SPSS (.sav)",
        "R (.rds)",
        "CSV",
        "LaTeX Tables",
        "Replication Package (ZIP)"
    ])

    if st.button("Generate Export"):
        file = db.export(format)
        st.download_button("Download", file)
```

---

## 📚 ACADEMIC RIGOR FEATURES

### 1. **Reproducibility Package Generator**

```python
def create_replication_package(analysis_id):
    """
    Generate complete replication package
    """
    package = {
        'data': export_data_subset(analysis_id),
        'code': {
            'python': get_analysis_code(analysis_id),
            'r': convert_to_r(analysis_id),
            'stata': convert_to_stata(analysis_id)
        },
        'results': get_all_outputs(analysis_id),
        'metadata': {
            'software_versions': get_environment(),
            'random_seed': get_seed(analysis_id),
            'timestamp': datetime.now()
        },
        'documentation': generate_codebook(),
        'readme': generate_readme(analysis_id)
    }

    return create_zip(package)
```

### 2. **Citation Generator**

```python
def generate_citations():
    """
    Auto-generate all data citations
    """
    citations = []

    for source in db.get_data_sources():
        citation = f"""
        {source.name} ({source.year}). {source.dataset_title}.
        Retrieved from {source.url} on {source.access_date}.
        License: {source.license}.
        """
        citations.append(citation)

    return format_bibtex(citations)
```

### 3. **Methodology Documentation**

Automatically generate methods section:
- Data sources and collection procedures
- Sample selection criteria
- Variable construction
- Statistical methods employed
- Robustness checks performed

---

## 🚀 IMMEDIATE IMPLEMENTATION PLAN

### Phase 1: Database Setup (Week 1)
```bash
# 1. Set up PostgreSQL database
createdb wage_gap_research

# 2. Create schema
psql wage_gap_research < schema.sql

# 3. Migrate existing CSV data
python migrate_csv_to_db.py

# 4. Set up automated backups
```

### Phase 2: API Integration (Week 2)
- Connect to Eurostat API
- Connect to World Bank API
- Connect to ILO API
- Add OECD data
- Set up automated fetching

### Phase 3: Factor Expansion (Week 3-4)
- Add 50+ determinant variables
- Implement factor computation pipeline
- Validate factor data quality
- Create factor documentation

### Phase 4: Analysis Engine (Week 5-6)
- Panel regression models
- Causal inference methods
- Machine learning models
- Robustness testing framework

### Phase 5: Research Interface (Week 7)
- Clean, professional UI
- SQL query builder
- Export to academic formats
- Replication package generator

---

## 🎓 SPECIFIC TO YOUR PHD

### Research Questions to Answer:
1. What are the TOP 10 determinants of wage gap?
2. How do determinants vary by region/development level?
3. Which policy interventions have strongest causal evidence?
4. Are there non-linear relationships (thresholds, diminishing returns)?
5. What are the interaction effects between factors?
6. How have determinants changed over time?
7. Can we predict future wage gaps based on current policies?

### Publications You Can Produce:
1. **Systematic literature review** with meta-analysis
2. **Determinants paper** - comprehensive factor analysis
3. **Policy evaluation** - causal inference study
4. **Methodology paper** - database and framework
5. **Country case studies** - deep dives
6. **Forecasting paper** - ML predictions

---

## 💻 NEXT STEPS - TELL ME:

1. **Database**: PostgreSQL or SQLite? (PostgreSQL recommended for research)
2. **Factors**: Which determinants are MOST important to you?
3. **Countries**: Expand beyond Balkans? (EU28, OECD, worldwide?)
4. **Timeline**: When do you need this operational?
5. **Start**: Should I begin implementing the database TODAY?

I can transform your current project into a **world-class PhD research platform**. Just tell me to start! 🚀
