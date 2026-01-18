# Gender Wage Gap Analysis Platform

**Comprehensive econometrics research platform for analyzing gender wage disparities across European Union countries.**

PhD research project by **Kiril Mickovski** - Data Scientist preparing for doctoral studies in Econometrics.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

---

## Overview

This platform provides interactive tools for econometric analysis of the gender wage gap, featuring 17 analytical modules covering everything from basic OLS regression to advanced causal inference methods. Designed for researchers, students, and policymakers studying labor market inequalities.

### Key Features

- **17 Interactive Pages**: Comprehensive coverage of econometric methods
- **Production-Ready**: Docker infrastructure with PostgreSQL database
- **Educational Focus**: Interactive tutorials and step-by-step learning modules
- **Real Data**: Eurostat and World Bank data integration
- **Reproducible Research**: Export results for academic papers

### Analytical Methods

| Category | Methods |
|----------|---------|
| **Causal Inference** | Difference-in-Differences (DiD), Instrumental Variables (IV/2SLS), Synthetic Control, Propensity Score Matching (PSM) |
| **Panel Econometrics** | Fixed Effects (FE), Random Effects (RE), Hausman Test, Dynamic Panels, GMM |
| **Machine Learning** | LASSO/Ridge Regression, Double Machine Learning (DML), Random Forest, Feature Selection |
| **Bayesian Methods** | Bayesian Regression, Hierarchical Models, MCMC Sampling, Credible Intervals |
| **Time Series** | ARIMA, Unit Root Tests (ADF, KPSS), Structural Breaks, Forecasting |

---

## Quick Start

### Option 1: One-Command Setup (Docker)

```bash
# Clone repository
git clone https://github.com/kiko1992-creator/gender-wage-gap-analysis.git
cd gender-wage-gap-analysis

# Start everything (PostgreSQL + Streamlit)
./quick-start.sh
```

**That's it!** App runs at http://localhost:8501

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## Project Structure

```
gender-wage-gap-analysis/
├── app.py                       # Main dashboard (pages 1-8)
├── database_connection.py       # PostgreSQL utilities
├── sample_data.py               # Fallback data
├── requirements.txt
│
├── pages/                       # Advanced econometric pages
│   ├── 09_🇪🇺_EU27_Database.py
│   ├── 10_🎨_Advanced_Visualizations.py
│   ├── 11_📊_Advanced_Statistics.py
│   ├── 12_🎯_Causal_Inference.py
│   ├── 13_📈_Panel_Econometrics.py
│   ├── 14_🤖_ML_Economics.py
│   ├── 15_🎲_Bayesian_Methods.py
│   ├── 16_📉_Time_Series.py
│   └── 17_📚_Week1_OLS_Tutorial.py
│
├── docker/                      # Database initialization
│   └── init-db/
│       ├── 01_create_tables.sql
│       └── 02_seed_data.sql
│
├── scripts/                     # Analysis utilities
│   ├── time_series.py
│   └── setup_database.py
│
├── data/                        # Research data
│   └── processed/
│
└── docs/                        # Documentation
    ├── DOCKER.md
    └── DEPLOYMENT.md
```

---

## Docker Infrastructure

### Start Services

```bash
# Quick start
make up

# With database UI (pgAdmin)
make up-dev
```

### Available Services

- **Streamlit**: http://localhost:8501
- **PostgreSQL**: localhost:5432
- **pgAdmin**: http://localhost:5050 (dev mode)

### Common Commands

```bash
make up           # Start all services
make down         # Stop all services
make logs         # View logs
make shell-db     # Open database shell
make db-backup    # Create backup
```

See [docs/DOCKER.md](docs/DOCKER.md) for details.

---

## Pages Overview

### Pages 1-8: Core Dashboard (app.py)

1. **Overview** - Country rankings and key metrics
2. **Country Profiles** - Detailed analysis by country
3. **Country Comparison** - Side-by-side comparisons
4. **Regional Analysis** - EU regions and clusters
5. **Time Series** - Trends and forecasting
6. **What-If Analysis** - Scenario modeling
7. **ML Insights** - Machine learning predictions
8. **Oaxaca-Blinder** - Wage decomposition
9. **Data Explorer** - Interactive tables

### Pages 9-17: Advanced Econometrics (pages/)

**Page 9: EU27 Database**
- PostgreSQL integration
- Data querying and exploration

**Page 10-11: Visualizations & Statistics**
- Interactive plots (Plotly)
- Descriptive statistics
- Distribution analysis

**Page 12: Causal Inference** ([View Code](pages/12_🎯_Causal_Inference.py))
- Difference-in-Differences (DiD)
- Instrumental Variables (IV/2SLS)
- Synthetic Control Method
- Propensity Score Matching (PSM)

**Page 13: Panel Econometrics** ([View Code](pages/13_📈_Panel_Econometrics.py))
- Fixed Effects (FE)
- Random Effects (RE)
- Hausman Test
- GMM Estimation

**Page 14: ML for Economics** ([View Code](pages/14_🤖_ML_Economics.py))
- LASSO/Ridge with cross-validation
- Double Machine Learning (Chernozhukov et al.)
- Random Forest for causal inference
- Feature importance analysis

**Page 15: Bayesian Methods** ([View Code](pages/15_🎲_Bayesian_Methods.py))
- Bayesian regression
- Hierarchical models
- MCMC sampling
- Credible intervals

**Page 16: Time Series** ([View Code](pages/16_📉_Time_Series.py))
- ARIMA modeling
- Unit root tests (ADF, KPSS)
- Structural break detection
- Forecasting

**Page 17: Interactive Tutorial** ([View Code](pages/17_📚_Week1_OLS_Tutorial.py))
- Week 1: OLS regression foundations
- Hands-on exercises with real data
- Assumption testing
- Interactive quizzes

---

## Data Sources

- **Eurostat**: EU labor force statistics
- **World Bank**: GDP, population, economic indicators
- **OECD**: Wage and employment data
- **National Statistical Offices**: Country-specific data

---

## For Researchers

### Using for Your Research

1. **Load Your Data**:
```python
import pandas as pd
from database_connection import get_connection

df = pd.read_csv('your_data.csv')
conn = get_connection()
df.to_sql('your_table', conn, if_exists='append')
```

2. **Run Analysis**: Use any of the 17 pages
3. **Export Results**: Download plots, tables, regression outputs
4. **Cite**: See [CITATION.md](CITATION.md)

### Academic Use

This platform is designed for academic research and education. If you use it in your papers:

```bibtex
@software{mickovski2024wagegap,
  author = {Mickovski, Kiril},
  title = {Gender Wage Gap Analysis Platform},
  year = {2024},
  url = {https://github.com/kiko1992-creator/gender-wage-gap-analysis}
}
```

---

## Development

### Local Development

```bash
# Clone repository
git clone https://github.com/kiko1992-creator/gender-wage-gap-analysis.git
cd gender-wage-gap-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### With Docker (Recommended)

```bash
# Start full stack (app + database)
./quick-start.sh

# Or using make
make up
```

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-analysis`
3. Commit changes: `git commit -m 'Add new analysis method'`
4. Push: `git push origin feature/my-analysis`
5. Create Pull Request

---

## Deployment

### Streamlit Cloud (Free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Deploy from your fork
4. Main file: `app.py`

**Note**: Uses `sample_data.py` fallback (no database needed).

### Production Deployment

**Railway.app** (Recommended):
```bash
# See docs/DEPLOYMENT.md for full guide
railway init
railway up
```

**Docker**:
```bash
docker-compose up -d
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for AWS, GCP, Azure guides.

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.11 |
| **Database** | PostgreSQL 16 |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Statistics** | statsmodels, scipy, sklearn |
| **Infrastructure** | Docker, docker-compose |
| **CI/CD** | GitHub Actions |

### Key Dependencies

```python
streamlit>=1.30.0
pandas>=2.0.0
plotly>=5.18.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
psycopg2-binary>=2.9.0
```

---

## Roadmap

### Phase 1: Foundation (Completed ✅)
- ✅ 17 interactive pages
- ✅ Docker infrastructure
- ✅ PostgreSQL integration
- ✅ Production deployment ready

### Phase 2: Automation (Next 2 months)
- ⏳ Automated Eurostat data fetching
- ⏳ Scheduled updates (GitHub Actions)
- ⏳ Testing framework
- ⏳ CI/CD pipeline

### Phase 3: Advanced Features (Months 3-6)
- ⏳ User authentication
- ⏳ RESTful API
- ⏳ Jupyter notebook integration
- ⏳ LaTeX export for papers

### Phase 4: Research Focus (Months 7-18)
- ⏳ Use for actual PhD research
- ⏳ Publish academic papers
- ⏳ Share with research community
- ⏳ Conference presentations

---

## Support & Contact

**Issues**: [GitHub Issues](https://github.com/kiko1992-creator/gender-wage-gap-analysis/issues)
**Discussions**: [GitHub Discussions](https://github.com/kiko1992-creator/gender-wage-gap-analysis/discussions)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

Available for academic research and educational purposes. Commercial use requires attribution.

---

## Acknowledgments

- Eurostat for providing open labor market data
- Streamlit for the excellent web framework
- PostgreSQL community for robust database system
- Economic research community for methodological foundations

---

**Built with 💡 for PhD research in Econometrics**

*Platform designed to enable students and researchers worldwide to conduct rigorous wage gap analysis for academic papers and policy research.*
