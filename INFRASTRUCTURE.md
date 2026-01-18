# Infrastructure & Automation Roadmap

## Vision
Build a robust, self-sufficient educational platform for econometrics research that students and researchers worldwide can use for learning and academic papers.

## Current State (Phase 1) ✅

### What We Have
- 17-page interactive Streamlit dashboard
- 9 advanced econometric methods (Causal Inference, Panel Data, ML, Bayesian, Time Series)
- PostgreSQL integration with fallback
- Basic deployment on Streamlit Cloud

### Limitations
- No persistent database in cloud
- Single-server deployment
- Manual data updates
- No user authentication
- No API access

## Infrastructure Evolution Plan

### Phase 2: Production-Ready Deployment (Weeks 1-2)

**Goal**: Reliable, always-available platform with persistent database

#### Option A: Railway.app (Recommended for Quick Start)
**Pros**: Easy, PostgreSQL included, reasonable cost (~$10-20/month)
**Setup**:
```bash
# 1. Create Railway account
# 2. Connect GitHub repo
# 3. Add PostgreSQL service
# 4. Deploy from branch
```

**Environment Variables**:
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
POSTGRES_HOST=containers-us-west-xxx.railway.app
POSTGRES_PORT=5432
POSTGRES_DB=practice_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=xxx
```

#### Option B: Docker + DigitalOcean/AWS (More Control)
**Pros**: Full control, scalable, professional
**Cost**: $12-40/month

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml** (for local development):
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: practice_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"

  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DB: practice_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    depends_on:
      - postgres
    volumes:
      - .:/app

volumes:
  postgres_data:
```

### Phase 3: Data Automation (Weeks 3-4)

**Goal**: Automatic data updates from Eurostat, World Bank, etc.

#### Automated Data Pipeline
```python
# scripts/automated_data_update.py
"""
Scheduled job to update wage gap data
Run daily via cron/GitHub Actions
"""

import requests
import pandas as pd
from database_connection import get_connection

def fetch_eurostat_data():
    """Fetch latest data from Eurostat API"""
    url = "https://ec.europa.eu/eurostat/api/dissemination/..."
    # Implementation

def update_database():
    """Update PostgreSQL with fresh data"""
    conn = get_connection()
    # Update logic

if __name__ == "__main__":
    fetch_eurostat_data()
    update_database()
```

#### GitHub Actions Workflow
```yaml
# .github/workflows/data-update.yml
name: Daily Data Update

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:  # Manual trigger

jobs:
  update-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run data update
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: python scripts/automated_data_update.py

      - name: Commit updated data
        run: |
          git config user.name "Data Bot"
          git config user.email "bot@example.com"
          git add data/
          git commit -m "chore: automated data update" || echo "No changes"
          git push
```

### Phase 4: Advanced Features (Weeks 5-8)

#### 4.1 User Authentication (for personalized research)
**Technology**: Streamlit-Authenticator or Auth0

```python
# Add to app.py
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'gender_wage_gap',
    'auth_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
    authenticator.logout('Logout', 'sidebar')
    # ... rest of app
```

**Benefits**:
- Save user analysis
- Personal notebooks
- Export research data
- Track learning progress

#### 4.2 RESTful API (for researchers to access data programmatically)
**Technology**: FastAPI

```python
# api/main.py
from fastapi import FastAPI
from typing import List, Optional

app = FastAPI(title="Gender Wage Gap API")

@app.get("/api/v1/countries")
def get_countries():
    """Get all countries in database"""
    # Implementation

@app.get("/api/v1/wage_gap/{country}")
def get_country_data(country: str, year_start: Optional[int] = None):
    """Get wage gap data for specific country"""
    # Implementation

@app.post("/api/v1/analysis/ols")
def run_ols_regression(data: dict):
    """Run OLS regression on provided data"""
    # Implementation
```

#### 4.3 Jupyter Notebook Integration
**Technology**: JupyterHub or Google Colab integration

```python
# notebooks/research_template.ipynb
"""
Template notebook for researchers
- Pre-configured database connection
- Common analysis patterns
- Export to LaTeX for papers
"""
```

#### 4.4 Advanced Monitoring & Logging
**Technology**: Prometheus + Grafana or Datadog

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram

page_views = Counter('page_views_total', 'Total page views', ['page_name'])
regression_time = Histogram('regression_duration_seconds', 'Time to run regression')

# Track usage
page_views.labels(page_name='Causal_Inference').inc()
```

### Phase 5: Scalability & Research Features (Weeks 9-12)

#### 5.1 Computational Cluster Integration
**For heavy computations**: Bootstrap, MCMC, cross-validation

```python
# Use Dask or Ray for distributed computing
from dask.distributed import Client

client = Client('scheduler-address:8786')

# Run 10,000 bootstrap iterations in parallel
futures = client.map(bootstrap_iteration, range(10000))
results = client.gather(futures)
```

#### 5.2 Research Paper Export
**Auto-generate LaTeX tables and figures**

```python
# utils/latex_export.py
def export_regression_table(model, filename="regression_table.tex"):
    """Export regression results to LaTeX table format"""
    # Generate publication-ready table

def export_plots_for_paper(figs, dpi=300):
    """Export high-resolution plots for papers"""
    # Save as PDF/PNG with proper sizing
```

#### 5.3 Collaborative Research Workspace
**Multiple researchers working on same project**

- Shared datasets
- Version control for analyses
- Comments and annotations
- Reproducible research pipelines

## Cost Estimation

| Phase | Infrastructure | Monthly Cost | One-time Setup |
|-------|---------------|--------------|----------------|
| 1 (Current) | Streamlit Cloud | Free | $0 |
| 2 | Railway.app + PostgreSQL | $15-25 | $0 |
| 2 (Alt) | DigitalOcean Droplet + DB | $20-30 | 4 hours |
| 3 | + GitHub Actions | $0 (free tier) | 2 hours |
| 4 | + Auth + API | $25-40 | 8 hours |
| 5 | + Compute cluster | $50-200 | 16 hours |

## Timeline for Your PhD Journey (18 months)

### Months 1-3: Foundation (Current)
- ✅ Basic Streamlit app
- ✅ Core econometric methods
- 🔄 Production deployment
- ⏳ Local VS Code setup

### Months 4-6: Automation
- Automated data pipeline
- Testing framework
- CI/CD with GitHub Actions
- Documentation

### Months 7-12: Advanced Features
- User authentication
- API development
- Jupyter integration
- Research export tools

### Months 13-18: Research Focus
- Use platform for actual PhD research
- Publish papers using the platform
- Share with academic community
- Scale based on usage

## Security Considerations

1. **Database Security**
   - SSL/TLS connections only
   - Password rotation
   - Backup strategy (daily)
   - Read-only users for public access

2. **Application Security**
   - Input validation
   - SQL injection prevention (use parameterized queries)
   - Rate limiting on API
   - HTTPS only

3. **Data Privacy**
   - No PII storage
   - GDPR compliance (if needed)
   - Clear data usage policy

## Backup & Disaster Recovery

```bash
# Automated PostgreSQL backups
# Run via cron: 0 3 * * * /path/to/backup.sh

#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup database
pg_dump -h localhost -U postgres practice_db | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Upload to S3/Cloud Storage
aws s3 cp $BACKUP_DIR/db_$DATE.sql.gz s3://your-bucket/backups/

# Keep only last 30 days
find $BACKUP_DIR -name "db_*.sql.gz" -mtime +30 -delete
```

## Next Steps

1. **This Week**: Fix Streamlit Cloud deployment
2. **Next Week**: Set up VS Code + local PostgreSQL
3. **Week 3-4**: Choose production deployment platform (Railway vs Docker)
4. **Month 2**: Implement automated data pipeline
5. **Month 3**: Add testing and CI/CD

## Resources

- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Railway.app Docs](https://docs.railway.app/)
- [Docker Docs](https://docs.docker.com/)
- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Don%27t_Do_This)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
