# Docker Setup Guide - Gender Wage Gap Analysis

## Quick Start (5 Minutes)

### Prerequisites
- Docker Desktop installed and running
- Git (to clone the repository)

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/kiko1992-creator/gender-wage-gap-analysis.git
cd gender-wage-gap-analysis

# Copy environment file
cp .env.example .env

# (Optional) Edit .env if you want custom passwords
```

### Step 2: Start Everything

```bash
# Build and start all services
make up

# Or without Makefile:
docker-compose up -d
```

That's it! Your application is now running at:
- **Streamlit App**: http://localhost:8501
- **PostgreSQL**: localhost:5432

### Step 3: Verify It's Working

```bash
# Check service status
make ps

# View logs
make logs

# Check health
make health
```

## What Just Happened?

Docker automatically:
1. ✅ Built the Streamlit application image
2. ✅ Started PostgreSQL 16 database
3. ✅ Created database tables (`countries`, `wage_gap_data`, etc.)
4. ✅ Loaded sample data (EU27 countries, wage gap records)
5. ✅ Connected Streamlit app to database
6. ✅ Made app available at http://localhost:8501

## Available Commands (Makefile)

### Basic Operations

```bash
make up              # Start all services
make down            # Stop all services
make restart         # Restart all services
make ps              # Show running containers
make logs            # View all logs
make logs-app        # View only Streamlit logs
make logs-db         # View only PostgreSQL logs
```

### Development Mode

```bash
make up-dev          # Start with pgAdmin (database UI)
```

After running, access:
- **pgAdmin**: http://localhost:5050
  - Email: `admin@wagegap.local`
  - Password: `admin`
  - PostgreSQL connection:
    - Host: `postgres`
    - Port: `5432`
    - Database: `practice_db`
    - Username: `postgres`
    - Password: `postgres`

### Database Operations

```bash
make shell-db        # Open PostgreSQL shell
make db-backup       # Create database backup
make db-restore FILE=backups/backup_xxx.sql  # Restore backup
make db-init         # Reinitialize database (if needed)
```

### Cleanup

```bash
make clean           # Remove containers and volumes
make clean-all       # Remove everything including images
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Docker Compose Orchestration               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌────────────────┐  │
│  │  Streamlit   │──────│  PostgreSQL 16 │  │
│  │  Container   │      │  Container     │  │
│  │              │      │                │  │
│  │  Port: 8501  │      │  Port: 5432    │  │
│  └──────────────┘      └────────────────┘  │
│         │                      │            │
│         │                      │            │
│    ┌────▼──────────────────────▼─────┐     │
│    │   Named Volumes (Persistent)    │     │
│    │   - postgres_data               │     │
│    │   - backups/                    │     │
│    └─────────────────────────────────┘     │
│                                             │
│  Optional (Development):                   │
│  ┌──────────────┐                          │
│  │  pgAdmin     │                          │
│  │  Port: 5050  │                          │
│  └──────────────┘                          │
└─────────────────────────────────────────────┘
```

## Directory Structure

```
gender-wage-gap-analysis/
├── Dockerfile                  # Production app image
├── docker-compose.yml          # Service orchestration
├── .dockerignore              # Files to exclude from image
├── .env.example               # Environment template
├── .env                       # Your local config (git ignored)
├── Makefile                   # Easy commands
├── docker/
│   └── init-db/               # Database initialization
│       ├── 01_create_tables.sql   # Table schemas
│       └── 02_seed_data.sql       # Sample data
└── backups/                   # Database backups
```

## Environment Variables

Edit `.env` file to customize:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=practice_db

# Ports (if 8501 is already used)
STREAMLIT_PORT=8502
POSTGRES_PORT=5433

# Development tools
PGADMIN_EMAIL=your@email.com
PGADMIN_PASSWORD=admin123
```

## Common Workflows

### 1. Daily Development

```bash
# Morning: Start everything
make up

# Code in your editor (VS Code)
# Streamlit auto-reloads on file changes

# View logs if something breaks
make logs-app

# Evening: Stop everything
make down
```

### 2. Database Changes

```bash
# Open database shell
make shell-db

# Run SQL commands
practice_db=# SELECT * FROM countries LIMIT 5;
practice_db=# \dt  -- List tables
practice_db=# \q   -- Exit

# Or backup before making changes
make db-backup
```

### 3. Fresh Start

```bash
# Nuclear option: delete everything and start fresh
make clean-all
make build
make up
```

## Troubleshooting

### Issue: Port 8501 already in use

**Solution 1**: Stop other Streamlit apps
```bash
# Find process using port 8501
lsof -ti:8501 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :8501   # Windows
```

**Solution 2**: Change port in `.env`
```bash
echo "STREAMLIT_PORT=8502" >> .env
make restart
```

### Issue: Database connection failed

**Check if PostgreSQL is running:**
```bash
make ps
# Should show postgres container as "Up" and "healthy"
```

**Check logs:**
```bash
make logs-db
```

**Reinitialize database:**
```bash
make down
make clean
make up
```

### Issue: Changes not reflected in app

**For code changes:**
- Streamlit auto-reloads Python files
- If not working, refresh browser (Ctrl+F5)

**For Dockerfile changes:**
```bash
make down
make build
make up
```

**For database schema changes:**
```bash
# Recreate database
make down
make clean
make up
```

### Issue: Out of disk space

**Clean old Docker resources:**
```bash
docker system prune -a --volumes
# WARNING: This removes ALL unused Docker data
```

## Production Deployment

### Deploy to Cloud (Railway.app Example)

1. **Push your code to GitHub**
```bash
git add .
git commit -m "Add Docker configuration"
git push origin master
```

2. **Create Railway.app account**
   - Visit https://railway.app
   - Connect GitHub account

3. **Create new project**
   - "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects docker-compose.yml

4. **Set environment variables**
   - Add `POSTGRES_PASSWORD` (secure password)
   - Railway auto-provisions PostgreSQL

5. **Deploy**
   - Railway builds and deploys automatically
   - Get public URL: `https://your-app.railway.app`

### Deploy to AWS/GCP/Azure

See `INFRASTRUCTURE.md` for detailed cloud deployment guides.

## Performance Optimization

### For Production

Edit `docker-compose.yml`:

```yaml
streamlit:
  # Remove development volume mounts (lines 45-49)
  # volumes:
  #   - ./app.py:/app/app.py  # Comment out

  # Add production environment
  environment:
    STREAMLIT_SERVER_FILE_WATCHER_TYPE: none
    STREAMLIT_SERVER_RUN_ON_SAVE: false
```

### Database Performance

```bash
# Open database shell
make shell-db

# Create additional indexes
CREATE INDEX idx_wage_gap_data_year ON wage_gap_data(year);
CREATE INDEX idx_wage_gap_data_country_code ON wage_gap_data(country_code);

# Analyze query performance
EXPLAIN ANALYZE SELECT * FROM wage_gap_data WHERE year = 2023;
```

## Data Persistence

### Where is data stored?

```bash
# View volumes
docker volume ls

# Inspect postgres volume
docker volume inspect gender-wage-gap-analysis_postgres_data
```

### Backup Strategy

**Automated daily backups** (add to crontab):
```bash
# Add to crontab -e
0 2 * * * cd /path/to/project && make db-backup
```

**Manual backup before changes:**
```bash
make db-backup
# Creates: backups/backup_20260118_143022.sql
```

## Next Steps

1. ✅ **You are here**: Docker setup complete
2. **Load your data**: Import real Eurostat/World Bank data
3. **Customize pages**: Edit pages in `pages/` directory
4. **Add features**: Implement new analysis methods
5. **Deploy to cloud**: Make it accessible worldwide

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [PostgreSQL in Docker](https://hub.docker.com/_/postgres)
- [Streamlit in Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

## Questions?

Check:
1. `DEPLOYMENT.md` - Cloud deployment guide
2. `INFRASTRUCTURE.md` - Long-term architecture plan
3. `VSCODE_SETUP.md` - Local development setup
4. GitHub Issues: https://github.com/kiko1992-creator/gender-wage-gap-analysis/issues
