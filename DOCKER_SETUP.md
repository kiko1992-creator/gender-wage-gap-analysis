# 🐳 Docker Setup Guide

Simple guide to run the Gender Wage Gap Analysis project using Docker.

## Prerequisites

- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- That's it! No need to install Python, PostgreSQL, or any dependencies manually.

## Quick Start (3 Simple Steps)

### 1. Start Everything

Open a terminal in this project folder and run:

```bash
docker-compose up
```

Wait 1-2 minutes for everything to start. You'll see logs from all services.

### 2. Access Your Applications

Once running, open your browser:

- **Streamlit Dashboard**: http://localhost:8501
- **Jupyter Notebooks**: http://localhost:8888
- **PostgreSQL Database**: localhost:5432

### 3. Stop Everything

Press `Ctrl+C` in the terminal, then run:

```bash
docker-compose down
```

## Common Commands

### Start in Background (Detached Mode)
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop All Services
```bash
docker-compose down
```

### Rebuild After Code Changes
```bash
docker-compose up --build
```

### Remove Everything (Including Database Data)
```bash
docker-compose down -v
```

## What's Running?

The `docker-compose up` command starts 3 services:

1. **PostgreSQL Database** (`postgres`)
   - Database name: `eu_wage_gap_research`
   - Port: 5432
   - User: postgres
   - Password: postgres123 (configurable)

2. **Streamlit App** (`streamlit`)
   - Interactive dashboard
   - Port: 8501
   - Auto-connects to database

3. **Jupyter Lab** (`jupyter`)
   - For running notebooks
   - Port: 8888
   - No password required (dev environment)

## Customization (Optional)

### Change Database Password

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and change `POSTGRES_PASSWORD`

3. Restart:
   ```bash
   docker-compose down
   docker-compose up
   ```

## Troubleshooting

### Port Already in Use

If you get an error like "port 8501 is already allocated":

1. Check what's using the port:
   ```bash
   # Windows
   netstat -ano | findstr :8501

   # Mac/Linux
   lsof -i :8501
   ```

2. Either stop that process or change the port in `docker-compose.yml`

### Database Connection Errors

If the app can't connect to the database:

1. Check if PostgreSQL is running:
   ```bash
   docker-compose ps
   ```

2. View database logs:
   ```bash
   docker-compose logs postgres
   ```

3. Wait a few more seconds - the database takes ~10 seconds to initialize

### Container Won't Start

1. View detailed logs:
   ```bash
   docker-compose logs
   ```

2. Try rebuilding:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## Development Tips

### Live Code Updates

Your code changes are automatically reflected (no rebuild needed) because:
- The project folder is mounted as a volume
- Streamlit watches for file changes
- Just refresh your browser!

### Access Database Directly

Use any PostgreSQL client with these credentials:
- Host: localhost
- Port: 5432
- Database: eu_wage_gap_research
- User: postgres
- Password: postgres123

### Run Commands in Container

```bash
# Open a shell in the Streamlit container
docker-compose exec streamlit bash

# Run a Python script
docker-compose exec streamlit python scripts/your_script.py

# Access PostgreSQL CLI
docker-compose exec postgres psql -U postgres -d eu_wage_gap_research
```

## Need Help?

- Docker not working? Make sure Docker Desktop is running
- Still stuck? Check the logs: `docker-compose logs`
- Want to start fresh? `docker-compose down -v && docker-compose up --build`

---

**That's it!** You now have a complete data analysis environment running with zero manual configuration. 🎉
