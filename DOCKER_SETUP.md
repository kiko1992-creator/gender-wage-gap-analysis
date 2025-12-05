# Docker Setup Guide (OPTIONAL)

## When to Use Docker

Use Docker if you want:
- ✓ Consistent environment across different machines
- ✓ Easy sharing with team members
- ✓ No Python installation needed on host
- ✓ Isolated environment

**Most users DON'T need this** - regular Python setup is simpler!

---

## Prerequisites

1. Install Docker Desktop: https://www.docker.com/products/docker-desktop
2. Ensure Docker is running (check system tray icon)

---

## Quick Start with Docker

### Method 1: Using Docker Compose (Recommended)

```bash
# 1. Clone repo
git clone https://github.com/kiko1992-creator/gender-wage-gap-analysis.git
cd gender-wage-gap-analysis

# 2. Build and start
docker-compose up --build

# 3. Open browser to: http://localhost:8888
```

**That's it!** Jupyter opens in browser, all notebooks ready.

### Method 2: Using Docker directly

```bash
# Build image
docker build -t gender-wage-gap .

# Run container
docker run -p 8888:8888 -v "%cd%":/app gender-wage-gap
```

---

## Access Jupyter

Once container is running:
1. Open browser
2. Go to: http://localhost:8888
3. All notebooks are in the `notebooks/` folder

---

## Useful Commands

```bash
# Stop container
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Remove everything (clean slate)
docker-compose down -v
docker system prune -a
```

---

## Updating Code

Changes you make in notebooks are automatically saved to your local folder (due to volume mounting).

```bash
# After making changes:
git add .
git commit -m "Your changes"
git push
```

---

## Troubleshooting

### Port 8888 already in use:
```bash
# Change port in docker-compose.yml:
ports:
  - "9999:8888"  # Use 9999 instead

# Then access: http://localhost:9999
```

### Container won't start:
```bash
# Check Docker is running
docker info

# View errors
docker-compose logs
```

### Out of disk space:
```bash
# Clean up unused Docker resources
docker system prune -a
```

---

## Comparison: Docker vs Local

| Feature | Local Setup | Docker |
|---------|-------------|--------|
| **Setup Time** | 5 minutes | 10-15 minutes (first time) |
| **Disk Space** | ~500 MB | ~2 GB |
| **Speed** | Faster | Slightly slower |
| **Consistency** | Depends on system | Guaranteed |
| **Ease of Use** | Simpler | More complex |
| **Best For** | Personal use | Teams, sharing |

**Recommendation:** Start with local setup. Use Docker only if you need it.

---

## Performance Notes

- First build takes ~5-10 minutes (downloads Python image)
- Subsequent builds are much faster (uses cache)
- Running notebooks is same speed as local
- Data persists in local folder (not lost when container stops)

---

**Still prefer simple setup?** Just use regular Python + pip (see main README.md)
