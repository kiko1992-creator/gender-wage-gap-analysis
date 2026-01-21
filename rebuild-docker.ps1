# Complete Docker Rebuild Script for Windows PowerShell
# This script completely removes and rebuilds all Docker containers and images

Write-Host "🔄 Starting complete Docker rebuild..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop all running containers
Write-Host "1️⃣ Stopping all containers..." -ForegroundColor Yellow
docker-compose down
if ($LASTEXITCODE -ne 0) {
    docker compose down
}

# Step 2: Remove all containers, networks, and volumes
Write-Host "2️⃣ Removing all containers, networks, and volumes..." -ForegroundColor Yellow
docker-compose down -v --remove-orphans
if ($LASTEXITCODE -ne 0) {
    docker compose down -v --remove-orphans
}

# Step 3: Remove Docker images for this project
Write-Host "3️⃣ Removing project Docker images..." -ForegroundColor Yellow
docker images | Select-String "gender-wage-gap-analysis" | ForEach-Object {
    $imageId = ($_ -split '\s+')[2]
    docker rmi -f $imageId
}

# Step 4: Prune Docker build cache
Write-Host "4️⃣ Pruning Docker build cache..." -ForegroundColor Yellow
docker builder prune -f

# Step 5: Rebuild from scratch
Write-Host "5️⃣ Building containers from scratch (this will take 3-5 minutes)..." -ForegroundColor Yellow
docker-compose build --no-cache
if ($LASTEXITCODE -ne 0) {
    docker compose build --no-cache
}

# Step 6: Start containers
Write-Host "6️⃣ Starting containers..." -ForegroundColor Yellow
docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    docker compose up -d
}

Write-Host ""
Write-Host "✅ Rebuild complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Your applications should be available at:" -ForegroundColor Cyan
Write-Host "   Streamlit: http://localhost:8501" -ForegroundColor White
Write-Host "   Jupyter:   http://localhost:8888" -ForegroundColor White
Write-Host ""
Write-Host "📊 View logs with: docker-compose logs -f" -ForegroundColor Yellow
Write-Host "🛑 Stop with: docker-compose down" -ForegroundColor Yellow
