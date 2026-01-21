#!/bin/bash
# Complete Docker Rebuild Script for Linux/Mac
# This script completely removes and rebuilds all Docker containers and images

echo "🔄 Starting complete Docker rebuild..."
echo ""

# Step 1: Stop all running containers
echo "1️⃣ Stopping all containers..."
docker-compose down 2>/dev/null || docker compose down

# Step 2: Remove all containers, networks, and volumes
echo "2️⃣ Removing all containers, networks, and volumes..."
docker-compose down -v --remove-orphans 2>/dev/null || docker compose down -v --remove-orphans

# Step 3: Remove Docker images for this project
echo "3️⃣ Removing project Docker images..."
docker images | grep gender-wage-gap-analysis | awk '{print $3}' | xargs -r docker rmi -f

# Step 4: Prune Docker build cache
echo "4️⃣ Pruning Docker build cache..."
docker builder prune -f

# Step 5: Rebuild from scratch
echo "5️⃣ Building containers from scratch (this will take 3-5 minutes)..."
docker-compose build --no-cache 2>/dev/null || docker compose build --no-cache

# Step 6: Start containers
echo "6️⃣ Starting containers..."
docker-compose up -d 2>/dev/null || docker compose up -d

echo ""
echo "✅ Rebuild complete!"
echo ""
echo "🌐 Your applications should be available at:"
echo "   Streamlit: http://localhost:8501"
echo "   Jupyter:   http://localhost:8888"
echo ""
echo "📊 View logs with: docker-compose logs -f"
echo "🛑 Stop with: docker-compose down"
