#!/bin/bash
# Quick Start Script for Gender Wage Gap Analysis
# Automatically sets up Docker environment

set -e  # Exit on error

echo "=================================="
echo "Gender Wage Gap Analysis"
echo "Docker Quick Start"
echo "=================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop"
    exit 1
fi

echo "✅ Docker is installed and running"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "💡 You can edit .env to customize passwords and ports"
else
    echo "✅ .env file already exists"
fi
echo ""

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "=================================="
    echo "✅ SUCCESS! Application is running"
    echo "=================================="
    echo ""
    echo "🌐 Streamlit App: http://localhost:8501"
    echo "🗄️  PostgreSQL:    localhost:5432"
    echo ""
    echo "Useful commands:"
    echo "  make logs       - View application logs"
    echo "  make logs-db    - View database logs"
    echo "  make down       - Stop all services"
    echo "  make shell-db   - Open database shell"
    echo ""
    echo "Opening browser..."

    # Try to open browser
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8501 &> /dev/null &
    elif command -v open &> /dev/null; then
        open http://localhost:8501 &> /dev/null &
    fi
else
    echo ""
    echo "⚠️  Services started but may not be healthy yet"
    echo "Run 'make logs' to see what's happening"
fi

echo ""
echo "To view logs: make logs"
echo "To stop:      make down"
echo ""
