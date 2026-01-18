.PHONY: help build up down restart logs ps clean clean-all db-backup db-restore test

# Default target
help:
	@echo "Gender Wage Gap Analysis - Docker Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build Docker images"
	@echo "  make up             - Start all services"
	@echo "  make up-dev         - Start with pgAdmin (development mode)"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View logs (all services)"
	@echo "  make logs-app       - View Streamlit app logs"
	@echo "  make logs-db        - View PostgreSQL logs"
	@echo "  make ps             - Show running containers"
	@echo "  make shell-app      - Open shell in Streamlit container"
	@echo "  make shell-db       - Open PostgreSQL shell"
	@echo "  make db-backup      - Backup PostgreSQL database"
	@echo "  make db-restore     - Restore PostgreSQL database"
	@echo "  make clean          - Remove containers and volumes"
	@echo "  make clean-all      - Remove everything including images"
	@echo "  make test           - Run tests in Docker"
	@echo ""

# Build Docker images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo "🌐 Streamlit app: http://localhost:8501"
	@echo "🗄️  PostgreSQL: localhost:5432"
	@echo ""
	@echo "Run 'make logs' to view logs"

# Start with development tools (pgAdmin)
up-dev:
	docker-compose --profile dev up -d
	@echo ""
	@echo "✅ Services started (development mode)!"
	@echo "🌐 Streamlit app: http://localhost:8501"
	@echo "🗄️  PostgreSQL: localhost:5432"
	@echo "🔧 pgAdmin: http://localhost:5050"
	@echo ""

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs (all services)
logs:
	docker-compose logs -f

# View Streamlit app logs
logs-app:
	docker-compose logs -f streamlit

# View PostgreSQL logs
logs-db:
	docker-compose logs -f postgres

# Show running containers
ps:
	docker-compose ps

# Open shell in Streamlit container
shell-app:
	docker-compose exec streamlit /bin/bash

# Open PostgreSQL shell
shell-db:
	docker-compose exec postgres psql -U postgres -d practice_db

# Backup database
db-backup:
	@mkdir -p backups
	@echo "Creating backup..."
	docker-compose exec -T postgres pg_dump -U postgres practice_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/ directory"

# Restore database (use: make db-restore FILE=backups/backup_xxx.sql)
db-restore:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Error: Please specify backup file: make db-restore FILE=backups/backup_xxx.sql"; \
		exit 1; \
	fi
	@echo "Restoring from $(FILE)..."
	docker-compose exec -T postgres psql -U postgres -d practice_db < $(FILE)
	@echo "✅ Database restored"

# Run tests
test:
	docker-compose exec streamlit pytest tests/ -v

# Clean containers and volumes
clean:
	docker-compose down -v
	@echo "✅ Containers and volumes removed"

# Clean everything including images
clean-all:
	docker-compose down -v --rmi all
	@echo "✅ Everything cleaned"

# Initialize database (if needed manually)
db-init:
	docker-compose exec -T postgres psql -U postgres -d practice_db < docker/init-db/01_create_tables.sql
	docker-compose exec -T postgres psql -U postgres -d practice_db < docker/init-db/02_seed_data.sql
	@echo "✅ Database initialized"

# Check health of services
health:
	@echo "Checking service health..."
	@docker-compose ps
	@echo ""
	@echo "PostgreSQL:"
	@docker-compose exec postgres pg_isready -U postgres || echo "❌ PostgreSQL not ready"
	@echo ""
	@echo "Streamlit:"
	@curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1 && echo "✅ Streamlit healthy" || echo "❌ Streamlit not ready"
