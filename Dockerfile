# Dockerfile for Gender Wage Gap Analysis Project
# Supports Streamlit app and Jupyter notebooks

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter for notebooks support
RUN pip install --no-cache-dir jupyter jupyterlab

# Copy the entire project
COPY . .

# Expose ports
# 8501 for Streamlit
# 8888 for Jupyter
EXPOSE 8501 8888

# Default command (can be overridden in docker-compose)
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
