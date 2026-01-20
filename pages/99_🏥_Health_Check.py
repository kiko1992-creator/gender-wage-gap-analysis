"""
System Health Check Dashboard
Monitor application health, performance, and resource usage

Features:
- Database connectivity check
- Cache statistics
- Memory and CPU usage
- Configuration display
- Performance metrics
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, configure_streamlit_page
from database_pool import db_pool, check_database_health, clear_all_caches
from utils.performance import MemoryMonitor, CacheMonitor, get_profiler
from utils.error_handler import ErrorContext
import psutil
from datetime import datetime

# Configure page
configure_streamlit_page("Health Check")

# ============================================================
# HEADER
# ============================================================

st.title("🏥 System Health Check")
st.write("Monitor application health, performance, and resource usage")

# ============================================================
# ENVIRONMENT INFO
# ============================================================

st.header("🌍 Environment")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Environment", config.environment.upper())

with col2:
    st.metric("Version", config.get('APP_VERSION'))

with col3:
    st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# ============================================================
# DATABASE HEALTH
# ============================================================

st.header("🗄️ Database Status")

db_health = check_database_health()

col1, col2, col3 = st.columns(3)

with col1:
    status_icon = "✅" if db_health['available'] else "❌"
    st.metric("Status", f"{status_icon} {'Connected' if db_health['available'] else 'Disconnected'}")

with col2:
    data_source = "Sample Data" if db_health['using_sample_data'] else "PostgreSQL"
    st.metric("Data Source", data_source)

with col3:
    conn_type = db_health.get('connection_type', 'N/A') if db_health['available'] else 'N/A'
    st.metric("Connection Type", conn_type.upper())

# Database connection details
with st.expander("🔍 Database Details"):
    if db_health['available']:
        st.success("✅ Database connection pool is active")
        st.write(f"**Connection Type:** {conn_type}")
        if conn_type == 'docker':
            st.write(f"**Host:** {os.getenv('POSTGRES_HOST', 'N/A')}")
            st.write(f"**Port:** {os.getenv('POSTGRES_PORT', 'N/A')}")
            st.write(f"**Database:** {os.getenv('POSTGRES_DB', 'N/A')}")
    else:
        st.warning("⚠️ Database not available - using sample data fallback")
        st.info("This is normal for cloud deployments or when PostgreSQL is not installed")

# ============================================================
# MEMORY & CPU
# ============================================================

st.header("💾 Resource Usage")

mem_info = MemoryMonitor.get_memory_usage()
process = psutil.Process(os.getpid())
cpu_percent = process.cpu_percent(interval=1)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Memory Usage",
        f"{mem_info['rss_mb']:.1f} MB",
        f"{mem_info['percent']:.1f}%"
    )

with col2:
    st.metric("CPU Usage", f"{cpu_percent:.1f}%")

with col3:
    # System memory
    system_mem = psutil.virtual_memory()
    st.metric("System Memory", f"{system_mem.percent:.1f}% used")

# Memory details
with st.expander("🔍 Memory Details"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Process Memory:**")
        st.write(f"- RSS: {mem_info['rss_mb']:.1f} MB")
        st.write(f"- VMS: {mem_info['vms_mb']:.1f} MB")
        st.write(f"- Percent: {mem_info['percent']:.1f}%")

    with col2:
        st.write("**System Memory:**")
        st.write(f"- Total: {system_mem.total / 1024 / 1024 / 1024:.1f} GB")
        st.write(f"- Available: {system_mem.available / 1024 / 1024 / 1024:.1f} GB")
        st.write(f"- Used: {system_mem.percent:.1f}%")

# ============================================================
# CACHE STATISTICS
# ============================================================

st.header("🗂️ Cache Performance")

cache_stats = CacheMonitor.get_stats()

if cache_stats:
    # Overall stats
    total_hits = sum(s['hits'] for s in cache_stats.values())
    total_misses = sum(s['misses'] for s in cache_stats.values())
    total_requests = total_hits + total_misses
    overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Hits", total_hits)

    with col2:
        st.metric("Total Misses", total_misses)

    with col3:
        st.metric("Total Requests", total_requests)

    with col4:
        st.metric("Hit Rate", f"{overall_hit_rate:.1f}%")

    # Per-cache details
    with st.expander("🔍 Cache Details"):
        for key, stats in cache_stats.items():
            st.subheader(f"📦 {key}")
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**Hits:** {stats['hits']}")
            col2.write(f"**Misses:** {stats['misses']}")
            col3.write(f"**Total:** {stats['total']}")
            col4.write(f"**Hit Rate:** {stats['hit_rate']:.1f}%")
else:
    st.info("No cache statistics available yet. Cache stats are recorded as you use the app.")

# Cache management
st.subheader("🔧 Cache Management")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear All Caches", type="primary"):
        with ErrorContext("Clearing caches"):
            clear_all_caches()
            CacheMonitor.reset_stats()
            st.success("✅ All caches cleared successfully!")
            st.rerun()

with col2:
    if st.button("Reset Cache Statistics"):
        CacheMonitor.reset_stats()
        st.success("✅ Cache statistics reset!")
        st.rerun()

# ============================================================
# CONFIGURATION
# ============================================================

st.header("⚙️ Configuration")

config_data = config.get_all()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 App Settings")
    st.write(f"**App Name:** {config_data.get('APP_NAME')}")
    st.write(f"**Version:** {config_data.get('APP_VERSION')}")
    st.write(f"**Author:** {config_data.get('APP_AUTHOR')}")
    st.write(f"**Layout:** {config_data.get('LAYOUT')}")

with col2:
    st.subheader("🔧 Performance Settings")
    st.write(f"**Cache TTL (Short):** {config_data.get('CACHE_TTL_SHORT')}s")
    st.write(f"**Cache TTL (Medium):** {config_data.get('CACHE_TTL_MEDIUM')}s")
    st.write(f"**Cache TTL (Long):** {config_data.get('CACHE_TTL_LONG')}s")
    st.write(f"**Max Upload Size:** {config_data.get('MAX_UPLOAD_SIZE_MB')} MB")

# Feature flags
with st.expander("🚀 Feature Flags"):
    st.write(f"**Database:** {'✅ Enabled' if config_data.get('ENABLE_DATABASE') else '❌ Disabled'}")
    st.write(f"**Dark Mode:** {'✅ Enabled' if config_data.get('ENABLE_DARK_MODE') else '❌ Disabled'}")
    st.write(f"**Data Export:** {'✅ Enabled' if config_data.get('ENABLE_DATA_EXPORT') else '❌ Disabled'}")
    st.write(f"**Advanced Stats:** {'✅ Enabled' if config_data.get('ENABLE_ADVANCED_STATS') else '❌ Disabled'}")
    st.write(f"**Profiling:** {'✅ Enabled' if config_data.get('ENABLE_PROFILING') else '❌ Disabled'}")
    st.write(f"**Debug Info:** {'✅ Enabled' if config_data.get('ENABLE_DEBUG_INFO') else '❌ Disabled'}")

# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.header("⚡ Performance Metrics")

profiler = get_profiler()

# Show recommendations
recommendations = profiler.get_recommendations()

for rec in recommendations:
    if "⚠️" in rec:
        st.warning(rec)
    else:
        st.success(rec)

# Detailed metrics
with st.expander("🔍 Detailed Metrics"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Page Load Times")
        if profiler.metrics['page_load_times']:
            for metric in profiler.metrics['page_load_times'][-5:]:  # Last 5
                st.write(f"- {metric['duration']:.3f}s at {metric['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("No page load metrics yet")

    with col2:
        st.subheader("🗄️ Query Times")
        if profiler.metrics['query_times']:
            for metric in profiler.metrics['query_times'][-5:]:  # Last 5
                st.write(f"- {metric['query']}: {metric['duration']:.3f}s")
        else:
            st.info("No query metrics yet")

# ============================================================
# SYSTEM INFO
# ============================================================

st.header("🖥️ System Information")

with st.expander("🔍 System Details"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Platform:**")
        st.write(f"- OS: {os.name}")
        st.write(f"- Python: {sys.version.split()[0]}")
        st.write(f"- PID: {os.getpid()}")

    with col2:
        st.write("**Paths:**")
        st.write(f"- Working Dir: {os.getcwd()}")
        st.write(f"- Script Dir: {Path(__file__).parent.parent}")

# ============================================================
# HEALTH CHECK SUMMARY
# ============================================================

st.header("✅ Health Check Summary")

# Calculate overall health score
health_score = 0
max_score = 5

# Database
if db_health['available'] or db_health['using_sample_data']:
    health_score += 1

# Memory
if mem_info['percent'] < 80:
    health_score += 1

# CPU
if cpu_percent < 80:
    health_score += 1

# Cache (if any hits)
if cache_stats and total_hits > 0:
    health_score += 1
else:
    health_score += 0.5  # Partial credit if no stats yet

# Configuration
if config.environment in ['development', 'production', 'cloud']:
    health_score += 1

# Display score
health_percentage = (health_score / max_score) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Health Score", f"{health_score}/{max_score}")

with col2:
    st.metric("Health Percentage", f"{health_percentage:.0f}%")

with col3:
    if health_percentage >= 80:
        st.success("✅ Healthy")
    elif health_percentage >= 60:
        st.warning("⚠️ Fair")
    else:
        st.error("❌ Needs Attention")

# Refresh button
st.divider()

col1, col2, col3 = st.columns(3)

with col2:
    if st.button("🔄 Refresh Health Check", type="primary"):
        st.rerun()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
