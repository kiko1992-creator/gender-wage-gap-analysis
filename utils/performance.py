"""
Performance Monitoring and Optimization Utilities
Track and optimize Streamlit app performance

Features:
- Execution time tracking
- Memory usage monitoring
- Cache hit rate tracking
- Performance profiling
- Optimization recommendations
"""

import streamlit as st
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
import psutil
import os
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# EXECUTION TIME TRACKING
# ============================================================

class PerformanceTimer:
    """Track execution time of operations"""

    def __init__(self, operation_name: str, show_in_ui: bool = False):
        self.operation_name = operation_name
        self.show_in_ui = show_in_ui
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏱️  Started: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        logger.info(f"✅ Completed: {self.operation_name} ({duration:.3f}s)")

        if self.show_in_ui:
            st.caption(f"⏱️ {self.operation_name}: {duration:.3f}s")

        return False

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def time_execution(operation_name: str = None, show_in_ui: bool = False):
    """
    Decorator to time function execution

    Usage:
        @time_execution("Load data")
        def load_data():
            return pd.read_csv('file.csv')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            with PerformanceTimer(name, show_in_ui):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# MEMORY MONITORING
# ============================================================

class MemoryMonitor:
    """Monitor memory usage"""

    @staticmethod
    def get_memory_usage() -> dict:
        """
        Get current memory usage statistics

        Returns:
            Dict with memory metrics in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }

    @staticmethod
    def show_memory_usage():
        """Display current memory usage in Streamlit"""
        mem = MemoryMonitor.get_memory_usage()
        st.caption(f"💾 Memory: {mem['rss_mb']:.1f} MB ({mem['percent']:.1f}%)")

    @staticmethod
    @contextmanager
    def track_memory(operation_name: str, show_in_ui: bool = False):
        """
        Context manager to track memory usage delta

        Usage:
            with MemoryMonitor.track_memory("Load data"):
                df = load_large_dataset()
        """
        start_mem = MemoryMonitor.get_memory_usage()
        logger.info(f"📊 Memory before {operation_name}: {start_mem['rss_mb']:.1f} MB")

        yield

        end_mem = MemoryMonitor.get_memory_usage()
        delta = end_mem['rss_mb'] - start_mem['rss_mb']

        logger.info(f"📊 Memory after {operation_name}: {end_mem['rss_mb']:.1f} MB (Δ {delta:+.1f} MB)")

        if show_in_ui:
            st.caption(f"💾 Memory change: {delta:+.1f} MB")


# ============================================================
# CACHE MONITORING
# ============================================================

class CacheMonitor:
    """Monitor Streamlit cache performance"""

    _cache_hits = {}
    _cache_misses = {}

    @classmethod
    def record_hit(cls, cache_key: str):
        """Record a cache hit"""
        cls._cache_hits[cache_key] = cls._cache_hits.get(cache_key, 0) + 1

    @classmethod
    def record_miss(cls, cache_key: str):
        """Record a cache miss"""
        cls._cache_misses[cache_key] = cls._cache_misses.get(cache_key, 0) + 1

    @classmethod
    def get_stats(cls) -> dict:
        """Get cache statistics"""
        all_keys = set(cls._cache_hits.keys()) | set(cls._cache_misses.keys())

        stats = {}
        for key in all_keys:
            hits = cls._cache_hits.get(key, 0)
            misses = cls._cache_misses.get(key, 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0

            stats[key] = {
                'hits': hits,
                'misses': misses,
                'total': total,
                'hit_rate': hit_rate
            }

        return stats

    @classmethod
    def show_stats(cls):
        """Display cache statistics in Streamlit"""
        stats = cls.get_stats()

        if not stats:
            st.info("No cache statistics available yet")
            return

        st.subheader("📊 Cache Performance")

        for key, data in stats.items():
            with st.expander(f"🔑 {key}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Hits", data['hits'])
                col2.metric("Misses", data['misses'])
                col3.metric("Total", data['total'])
                col4.metric("Hit Rate", f"{data['hit_rate']:.1f}%")

    @classmethod
    def reset_stats(cls):
        """Reset all cache statistics"""
        cls._cache_hits.clear()
        cls._cache_misses.clear()
        logger.info("Cache statistics reset")


def cached_with_monitoring(ttl: int = 3600, cache_key: str = None):
    """
    Decorator combining Streamlit caching with monitoring

    Usage:
        @cached_with_monitoring(ttl=3600, cache_key="data_load")
        def load_data():
            return expensive_operation()
    """
    def decorator(func: Callable) -> Callable:
        # Apply Streamlit cache
        cached_func = st.cache_data(ttl=ttl)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache_key or func.__name__

            # Check if result is cached (approximation)
            try:
                result = cached_func(*args, **kwargs)
                CacheMonitor.record_hit(key)
                logger.debug(f"Cache HIT: {key}")
            except:
                CacheMonitor.record_miss(key)
                logger.debug(f"Cache MISS: {key}")
                raise

            return result

        return wrapper
    return decorator


# ============================================================
# PERFORMANCE PROFILING
# ============================================================

class PerformanceProfiler:
    """Profile app performance and provide recommendations"""

    def __init__(self):
        self.metrics = {
            'page_load_times': [],
            'query_times': [],
            'render_times': [],
            'memory_snapshots': []
        }

    def record_page_load(self, duration: float):
        """Record page load time"""
        self.metrics['page_load_times'].append({
            'timestamp': datetime.now(),
            'duration': duration
        })

    def record_query(self, query_name: str, duration: float):
        """Record database query time"""
        self.metrics['query_times'].append({
            'timestamp': datetime.now(),
            'query': query_name,
            'duration': duration
        })

    def record_render(self, component: str, duration: float):
        """Record component render time"""
        self.metrics['render_times'].append({
            'timestamp': datetime.now(),
            'component': component,
            'duration': duration
        })

    def snapshot_memory(self):
        """Take memory snapshot"""
        mem = MemoryMonitor.get_memory_usage()
        self.metrics['memory_snapshots'].append({
            'timestamp': datetime.now(),
            'memory_mb': mem['rss_mb']
        })

    def get_recommendations(self) -> list:
        """
        Analyze metrics and provide optimization recommendations

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check page load times
        if self.metrics['page_load_times']:
            avg_load = sum(m['duration'] for m in self.metrics['page_load_times']) / len(self.metrics['page_load_times'])
            if avg_load > 3.0:
                recommendations.append("⚠️ Average page load time > 3s. Consider implementing lazy loading.")

        # Check query times
        if self.metrics['query_times']:
            slow_queries = [q for q in self.metrics['query_times'] if q['duration'] > 1.0]
            if slow_queries:
                recommendations.append(f"⚠️ {len(slow_queries)} slow queries detected. Consider adding database indexes.")

        # Check memory usage
        if self.metrics['memory_snapshots']:
            max_mem = max(m['memory_mb'] for m in self.metrics['memory_snapshots'])
            if max_mem > 500:
                recommendations.append("⚠️ High memory usage (>500 MB). Consider data chunking or pagination.")

        # Check render times
        if self.metrics['render_times']:
            slow_renders = [r for r in self.metrics['render_times'] if r['duration'] > 0.5]
            if slow_renders:
                recommendations.append(f"⚠️ {len(slow_renders)} slow renders. Consider optimizing visualizations.")

        if not recommendations:
            recommendations.append("✅ No performance issues detected!")

        return recommendations

    def show_report(self):
        """Display performance report in Streamlit"""
        st.subheader("📊 Performance Report")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        if self.metrics['page_load_times']:
            avg_load = sum(m['duration'] for m in self.metrics['page_load_times']) / len(self.metrics['page_load_times'])
            col1.metric("Avg Page Load", f"{avg_load:.2f}s")

        if self.metrics['query_times']:
            avg_query = sum(m['duration'] for m in self.metrics['query_times']) / len(self.metrics['query_times'])
            col2.metric("Avg Query Time", f"{avg_query:.3f}s")

        if self.metrics['memory_snapshots']:
            current_mem = self.metrics['memory_snapshots'][-1]['memory_mb']
            col3.metric("Current Memory", f"{current_mem:.1f} MB")

        col4.metric("Total Queries", len(self.metrics['query_times']))

        # Recommendations
        st.subheader("💡 Recommendations")
        for rec in self.get_recommendations():
            st.write(rec)


# ============================================================
# GLOBAL PROFILER INSTANCE
# ============================================================

# Create global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    return _profiler


# ============================================================
# STREAMLIT INTEGRATION
# ============================================================

def show_performance_sidebar():
    """Show performance metrics in sidebar"""
    with st.sidebar:
        with st.expander("⚡ Performance", expanded=False):
            MemoryMonitor.show_memory_usage()
            CacheMonitor.show_stats()

            if st.button("Clear Cache"):
                st.cache_data.clear()
                CacheMonitor.reset_stats()
                st.success("Cache cleared!")


def enable_performance_monitoring():
    """
    Enable performance monitoring for the app
    Call this at the top of your main app
    """
    # Show performance info in sidebar
    show_performance_sidebar()

    # Return profiler for custom tracking
    return get_profiler()
