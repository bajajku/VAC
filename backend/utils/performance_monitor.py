"""
Performance monitoring utilities for tracking streaming optimizations.
"""
import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics

class PerformanceMonitor:
    """Monitor and track performance metrics for streaming operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "stream_init_time": [],
            "first_chunk_time": [],
            "total_stream_time": [],
            "db_operation_time": [],
            "retrieval_time": [],
            "session_validation_time": []
        }
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> Optional[float]:
        """End timing an operation and record the duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation in self.metrics:
                self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return None
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        # Get recent values within window
        values = self.metrics[metric_name][-window_size:]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "recent": values[-1] if values else 0
        }
    
    def get_all_stats(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            metric: self.get_stats(metric, window_size) 
            for metric in self.metrics.keys()
        }
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        for metric in self.metrics:
            self.metrics[metric].clear()
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        stats = self.get_all_stats()
        
        summary = "🚀 **Performance Summary**\n\n"
        
        for metric, data in stats.items():
            if not data:
                continue
                
            metric_display = metric.replace("_", " ").title()
            summary += f"**{metric_display}:**\n"
            summary += f"  - Average: {data['avg']:.3f}s\n"
            summary += f"  - Median: {data['median']:.3f}s\n"
            summary += f"  - P95: {data['p95']:.3f}s\n"
            summary += f"  - Recent: {data['recent']:.3f}s\n"
            summary += f"  - Count: {data['count']}\n\n"
        
        return summary

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def track_performance(operation_name: str):
    """Decorator to track performance of functions."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                performance_monitor.start_timer(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    performance_monitor.end_timer(operation_name)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                performance_monitor.start_timer(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    performance_monitor.end_timer(operation_name)
            return sync_wrapper
    return decorator