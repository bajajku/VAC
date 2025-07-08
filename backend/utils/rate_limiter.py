# Rate limiting setup
from fastapi import Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import Dict, List, Set
import asyncio
import time

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        
    async def is_rate_limited(self, ip: str) -> bool:
        current_time = time.time()
        
        if ip not in self.requests:
            self.requests[ip] = [current_time]
            return False
            
        # Filter out requests older than window
        self.requests[ip] = [t for t in self.requests[ip] 
                             if t > current_time - self.window_seconds]
        
        # Check if too many requests
        if len(self.requests[ip]) >= self.max_requests:
            return True
            
        # Add current request
        self.requests[ip].append(current_time)
        return False
        
    async def clean_old_entries(self):
        current_time = time.time()
        for ip in list(self.requests.keys()):
            self.requests[ip] = [t for t in self.requests[ip] 
                                if t > current_time - self.window_seconds]
            if not self.requests[ip]:
                del self.requests[ip]

# Create rate limiter for login attempts
login_rate_limiter = RateLimiter(max_requests=5, window_seconds=60)

# Automatic periodic cleanup task for rate limiter
async def cleanup_rate_limiter():
    while True:
        await asyncio.sleep(300)  # Clean every 5 minutes
        await login_rate_limiter.clean_old_entries()
