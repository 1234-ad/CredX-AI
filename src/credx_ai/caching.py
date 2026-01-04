"""
Caching and Performance Optimization
Intelligent caching system for AI model outputs and API responses
"""

import hashlib
import json
import time
from typing import Any, Optional, Dict, Callable
from functools import wraps
import pickle
import logging
from collections import OrderedDict
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    """
    Least Recently Used (LRU) Cache implementation.
    Automatically evicts least recently used items when capacity is reached.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to store
        """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]["value"]

    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        # Remove if exists
        if key in self.cache:
            self.cache.move_to_end(key)

        # Add new item
        expires_at = time.time() + ttl if ttl else None
        self.cache[key] = {"value": value, "expires_at": expires_at}

        # Evict oldest if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def is_expired(self, key: str) -> bool:
        """Check if cached item has expired."""
        if key not in self.cache:
            return True

        expires_at = self.cache[key]["expires_at"]
        if expires_at and time.time() > expires_at:
            del self.cache[key]
            return True

        return False

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
        }


class SemanticCache:
    """
    Semantic cache for AI model outputs.
    Caches results based on semantic similarity of inputs.
    """

    def __init__(self, capacity: int = 500, similarity_threshold: float = 0.95):
        """
        Initialize semantic cache.

        Args:
            capacity: Maximum number of items to store
            similarity_threshold: Minimum similarity to consider a cache hit
        """
        self.cache = LRUCache(capacity)
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache = {}

    def _generate_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        """
        Get cached result for text.

        Args:
            text: Input text

        Returns:
            Cached result or None
        """
        key = self._generate_key(text)
        return self.cache.get(key)

    def put(self, text: str, result: Any, ttl: int = 3600):
        """
        Cache result for text.

        Args:
            text: Input text
            result: Result to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        key = self._generate_key(text)
        self.cache.put(key, result, ttl)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()


class ModelCache:
    """
    Cache for AI model outputs (embeddings, predictions, etc.).
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize model cache.

        Args:
            capacity: Maximum number of items to store
        """
        self.embedding_cache = LRUCache(capacity)
        self.prediction_cache = LRUCache(capacity)
        self.resume_cache = LRUCache(capacity // 2)  # Smaller capacity for large objects

    def cache_embedding(self, text: str, embedding: Any, ttl: int = 7200):
        """
        Cache text embedding.

        Args:
            text: Input text
            embedding: Computed embedding
            ttl: Time to live in seconds (default: 2 hours)
        """
        key = hashlib.sha256(text.encode()).hexdigest()
        self.embedding_cache.put(key, embedding, ttl)

    def get_embedding(self, text: str) -> Optional[Any]:
        """Get cached embedding for text."""
        key = hashlib.sha256(text.encode()).hexdigest()
        return self.embedding_cache.get(key)

    def cache_prediction(self, input_data: Dict, prediction: Any, ttl: int = 3600):
        """
        Cache model prediction.

        Args:
            input_data: Input data dictionary
            prediction: Model prediction
            ttl: Time to live in seconds (default: 1 hour)
        """
        key = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        self.prediction_cache.put(key, prediction, ttl)

    def get_prediction(self, input_data: Dict) -> Optional[Any]:
        """Get cached prediction for input data."""
        key = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
        return self.prediction_cache.get(key)

    def cache_resume_parse(self, resume_hash: str, parsed_data: Dict, ttl: int = 86400):
        """
        Cache parsed resume data.

        Args:
            resume_hash: Hash of resume content
            parsed_data: Parsed resume data
            ttl: Time to live in seconds (default: 24 hours)
        """
        self.resume_cache.put(resume_hash, parsed_data, ttl)

    def get_resume_parse(self, resume_hash: str) -> Optional[Dict]:
        """Get cached parsed resume data."""
        return self.resume_cache.get(resume_hash)

    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "prediction_cache": self.prediction_cache.get_stats(),
            "resume_cache": self.resume_cache.get_stats(),
        }


class PerformanceMonitor:
    """
    Monitor and track performance metrics.
    """

    def __init__(self):
        self.metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_response_time": 0,
            "endpoint_stats": {},
        }
        self.start_time = time.time()

    def record_api_call(self, endpoint: str, response_time: float, cache_hit: bool):
        """
        Record API call metrics.

        Args:
            endpoint: API endpoint called
            response_time: Response time in seconds
            cache_hit: Whether result was from cache
        """
        self.metrics["api_calls"] += 1
        self.metrics["total_response_time"] += response_time

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

        # Track per-endpoint stats
        if endpoint not in self.metrics["endpoint_stats"]:
            self.metrics["endpoint_stats"][endpoint] = {
                "calls": 0,
                "total_time": 0,
                "cache_hits": 0,
            }

        stats = self.metrics["endpoint_stats"][endpoint]
        stats["calls"] += 1
        stats["total_time"] += response_time
        if cache_hit:
            stats["cache_hits"] += 1

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        uptime = time.time() - self.start_time
        total_calls = self.metrics["api_calls"]
        avg_response_time = (
            self.metrics["total_response_time"] / total_calls if total_calls > 0 else 0
        )
        cache_hit_rate = (
            self.metrics["cache_hits"] / total_calls * 100 if total_calls > 0 else 0
        )

        # Calculate per-endpoint averages
        endpoint_stats = {}
        for endpoint, stats in self.metrics["endpoint_stats"].items():
            endpoint_stats[endpoint] = {
                "calls": stats["calls"],
                "avg_response_time": stats["total_time"] / stats["calls"],
                "cache_hit_rate": f"{stats['cache_hits'] / stats['calls'] * 100:.2f}%",
            }

        return {
            "uptime_seconds": uptime,
            "total_api_calls": total_calls,
            "avg_response_time": f"{avg_response_time:.3f}s",
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "endpoint_stats": endpoint_stats,
        }


# Global instances
model_cache = ModelCache()
semantic_cache = SemanticCache()
performance_monitor = PerformanceMonitor()


def cached(ttl: int = 3600, cache_type: str = "semantic"):
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds
        cache_type: Type of cache to use (semantic, model)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key_hash = hashlib.sha256(cache_key.encode()).hexdigest()

            # Try to get from cache
            if cache_type == "semantic":
                cached_result = semantic_cache.cache.get(cache_key_hash)
            else:
                cached_result = model_cache.prediction_cache.get(cache_key_hash)

            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Cache result
            if cache_type == "semantic":
                semantic_cache.cache.put(cache_key_hash, result, ttl)
            else:
                model_cache.prediction_cache.put(cache_key_hash, result, ttl)

            logger.info(f"Cache miss for {func.__name__} (executed in {execution_time:.3f}s)")
            return result

        return wrapper

    return decorator


def monitor_performance(endpoint: str):
    """
    Decorator to monitor endpoint performance.

    Args:
        endpoint: Endpoint name
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Check if result is from cache (simplified check)
            cache_hit = False  # This should be determined by actual cache check

            result = func(*args, **kwargs)

            response_time = time.time() - start_time
            performance_monitor.record_api_call(endpoint, response_time, cache_hit)

            return result

        return wrapper

    return decorator


def batch_cache_embeddings(texts: list, embedding_func: Callable, ttl: int = 7200):
    """
    Batch cache embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        embedding_func: Function to compute embeddings
        ttl: Time to live in seconds

    Returns:
        List of embeddings
    """
    embeddings = []
    uncached_texts = []
    uncached_indices = []

    # Check cache for each text
    for i, text in enumerate(texts):
        cached_embedding = model_cache.get_embedding(text)
        if cached_embedding is not None:
            embeddings.append(cached_embedding)
        else:
            embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Compute embeddings for uncached texts
    if uncached_texts:
        new_embeddings = embedding_func(uncached_texts)

        # Cache and insert new embeddings
        for idx, embedding in zip(uncached_indices, new_embeddings):
            model_cache.cache_embedding(texts[idx], embedding, ttl)
            embeddings[idx] = embedding

    return embeddings


def clear_all_caches():
    """Clear all caches."""
    model_cache.embedding_cache.clear()
    model_cache.prediction_cache.clear()
    model_cache.resume_cache.clear()
    semantic_cache.cache.clear()
    logger.info("All caches cleared")


def get_all_cache_stats() -> Dict:
    """Get statistics for all caches."""
    return {
        "model_cache": model_cache.get_stats(),
        "semantic_cache": semantic_cache.get_stats(),
        "performance": performance_monitor.get_stats(),
    }
