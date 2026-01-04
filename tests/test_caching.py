"""
Unit Tests for Caching Module
Tests for LRU cache, semantic cache, model cache, and performance monitoring
"""

import pytest
import time
from src.credx_ai.caching import (
    LRUCache,
    SemanticCache,
    ModelCache,
    PerformanceMonitor,
    model_cache,
    semantic_cache,
    performance_monitor,
    cached,
    batch_cache_embeddings,
)


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_lru_cache_initialization(self):
        """Test LRU cache initialization."""
        cache = LRUCache(capacity=10)
        assert cache.capacity == 10
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_lru_cache_put_and_get(self):
        """Test putting and getting items."""
        cache = LRUCache(capacity=10)

        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_lru_cache_miss(self):
        """Test cache miss."""
        cache = LRUCache(capacity=10)

        result = cache.get("nonexistent")

        assert result is None
        assert cache.misses == 1

    def test_lru_cache_eviction(self):
        """Test LRU eviction when capacity is exceeded."""
        cache = LRUCache(capacity=3)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Add one more (should evict key1)
        cache.put("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key4") == "value4"  # Present

    def test_lru_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCache(capacity=10)

        cache.put("key1", "value1", ttl=1)  # 1 second TTL
        time.sleep(1.1)  # Wait for expiration

        assert cache.is_expired("key1") is True
        assert cache.get("key1") is None

    def test_lru_cache_update_existing(self):
        """Test updating existing key."""
        cache = LRUCache(capacity=10)

        cache.put("key1", "value1")
        cache.put("key1", "value2")  # Update

        assert cache.get("key1") == "value2"
        assert len(cache.cache) == 1  # Still only one item

    def test_lru_cache_clear(self):
        """Test clearing cache."""
        cache = LRUCache(capacity=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_lru_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(capacity=10)

        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["capacity"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert "50.00%" in stats["hit_rate"]


class TestSemanticCache:
    """Tests for SemanticCache class."""

    def test_semantic_cache_initialization(self):
        """Test semantic cache initialization."""
        cache = SemanticCache(capacity=100, similarity_threshold=0.95)
        assert cache.similarity_threshold == 0.95

    def test_semantic_cache_put_and_get(self):
        """Test putting and getting items."""
        cache = SemanticCache()

        cache.put("test query", {"result": "data"})
        result = cache.get("test query")

        assert result == {"result": "data"}

    def test_semantic_cache_different_queries(self):
        """Test that different queries don't match."""
        cache = SemanticCache()

        cache.put("query1", "result1")
        result = cache.get("query2")

        assert result is None

    def test_semantic_cache_stats(self):
        """Test semantic cache statistics."""
        cache = SemanticCache()

        cache.put("query1", "result1")
        cache.get("query1")  # Hit
        cache.get("query2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestModelCache:
    """Tests for ModelCache class."""

    def test_model_cache_initialization(self):
        """Test model cache initialization."""
        cache = ModelCache(capacity=100)
        assert cache.embedding_cache.capacity == 100
        assert cache.prediction_cache.capacity == 100

    def test_model_cache_embedding(self):
        """Test caching embeddings."""
        cache = ModelCache()

        embedding = [0.1, 0.2, 0.3]
        cache.cache_embedding("test text", embedding)
        result = cache.get_embedding("test text")

        assert result == embedding

    def test_model_cache_prediction(self):
        """Test caching predictions."""
        cache = ModelCache()

        input_data = {"skill": "Python", "experience": "5 years"}
        prediction = {"score": 0.95}

        cache.cache_prediction(input_data, prediction)
        result = cache.get_prediction(input_data)

        assert result == prediction

    def test_model_cache_resume_parse(self):
        """Test caching resume parsing."""
        cache = ModelCache()

        resume_hash = "abc123"
        parsed_data = {"name": "John Doe", "skills": ["Python", "ML"]}

        cache.cache_resume_parse(resume_hash, parsed_data)
        result = cache.get_resume_parse(resume_hash)

        assert result == parsed_data

    def test_model_cache_stats(self):
        """Test model cache statistics."""
        cache = ModelCache()

        cache.cache_embedding("text1", [0.1, 0.2])
        cache.get_embedding("text1")  # Hit

        stats = cache.get_stats()

        assert "embedding_cache" in stats
        assert "prediction_cache" in stats
        assert "resume_cache" in stats


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.metrics["api_calls"] == 0
        assert monitor.metrics["cache_hits"] == 0

    def test_performance_monitor_record_api_call(self):
        """Test recording API calls."""
        monitor = PerformanceMonitor()

        monitor.record_api_call("/api/test", 0.5, cache_hit=True)

        assert monitor.metrics["api_calls"] == 1
        assert monitor.metrics["cache_hits"] == 1
        assert monitor.metrics["total_response_time"] == 0.5

    def test_performance_monitor_endpoint_stats(self):
        """Test per-endpoint statistics."""
        monitor = PerformanceMonitor()

        monitor.record_api_call("/api/endpoint1", 0.5, cache_hit=True)
        monitor.record_api_call("/api/endpoint1", 0.3, cache_hit=False)
        monitor.record_api_call("/api/endpoint2", 0.2, cache_hit=True)

        stats = monitor.get_stats()

        assert "/api/endpoint1" in stats["endpoint_stats"]
        assert stats["endpoint_stats"]["/api/endpoint1"]["calls"] == 2

    def test_performance_monitor_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        monitor = PerformanceMonitor()

        monitor.record_api_call("/api/test", 0.1, cache_hit=True)
        monitor.record_api_call("/api/test", 0.1, cache_hit=True)
        monitor.record_api_call("/api/test", 0.1, cache_hit=False)

        stats = monitor.get_stats()

        assert "66.67%" in stats["cache_hit_rate"]

    def test_performance_monitor_avg_response_time(self):
        """Test average response time calculation."""
        monitor = PerformanceMonitor()

        monitor.record_api_call("/api/test", 0.5, cache_hit=False)
        monitor.record_api_call("/api/test", 0.3, cache_hit=False)

        stats = monitor.get_stats()

        assert "0.400s" in stats["avg_response_time"]


class TestCachedDecorator:
    """Tests for cached decorator."""

    def test_cached_decorator_caches_result(self):
        """Test that decorator caches function results."""
        call_count = [0]

        @cached(ttl=60, cache_type="semantic")
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count[0] == 1

        # Second call (should be cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count[0] == 1  # Not called again

    def test_cached_decorator_different_args(self):
        """Test that different arguments create different cache entries."""
        call_count = [0]

        @cached(ttl=60, cache_type="semantic")
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count[0] == 2  # Called twice for different args


class TestBatchCacheEmbeddings:
    """Tests for batch_cache_embeddings function."""

    def test_batch_cache_embeddings_all_cached(self):
        """Test batch caching when all embeddings are cached."""
        cache = ModelCache()

        # Pre-cache embeddings
        texts = ["text1", "text2", "text3"]
        for i, text in enumerate(texts):
            cache.cache_embedding(text, [i, i, i])

        # Mock embedding function (should not be called)
        call_count = [0]

        def mock_embedding_func(texts):
            call_count[0] += 1
            return [[0, 0, 0]] * len(texts)

        # Get embeddings
        embeddings = batch_cache_embeddings(texts, mock_embedding_func)

        assert len(embeddings) == 3
        assert call_count[0] == 0  # Not called (all cached)

    def test_batch_cache_embeddings_partial_cached(self):
        """Test batch caching with partial cache hits."""
        cache = ModelCache()

        # Pre-cache only first embedding
        cache.cache_embedding("text1", [0, 0, 0])

        texts = ["text1", "text2", "text3"]

        def mock_embedding_func(texts):
            return [[i, i, i] for i in range(len(texts))]

        # Get embeddings
        embeddings = batch_cache_embeddings(texts, mock_embedding_func)

        assert len(embeddings) == 3
        assert embeddings[0] == [0, 0, 0]  # From cache


# Fixtures


@pytest.fixture
def clean_lru_cache():
    """Provide a clean LRU cache instance."""
    return LRUCache(capacity=10)


@pytest.fixture
def clean_semantic_cache():
    """Provide a clean semantic cache instance."""
    return SemanticCache(capacity=10)


@pytest.fixture
def clean_model_cache():
    """Provide a clean model cache instance."""
    return ModelCache(capacity=10)


@pytest.fixture
def clean_performance_monitor():
    """Provide a clean performance monitor instance."""
    return PerformanceMonitor()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
