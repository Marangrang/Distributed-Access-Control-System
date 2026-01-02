"""
Prometheus metrics for face verification service
Monitors inference performance and system health
"""
from prometheus_client import Counter, Histogram, Gauge

# Metric: Total verification requests
verification_requests_total = Counter(
    'verification_requests_total',
    'Total number of face verification requests',
    ['status']  # success, failure, no_face_detected, invalid_request
)

# Metric: Verification latency
verification_latency_ms = Histogram(
    'verification_latency_ms',
    'Face verification latency in milliseconds',
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]  # milliseconds
)

# Metric: Model sync lag
model_sync_lag_seconds = Gauge(
    'model_sync_lag_seconds',
    'Seconds since last model update from MinIO'
)

# Metric: Active connections
active_connections = Gauge(
    'active_connections',
    'Number of active API connections'
)

# Metric: Similarity score distribution
similarity_score = Histogram(
    'similarity_score',
    'Distribution of similarity scores (0.0-1.0)',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metric: Verification results
verification_results = Counter(
    'verification_results_total',
    'Verification outcomes (verified, rejected, unknown)',
    ['result']
)


# ============================================================================
# Helper functions to record metrics
# ============================================================================

def record_request(status: str = 'success'):
    """Record a verification request with status"""
    verification_requests_total.labels(status=status).inc()


def observe_latency(latency_ms: float):
    """Record verification latency in milliseconds"""
    verification_latency_ms.observe(latency_ms)


def set_sync_lag(lag_seconds: float):
    """Set model sync lag in seconds since last update"""
    model_sync_lag_seconds.set(lag_seconds)


def record_similarity(score: float):
    """Record similarity score (0.0-1.0)"""
    similarity_score.observe(score)


def record_verification_result(verified: bool, similarity: float):
    """Record verification result based on outcome"""
    if verified:
        verification_results.labels(result='verified').inc()
    elif similarity < 0.3:
        verification_results.labels(result='unknown').inc()
    else:
        verification_results.labels(result='rejected').inc()


def increment_active_connections():
    """Increment active connections gauge"""
    active_connections.inc()


def decrement_active_connections():
    """Decrement active connections gauge"""
    active_connections.dec()
