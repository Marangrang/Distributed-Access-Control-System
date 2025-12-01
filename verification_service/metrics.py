# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

requests_total = Counter(
    'face_verification_requests_total',
    'Total number of face verification requests')
verify_latency_ms = Histogram(
    'verify_latency_ms',
    'Face verification latency in milliseconds')
sync_lag_s = Gauge('sync_lag_s', 'Seconds since last successful sync')


def record_request():
    requests_total.inc()


def observe_latency(latency_ms):
    verify_latency_ms.observe(latency_ms)


def set_sync_lag(lag_s):
    sync_lag_s.set(lag_s)


def get_metrics():
    return generate_latest()
