"""
Gunicorn WSGI server configuration
Production-ready settings for FastAPI application
"""
import multiprocessing
import os

# Server Socket
bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8080')}"
backlog = 2048

# Worker Processes
workers = int(os.getenv('API_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'uvicorn.workers.UvicornWorker'  # ASGI worker for FastAPI
worker_connections = 1000
max_requests = int(os.getenv('MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', 50))
timeout = int(os.getenv('WORKER_TIMEOUT', 60))
keepalive = int(os.getenv('KEEPALIVE', 5))
graceful_timeout = 30

# Logging
accesslog = os.getenv('ACCESS_LOG', 'logs/access.log')
errorlog = os.getenv('ERROR_LOG', 'logs/error.log')
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process Naming
proc_name = 'face_verification_api'

# Server Mechanics
daemon = False  # Don't run as daemon (let Docker/systemd handle)
pidfile = 'logs/gunicorn.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Server Hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Face Verification Service")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Bind: {bind}")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Server is ready. Spawning workers")


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Server reloading...")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")


def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    worker.log.info(f"Worker received INT or QUIT signal (pid: {worker.pid})")


def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info(f"Worker received SIGABRT signal (pid: {worker.pid})")


def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug(f"{req.method} {req.path}")


def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    pass


def child_exit(server, worker):
    """Called just after a worker has been exited, in the master process."""
    server.log.info(f"Worker exited (pid: {worker.pid})")


def worker_exit(server, worker):
    """Called just after a worker has been exited, in the worker process."""
    pass


def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info(f"Number of workers changed from {old_value} to {new_value}")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    server.log.info("Shutting down Face Verification Service")


# Development vs Production
if os.getenv('APP_ENV') == 'development':
    reload = True
    reload_extra_files = ['config/config.yaml']
    loglevel = 'debug'
else:
    reload = False
    preload_app = True  # Load app before forking workers (saves memory)
