import os

bind = "0.0.0.0:8080"
workers = int(os.getenv("UVICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
timeout = 120
accesslog = "-"   # stdout
errorlog = "-"    # stderr
loglevel = "info"
worker_class = "uvicorn.workers.UvicornWorker"