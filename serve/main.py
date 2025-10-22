# shim so `serve.main:app` imports work (keeps compatibility with existing Docker/Gunicorn commands)
from verification_service.main import app

__all__ = ["app"]