# Use the real app from verification_service
from verification_service.main import app

# Expose names expected by gunicorn/uvicorn
application = app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "verification_service.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False)
