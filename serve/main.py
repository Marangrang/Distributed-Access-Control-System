from verification_service.main import app

@app.get("/health")
async def health():
    return {"status": "healthy"}

__all__ = ["app"]
