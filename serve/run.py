import os
import uvicorn


def app_entry():
    workers = int(os.getenv("UVICORN_WORKERS", "2"))
    uvicorn.run(
        "verification_service.main:app",
        host="0.0.0.0",
        port=8080,
        workers=workers)


if __name__ == "__main__":
    app_entry()
