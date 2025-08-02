#!/usr/bin/env python3
"""
Test script to verify graceful shutdown behavior
"""
import asyncio
import signal
import time
from fastapi import FastAPI
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    This is the modern way to handle startup and shutdown events.
    """
    # Startup
    print("🚀 Application starting up...")
    yield
    # Shutdown
    print("🛑 Application shutting down...")


def main():
    """Test the shutdown behavior with a simple FastAPI app."""
    import uvicorn
    
    app = FastAPI(lifespan=lifespan)
    
    @app.get("/")
    async def root():
        return {"message": "Hello World"}
    
    print("Starting test server...")
    print("Press Ctrl+C to test graceful shutdown")
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✅ Graceful shutdown completed")
    except Exception as e:
        print(f"❌ Error during shutdown: {e}")


if __name__ == "__main__":
    main()