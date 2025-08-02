"""
Auto-Paraphrasing System Main Entry Point
"""
import uvicorn
import signal
import sys
from app.core.config import settings


def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    print("\nReceived shutdown signal. Shutting down gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        uvicorn.run(
            "app.api.routes:app",
            host="0.0.0.0",
            port=8000,
            reload=settings.debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("Application shutdown complete")
