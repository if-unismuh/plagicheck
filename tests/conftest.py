"""
Test Configuration and Setup
"""
import pytest
import os
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.core.database import Base, get_db
from app.api.routes import app

# Import enhanced routes only when needed to avoid initialization issues
# from app.api import enhanced_routes


# Test database URL (using SQLite for tests)
TEST_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
test_engine = create_engine(
    TEST_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override the dependency
app.dependency_overrides[get_db] = override_get_db

# Include enhanced routes in the app for testing (lazy import)
try:
    from app.api import enhanced_routes
    app.include_router(enhanced_routes.router)
except Exception as e:
    print(f"Warning: Could not load enhanced routes for testing: {e}")
    # Continue without enhanced routes for basic testing


@pytest.fixture(scope="function")
def test_db():
    """Create test database tables."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_text_file(temp_upload_dir):
    """Create a sample text file for testing."""
    file_path = temp_upload_dir / "sample.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("This is a sample text for testing the paraphrasing system.")
    return file_path
