"""
Test Configuration and Setup
"""
import pytest
from fastapi.testclient import TestClient
from app.api.routes import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_text():
    """Sample Indonesian text for testing."""
    return "Penelitian ini menggunakan metode kualitatif untuk menganalisis data yang dikumpulkan."


@pytest.fixture
def sample_file_content():
    """Sample file content for upload testing."""
    return "Ini adalah contoh dokumen untuk diuji dalam sistem parafrase otomatis."
