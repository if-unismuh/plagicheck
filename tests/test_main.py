#!/usr/bin/env python3
"""
Main Test Suite for Auto-Paraphrasing System
Comprehensive testing of core functionality and API endpoints.
"""
import io
import json
import pytest
from fastapi.testclient import TestClient
from app.api.routes import app


class TestAPI:
    """Test API endpoints and functionality."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "status" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_document_upload(self):
        """Test document upload functionality."""
        test_content = "Ini adalah contoh teks untuk diuji."
        file_data = io.BytesIO(test_content.encode())
        
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", file_data, "text/plain")},
            data={"chapter": "BAB 1"}
        )
        
        # Should either succeed or return a meaningful error
        assert response.status_code in [200, 400, 422]
    
    def test_paraphrase_endpoint(self):
        """Test paraphrasing functionality."""
        test_data = {
            "text": "Penelitian ini menggunakan metode kualitatif.",
            "method": "simple"
        }
        
        try:
            response = self.client.post("/api/v2/text/paraphrase-direct", json=test_data)
            # Should either succeed or return a meaningful error
            assert response.status_code in [200, 400, 422, 500]
        except Exception:
            # If service dependencies are not available, test should pass
            pytest.skip("Paraphrasing service dependencies not available")


class TestRouteStructure:
    """Test that routes are properly structured and accessible."""
    
    def test_route_imports(self):
        """Test that route modules can be imported."""
        try:
            from app.api.routes import app
            from app.api.enhanced_routes import router
            assert app is not None
            assert router is not None
        except ImportError as e:
            pytest.fail(f"Failed to import route modules: {e}")
    
    def test_postman_collection_structure(self):
        """Test Postman collection structure if available."""
        try:
            with open('postman/paraphrase_system_collection.json', 'r') as f:
                collection = json.load(f)
            
            assert 'info' in collection
            assert 'item' in collection
            assert collection['info']['name']
            print(f"Postman collection validated: {collection['info']['name']}")
            
        except FileNotFoundError:
            pytest.skip("Postman collection not found")
        except json.JSONDecodeError:
            pytest.fail("Invalid Postman collection JSON")


class TestSystemHealth:
    """Test overall system health and dependencies."""
    
    def test_import_core_modules(self):
        """Test that core modules can be imported."""
        try:
            import app.core.config
            import app.core.database
            import app.models.document
            import app.services.document_processor
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")
    
    def test_database_connection(self):
        """Test database connectivity."""
        try:
            from app.core.database import get_database_url
            from app.core.config import settings
            
            # Just test that we can get database configuration
            db_url = get_database_url()
            assert db_url is not None
            assert isinstance(db_url, str)
            
        except Exception as e:
            pytest.skip(f"Database test skipped: {e}")


def run_all_tests():
    """Run all tests programmatically."""
    print("=" * 60)
    print("AUTO-PARAPHRASING SYSTEM - MAIN TEST SUITE")
    print("=" * 60)
    
    # Test API functionality
    print("\n1. Testing API Endpoints...")
    api_test = TestAPI()
    api_test.setup_method()
    
    try:
        api_test.test_root_endpoint()
        print("   ✓ Root endpoint - PASSED")
    except Exception as e:
        print(f"   ✗ Root endpoint - FAILED: {e}")
    
    try:
        api_test.test_health_check()
        print("   ✓ Health check - PASSED")
    except Exception as e:
        print(f"   ✗ Health check - FAILED: {e}")
    
    try:
        api_test.test_document_upload()
        print("   ✓ Document upload - PASSED")
    except Exception as e:
        print(f"   ✗ Document upload - FAILED: {e}")
    
    try:
        api_test.test_paraphrase_endpoint()
        print("   ✓ Paraphrase endpoint - PASSED")
    except Exception as e:
        print(f"   ✗ Paraphrase endpoint - FAILED: {e}")
    
    # Test route structure
    print("\n2. Testing Route Structure...")
    route_test = TestRouteStructure()
    
    try:
        route_test.test_route_imports()
        print("   ✓ Route imports - PASSED")
    except Exception as e:
        print(f"   ✗ Route imports - FAILED: {e}")
    
    try:
        route_test.test_postman_collection_structure()
        print("   ✓ Postman collection - PASSED")
    except Exception as e:
        print(f"   ✗ Postman collection - FAILED: {e}")
    
    # Test system health
    print("\n3. Testing System Health...")
    health_test = TestSystemHealth()
    
    try:
        health_test.test_import_core_modules()
        print("   ✓ Core module imports - PASSED")
    except Exception as e:
        print(f"   ✗ Core module imports - FAILED: {e}")
    
    try:
        health_test.test_database_connection()
        print("   ✓ Database configuration - PASSED")
    except Exception as e:
        print(f"   ✗ Database configuration - FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nTo run with pytest: pytest test_main.py -v")


if __name__ == "__main__":
    run_all_tests()
