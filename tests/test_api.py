"""
Test the API endpoints
"""
import io
import uuid
import pytest
from fastapi.testclient import TestClient

from app.models.document import DocumentStatus


@pytest.mark.unit
def test_root_endpoint(client: TestClient):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Auto-Paraphrasing System"
    assert data["status"] == "running"


@pytest.mark.unit
def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.unit
def test_upload_document(client: TestClient):
    """Test document upload."""
    # Create a test file
    test_content = "This is a test document for the paraphrasing system."
    file_data = io.BytesIO(test_content.encode())
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", file_data, "text/plain")},
        data={"chapter": "BAB 1"}
    )
    
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["chapter"] == "BAB 1"
    assert data["status"] == DocumentStatus.PENDING.value


@pytest.mark.unit
def test_upload_invalid_file(client: TestClient):
    """Test upload with invalid file type."""
    file_data = io.BytesIO(b"fake content")
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.xyz", file_data, "application/unknown")}
    )
    
    assert response.status_code == 400


@pytest.mark.unit
def test_get_document_not_found(client: TestClient):
    """Test getting non-existent document."""
    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/documents/{fake_id}")
    assert response.status_code == 404


@pytest.mark.unit
def test_list_documents_empty(client: TestClient):
    """Test listing documents when none exist."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data == []


@pytest.mark.integration
def test_document_workflow(client: TestClient):
    """Test complete document workflow."""
    # 1. Upload document
    test_content = "This is a comprehensive test of the document processing workflow."
    file_data = io.BytesIO(test_content.encode())
    
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("workflow_test.txt", file_data, "text/plain")},
        data={"chapter": "BAB 2"}
    )
    
    assert upload_response.status_code == 200
    document_data = upload_response.json()
    document_id = document_data["id"]
    
    # 2. Get document details
    detail_response = client.get(f"/api/documents/{document_id}")
    assert detail_response.status_code == 200
    detail_data = detail_response.json()
    assert detail_data["original_content"] == test_content
    
    # 3. Check document status
    status_response = client.get(f"/api/documents/{document_id}/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == DocumentStatus.PENDING.value
    
    # 4. List documents
    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert len(list_data) == 1
    assert list_data[0]["id"] == document_id
    
    # 5. Delete document
    delete_response = client.delete(f"/api/documents/{document_id}")
    assert delete_response.status_code == 200
    
    # 6. Verify deletion
    final_list_response = client.get("/api/documents")
    assert final_list_response.status_code == 200
    final_list_data = final_list_response.json()
    assert len(final_list_data) == 0
