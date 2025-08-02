"""
Enhanced API Tests
Comprehensive tests for all API endpoints including enhanced routes.
"""
import io
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.models.document import DocumentStatus, ParaphraseMethod


def test_enhanced_document_upload(client: TestClient):
    """Test enhanced document upload endpoint."""
    test_content = "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji dampak teknologi digital."
    file_data = io.BytesIO(test_content.encode())
    
    response = client.post(
        "/api/v2/documents/upload-enhanced",
        files={"file": ("test_research.txt", file_data, "text/plain")},
        data={
            "chapter": "BAB 1",
            "preserve_structure": "true",
            "extract_academic_terms": "true"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "filename" in data
    assert data["message"] == "Document uploaded and processed successfully"
    assert data["filename"] == "test_research.txt"


def test_enhanced_document_upload_invalid_file(client: TestClient):
    """Test enhanced upload with invalid file."""
    response = client.post(
        "/api/v2/documents/upload-enhanced",
        files={"file": ("", io.BytesIO(b""), "text/plain")},
        data={"preserve_structure": "true"}
    )
    
    assert response.status_code == 400


def test_text_analysis_nlp(client: TestClient):
    """Test NLP text analysis endpoint."""
    request_data = {
        "text": "Penelitian ini menggunakan metode analisis kualitatif. Data dikumpulkan melalui wawancara mendalam.",
        "extract_academic_terms": True
    }
    
    with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
        # Mock analysis result
        mock_result = MagicMock()
        mock_result.sentences = [
            MagicMock(priority_for_paraphrasing=0.8, complexity_score=0.7, readability_score=0.6),
            MagicMock(priority_for_paraphrasing=0.9, complexity_score=0.8, readability_score=0.5)
        ]
        mock_result.overall_readability = 0.55
        mock_result.overall_complexity = 0.75
        mock_result.academic_terms = {"penelitian", "metode", "analisis"}
        mock_result.named_entities = {"kualitatif"}
        mock_result.quality_metrics = {"score": 0.7}
        mock_result.paraphrasing_priorities = [0, 1]
        mock_analyze.return_value = mock_result
        
        response = client.post(
            "/api/v2/text/analyze",
            json=request_data
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_sentences"] == 2
    assert data["overall_readability"] == 0.55
    assert data["overall_complexity"] == 0.75
    assert data["academic_terms_count"] == 3
    assert data["named_entities_count"] == 1
    assert "high_priority_sentences" in data


def test_text_quality_assessment(client: TestClient):
    """Test text quality assessment endpoint."""
    test_text = "Ini adalah contoh teks untuk penilaian kualitas."
    
    with patch('app.services.rule_based_paraphraser.paraphrase') as mock_paraphrase:
        # Mock quality result
        mock_quality = MagicMock()
        mock_quality.readability_score = 0.7
        mock_quality.grammar_score = 0.8
        mock_quality.academic_tone_score = 0.6
        mock_quality.similarity_score = 0.9
        mock_quality.overall_score = 0.75
        mock_quality.issues = ["No major issues"]
        mock_paraphrase.return_value = [("paraphrased text", mock_quality)]
        
        response = client.post(
            "/api/v2/text/quality-assessment",
            params={"text": test_text}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["readability_score"] == 0.7
    assert data["grammar_score"] == 0.8
    assert data["academic_tone_score"] == 0.6
    assert data["overall_quality"] == 0.75
    assert isinstance(data["recommendations"], list)


def test_direct_text_paraphrasing_indot5(client: TestClient):
    """Test direct text paraphrasing with IndoT5 method."""
    request_data = {
        "text": "Penelitian ini menggunakan metode eksperimen.",
        "method": "indot5",
        "preserve_academic_terms": True,
        "preserve_citations": True,
        "num_variants": 2
    }
    
    with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
        # Mock paraphrase result
        mock_result = MagicMock()
        mock_result.paraphrased_variants = [
            "Studi ini menggunakan pendekatan eksperimental.",
            "Riset ini menerapkan metode percobaan."
        ]
        mock_result.similarity_scores = [0.7, 0.65]
        mock_result.quality_scores = [0.85, 0.8]
        mock_result.best_variant = "Studi ini menggunakan pendekatan eksperimental."
        mock_result.metadata = {"processing_time": 1.2}
        mock_paraphrase.return_value = mock_result
        
        response = client.post(
            "/api/v2/text/paraphrase-direct",
            json=request_data
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == request_data["text"]
    assert len(data["variants"]) == 2
    assert data["best_variant"] == "Studi ini menggunakan pendekatan eksperimental."
    assert "processing_time" in data
    assert "metadata" in data


def test_direct_text_paraphrasing_rule_based(client: TestClient):
    """Test direct text paraphrasing with rule-based method."""
    request_data = {
        "text": "Metode penelitian yang digunakan adalah kualitatif.",
        "method": "rule_based",
        "preserve_academic_terms": True,
        "num_variants": 2
    }
    
    with patch('app.services.rule_based_paraphraser.paraphrase') as mock_paraphrase:
        # Mock quality objects
        mock_quality1 = MagicMock()
        mock_quality1.similarity_score = 0.6
        mock_quality1.overall_score = 0.8
        
        mock_quality2 = MagicMock()
        mock_quality2.similarity_score = 0.65
        mock_quality2.overall_score = 0.75
        
        mock_paraphrase.return_value = [
            ("Pendekatan penelitian yang diterapkan bersifat kualitatif.", mock_quality1),
            ("Teknik riset yang digunakan adalah kualitatif.", mock_quality2)
        ]
        
        response = client.post(
            "/api/v2/text/paraphrase-direct",
            json=request_data
        )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["variants"]) == 2
    assert all(variant["method_used"] == "rule_based" for variant in data["variants"])


def test_direct_text_paraphrasing_hybrid(client: TestClient):
    """Test direct text paraphrasing with hybrid method."""
    request_data = {
        "text": "Hasil penelitian menunjukkan korelasi positif.",
        "method": "hybrid",
        "num_variants": 2
    }
    
    with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_indot5, \
         patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule:
        
        # Mock IndoT5 result
        mock_indot5_result = MagicMock()
        mock_indot5_result.paraphrased_variants = ["Temuan riset menunjukkan hubungan positif."]
        mock_indot5_result.similarity_scores = [0.7]
        mock_indot5_result.quality_scores = [0.85]
        mock_indot5.return_value = mock_indot5_result
        
        # Mock rule-based result
        mock_quality = MagicMock()
        mock_quality.similarity_score = 0.65
        mock_quality.overall_score = 0.8
        mock_rule.return_value = [("Outcome penelitian menampilkan korelasi positif.", mock_quality)]
        
        response = client.post(
            "/api/v2/text/paraphrase-direct",
            json=request_data
        )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["variants"]) == 2
    assert data["metadata"]["hybrid_approach"] is True


def test_direct_text_paraphrasing_invalid_method(client: TestClient):
    """Test direct text paraphrasing with invalid method."""
    request_data = {
        "text": "Test text",
        "method": "invalid_method",
        "num_variants": 1
    }
    
    response = client.post(
        "/api/v2/text/paraphrase-direct",
        json=request_data
    )
    
    assert response.status_code == 400
    assert "Invalid paraphrasing method" in response.json()["detail"]


def test_performance_stats(client: TestClient):
    """Test performance statistics endpoint."""
    with patch('app.services.enhanced_indot5_paraphraser.get_performance_stats') as mock_indot5_stats, \
         patch('app.services.rule_based_paraphraser.get_transformation_stats') as mock_rule_stats:
        
        mock_indot5_stats.return_value = {"total_requests": 10, "avg_time": 1.5}
        mock_rule_stats.return_value = {"total_transformations": 20, "success_rate": 0.95}
        
        with patch('app.services.enhanced_indot5_paraphraser.model_loaded', True), \
             patch('app.services.indonesian_nlp_pipeline.nlp_id', "test_nlp"):
            
            response = client.get("/api/v2/performance/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert "enhanced_indot5" in data
    assert "rule_based" in data
    assert "system_status" in data
    assert data["system_status"]["services_loaded"] is True


def test_clear_cache(client: TestClient):
    """Test cache clearing endpoint."""
    with patch('app.services.enhanced_indot5_paraphraser.clear_cache') as mock_clear:
        response = client.post("/api/v2/performance/clear-cache")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "timestamp" in data
    mock_clear.assert_called_once()


def test_demo_sample_analysis(client: TestClient):
    """Test demo sample analysis endpoint."""
    with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze, \
         patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_indot5, \
         patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule:
        
        # Mock analysis
        mock_analysis = MagicMock()
        mock_analysis.sentences = [MagicMock(), MagicMock()]
        mock_analysis.overall_readability = 0.7
        mock_analysis.overall_complexity = 0.6
        mock_analysis.academic_terms = {"penelitian", "metode"}
        mock_analysis.named_entities = {"Smith"}
        mock_analyze.return_value = mock_analysis
        
        # Mock IndoT5
        mock_indot5_result = MagicMock()
        mock_indot5_result.best_variant = "Paraphrased sample text"
        mock_indot5_result.similarity_scores = [0.7]
        mock_indot5_result.quality_scores = [0.8]
        mock_indot5.return_value = mock_indot5_result
        
        # Mock rule-based
        mock_quality = MagicMock()
        mock_quality.overall_score = 0.75
        mock_quality.similarity_score = 0.65
        mock_rule.return_value = [("Rule-based paraphrase", mock_quality)]
        
        response = client.get("/api/v2/demo/sample-analysis")
    
    assert response.status_code == 200
    data = response.json()
    assert "sample_text" in data
    assert "nlp_analysis" in data
    assert "paraphrasing_results" in data
    assert "indot5" in data["paraphrasing_results"]
    assert "rule_based" in data["paraphrasing_results"]


def test_enhanced_document_paraphrasing(client: TestClient):
    """Test enhanced document paraphrasing endpoint."""
    # First upload a document
    test_content = "Test document content for paraphrasing."
    file_data = io.BytesIO(test_content.encode())
    
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", file_data, "text/plain")}
    )
    
    assert upload_response.status_code == 200
    document_id = upload_response.json()["id"]
    
    # Test enhanced paraphrasing
    request_data = {
        "document_id": document_id,
        "method": ParaphraseMethod.INDOT5.value,
        "use_nlp_analysis": True,
        "preserve_academic_terms": True,
        "preserve_citations": True,
        "num_variants": 2,
        "quality_threshold": 0.7
    }
    
    with patch('app.services.enhanced_paraphrasing_service.paraphrase_document_enhanced') as mock_paraphrase:
        # Mock session result
        mock_session = MagicMock()
        mock_session.id = str(uuid.uuid4())
        mock_session.method_used = ParaphraseMethod.INDOT5.value
        mock_session.similarity_score = 0.75
        mock_session.processing_time = 5000
        mock_session.token_usage = {"enhanced": True}
        mock_paraphrase.return_value = mock_session
        
        response = client.post(
            f"/api/v2/documents/{document_id}/paraphrase-enhanced",
            json=request_data
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Enhanced paraphrasing completed"
    assert "session_id" in data
    assert data["document_id"] == document_id


def test_paraphrasing_request_validation(client: TestClient):
    """Test request validation for paraphrasing endpoints."""
    # Test with invalid num_variants
    request_data = {
        "text": "Test text",
        "method": "indot5",
        "num_variants": 15  # Max is 5 for direct paraphrasing
    }
    
    response = client.post(
        "/api/v2/text/paraphrase-direct",
        json=request_data
    )
    
    assert response.status_code == 422  # Validation error


def test_text_analysis_empty_text(client: TestClient):
    """Test text analysis with empty text."""
    request_data = {
        "text": "",
        "extract_academic_terms": True
    }
    
    response = client.post(
        "/api/v2/text/analyze",
        json=request_data
    )
    
    assert response.status_code == 422  # Validation error


def test_file_upload_size_limit(client: TestClient):
    """Test file upload with size exceeding limit."""
    # Create a large file (simulate 60MB)
    large_content = "x" * (60 * 1024 * 1024)
    file_data = io.BytesIO(large_content.encode())
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("large_file.txt", file_data, "text/plain")}
    )
    
    assert response.status_code == 413  # File too large


def test_paraphrase_nonexistent_document(client: TestClient):
    """Test paraphrasing non-existent document."""
    fake_id = str(uuid.uuid4())
    request_data = {
        "method": ParaphraseMethod.INDOT5.value
    }
    
    response = client.post(
        f"/api/documents/{fake_id}/paraphrase",
        json=request_data
    )
    
    assert response.status_code == 404


def test_document_list_filtering(client: TestClient):
    """Test document listing with filters."""
    # Test with status filter
    response = client.get(
        "/api/documents",
        params={"status": DocumentStatus.PENDING.value, "limit": 10}
    )
    
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_document_sessions_listing(client: TestClient):
    """Test listing paraphrase sessions for a document."""
    # First upload a document
    test_content = "Test document for sessions."
    file_data = io.BytesIO(test_content.encode())
    
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("sessions_test.txt", file_data, "text/plain")}
    )
    
    assert upload_response.status_code == 200
    document_id = upload_response.json()["id"]
    
    # Test getting sessions (should be empty initially)
    response = client.get(f"/api/documents/{document_id}/sessions")
    
    assert response.status_code == 200
    assert response.json() == []


def test_concurrent_document_processing(client: TestClient):
    """Test handling concurrent document processing attempts."""
    # Upload document
    test_content = "Test concurrent processing."
    file_data = io.BytesIO(test_content.encode())
    
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("concurrent_test.txt", file_data, "text/plain")}
    )
    
    document_id = upload_response.json()["id"]
    
    # Mock document status as processing
    with patch('app.services.document_processor.get_document') as mock_get_doc:
        mock_doc = MagicMock()
        mock_doc.status = DocumentStatus.PROCESSING
        mock_get_doc.return_value = mock_doc
        
        request_data = {"method": ParaphraseMethod.INDOT5.value}
        response = client.post(
            f"/api/documents/{document_id}/paraphrase",
            json=request_data
        )
    
    assert response.status_code == 409  # Conflict - already processing
