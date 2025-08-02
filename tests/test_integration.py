"""
Integration Tests
End-to-end testing of complete workflows and service interactions.
"""
import io
import time
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.models.document import DocumentStatus, ParaphraseMethod


class TestDocumentWorkflow:
    """Test complete document processing workflows."""
    
    def test_full_document_workflow_basic(self, client: TestClient):
        """Test complete workflow: upload -> process -> paraphrase -> retrieve."""
        # Step 1: Upload document
        test_content = "Penelitian ini menggunakan metode kualitatif untuk menganalisis data yang dikumpulkan."
        file_data = io.BytesIO(test_content.encode())
        
        upload_response = client.post(
            "/api/documents/upload",
            files={"file": ("workflow_test.txt", file_data, "text/plain")},
            data={"chapter": "BAB 1"}
        )
        
        assert upload_response.status_code == 200
        document_data = upload_response.json()
        document_id = document_data["id"]
        
        # Step 2: Verify document details
        detail_response = client.get(f"/api/documents/{document_id}")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()
        assert detail_data["original_content"] == test_content
        assert detail_data["status"] == DocumentStatus.PENDING.value
        
        # Step 3: Start paraphrasing
        with patch('app.services.paraphrasing_service.paraphrase_document') as mock_paraphrase:
            mock_session = MagicMock()
            mock_session.id = str(uuid.uuid4())
            mock_session.document_id = document_id
            mock_session.method_used = ParaphraseMethod.INDOT5
            mock_session.similarity_score = 0.75
            mock_session.processing_time = 3000
            mock_session.token_usage = {"input_tokens": 50, "output_tokens": 45}
            mock_session.created_at = "2024-01-01T00:00:00"
            mock_paraphrase.return_value = mock_session
            
            paraphrase_request = {"method": ParaphraseMethod.INDOT5.value}
            paraphrase_response = client.post(
                f"/api/documents/{document_id}/paraphrase",
                json=paraphrase_request
            )
        
        assert paraphrase_response.status_code == 200
        session_data = paraphrase_response.json()
        assert session_data["document_id"] == document_id
        assert session_data["similarity_score"] == 0.75
        
        # Step 4: Check processing status
        with patch('app.services.document_processor.get_document') as mock_get_doc:
            mock_doc = MagicMock()
            mock_doc.status = DocumentStatus.COMPLETED
            mock_get_doc.return_value = mock_doc
            
            status_response = client.get(f"/api/documents/{document_id}/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] == DocumentStatus.COMPLETED.value
        
        # Step 5: Retrieve sessions
        with patch('app.services.paraphrasing_service.get_document_sessions') as mock_get_sessions:
            mock_get_sessions.return_value = [mock_session]
            
            sessions_response = client.get(f"/api/documents/{document_id}/sessions")
            assert sessions_response.status_code == 200
            sessions_data = sessions_response.json()
            assert len(sessions_data) == 1
            assert sessions_data[0]["method_used"] == ParaphraseMethod.INDOT5.value
    
    def test_enhanced_document_workflow(self, client: TestClient):
        """Test enhanced document workflow with all features."""
        # Step 1: Enhanced upload
        test_content = """
        Penelitian ini bertujuan untuk menganalisis dampak teknologi digital terhadap pembelajaran mahasiswa. 
        Metode yang digunakan adalah mixed-method research dengan pendekatan kualitatif dan kuantitatif. 
        Data dikumpulkan melalui survei (n=200) dan wawancara mendalam (n=30) dengan mahasiswa dari berbagai fakultas.
        """
        file_data = io.BytesIO(test_content.encode())
        
        with patch('app.services.enhanced_document_processor.process_document_enhanced') as mock_process:
            mock_doc = MagicMock()
            mock_doc.id = str(uuid.uuid4())
            mock_doc.filename = "enhanced_test.txt"
            mock_doc.status = DocumentStatus.PENDING
            mock_doc.document_metadata = {
                "academic_terms": ["penelitian", "metode", "kualitatif", "kuantitatif"],
                "structure_preserved": True,
                "sentence_count": 3,
                "word_count": 45
            }
            mock_process.return_value = mock_doc
            
            upload_response = client.post(
                "/api/v2/documents/upload-enhanced",
                files={"file": ("enhanced_test.txt", file_data, "text/plain")},
                data={
                    "chapter": "BAB 2",
                    "preserve_structure": "true",
                    "extract_academic_terms": "true"
                }
            )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        document_id = upload_data["document_id"]
        assert "academic_terms" in upload_data["metadata"]
        
        # Step 2: NLP Analysis
        analysis_request = {
            "text": test_content,
            "extract_academic_terms": True
        }
        
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
            mock_analysis = MagicMock()
            mock_analysis.sentences = [
                MagicMock(priority_for_paraphrasing=0.8, complexity_score=0.7, readability_score=0.6),
                MagicMock(priority_for_paraphrasing=0.9, complexity_score=0.8, readability_score=0.5),
                MagicMock(priority_for_paraphrasing=0.7, complexity_score=0.6, readability_score=0.7)
            ]
            mock_analysis.overall_readability = 0.6
            mock_analysis.overall_complexity = 0.73
            mock_analysis.academic_terms = {"penelitian", "metode", "kualitatif", "kuantitatif", "data"}
            mock_analysis.named_entities = {"mahasiswa"}
            mock_analysis.quality_metrics = {"overall_score": 0.75}
            mock_analysis.paraphrasing_priorities = [1, 0, 2]
            mock_analyze.return_value = mock_analysis
            
            analysis_response = client.post(
                "/api/v2/text/analyze",
                json=analysis_request
            )
        
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        assert analysis_data["total_sentences"] == 3
        assert analysis_data["academic_terms_count"] == 5
        assert len(analysis_data["high_priority_sentences"]) > 0
        
        # Step 3: Enhanced paraphrasing
        paraphrase_request = {
            "document_id": document_id,
            "method": ParaphraseMethod.HYBRID.value,
            "use_nlp_analysis": True,
            "preserve_academic_terms": True,
            "preserve_citations": True,
            "num_variants": 3,
            "quality_threshold": 0.7
        }
        
        with patch('app.services.enhanced_paraphrasing_service.paraphrase_document_enhanced') as mock_enhanced_paraphrase:
            mock_session = MagicMock()
            mock_session.id = str(uuid.uuid4())
            mock_session.method_used = ParaphraseMethod.HYBRID
            mock_session.similarity_score = 0.68
            mock_session.processing_time = 8500
            mock_session.token_usage = {
                "enhanced_features": True,
                "variants_generated": 3,
                "quality_filtered": 2,
                "nlp_analysis_used": True
            }
            mock_enhanced_paraphrase.return_value = mock_session
            
            enhanced_paraphrase_response = client.post(
                f"/api/v2/documents/{document_id}/paraphrase-enhanced",
                json=paraphrase_request
            )
        
        assert enhanced_paraphrase_response.status_code == 200
        enhanced_data = enhanced_paraphrase_response.json()
        assert enhanced_data["method_used"] == ParaphraseMethod.HYBRID.value
        assert enhanced_data["similarity_score"] == 0.68
        assert enhanced_data["enhanced_metadata"]["variants_generated"] == 3
    
    def test_error_handling_workflow(self, client: TestClient):
        """Test error handling throughout the workflow."""
        # Test 1: Invalid file upload
        invalid_response = client.post(
            "/api/documents/upload",
            files={"file": ("", io.BytesIO(b""), "text/plain")}
        )
        assert invalid_response.status_code == 400
        
        # Test 2: Paraphrasing non-existent document
        fake_id = str(uuid.uuid4())
        paraphrase_request = {"method": ParaphraseMethod.INDOT5.value}
        
        nonexistent_response = client.post(
            f"/api/documents/{fake_id}/paraphrase",
            json=paraphrase_request
        )
        assert nonexistent_response.status_code == 404
        
        # Test 3: Service failure simulation
        test_content = "Test content for service failure"
        file_data = io.BytesIO(test_content.encode())
        
        upload_response = client.post(
            "/api/documents/upload",
            files={"file": ("service_fail_test.txt", file_data, "text/plain")}
        )
        
        document_id = upload_response.json()["id"]
        
        with patch('app.services.paraphrasing_service.paraphrase_document') as mock_paraphrase:
            mock_paraphrase.side_effect = Exception("Service unavailable")
            
            failure_response = client.post(
                f"/api/documents/{document_id}/paraphrase",
                json=paraphrase_request
            )
            
            assert failure_response.status_code == 500
            assert "Paraphrasing failed" in failure_response.json()["detail"]


class TestDirectParaphrasingWorkflow:
    """Test direct text paraphrasing workflows."""
    
    def test_multiple_method_comparison(self, client: TestClient):
        """Test comparing different paraphrasing methods on the same text."""
        test_text = "Penelitian ini menggunakan metodologi kualitatif dengan pendekatan fenomenologi untuk memahami pengalaman subjektif partisipan."
        
        # Test IndoT5 method
        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_indot5:
            mock_result = MagicMock()
            mock_result.paraphrased_variants = [
                "Studi ini menerapkan metodologi kualitatif dengan pendekatan fenomenologis untuk memahami pengalaman subjektif peserta.",
                "Riset ini menggunakan metodologi kualitatif dengan pendekatan fenomenologi guna memahami pengalaman subjektif partisipan."
            ]
            mock_result.similarity_scores = [0.72, 0.68]
            mock_result.quality_scores = [0.85, 0.82]
            mock_result.best_variant = mock_result.paraphrased_variants[0]
            mock_result.metadata = {"model": "indot5", "processing_time": 1.2}
            mock_indot5.return_value = mock_result
            
            indot5_request = {
                "text": test_text,
                "method": "indot5",
                "num_variants": 2
            }
            
            indot5_response = client.post(
                "/api/v2/text/paraphrase-direct",
                json=indot5_request
            )
        
        assert indot5_response.status_code == 200
        indot5_data = indot5_response.json()
        
        # Test rule-based method
        with patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule:
            mock_quality1 = MagicMock()
            mock_quality1.similarity_score = 0.65
            mock_quality1.overall_score = 0.78
            
            mock_quality2 = MagicMock()
            mock_quality2.similarity_score = 0.62
            mock_quality2.overall_score = 0.75
            
            mock_rule.return_value = [
                ("Kajian ini menggunakan metodologi kualitatif dengan pendekatan fenomenologi untuk memahami pengalaman subjektif partisipan.", mock_quality1),
                ("Penelitian ini menerapkan metodologi kualitatif dengan pendekatan fenomenologis untuk memahami pengalaman subjektif peserta.", mock_quality2)
            ]
            
            rule_request = {
                "text": test_text,
                "method": "rule_based",
                "num_variants": 2
            }
            
            rule_response = client.post(
                "/api/v2/text/paraphrase-direct",
                json=rule_request
            )
        
        assert rule_response.status_code == 200
        rule_data = rule_response.json()
        
        # Test hybrid method
        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_indot5_hybrid, \
             patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule_hybrid:
            
            mock_indot5_hybrid.return_value = mock_result
            mock_rule_hybrid.return_value = [
                ("Hybrid paraphrase result", mock_quality1)
            ]
            
            hybrid_request = {
                "text": test_text,
                "method": "hybrid",
                "num_variants": 2
            }
            
            hybrid_response = client.post(
                "/api/v2/text/paraphrase-direct",
                json=hybrid_request
            )
        
        assert hybrid_response.status_code == 200
        hybrid_data = hybrid_response.json()
        
        # Compare results
        assert len(indot5_data["variants"]) == 2
        assert len(rule_data["variants"]) == 2
        assert len(hybrid_data["variants"]) >= 2  # Combines both methods
        assert hybrid_data["metadata"]["hybrid_approach"] is True
        
        # Verify different methods produce different results
        assert indot5_data["best_variant"] != rule_data["best_variant"]
    
    def test_quality_assessment_workflow(self, client: TestClient):
        """Test text quality assessment workflow."""
        test_text = "Ini adalah contoh teks yang akan dianalisis kualitasnya untuk menentukan apakah perlu diparafrase."
        
        # Step 1: Quality assessment
        with patch('app.services.rule_based_paraphraser.paraphrase') as mock_quality_check:
            mock_quality = MagicMock()
            mock_quality.readability_score = 0.6
            mock_quality.grammar_score = 0.85
            mock_quality.academic_tone_score = 0.4  # Low academic tone
            mock_quality.similarity_score = 0.95  # High similarity (needs paraphrasing)
            mock_quality.overall_score = 0.65
            mock_quality.issues = ["Low academic tone", "High similarity to original"]
            mock_quality_check.return_value = [("quality test", mock_quality)]
            
            quality_response = client.post(
                "/api/v2/text/quality-assessment",
                params={"text": test_text}
            )
        
        assert quality_response.status_code == 200
        quality_data = quality_response.json()
        
        # Step 2: Based on quality assessment, decide to paraphrase
        if quality_data["academic_tone_score"] < 0.5 or quality_data["overall_quality"] < 0.7:
            # Text needs improvement, proceed with paraphrasing
            with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_improve:
                mock_improved_result = MagicMock()
                mock_improved_result.paraphrased_variants = [
                    "Berikut merupakan contoh teks yang akan dianalisis kualitasnya untuk menentukan kebutuhan parafrase."
                ]
                mock_improved_result.similarity_scores = [0.7]
                mock_improved_result.quality_scores = [0.85]
                mock_improved_result.best_variant = mock_improved_result.paraphrased_variants[0]
                mock_improved_result.metadata = {"improvement_target": "academic_tone"}
                mock_improve.return_value = mock_improved_result
                
                paraphrase_request = {
                    "text": test_text,
                    "method": "indot5",
                    "preserve_academic_terms": True,
                    "num_variants": 1
                }
                
                improve_response = client.post(
                    "/api/v2/text/paraphrase-direct",
                    json=paraphrase_request
                )
            
            assert improve_response.status_code == 200
            improved_data = improve_response.json()
            
            # Step 3: Re-assess quality of improved text
            improved_text = improved_data["best_variant"]
            
            with patch('app.services.rule_based_paraphraser.paraphrase') as mock_reassess:
                mock_improved_quality = MagicMock()
                mock_improved_quality.readability_score = 0.7
                mock_improved_quality.grammar_score = 0.9
                mock_improved_quality.academic_tone_score = 0.8  # Improved
                mock_improved_quality.similarity_score = 0.7  # Good difference
                mock_improved_quality.overall_score = 0.82  # Improved
                mock_improved_quality.issues = []
                mock_reassess.return_value = [("reassessed", mock_improved_quality)]
                
                reassess_response = client.post(
                    "/api/v2/text/quality-assessment",
                    params={"text": improved_text}
                )
            
            assert reassess_response.status_code == 200
            reassess_data = reassess_response.json()
            
            # Verify improvement
            assert reassess_data["academic_tone_score"] > quality_data["academic_tone_score"]
            assert reassess_data["overall_quality"] > quality_data["overall_quality"]


class TestSystemIntegration:
    """Test system-wide integration scenarios."""
    
    def test_concurrent_processing(self, client: TestClient):
        """Test system behavior under concurrent load."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def process_document(thread_id):
            try:
                # Each thread processes a different document
                content = f"Test document {thread_id} content for concurrent processing."
                file_data = io.BytesIO(content.encode())
                
                response = client.post(
                    "/api/documents/upload",
                    files={"file": (f"concurrent_test_{thread_id}.txt", file_data, "text/plain")}
                )
                
                results.put((thread_id, response.status_code == 200))
            except Exception as e:
                results.put((thread_id, False, str(e)))
        
        # Start multiple threads
        threads = []
        num_threads = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_document, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        successful_uploads = 0
        while not results.empty():
            result = results.get()
            if len(result) >= 2 and result[1]:
                successful_uploads += 1
        
        # All uploads should succeed
        assert successful_uploads == num_threads
    
    def test_service_health_monitoring(self, client: TestClient):
        """Test service health and monitoring endpoints."""
        # Check basic health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Check performance stats
        with patch('app.services.enhanced_indot5_paraphraser.get_performance_stats') as mock_stats, \
             patch('app.services.rule_based_paraphraser.get_transformation_stats') as mock_rule_stats:
            
            mock_stats.return_value = {
                "total_requests": 100,
                "successful_requests": 95,
                "average_processing_time": 1.5,
                "cache_hit_rate": 0.3
            }
            
            mock_rule_stats.return_value = {
                "total_transformations": 200,
                "successful_transformations": 190,
                "average_transformation_time": 0.8
            }
            
            with patch('app.services.enhanced_indot5_paraphraser.model_loaded', True), \
                 patch('app.services.indonesian_nlp_pipeline.nlp_id', "test_nlp"):
                
                stats_response = client.get("/api/v2/performance/stats")
        
        assert stats_response.status_code == 200
        stats_data = stats_response.json()
        
        # Verify service status
        assert stats_data["system_status"]["services_loaded"] is True
        assert stats_data["system_status"]["models_ready"]["indot5"] is True
        assert stats_data["system_status"]["models_ready"]["nlp_pipeline"] is True
        
        # Verify performance metrics
        assert stats_data["enhanced_indot5"]["total_requests"] == 100
        assert stats_data["rule_based"]["total_transformations"] == 200
    
    def test_demo_endpoint_integration(self, client: TestClient):
        """Test demo endpoint with full service integration."""
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze, \
             patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_indot5, \
             patch('app.services.rule_based_paraphraser.paraphrase') as mock_rule:
            
            # Mock comprehensive analysis
            mock_analysis = MagicMock()
            mock_analysis.sentences = [MagicMock() for _ in range(4)]
            mock_analysis.overall_readability = 0.65
            mock_analysis.overall_complexity = 0.75
            mock_analysis.academic_terms = {"penelitian", "metode", "analisis", "data", "responden"}
            mock_analysis.named_entities = {"Smith", "mahasiswa"}
            mock_analyze.return_value = mock_analysis
            
            # Mock IndoT5 paraphrasing
            mock_indot5_result = MagicMock()
            mock_indot5_result.best_variant = "Enhanced paraphrased version with improved academic tone and clarity."
            mock_indot5_result.similarity_scores = [0.65, 0.68]
            mock_indot5_result.quality_scores = [0.85, 0.82]
            mock_indot5.return_value = mock_indot5_result
            
            # Mock rule-based paraphrasing
            mock_quality1 = MagicMock()
            mock_quality1.overall_score = 0.78
            mock_quality1.similarity_score = 0.62
            
            mock_quality2 = MagicMock()
            mock_quality2.overall_score = 0.75
            mock_quality2.similarity_score = 0.65
            
            mock_rule.return_value = [
                ("Rule-based paraphrase with lexical substitutions.", mock_quality1),
                ("Alternative rule-based transformation result.", mock_quality2)
            ]
            
            demo_response = client.get("/api/v2/demo/sample-analysis")
        
        assert demo_response.status_code == 200
        demo_data = demo_response.json()
        
        # Verify comprehensive demo response
        assert "sample_text" in demo_data
        assert "nlp_analysis" in demo_data
        assert "paraphrasing_results" in demo_data
        
        # Check NLP analysis results
        nlp_analysis = demo_data["nlp_analysis"]
        assert nlp_analysis["total_sentences"] == 4
        assert nlp_analysis["readability_score"] == 0.65
        assert len(nlp_analysis["academic_terms"]) == 5
        
        # Check paraphrasing results
        paraphrasing_results = demo_data["paraphrasing_results"]
        assert "indot5" in paraphrasing_results
        assert "rule_based" in paraphrasing_results
        
        indot5_results = paraphrasing_results["indot5"]
        assert indot5_results["best_variant"] is not None
        assert len(indot5_results["similarity_scores"]) == 2
        
        rule_based_results = paraphrasing_results["rule_based"]
        assert len(rule_based_results) == 2
        assert all("quality_score" in result for result in rule_based_results)
    
    def test_full_pipeline_integration(self, client: TestClient):
        """Test the complete pipeline from upload to final output."""
        # Academic paper excerpt
        academic_text = """
        Penelitian ini bertujuan untuk menganalisis pengaruh teknologi digital terhadap efektivitas pembelajaran 
        di perguruan tinggi. Metode yang digunakan adalah mixed-method research dengan mengkombinasikan pendekatan 
        kualitatif dan kuantitatif. Populasi penelitian adalah mahasiswa aktif di tiga perguruan tinggi negeri 
        di Jakarta dengan sampel sebanyak 300 responden yang dipilih menggunakan teknik stratified random sampling.
        """
        
        # Step 1: Enhanced document upload
        file_data = io.BytesIO(academic_text.encode())
        
        with patch('app.services.enhanced_document_processor.process_document_enhanced') as mock_process:
            mock_doc = MagicMock()
            mock_doc.id = str(uuid.uuid4())
            mock_doc.filename = "academic_paper.txt"
            mock_doc.status = DocumentStatus.PENDING
            mock_doc.document_metadata = {
                "academic_terms": ["penelitian", "metode", "mixed-method", "kualitatif", "kuantitatif"],
                "sentence_count": 3,
                "word_count": 67,
                "complexity_score": 0.8
            }
            mock_process.return_value = mock_doc
            
            upload_response = client.post(
                "/api/v2/documents/upload-enhanced",
                files={"file": ("academic_paper.txt", file_data, "text/plain")},
                data={
                    "preserve_structure": "true",
                    "extract_academic_terms": "true"
                }
            )
        
        document_id = upload_response.json()["document_id"]
        
        # Step 2: NLP Analysis
        with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
            mock_analysis = MagicMock()
            mock_analysis.sentences = [MagicMock() for _ in range(3)]
            mock_analysis.overall_readability = 0.55
            mock_analysis.overall_complexity = 0.8
            mock_analysis.academic_terms = {"penelitian", "metode", "kualitatif", "kuantitatif", "responden"}
            mock_analysis.named_entities = {"Jakarta"}
            mock_analysis.quality_metrics = {"overall_score": 0.7}
            mock_analysis.paraphrasing_priorities = [0, 1, 2]
            mock_analyze.return_value = mock_analysis
            
            analysis_response = client.post(
                "/api/v2/text/analyze",
                json={"text": academic_text, "extract_academic_terms": True}
            )
        
        # Step 3: Quality assessment
        with patch('app.services.rule_based_paraphraser.paraphrase') as mock_quality:
            mock_quality_obj = MagicMock()
            mock_quality_obj.readability_score = 0.55
            mock_quality_obj.grammar_score = 0.85
            mock_quality_obj.academic_tone_score = 0.9
            mock_quality_obj.similarity_score = 0.95
            mock_quality_obj.overall_score = 0.75
            mock_quality_obj.issues = ["Complex sentence structure"]
            mock_quality.return_value = [("quality check", mock_quality_obj)]
            
            quality_response = client.post(
                "/api/v2/text/quality-assessment",
                params={"text": academic_text}
            )
        
        # Step 4: Enhanced paraphrasing
        with patch('app.services.enhanced_paraphrasing_service.paraphrase_document_enhanced') as mock_enhanced:
            mock_session = MagicMock()
            mock_session.id = str(uuid.uuid4())
            mock_session.method_used = ParaphraseMethod.HYBRID
            mock_session.similarity_score = 0.68
            mock_session.processing_time = 12000
            mock_session.token_usage = {
                "total_variants_generated": 5,
                "quality_filtered_variants": 3,
                "final_selected_variant": 1,
                "academic_terms_preserved": 5,
                "nlp_analysis_applied": True
            }
            mock_enhanced.return_value = mock_session
            
            enhanced_paraphrase_response = client.post(
                f"/api/v2/documents/{document_id}/paraphrase-enhanced",
                json={
                    "document_id": document_id,
                    "method": ParaphraseMethod.HYBRID.value,
                    "use_nlp_analysis": True,
                    "preserve_academic_terms": True,
                    "preserve_citations": True,
                    "num_variants": 5,
                    "quality_threshold": 0.7
                }
            )
        
        # Verify complete pipeline
        assert upload_response.status_code == 200
        assert analysis_response.status_code == 200
        assert quality_response.status_code == 200
        assert enhanced_paraphrase_response.status_code == 200
        
        # Verify data flow
        upload_data = upload_response.json()
        analysis_data = analysis_response.json()
        quality_data = quality_response.json()
        paraphrase_data = enhanced_paraphrase_response.json()
        
        # Check that academic terms are consistently identified
        assert len(upload_data["metadata"]["academic_terms"]) >= 3
        assert analysis_data["academic_terms_count"] >= 3
        
        # Check quality assessment identifies areas for improvement
        assert quality_data["academic_tone_score"] > 0.8  # High academic tone
        assert quality_data["overall_quality"] > 0.7
        
        # Check enhanced paraphrasing results
        assert paraphrase_data["method_used"] == ParaphraseMethod.HYBRID.value
        assert paraphrase_data["enhanced_metadata"]["nlp_analysis_applied"] is True
        assert paraphrase_data["enhanced_metadata"]["academic_terms_preserved"] >= 3
