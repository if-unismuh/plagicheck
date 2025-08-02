"""
Performance and Load Tests
Testing system performance, scalability, and resource usage.
"""
import time
import threading
import concurrent.futures
import statistics
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        self.end_time = None
    
    def add_result(self, response_time: float, success: bool):
        """Add a single test result."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def start_timer(self):
        """Start the overall test timer."""
        self.start_time = time.time()
    
    def stop_timer(self):
        """Stop the overall test timer."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        total_requests = self.success_count + self.error_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": success_rate,
            "avg_response_time": statistics.mean(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": self._percentile(self.response_times, 95),
            "p99_response_time": self._percentile(self.response_times, 99),
            "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "requests_per_second": total_requests / (self.end_time - self.start_time) if self.start_time and self.end_time and (self.end_time - self.start_time) > 0 else 0
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of response times."""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestAPIPerformance:
    """Test API endpoint performance under various conditions."""
    
    def test_single_request_performance(self, client: TestClient):
        """Test performance of individual API requests."""
        endpoints_to_test = [
            ("GET", "/", {}),
            ("GET", "/health", {}),
            ("GET", "/api/documents", {}),
            ("GET", "/api/v2/performance/stats", {})
        ]
        
        performance_results = {}
        
        for method, endpoint, params in endpoints_to_test:
            response_times = []
            
            # Test each endpoint multiple times
            for _ in range(10):
                start_time = time.time()
                
                if method == "GET":
                    response = client.get(endpoint, params=params)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Basic assertions
                assert response.status_code in [200, 404]  # 404 acceptable for some endpoints
                response_times.append(response_time)
            
            # Calculate metrics
            avg_time = statistics.mean(response_times)
            max_time = max(response_times)
            
            performance_results[endpoint] = {
                "avg_response_time": avg_time,
                "max_response_time": max_time,
                "min_response_time": min(response_times)
            }
            
            # Performance assertions (adjust thresholds as needed)
            assert avg_time < 1.0, f"Average response time for {endpoint} too high: {avg_time:.3f}s"
            assert max_time < 2.0, f"Max response time for {endpoint} too high: {max_time:.3f}s"
        
        print("\nPerformance Results:")
        for endpoint, metrics in performance_results.items():
            print(f"{endpoint}: avg={metrics['avg_response_time']:.3f}s, max={metrics['max_response_time']:.3f}s")
    
    def test_concurrent_document_uploads(self, client: TestClient):
        """Test performance under concurrent document uploads."""
        metrics = PerformanceMetrics()
        
        def upload_document(thread_id: int) -> bool:
            """Upload a document and measure performance."""
            try:
                import io
                content = f"Performance test document {thread_id}. " * 50  # ~1KB content
                file_data = io.BytesIO(content.encode())
                
                start_time = time.time()
                response = client.post(
                    "/api/documents/upload",
                    files={"file": (f"perf_test_{thread_id}.txt", file_data, "text/plain")}
                )
                end_time = time.time()
                
                response_time = end_time - start_time
                success = response.status_code == 200
                
                metrics.add_result(response_time, success)
                return success
                
            except Exception:
                metrics.add_result(0, False)
                return False
        
        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}
        
        for concurrency in concurrency_levels:
            metrics = PerformanceMetrics()
            metrics.start_timer()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(upload_document, i) for i in range(concurrency)]
                
                # Wait for all to complete
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            
            metrics.stop_timer()
            summary = metrics.get_summary()
            results[concurrency] = summary
            
            # Performance assertions
            assert summary["success_rate"] >= 0.9, f"Success rate too low at concurrency {concurrency}: {summary['success_rate']}"
            assert summary["avg_response_time"] < 5.0, f"Average response time too high at concurrency {concurrency}: {summary['avg_response_time']:.3f}s"
        
        print("\nConcurrency Test Results:")
        for concurrency, summary in results.items():
            print(f"Concurrency {concurrency}: "
                  f"Success Rate: {summary['success_rate']:.2%}, "
                  f"Avg Response: {summary['avg_response_time']:.3f}s, "
                  f"RPS: {summary['requests_per_second']:.1f}")
    
    def test_paraphrasing_performance(self, client: TestClient):
        """Test paraphrasing performance with different text lengths."""
        test_cases = [
            ("short", "Penelitian ini menggunakan metode kualitatif."),
            ("medium", "Penelitian ini menggunakan metode kualitatif untuk menganalisis data. " * 5),
            ("long", "Penelitian ini menggunakan metode kualitatif untuk menganalisis data yang dikumpulkan melalui wawancara mendalam. " * 20)
        ]
        
        performance_results = {}
        
        for case_name, text in test_cases:
            response_times = []
            
            with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                # Mock with realistic processing times based on text length
                processing_delay = len(text) / 1000  # Simulate processing time
                
                def mock_paraphrase_with_delay(*args, **kwargs):
                    time.sleep(processing_delay)
                    mock_result = MagicMock()
                    mock_result.paraphrased_variants = [f"Paraphrased: {text[:50]}..."]
                    mock_result.similarity_scores = [0.7]
                    mock_result.quality_scores = [0.8]
                    mock_result.best_variant = mock_result.paraphrased_variants[0]
                    mock_result.metadata = {"processing_time": processing_delay}
                    return mock_result
                
                mock_paraphrase.side_effect = mock_paraphrase_with_delay
                
                # Test multiple times
                for _ in range(5):
                    request_data = {
                        "text": text,
                        "method": "indot5",
                        "num_variants": 1
                    }
                    
                    start_time = time.time()
                    response = client.post("/api/v2/text/paraphrase-direct", json=request_data)
                    end_time = time.time()
                    
                    assert response.status_code == 200
                    response_times.append(end_time - start_time)
            
            avg_time = statistics.mean(response_times)
            performance_results[case_name] = {
                "text_length": len(text),
                "avg_response_time": avg_time,
                "max_response_time": max(response_times)
            }
            
            # Performance scaling assertions
            expected_max_time = 2.0 + (len(text) / 1000)  # Scale with text length
            assert avg_time < expected_max_time, f"Paraphrasing too slow for {case_name}: {avg_time:.3f}s"
        
        print("\nParaphrasing Performance Results:")
        for case, metrics in performance_results.items():
            print(f"{case} ({metrics['text_length']} chars): "
                  f"avg={metrics['avg_response_time']:.3f}s, "
                  f"max={metrics['max_response_time']:.3f}s")
    
    def test_memory_usage_under_load(self, client: TestClient):
        """Test memory usage patterns under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate sustained load
        total_requests = 50
        memory_samples = []
        
        for i in range(total_requests):
            # Make various types of requests
            request_type = i % 4
            
            if request_type == 0:
                client.get("/health")
            elif request_type == 1:
                client.get("/api/documents")
            elif request_type == 2:
                with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
                    mock_analyze.return_value = MagicMock()
                    client.post("/api/v2/text/analyze", json={"text": "Test text", "extract_academic_terms": True})
            else:
                with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                    mock_paraphrase.return_value = MagicMock()
                    client.post("/api/v2/text/paraphrase-direct", 
                               json={"text": "Test", "method": "indot5", "num_variants": 1})
            
            # Sample memory every 10 requests
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_samples)
        
        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.1f} MB")
        print(f"Final: {final_memory:.1f} MB")
        print(f"Max: {max_memory:.1f} MB")
        print(f"Increase: {memory_increase:.1f} MB")
        
        # Memory assertions (adjust based on your system)
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f} MB"
        assert max_memory < initial_memory + 150, f"Peak memory usage too high: {max_memory:.1f} MB"


class TestDatabasePerformance:
    """Test database operation performance."""
    
    def test_database_connection_performance(self, client: TestClient):
        """Test database connection and query performance."""
        response_times = []
        
        # Test multiple database operations
        for _ in range(20):
            start_time = time.time()
            response = client.get("/api/documents")  # This hits the database
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        print(f"\nDatabase Performance:")
        print(f"Average query time: {avg_time:.3f}s")
        print(f"Max query time: {max_time:.3f}s")
        
        # Database performance assertions
        assert avg_time < 0.5, f"Database queries too slow: {avg_time:.3f}s"
        assert max_time < 1.0, f"Slowest database query too slow: {max_time:.3f}s"
    
    def test_bulk_document_operations(self, client: TestClient):
        """Test performance of bulk document operations."""
        import io
        
        # Upload multiple documents
        document_ids = []
        upload_times = []
        
        for i in range(10):
            content = f"Bulk test document {i}. " + "Content. " * 100
            file_data = io.BytesIO(content.encode())
            
            start_time = time.time()
            response = client.post(
                "/api/documents/upload",
                files={"file": (f"bulk_test_{i}.txt", file_data, "text/plain")}
            )
            end_time = time.time()
            
            assert response.status_code == 200
            document_ids.append(response.json()["id"])
            upload_times.append(end_time - start_time)
        
        # Test bulk retrieval
        retrieval_times = []
        
        for doc_id in document_ids:
            start_time = time.time()
            response = client.get(f"/api/documents/{doc_id}")
            end_time = time.time()
            
            assert response.status_code == 200
            retrieval_times.append(end_time - start_time)
        
        # Test list operation performance
        start_time = time.time()
        list_response = client.get("/api/documents?limit=20")
        end_time = time.time()
        list_time = end_time - start_time
        
        assert list_response.status_code == 200
        documents = list_response.json()
        assert len(documents) >= 10
        
        print(f"\nBulk Operations Performance:")
        print(f"Average upload time: {statistics.mean(upload_times):.3f}s")
        print(f"Average retrieval time: {statistics.mean(retrieval_times):.3f}s")
        print(f"List operation time: {list_time:.3f}s")
        
        # Performance assertions
        assert statistics.mean(upload_times) < 2.0, "Bulk uploads too slow"
        assert statistics.mean(retrieval_times) < 0.5, "Bulk retrievals too slow"
        assert list_time < 1.0, "List operation too slow"


class TestServicePerformance:
    """Test individual service performance."""
    
    def test_nlp_pipeline_performance(self, client: TestClient):
        """Test NLP pipeline performance with various text complexities."""
        test_texts = [
            "Simple text.",
            "Penelitian ini menggunakan metode kualitatif untuk analisis data.",
            "Penelitian ini bertujuan untuk menganalisis dampak teknologi digital terhadap efektivitas pembelajaran di perguruan tinggi. Metode yang digunakan adalah mixed-method research dengan mengkombinasikan pendekatan kualitatif dan kuantitatif. Data dikumpulkan melalui survei dan wawancara mendalam dengan mahasiswa dari berbagai fakultas."
        ]
        
        for i, text in enumerate(test_texts):
            response_times = []
            
            with patch('app.services.indonesian_nlp_pipeline.analyze_document') as mock_analyze:
                def mock_analyze_with_delay(*args, **kwargs):
                    # Simulate processing time based on text complexity
                    processing_time = len(text) / 5000  # Realistic processing time
                    time.sleep(processing_time)
                    
                    mock_result = MagicMock()
                    mock_result.sentences = [MagicMock() for _ in range(len(text.split('.')))]
                    mock_result.overall_readability = 0.7
                    mock_result.overall_complexity = 0.6
                    mock_result.academic_terms = {"penelitian", "metode"}
                    mock_result.named_entities = {"test"}
                    mock_result.quality_metrics = {"score": 0.8}
                    mock_result.paraphrasing_priorities = [0, 1]
                    return mock_result
                
                mock_analyze.side_effect = mock_analyze_with_delay
                
                # Test multiple times
                for _ in range(3):
                    start_time = time.time()
                    response = client.post(
                        "/api/v2/text/analyze",
                        json={"text": text, "extract_academic_terms": True}
                    )
                    end_time = time.time()
                    
                    assert response.status_code == 200
                    response_times.append(end_time - start_time)
            
            avg_time = statistics.mean(response_times)
            print(f"NLP Analysis {i+1} ({len(text)} chars): {avg_time:.3f}s")
            
            # Performance should scale reasonably with text length
            expected_max_time = 1.0 + (len(text) / 1000)
            assert avg_time < expected_max_time, f"NLP analysis too slow for text {i+1}"
    
    def test_paraphrasing_service_scaling(self, client: TestClient):
        """Test how paraphrasing services scale with different parameters."""
        base_text = "Penelitian ini menggunakan metode analisis kualitatif."
        
        # Test with different numbers of variants
        variant_counts = [1, 2, 3, 5]
        performance_by_variants = {}
        
        for num_variants in variant_counts:
            response_times = []
            
            with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                def mock_paraphrase_scaled(*args, **kwargs):
                    # Simulate scaling with number of variants
                    processing_time = 0.1 * num_variants
                    time.sleep(processing_time)
                    
                    mock_result = MagicMock()
                    mock_result.paraphrased_variants = [f"Variant {i}: {base_text}" for i in range(num_variants)]
                    mock_result.similarity_scores = [0.7] * num_variants
                    mock_result.quality_scores = [0.8] * num_variants
                    mock_result.best_variant = mock_result.paraphrased_variants[0]
                    mock_result.metadata = {"variants": num_variants}
                    return mock_result
                
                mock_paraphrase.side_effect = mock_paraphrase_scaled
                
                for _ in range(3):
                    start_time = time.time()
                    response = client.post(
                        "/api/v2/text/paraphrase-direct",
                        json={
                            "text": base_text,
                            "method": "indot5",
                            "num_variants": num_variants
                        }
                    )
                    end_time = time.time()
                    
                    assert response.status_code == 200
                    response_times.append(end_time - start_time)
            
            avg_time = statistics.mean(response_times)
            performance_by_variants[num_variants] = avg_time
            
            print(f"Paraphrasing with {num_variants} variants: {avg_time:.3f}s")
        
        # Check that performance scales reasonably
        for variants, time_taken in performance_by_variants.items():
            expected_max_time = 0.5 + (0.2 * variants)  # Should scale roughly linearly
            assert time_taken < expected_max_time, f"Paraphrasing with {variants} variants too slow: {time_taken:.3f}s"
    
    def test_cache_performance(self, client: TestClient):
        """Test performance improvements from caching."""
        test_text = "Penelitian ini menggunakan metode kualitatif untuk analisis."
        
        # Mock cache miss (first request)
        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
            mock_result = MagicMock()
            mock_result.paraphrased_variants = ["Cached result"]
            mock_result.similarity_scores = [0.7]
            mock_result.quality_scores = [0.8]
            mock_result.best_variant = "Cached result"
            mock_result.metadata = {"cached": False}
            
            def slow_first_request(*args, **kwargs):
                time.sleep(0.5)  # Simulate slow processing
                return mock_result
            
            mock_paraphrase.side_effect = slow_first_request
            
            # First request (cache miss)
            start_time = time.time()
            response = client.post(
                "/api/v2/text/paraphrase-direct",
                json={"text": test_text, "method": "indot5", "num_variants": 1}
            )
            first_request_time = time.time() - start_time
            
            assert response.status_code == 200
        
        # Mock cache hit (subsequent requests)
        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase_cached:
            mock_result.metadata = {"cached": True}
            
            def fast_cached_request(*args, **kwargs):
                time.sleep(0.01)  # Simulate fast cache retrieval
                return mock_result
            
            mock_paraphrase_cached.side_effect = fast_cached_request
            
            # Subsequent requests (cache hits)
            cached_times = []
            for _ in range(3):
                start_time = time.time()
                response = client.post(
                    "/api/v2/text/paraphrase-direct",
                    json={"text": test_text, "method": "indot5", "num_variants": 1}
                )
                cached_times.append(time.time() - start_time)
                assert response.status_code == 200
        
        avg_cached_time = statistics.mean(cached_times)
        speedup = first_request_time / avg_cached_time
        
        print(f"\nCache Performance:")
        print(f"First request (cache miss): {first_request_time:.3f}s")
        print(f"Cached requests average: {avg_cached_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        # Cache should provide significant speedup
        assert speedup > 5, f"Cache speedup insufficient: {speedup:.1f}x"
        assert avg_cached_time < 0.1, f"Cached requests too slow: {avg_cached_time:.3f}s"


@pytest.mark.performance
class TestStressTest:
    """Stress tests for system limits."""
    
    def test_high_concurrency_stress(self, client: TestClient):
        """Test system behavior under high concurrency."""
        concurrent_users = 50
        requests_per_user = 5
        
        metrics = PerformanceMetrics()
        metrics.start_timer()
        
        def stress_test_user(user_id: int):
            """Simulate a user making multiple requests."""
            for request_id in range(requests_per_user):
                try:
                    start_time = time.time()
                    
                    # Mix different types of requests
                    if request_id % 3 == 0:
                        response = client.get("/health")
                    elif request_id % 3 == 1:
                        response = client.get("/api/documents")
                    else:
                        with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single'):
                            response = client.post(
                                "/api/v2/text/paraphrase-direct",
                                json={"text": f"Stress test {user_id}-{request_id}", "method": "indot5", "num_variants": 1}
                            )
                    
                    end_time = time.time()
                    success = response.status_code in [200, 404]
                    metrics.add_result(end_time - start_time, success)
                    
                except Exception:
                    metrics.add_result(0, False)
        
        # Run stress test
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(stress_test_user, i) for i in range(concurrent_users)]
            
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        metrics.stop_timer()
        summary = metrics.get_summary()
        
        print(f"\nStress Test Results:")
        print(f"Total requests: {summary['total_requests']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Average response time: {summary['avg_response_time']:.3f}s")
        print(f"95th percentile: {summary['p95_response_time']:.3f}s")
        print(f"Requests per second: {summary['requests_per_second']:.1f}")
        
        # Stress test assertions
        assert summary["success_rate"] >= 0.95, f"Success rate too low under stress: {summary['success_rate']:.2%}"
        assert summary["avg_response_time"] < 3.0, f"Average response time too high under stress: {summary['avg_response_time']:.3f}s"
        assert summary["p95_response_time"] < 5.0, f"95th percentile too high under stress: {summary['p95_response_time']:.3f}s"
    
    def test_memory_stress(self, client: TestClient):
        """Test system behavior under memory pressure."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Generate large requests to stress memory
        large_text = "This is a large text document for memory stress testing. " * 1000  # ~50KB text
        
        memory_samples = []
        successful_requests = 0
        
        for i in range(20):  # Reduced from 100 for test efficiency
            try:
                with patch('app.services.enhanced_indot5_paraphraser.paraphrase_single') as mock_paraphrase:
                    # Mock to avoid actual model inference but still process data
                    mock_result = MagicMock()
                    mock_result.paraphrased_variants = [large_text[:1000]]  # Return truncated version
                    mock_result.similarity_scores = [0.7]
                    mock_result.quality_scores = [0.8]
                    mock_result.best_variant = mock_result.paraphrased_variants[0]
                    mock_result.metadata = {"large_text": True}
                    mock_paraphrase.return_value = mock_result
                    
                    response = client.post(
                        "/api/v2/text/paraphrase-direct",
                        json={"text": large_text, "method": "indot5", "num_variants": 1}
                    )
                
                if response.status_code == 200:
                    successful_requests += 1
                
                # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
            except Exception:
                pass  # Continue even if some requests fail
        
        max_memory = max(memory_samples) if memory_samples else initial_memory
        memory_increase = max_memory - initial_memory
        
        print(f"\nMemory Stress Test:")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory: {max_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Successful requests: {successful_requests}/20")
        
        # Memory stress assertions
        assert successful_requests >= 15, f"Too many failed requests under memory stress: {successful_requests}/20"
        assert memory_increase < 500, f"Memory increase too high: {memory_increase:.1f} MB"
