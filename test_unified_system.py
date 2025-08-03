"""
Test Script for Unified Paraphrasing System
Validates the new unified architecture and API endpoints.
"""
import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_unified_paraphraser():
    """Test the unified paraphraser directly."""
    try:
        from app.services.unified_paraphraser import (
            get_unified_paraphraser, UnifiedInput, UnifiedOptions, QualityCriteria
        )
        
        logger.info("Testing Unified Paraphraser...")
        
        # Initialize with custom synonyms
        synonyms_path = Path(__file__).parent / "synonyms.json"
        unified_paraphraser = get_unified_paraphraser(str(synonyms_path))
        
        # Test input
        test_text = """
        Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji 
        faktor-faktor yang mempengaruhi kualitas pembelajaran. Hasil penelitian 
        menunjukkan bahwa terdapat hubungan signifikan antara metode pengajaran 
        dan tingkat pemahaman siswa.
        """
        
        # Configure options
        options = UnifiedOptions(
            num_variants=3,
            preserve_academic_terms=True,
            preserve_citations=True,
            use_custom_synonyms=True
        )
        
        quality_criteria = QualityCriteria(
            quality_threshold=0.7
        )
        
        unified_input = UnifiedInput(
            text=test_text.strip(),
            options=options,
            quality_criteria=quality_criteria
        )
        
        # Execute paraphrasing
        logger.info("Executing unified paraphrasing...")
        result = await unified_paraphraser.unified_paraphrase(unified_input)
        
        # Display results
        logger.info("=== UNIFIED PARAPHRASING RESULTS ===")
        logger.info(f"Original: {test_text.strip()}")
        logger.info(f"Best Variant: {result.best_variant}")
        logger.info(f"Processing Time: {result.processing_time:.2f}s")
        logger.info(f"Quality Score: {result.quality_assessment.overall_score:.3f}")
        logger.info(f"Method Contributions: {result.method_contributions}")
        
        # Test performance stats
        stats = unified_paraphraser.get_performance_stats()
        logger.info(f"Performance Stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unified paraphraser test failed: {e}")
        return False


async def test_api_models():
    """Test the unified API models."""
    try:
        from app.api.unified_models import (
            UnifiedParaphraseRequest, UnifiedParaphraseResponse,
            QualityAssessment, QualityDimensions
        )
        
        logger.info("Testing Unified API Models...")
        
        # Test request model
        request = UnifiedParaphraseRequest(
            text="Test text for paraphrasing",
            num_variants=3,
            preserve_academic_terms=True
        )
        
        # Validate input
        request.validate_input()
        logger.info("✅ Request model validation passed")
        
        # Test response model structure
        quality_assessment = QualityAssessment(
            overall_score=0.85,
            dimension_scores=QualityDimensions(
                semantic_similarity=0.8,
                grammar_correctness=0.9,
                academic_tone_preservation=0.8,
                readability_score=0.85,
                structural_diversity=0.7,
                context_appropriateness=0.9
            ),
            confidence_level=0.88,
            recommendations=["Good quality paraphrase"],
            meets_threshold=True
        )
        
        response = UnifiedParaphraseResponse(
            original_text="Test text",
            best_variant="Paraphrased test text",
            processing_time=2.5,
            quality_assessment=quality_assessment
        )
        
        logger.info("✅ Response model creation passed")
        logger.info(f"Sample response: {response.model_dump()}")
        
        return True
        
    except Exception as e:
        logger.error(f"API models test failed: {e}")
        return False


def test_synonyms_loading():
    """Test custom synonyms loading."""
    try:
        from app.services.unified_paraphraser import UnifiedParaphraser
        
        logger.info("Testing Custom Synonyms Loading...")
        
        # Test with existing synonyms file
        synonyms_path = Path(__file__).parent / "synonyms.json"
        
        if synonyms_path.exists():
            paraphraser = UnifiedParaphraser(str(synonyms_path))
            logger.info(f"✅ Loaded {len(paraphraser.custom_synonyms)} custom synonyms")
            
            # Display sample synonyms
            sample_words = list(paraphraser.custom_synonyms.keys())[:5]
            for word in sample_words:
                synonyms = paraphraser.custom_synonyms[word]['sinonim']
                logger.info(f"  {word}: {synonyms}")
        else:
            logger.warning("synonyms.json not found, testing without custom synonyms")
            paraphraser = UnifiedParaphraser()
            logger.info("✅ Initialized without custom synonyms")
        
        return True
        
    except Exception as e:
        logger.error(f"Synonyms loading test failed: {e}")
        return False


async def test_integration():
    """Test full integration."""
    try:
        logger.info("Running Full Integration Test...")
        
        # Test 1: Unified Paraphraser
        result1 = await test_unified_paraphraser()
        
        # Test 2: API Models
        result2 = await test_api_models()
        
        # Test 3: Synonyms Loading
        result3 = test_synonyms_loading()
        
        # Summary
        logger.info("\n=== INTEGRATION TEST SUMMARY ===")
        logger.info(f"Unified Paraphraser: {'✅ PASS' if result1 else '❌ FAIL'}")
        logger.info(f"API Models: {'✅ PASS' if result2 else '❌ FAIL'}")
        logger.info(f"Synonyms Loading: {'✅ PASS' if result3 else '❌ FAIL'}")
        
        overall_success = all([result1, result2, result3])
        logger.info(f"Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing PlagiCheck Unified System")
    print("=" * 50)
    
    # Run tests
    success = asyncio.run(test_integration())
    
    print("=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ Unified system is ready for deployment")
    else:
        print("⚠️  Some tests failed")
        print("❌ Check logs for details")
    
    exit(0 if success else 1)
