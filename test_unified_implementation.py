#!/usr/bin/env python3
"""
Test Script for Unified Paraphrasing Implementation
Test the unified paraphrasing system end-to-end
"""
import asyncio
import json
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_unified_system():
    """Test the unified paraphrasing system."""
    print("🚀 Testing Unified Paraphrasing System")
    print("=" * 50)
    
    try:
        # Import required modules
        from app.services.unified_paraphraser import (
            get_unified_paraphraser, 
            UnifiedInput, 
            UnifiedOptions, 
            QualityCriteria,
            FormalityLevel,
            ComplexityTarget
        )
        
        # Test texts
        test_texts = [
            "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji faktor-faktor yang mempengaruhi kualitas pembelajaran.",
            "Berdasarkan hasil analisis data, dapat disimpulkan bahwa teknologi digital memiliki dampak signifikan terhadap proses pembelajaran.",
            "Metode penelitian yang digunakan dalam studi ini adalah pendekatan kuantitatif dengan teknik survei."
        ]
        
        # Initialize unified paraphraser
        print("🔧 Initializing Unified Paraphraser...")
        synonyms_path = Path("synonyms.json")
        if not synonyms_path.exists():
            print("⚠️  Custom synonyms file not found, creating sample...")
            create_sample_synonyms(synonyms_path)
        
        unified_paraphraser = get_unified_paraphraser(str(synonyms_path))
        print("✅ Unified Paraphraser initialized successfully")
        
        # Test each text
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 Test {i}: Processing text...")
            print(f"Original: {text}")
            
            # Configure options
            options = UnifiedOptions(
                num_variants=3,
                preserve_academic_terms=True,
                preserve_citations=True,
                formality_level=FormalityLevel.ACADEMIC,
                target_complexity=ComplexityTarget.MAINTAIN,
                use_indot5=True,
                use_rule_based=True,
                use_custom_synonyms=True,
                use_structural_transformation=True
            )
            
            # Configure quality criteria
            quality_criteria = QualityCriteria(
                quality_threshold=0.7,
                min_semantic_similarity=0.3,
                max_semantic_similarity=0.9,
                min_grammar_score=0.6,
                require_academic_tone=True
            )
            
            # Create unified input
            unified_input = UnifiedInput(
                text=text,
                options=options,
                quality_criteria=quality_criteria
            )
            
            # Execute paraphrasing
            start_time = time.time()
            try:
                result = await unified_paraphraser.unified_paraphrase(unified_input)
                processing_time = time.time() - start_time
                
                print(f"✅ Processing completed in {processing_time:.2f}s")
                print(f"📊 Best variant: {result.best_variant}")
                print(f"⭐ Quality score: {result.quality_assessment.overall_score:.3f}")
                print(f"🏆 Confidence: {result.quality_assessment.confidence_level:.3f}")
                print(f"🎯 Meets threshold: {result.quality_assessment.meets_threshold}")
                print(f"🔧 Methods used: {list(result.method_contributions.keys())}")
                
                if result.quality_assessment.recommendations:
                    print(f"💡 Recommendations: {', '.join(result.quality_assessment.recommendations)}")
                
            except Exception as e:
                print(f"❌ Error processing text: {e}")
                logger.error(f"Processing failed: {e}", exc_info=True)
        
        # Test performance stats
        print(f"\n📈 Performance Statistics:")
        stats = unified_paraphraser.get_performance_stats()
        print(json.dumps(stats, indent=2))
        
        print("\n🎉 All tests completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are available")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test execution failed: {e}", exc_info=True)


def create_sample_synonyms(path: Path):
    """Create sample synonyms.json for testing."""
    sample_synonyms = {
        "penelitian": {
            "tag": "NN",
            "sinonim": ["riset", "studi", "kajian", "investigasi"]
        },
        "analisis": {
            "tag": "NN", 
            "sinonim": ["analisa", "pemeriksaan", "penelaahan", "pengkajian"]
        },
        "metode": {
            "tag": "NN",
            "sinonim": ["cara", "pendekatan", "teknik", "sistem"]
        },
        "hasil": {
            "tag": "NN",
            "sinonim": ["output", "luaran", "produk", "kesimpulan"]
        },
        "menunjukkan": {
            "tag": "VB",
            "sinonim": ["memperlihatkan", "mengindikasikan", "menyatakan", "membuktikan"]
        },
        "signifikan": {
            "tag": "JJ",
            "sinonim": ["penting", "berarti", "bermakna", "relevan"]
        },
        "pembelajaran": {
            "tag": "NN",
            "sinonim": ["pendidikan", "edukasi", "pengajaran", "pelatihan"]
        }
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sample_synonyms, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Sample synonyms created at {path}")


async def test_api_integration():
    """Test API integration (if server is running)."""
    print("\n🌐 Testing API Integration...")
    print("=" * 30)
    
    import aiohttp
    
    # Test data
    test_request = {
        "text": "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji pembelajaran.",
        "num_variants": 3,
        "preserve_academic_terms": True,
        "preserve_citations": True,
        "include_detailed_analysis": True,
        "include_method_insights": True,
        "formality_level": "academic",
        "target_complexity": "maintain",
        "use_custom_synonyms": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test unified endpoint
            async with session.post(
                "http://localhost:8000/api/paraphrase",
                json=test_request,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ API test successful!")
                    print(f"📊 Best variant: {result.get('best_variant', 'N/A')}")
                    print(f"⭐ Quality score: {result.get('quality_assessment', {}).get('overall_score', 'N/A')}")
                    print(f"⏱️  Processing time: {result.get('processing_time', 'N/A')}s")
                else:
                    error_text = await response.text()
                    print(f"❌ API test failed: {response.status}")
                    print(f"Error: {error_text}")
            
            # Test health endpoint
            async with session.get("http://localhost:8000/api/paraphrase/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"💚 Health check: {health.get('status', 'unknown')}")
                    print(f"🔧 Methods available: {health.get('methods_available', {})}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    
    except aiohttp.ClientConnectorError:
        print("⚠️  API server not running. Please start the server with:")
        print("   python main.py")
    except Exception as e:
        print(f"❌ API test error: {e}")


if __name__ == "__main__":
    print("🧪 Unified Paraphrasing System Test Suite")
    print("=" * 50)
    
    # Run system tests
    asyncio.run(test_unified_system())
    
    # Run API tests
    asyncio.run(test_api_integration())
    
    print("\n✨ Test suite completed!")
