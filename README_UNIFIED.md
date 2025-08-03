# 🎯 PlagiCheck - Unified Paraphrasing System v2.0

**Advanced Indonesian Academic Text Paraphrasing with Unified Methodology**

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## 🚀 What's New in v2.0 - Unified System

### ✨ Major Improvements

- **🔄 Single Unified Endpoint** - All paraphrasing operations through one powerful `/api/paraphrase` endpoint
- **🧠 Intelligent Method Fusion** - Combines IndoT5, rule-based, and custom synonyms automatically
- **📚 Custom Synonyms Integration** - Upload your own `synonyms.json` for domain-specific paraphrasing
- **📊 Advanced Quality Assessment** - 6-dimensional quality scoring with confidence levels
- **🏥 Comprehensive Health Monitoring** - Real-time system status and performance metrics
- **⚡ Performance Optimizations** - Lazy loading, caching, and parallel processing
- **🔧 Simplified Configuration** - Streamlined settings for easier deployment

### 🎯 Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Unified API** | Single endpoint for all paraphrasing needs | ✅ |
| **IndoT5 Neural Model** | GPU-optimized Indonesian transformer | ✅ |
| **Rule-Based Engine** | Linguistic transformation rules | ✅ |
| **Custom Synonyms** | User-provided synonym dictionaries | ✅ |
| **Quality Assessment** | Multi-dimensional quality scoring | ✅ |
| **Academic Focus** | Preserves academic terminology | ✅ |
| **Citation Protection** | Maintains citation formats | ✅ |
| **Health Monitoring** | System status and metrics | ✅ |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│             🌐 Unified API Layer            │
│          /api/paraphrase (Single Endpoint)  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│        🧠 Unified Paraphraser Engine        │
│     • Method Selection & Orchestration      │
│     • Quality Assessment & Optimization     │
│     • Content Protection & Restoration      │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐ ┌─────▼─────┐ ┌─────▼─────┐
│IndoT5  │ │Rule-Based │ │Custom     │
│Neural  │ │Transform  │ │Synonyms   │
│Engine  │ │Engine     │ │Engine     │
└────────┘ └───────────┘ └───────────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│       📊 Quality Assessment Framework       │
│  • Semantic Similarity  • Grammar Quality  │
│  • Academic Tone        • Readability      │
│  • Structural Diversity • Context Fit      │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/plagicheck.git
cd plagicheck

# Install dependencies
pip install -r requirements.txt

# Run migration (if upgrading from v1.x)
python migration_unified.py

# Start the server
python main.py
```

### 2. Basic Usage

#### Simple Text Paraphrasing
```bash
curl -X POST "http://localhost:8000/api/paraphrase" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji faktor-faktor yang mempengaruhi kualitas pembelajaran.",
    "num_variants": 3,
    "preserve_academic_terms": true,
    "preserve_citations": true
  }'
```

#### With Custom Synonyms
```bash
curl -X POST "http://localhost:8000/api/paraphrase" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Penelitian ini menunjukkan hasil yang signifikan.",
    "use_custom_synonyms": true,
    "custom_synonyms_path": "synonyms.json",
    "include_method_insights": true
  }'
```

#### Document-Based Paraphrasing
```bash
curl -X POST "http://localhost:8000/api/paraphrase" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "quality_threshold": 0.8,
    "include_detailed_analysis": true,
    "include_quality_breakdown": true
  }'
```

## 📚 Custom Synonyms Integration

### Creating synonyms.json

```json
{
  "penelitian": {
    "tag": "NOUN",
    "sinonim": ["riset", "studi", "kajian", "eksplorasi"]
  },
  "analisis": {
    "tag": "NOUN", 
    "sinonim": ["kajian", "telaah", "pemeriksaan", "evaluasi"]
  },
  "menunjukkan": {
    "tag": "VERB",
    "sinonim": ["mengindikasikan", "memperlihatkan", "mendemonstrasikan"]
  }
}
```

### Benefits of Custom Synonyms

- **Domain-Specific** - Tailored vocabulary for your field
- **High Priority** - Custom synonyms get preference (confidence: 0.95)
- **Context-Aware** - POS tags ensure grammatical consistency
- **Academic Focus** - Maintains scholarly terminology

## 🔧 API Reference

### Main Endpoint: `/api/paraphrase`

#### Request Model
```typescript
interface UnifiedParaphraseRequest {
  // Input (choose one)
  text?: string;                    // Direct text input
  document_id?: string;             // Previously uploaded document
  
  // Processing Configuration
  num_variants?: number;            // Default: 3, Range: 1-5
  quality_threshold?: number;       // Default: 0.7, Range: 0.0-1.0
  preserve_academic_terms?: boolean; // Default: true
  preserve_citations?: boolean;     // Default: true
  
  // Output Customization
  include_detailed_analysis?: boolean;  // Default: false
  include_quality_breakdown?: boolean;  // Default: false
  include_method_insights?: boolean;    // Default: false
  
  // Advanced Options
  formality_level?: "academic" | "formal" | "neutral";
  target_complexity?: "simplify" | "maintain" | "enhance";
  
  // Method Selection
  use_indot5?: boolean;             // Default: true
  use_rule_based?: boolean;         // Default: true
  use_custom_synonyms?: boolean;    // Default: true
  custom_synonyms_path?: string;    // Optional
}
```

#### Response Model
```typescript
interface UnifiedParaphraseResponse {
  original_text: string;
  best_variant: string;
  processing_time: number;
  
  quality_assessment: {
    overall_score: number;
    dimension_scores: {
      semantic_similarity: number;
      grammar_correctness: number;
      academic_tone_preservation: number;
      readability_score: number;
      structural_diversity: number;
      context_appropriateness: number;
    };
    confidence_level: number;
    recommendations: string[];
    meets_threshold: boolean;
  };
  
  // Optional detailed information
  all_variants?: VariantInfo[];
  nlp_analysis?: NLPAnalysisSummary;
  method_insights?: MethodContributions;
  
  metadata: Record<string, any>;
  system_info: {
    version: string;
    unified_system: boolean;
    methods_available: string[];
  };
}
```

### Health Check: `/api/health`

```bash
curl -X GET "http://localhost:8000/api/health"
```

Returns comprehensive system status including:
- Component health status
- Performance metrics
- Method availability  
- System recommendations

## 🏥 Monitoring & Performance

### Performance Metrics

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| **Response Time** | 3-8 seconds | Depends on text length and methods used |
| **Throughput** | 50+ concurrent requests | With GPU acceleration |
| **Memory Usage** | 4-8GB | Including loaded models |
| **GPU Memory** | 2-6GB | For IndoT5 model |

### Health Monitoring

The unified system provides comprehensive health monitoring:

```bash
# Check system health
curl /api/health

# Monitor performance
curl /api/health | jq '.performance'

# Check component status
curl /api/health | jq '.components'
```

## 🔄 Migration from v1.x

### Automatic Migration

```bash
# Run the migration script
python migration_unified.py
```

The migration script:
- ✅ Creates backup of legacy files
- ✅ Analyzes existing endpoints
- ✅ Generates migration documentation
- ✅ Validates unified system setup

### Legacy Compatibility

Legacy endpoints are maintained for backward compatibility:

| Legacy Endpoint | Status | Redirect |
|----------------|--------|----------|
| `/api/documents/{id}/paraphrase` | ✅ Supported | → `/api/paraphrase` |
| `/api/v2/text/paraphrase-direct` | ⚠️ Deprecated | → `/api/paraphrase` |
| `/api/v2/documents/{id}/paraphrase-enhanced` | ⚠️ Deprecated | → `/api/paraphrase` |

## ⚙️ Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/plagicheck
DATABASE_HOST=localhost
DATABASE_PORT=5432

# AI Models
HUGGINGFACE_API_KEY=your_hf_key_here
GEMINI_API_KEY=your_gemini_key_here

# Processing
MAX_CONCURRENT_JOBS=5
SIMILARITY_THRESHOLD=0.7

# File Upload
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=[".pdf", ".docx", ".txt"]
```

### Unified System Config

```python
UNIFIED_PARAPHRASE_CONFIG = {
    'custom_synonyms_path': 'synonyms.json',
    'quality_threshold_default': 0.7,
    'max_variants_per_request': 5,
    'enable_gpu_acceleration': True,
    'enable_performance_caching': True,
    'academic_focus_mode': True,
    'preserve_formatting': True
}
```

## 🧪 Testing

### Run System Tests

```bash
# Test unified system
python test_unified_system.py

# Test specific components
python -m pytest tests/test_unified_paraphraser.py
python -m pytest tests/test_api_models.py
```

### Example Test Output

```
🚀 Testing PlagiCheck Unified System
==================================================
INFO - Testing Unified Paraphraser...
INFO - Executing unified paraphrasing...
INFO - === UNIFIED PARAPHRASING RESULTS ===
INFO - Best Variant: Riset ini menerapkan teknik kajian kualitatif...
INFO - Processing Time: 3.45s
INFO - Quality Score: 0.856
INFO - Method Contributions: {'indot5': 3, 'rule_based': 2, 'custom_synonyms': 1}
INFO - ✅ Unified Paraphraser: PASS
INFO - ✅ API Models: PASS  
INFO - ✅ Synonyms Loading: PASS
==================================================
🎉 All tests completed successfully!
✅ Unified system is ready for deployment
```

## 📈 Performance Optimization

### GPU Acceleration

```python
# Enable GPU for IndoT5 model
CUDA_VISIBLE_DEVICES=0 python main.py
```

### Memory Management

```python
# Configure model loading
UNIFIED_PARAPHRASE_CONFIG = {
    'enable_gpu_acceleration': True,
    'enable_performance_caching': True,
    'max_cache_size': 1000
}
```

### Scaling

For production deployment:

```bash
# Use Gunicorn with multiple workers
gunicorn app.api.routes:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h
```

#### 2. Memory Issues
```bash
# Monitor memory usage
watch -n 1 "ps aux | grep python | head -5"

# Reduce concurrent requests
MAX_CONCURRENT_JOBS=2
```

#### 3. Slow Performance
```bash
# Check system health
curl localhost:8000/api/health

# Monitor processing times
tail -f logs/application.log | grep "processing_time"
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=true python main.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/plagicheck.git
cd plagicheck

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/plagicheck/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/plagicheck/discussions)

---

## 🎯 Roadmap

### v2.1 (Planned)
- [ ] Multi-language support expansion
- [ ] Advanced caching strategies
- [ ] Real-time collaboration features
- [ ] Enhanced quality metrics

### v2.2 (Future)
- [ ] Machine learning quality prediction
- [ ] Automated A/B testing for methods
- [ ] Advanced document structure preservation
- [ ] Integration with popular academic tools

---

**Made with ❤️ for Indonesian Academic Community**

*PlagiCheck v2.0 - Unified Paraphrasing System*
