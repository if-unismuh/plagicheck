# 📖 API Documentation - PlagiCheck

Dokumentasi lengkap untuk semua API endpoints yang tersedia di PlagiCheck - Auto Paraphrasing System.

## 🌐 Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## 📊 API Overview

PlagiCheck menyediakan dua set API endpoints:

- **Basic API** (`/api`): Endpoints standar untuk operasi dasar
- **Enhanced API** (`/api/v2`): Endpoints canggih dengan fitur NLP lanjutan

## 🔧 Authentication

Saat ini API bersifat open (tidak memerlukan authentication). Untuk production, implementasi authentication dapat diaktifkan melalui konfigurasi.

## 📋 Response Format

Semua API menggunakan format JSON untuk request dan response:

```json
{
  "status": "success|error",
  "message": "Description",
  "data": {}, 
  "metadata": {}
}
```

## 🏥 Health & Status Endpoints

### GET / - Root Endpoint
Informasi dasar aplikasi.

**Response:**
```json
{
  "name": "PlagiCheck - Auto Paraphrasing System",
  "version": "2.0.0",
  "status": "running",
  "docs_url": "/docs"
}
```

### GET /health - Health Check
Status kesehatan aplikasi.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### GET /api/v2/performance/stats - Performance Statistics
Statistik performa sistem.

**Response:**
```json
{
  "enhanced_indot5": {
    "total_requests": 150,
    "average_processing_time": 8.5,
    "success_rate": 0.95
  },
  "rule_based": {
    "transformations_applied": 1200,
    "success_rate": 0.98
  },
  "system_status": {
    "services_loaded": true,
    "gpu_available": true,
    "models_ready": {
      "indot5": true,
      "nlp_pipeline": true,
      "rule_based": true
    }
  }
}
```

## 📄 Document Management

### POST /api/documents/upload - Basic Document Upload
Upload dokumen dengan pemrosesan standar.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "chapter=BAB 1"
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "chapter": "BAB 1",
  "status": "completed",
  "upload_date": "2025-01-15T10:30:00Z",
  "processed_date": "2025-01-15T10:31:00Z",
  "metadata": {
    "file_size": 1024000,
    "pages": 10,
    "word_count": 2500
  }
}
```

### POST /api/v2/documents/upload-enhanced - Enhanced Document Upload
Upload dengan fitur pemrosesan canggih.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v2/documents/upload-enhanced" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "chapter=BAB 1" \
  -F "preserve_structure=true" \
  -F "extract_academic_terms=true"
```

**Response:**
```json
{
  "message": "Document uploaded and processed successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "status": "completed",
  "metadata": {
    "file_size": 1024000,
    "academic_terms_extracted": 45,
    "structure_preserved": true,
    "processing_time": 12.5
  }
}
```

### GET /api/documents/{document_id} - Get Document Details
Retrieve informasi detail dokumen.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "chapter": "BAB 1",
  "status": "completed",
  "upload_date": "2025-01-15T10:30:00Z",
  "processed_date": "2025-01-15T10:31:00Z",
  "original_content": "Penelitian ini menggunakan...",
  "paraphrased_content": "Studi ini menerapkan...",
  "metadata": {
    "word_count": 2500,
    "academic_terms": ["metodologi", "analisis", "penelitian"]
  }
}
```

### GET /api/documents - List Documents
List semua dokumen dengan filtering.

**Query Parameters:**
- `status`: Filter berdasarkan status (pending, processing, completed, failed)
- `chapter`: Filter berdasarkan chapter
- `limit`: Jumlah maksimum hasil (default: 100)
- `offset`: Offset untuk pagination (default: 0)

**Request:**
```bash
curl "http://localhost:8000/api/documents?status=completed&limit=10"
```

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "document1.pdf",
    "chapter": "BAB 1",
    "status": "completed",
    "upload_date": "2025-01-15T10:30:00Z"
  }
]
```

### DELETE /api/documents/{document_id} - Delete Document
Hapus dokumen dari sistem.

**Response:**
```json
{
  "message": "Document deleted successfully"
}
```

### GET /api/documents/{document_id}/status - Document Status
Check status pemrosesan dokumen.

**Response:**
```json
{
  "status": "processing",
  "message": "Document is being processed",
  "progress": 0.75
}
```

## 🔍 Text Analysis

### POST /api/v2/text/analyze - NLP Text Analysis
Analisis NLP komprehensif untuk teks.

**Request:**
```json
{
  "text": "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji dampak teknologi digital terhadap pembelajaran mahasiswa.",
  "extract_academic_terms": true
}
```

**Response:**
```json
{
  "total_sentences": 1,
  "overall_readability": 0.72,
  "overall_complexity": 0.68,
  "academic_terms_count": 6,
  "named_entities_count": 2,
  "quality_metrics": {
    "readability_score": 0.72,
    "academic_tone": 0.85,
    "complexity_index": 0.68
  },
  "paraphrasing_priorities": [0, 1, 2],
  "high_priority_sentences": [
    {
      "index": 0,
      "text": "Penelitian ini menggunakan metode analisis kualitatif...",
      "priority": 0.8,
      "complexity": 0.75,
      "readability": 0.65
    }
  ]
}
```

### POST /api/v2/text/quality-assessment - Text Quality Assessment
Evaluasi kualitas teks pada multiple dimensi.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v2/text/quality-assessment" \
  -H "Content-Type: application/json" \
  -d '{"text": "Penelitian ini menggunakan metode kualitatif."}'
```

**Response:**
```json
{
  "readability_score": 0.75,
  "complexity_score": 0.65,
  "academic_tone_score": 0.85,
  "grammar_score": 0.92,
  "overall_quality": 0.79,
  "issues": [],
  "recommendations": [
    "Consider adding more detailed methodology description",
    "Academic terminology is appropriate for the context"
  ]
}
```

## 🔄 Paraphrasing Operations

### POST /api/documents/{document_id}/paraphrase - Basic Paraphrasing
Mulai parafrase dokumen dengan method standar.

**Request:**
```json
{
  "method": "indot5"
}
```

**Response:**
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "method_used": "indot5",
  "similarity_score": 0.75,
  "processing_time": 8500,
  "created_at": "2025-01-15T10:35:00Z"
}
```

### POST /api/v2/documents/{document_id}/paraphrase-enhanced - Enhanced Paraphrasing
Parafrase canggih dengan opsi komprehensif.

**Request:**
```json
{
  "method": "hybrid",
  "use_nlp_analysis": true,
  "preserve_academic_terms": true,
  "preserve_citations": true,
  "num_variants": 3,
  "quality_threshold": 0.7
}
```

**Response:**
```json
{
  "message": "Enhanced paraphrasing completed",
  "session_id": "660e8400-e29b-41d4-a716-446655440001",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "method_used": "hybrid",
  "similarity_score": 0.78,
  "processing_time": 15.5,
  "enhanced_metadata": {
    "variants_generated": 3,
    "quality_filtered": true,
    "academic_terms_preserved": 12
  }
}
```

### POST /api/v2/text/paraphrase-direct - Direct Text Paraphrasing
Parafrase langsung tanpa upload dokumen.

**Request:**
```json
{
  "text": "Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji dampak teknologi digital terhadap pembelajaran mahasiswa.",
  "method": "indot5",
  "preserve_academic_terms": true,
  "preserve_citations": true,
  "num_variants": 3
}
```

**Response:**
```json
{
  "original_text": "Penelitian ini menggunakan metode analisis kualitatif...",
  "variants": [
    {
      "text": "Studi ini menerapkan pendekatan analisis kualitatif...",
      "similarity_score": 0.72,
      "quality_score": 0.85,
      "method_used": "indot5"
    },
    {
      "text": "Riset ini mengadopsi metode analisis kualitatif...",
      "similarity_score": 0.75,
      "quality_score": 0.82,
      "method_used": "indot5"
    }
  ],
  "best_variant": "Studi ini menerapkan pendekatan analisis kualitatif...",
  "processing_time": 5.2,
  "metadata": {
    "model_used": "enhanced_indot5",
    "inference_time": 3.8,
    "post_processing_time": 1.4
  }
}
```

## 📊 Session Management

### GET /api/sessions/{session_id} - Get Paraphrase Session
Detail session parafrase.

**Response:**
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "method_used": "hybrid",
  "similarity_score": 0.78,
  "processing_time": 15500,
  "token_usage": {
    "input_tokens": 1024,
    "output_tokens": 896,
    "total_tokens": 1920
  },
  "created_at": "2025-01-15T10:35:00Z"
}
```

### GET /api/documents/{document_id}/sessions - Document Sessions
Semua session untuk dokumen tertentu.

**Response:**
```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "method_used": "indot5",
    "similarity_score": 0.75,
    "created_at": "2025-01-15T10:35:00Z"
  },
  {
    "id": "660e8400-e29b-41d4-a716-446655440002",
    "method_used": "rule_based",
    "similarity_score": 0.82,
    "created_at": "2025-01-15T11:00:00Z"
  }
]
```

## 🎯 Demo & Examples

### GET /api/v2/demo/sample-analysis - Sample Analysis Demo
Demo analisis kemampuan sistem.

**Response:**
```json
{
  "sample_text": "Penelitian ini menggunakan metode analisis kualitatif...",
  "nlp_analysis": {
    "total_sentences": 4,
    "readability_score": 0.72,
    "complexity_score": 0.68,
    "academic_terms": ["penelitian", "metode", "analisis", "kualitatif"],
    "named_entities": ["mahasiswa", "teknologi digital"]
  },
  "paraphrasing_results": {
    "indot5": {
      "best_variant": "Studi ini menerapkan pendekatan analisis kualitatif...",
      "similarity_scores": [0.72, 0.75],
      "quality_scores": [0.85, 0.82]
    },
    "rule_based": [
      {
        "text": "Riset ini menggunakan metode analisis kualitatif...",
        "quality_score": 0.80,
        "similarity_score": 0.78
      }
    ]
  }
}
```

### POST /api/v2/performance/clear-cache - Clear Cache
Bersihkan cache performa untuk mengoptimalkan memory.

**Response:**
```json
{
  "message": "Performance caches cleared successfully",
  "timestamp": "2025-01-15T10:45:00Z"
}
```

## ❌ Error Handling

### Error Response Format
```json
{
  "detail": "Error description",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

### Common Error Codes

| Code | Description | Possible Causes |
|------|-------------|----------------|
| 400 | Bad Request | Invalid input, malformed JSON |
| 404 | Not Found | Document/session not exists |
| 422 | Validation Error | Invalid parameters |
| 500 | Internal Error | Server/model errors |
| 503 | Service Unavailable | Model loading, maintenance |

### Error Examples

**400 - Bad Request:**
```json
{
  "detail": "File format not supported",
  "status_code": 400
}
```

**404 - Not Found:**
```json
{
  "detail": "Document not found",
  "status_code": 404
}
```

**422 - Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "num_variants"],
      "msg": "ensure this value is less than or equal to 10",
      "type": "value_error.number.not_le"
    }
  ],
  "status_code": 422
}
```

## 📚 SDK & Libraries

### Python SDK Example
```python
import requests

class PlagiCheckClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_document(self, file_path, chapter=None):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'chapter': chapter} if chapter else {}
            response = requests.post(
                f"{self.base_url}/api/documents/upload",
                files=files, data=data
            )
        return response.json()
    
    def paraphrase_text(self, text, method="indot5", num_variants=3):
        data = {
            "text": text,
            "method": method,
            "num_variants": num_variants
        }
        response = requests.post(
            f"{self.base_url}/api/v2/text/paraphrase-direct",
            json=data
        )
        return response.json()

# Usage
client = PlagiCheckClient()
result = client.paraphrase_text("Penelitian ini menggunakan metode kualitatif.")
print(result['best_variant'])
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');

class PlagiCheckClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async paraphraseText(text, method = 'indot5', numVariants = 3) {
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v2/text/paraphrase-direct`,
                {
                    text: text,
                    method: method,
                    num_variants: numVariants
                }
            );
            return response.data;
        } catch (error) {
            throw new Error(`API Error: ${error.response.data.detail}`);
        }
    }
}

// Usage
const client = new PlagiCheckClient();
client.paraphraseText('Penelitian ini menggunakan metode kualitatif.')
    .then(result => console.log(result.best_variant))
    .catch(error => console.error(error));
```

## 🔍 Rate Limiting

API memiliki rate limiting untuk mencegah abuse:

- **Default**: 60 requests per minute per IP
- **Burst**: 10 requests per 10 seconds
- **Headers**: Response includes rate limit headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642253400
```

## 📖 Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Interactive documentation memungkinkan untuk test API langsung dari browser dengan interface yang user-friendly.

---

🔗 **Links:**
- [Postman Collection](../postman/README.md)
- [Testing Guide](../TESTING.md)
- [Configuration](CONFIGURATION.md)
