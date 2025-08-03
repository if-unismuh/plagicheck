# 🎯 PlagiCheck - Sistem Parafrase Akademik Otomatis

**PlagiCheck** adalah sistem parafrase dokumen akademik yang komprehensif dengan kemampuan NLP canggih untuk teks Bahasa Indonesia dan Inggris. Sistem ini dirancang khusus untuk membantu mahasiswa, peneliti, dan akademisi dalam menghasilkan parafrase berkualitas tinggi dengan tetap mempertahankan makna dan tone akademik.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 🌟 Fitur Utama

### 📄 Pemrosesan Dokumen Multi-Format
- **Format Support**: PDF, DOCX, TXT dengan preservasi struktur dokumen
- **Text Preprocessing**: Pembersihan, normalisasi, dan segmentasi kalimat otomatis
- **Academic Term Detection**: Identifikasi dan preservasi terminologi akademik
- **Citation Preservation**: Mempertahankan format referensi selama pemrosesan
- **Document Reconstruction**: Konversi teks yang telah diproses kembali ke format DOCX

### 🔍 Pipeline NLP Bahasa Indonesia
- **Indonesian Language Support**: Integrasi spaCy dengan UDPipe untuk analisis mendalam
- **Comprehensive Analysis**: POS tagging, Named Entity Recognition, dependency parsing
- **Quality Metrics**: Skor readability, analisis kompleksitas, dan evaluasi tone akademik
- **Academic Writing Detection**: Identifikasi pola bahasa akademik formal
- **Sentence Prioritization**: Penentuan kalimat yang memerlukan parafrase prioritas tinggi

### 🚀 Engine Parafrase Multi-Method
- **Enhanced IndoT5 Paraphraser**: Optimisasi GPU, batch processing, multiple variants
- **Hybrid Rule-Based Paraphraser**: Substitusi leksikal dan transformasi sintaktik
- **Quality Filtering**: Scoring similarity semantik dan evaluasi grammar
- **Academic Tone Preservation**: Mempertahankan bahasa akademik formal
- **Real-time Processing**: Status tracking dan monitoring progress

### 🎯 API Endpoints Komprehensif
- **RESTful API**: 22+ endpoints dengan dokumentasi Swagger otomatis
- **Basic & Enhanced Routes**: Pemrosesan standar dan canggih
- **Direct Text Paraphrasing**: Parafrase langsung tanpa upload dokumen
- **Performance Monitoring**: Statistik performa dan manajemen cache
- **Demo Endpoints**: Contoh analisis dan testing capabilities

## 🔧 Arsitektur Sistem

```
PlagiCheck/
├── 🌐 FastAPI Application
│   ├── 📡 Basic API Routes (/api/v1)
│   └── 🚀 Enhanced API Routes (/api/v2)
├── 🧠 AI/ML Services
│   ├── 🇮🇩 Indonesian NLP Pipeline
│   ├── 🔄 Enhanced IndoT5 Paraphraser
│   ├── 📝 Rule-Based Paraphraser
│   └── 📊 Quality Assessment Engine
├── 💾 Database Layer
│   ├── 📋 Document Management
│   ├── 🔄 Paraphrase Sessions
│   └── 📈 Performance Metrics
└── 🛠️ Processing Pipeline
    ├── 📄 Document Processor
    ├── 🔍 Text Analyzer
    └── 📊 Quality Evaluator
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.9+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Database**: PostgreSQL 12+ atau SQLite untuk development

### Recommended Specifications
- **GPU**: NVIDIA CUDA-compatible untuk inferensi model lokal
- **RAM**: 16GB+ untuk pemrosesan batch
- **CPU**: Multi-core processor untuk parallel processing
- **Network**: Stabil untuk download model dan updates

## ⚡ Quick Start Guide

### 🚀 Setup Otomatis (Recommended)
```bash
# Clone repository
git clone https://github.com/if-unismuh/plagicheck.git
cd plagicheck

# Setup otomatis (install dependencies, setup database, download models)
python setup.py

# Jalankan aplikasi
python main.py
```

### � Akses Aplikasi
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🛠️ Manual Installation

### 1️⃣ Environment Setup
```bash
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -c "import spacy; spacy.download('en_core_web_sm')"
```

### 3️⃣ Database Configuration
```bash
# PostgreSQL (Production)
createdb plagicheck_db
python migrate_db.py create
python migrate_db.py migrate

# SQLite (Development)
python migrate_db.py migrate
```

### 4️⃣ Environment Variables
```bash
# Buat file .env
cat > .env << EOF
# Application Settings
APP_NAME="PlagiCheck - Auto Paraphrasing System"
APP_VERSION="2.0.0"
DEBUG=true

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/plagicheck_db
# Atau untuk SQLite:
# DATABASE_URL=sqlite:///./plagicheck.db

# File Upload Settings
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# AI Model Configuration
USE_GPU=true
MAX_CONCURRENT_JOBS=5
SIMILARITY_THRESHOLD=0.7

# API Keys (Optional)
HUGGINGFACE_API_KEY=your_huggingface_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
EOF
```

## 🔧 Konfigurasi Lanjutan

### Database Setup Options

#### PostgreSQL (Production)
```env
DATABASE_URL=postgresql://username:password@localhost/plagicheck_db
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=plagicheck_db
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password
```

#### SQLite (Development)
```env
DATABASE_URL=sqlite:///./plagicheck.db
```

### Model Configuration
```env
# GPU Configuration
USE_GPU=true
CUDA_VISIBLE_DEVICES=0

# Model Paths (akan di-download otomatis)
INDOT5_MODEL_PATH=Wikidepia/IndoT5-base-paraphrase
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

## 📖 API Documentation

### 🔧 Basic API Endpoints (/api)

#### Document Management
```http
POST   /api/documents/upload              # Upload dokumen basic
GET    /api/documents/{document_id}       # Detail dokumen
GET    /api/documents                     # List semua dokumen  
DELETE /api/documents/{document_id}       # Hapus dokumen
GET    /api/documents/{document_id}/status # Status pemrosesan
```

#### Paraphrasing Operations
```http
POST   /api/documents/{document_id}/paraphrase  # Mulai parafrase
GET    /api/sessions/{session_id}               # Detail session
GET    /api/documents/{document_id}/sessions    # Session history
```

### 🚀 Enhanced API Endpoints (/api/v2)

#### Advanced Document Processing
```http
POST   /api/v2/documents/upload-enhanced    # Upload dengan fitur enhanced
POST   /api/v2/text/analyze                 # Analisis NLP komprehensif
POST   /api/v2/text/quality-assessment      # Evaluasi kualitas teks
```

#### Advanced Paraphrasing
```http
POST   /api/v2/documents/{document_id}/paraphrase-enhanced  # Parafrase enhanced
POST   /api/v2/text/paraphrase-direct                      # Parafrase langsung
```

#### Performance & Monitoring
```http
GET    /api/v2/performance/stats      # Statistik performa
POST   /api/v2/performance/clear-cache # Clear cache
GET    /api/v2/demo/sample-analysis   # Demo analisis
```

### 📊 Request/Response Examples

#### Upload Document Enhanced
```bash
curl -X POST "http://localhost:8000/api/v2/documents/upload-enhanced" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "chapter=BAB 1" \
  -F "preserve_structure=true" \
  -F "extract_academic_terms=true"
```

#### Direct Text Paraphrasing
```bash
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Penelitian ini menggunakan metode kualitatif untuk menganalisis data.",
    "method": "indot5",
    "preserve_academic_terms": true,
    "num_variants": 3
  }'
```

## 🧪 Testing & Development

### Running Tests
```bash
# Jalankan semua test
pytest

# Test dengan coverage
pytest --cov=app --cov-report=html

# Test spesifik kategori
pytest -m api          # API tests
pytest -m integration  # Integration tests
pytest -m performance  # Performance tests
```

### Development Mode
```bash
# Jalankan dengan auto-reload
uvicorn app.api.routes:app --reload --host 0.0.0.0 --port 8000

# Debug mode
DEBUG=true python main.py
```

### 📮 Postman Testing
```bash
# Import Postman collection
postman/paraphrase_system_collection.json

# Environment files
postman/environments/development.json    # Development
postman/environments/production.json     # Production
```

## 📊 Database Management

### Migration Commands
```bash
# Status migrasi
python migrate_db.py status

# Jalankan migrasi
python migrate_db.py migrate

# Reset database (HATI-HATI!)
python migrate_db.py reset

# Seed data sample
python migrate_db.py seed

# Validasi koneksi
python migrate_db.py validate
```

### Database Schema
```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    chapter VARCHAR(50),
    original_content TEXT NOT NULL,
    paraphrased_content TEXT,
    status VARCHAR(20) NOT NULL,
    document_metadata JSONB,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP
);

-- Paraphrase Sessions table  
CREATE TABLE paraphrase_sessions (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    method_used VARCHAR(50) NOT NULL,
    similarity_score FLOAT,
    processing_time INTEGER,
    token_usage JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔄 Workflow Penggunaan

### 1. Upload & Analisis Dokumen
```mermaid
graph TD
    A[Upload Dokumen] --> B[Text Extraction]
    B --> C[NLP Analysis]
    C --> D[Academic Term Detection]
    D --> E[Quality Assessment]
    E --> F[Document Ready]
```

### 2. Paraphrasing Process
```mermaid
graph TD
    A[Pilih Method] --> B{Method Type}
    B -->|IndoT5| C[Enhanced IndoT5 Model]
    B -->|Rule-Based| D[Lexical/Syntactic Rules]
    B -->|Hybrid| E[Combined Approach]
    C --> F[Quality Filtering]
    D --> F
    E --> F
    F --> G[Best Variant Selection]
    G --> H[Result with Metadata]
```

## 🎯 Use Cases

### 📚 Academic Writing
- **Thesis/Dissertation**: Parafrase bab atau section untuk menghindari self-plagiarism
- **Research Papers**: Reformulasi metodologi dan hasil penelitian
- **Literature Review**: Parafrase sumber rujukan dengan preservasi meaning

### 📝 Content Creation  
- **Article Rewriting**: Variasi konten untuk multiple platform
- **SEO Content**: Multiple variant untuk optimasi search engine
- **Educational Materials**: Adaptasi level kesulitan bahasa

### 🔍 Quality Assurance
- **Plagiarism Prevention**: Pre-check sebelum submission
- **Readability Improvement**: Optimasi tingkat keterbacaan
- **Academic Tone**: Konsistensi formal academic writing

## 🚀 Performance Optimization

### Model Optimization
```python
# GPU Configuration
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=8                    # Sesuaikan dengan VRAM
MAX_SEQUENCE_LENGTH=512
NUM_BEAMS=5
```

### Caching Strategy
```python
# Redis untuk production (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# Memory cache untuk development
USE_MEMORY_CACHE=true
MAX_CACHE_SIZE=1000
```

### Monitoring & Logging
```python
# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/plagicheck.log
ENABLE_PERFORMANCE_LOGGING=true

# Metrics collection
ENABLE_METRICS=true
METRICS_ENDPOINT=/metrics
```

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Problems
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Re-download models
python -c "from app.services.enhanced_indot5_paraphraser import get_enhanced_indot5_paraphraser; get_enhanced_indot5_paraphraser()"
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
python -c "from app.core.database import engine; print(engine.execute('SELECT 1').scalar())"

# Reset database
python migrate_db.py reset
```

#### Memory Issues
```bash
# Monitor memory usage
htop

# Clear Python cache
python -c "import gc; gc.collect()"

# Restart services
sudo systemctl restart plagicheck
```

#### GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

## 📈 Performance Metrics

### Response Times (Typical)
- **Document Upload**: 2-5 seconds
- **NLP Analysis**: 3-8 seconds
- **IndoT5 Paraphrasing**: 5-15 seconds per variant
- **Rule-based Paraphrasing**: 1-3 seconds per variant
- **Quality Assessment**: 2-4 seconds

### Throughput
- **Concurrent Users**: Up to 50 (with GPU)
- **Documents per Hour**: 200-500 (depending on size)
- **API Requests**: 1000+ per minute

### Resource Usage
- **RAM**: 4-8GB (with models loaded)
- **GPU VRAM**: 2-6GB (IndoT5 model)
- **CPU**: 2-4 cores recommended
- **Storage**: 10GB+ for models and cache

## 🔐 Security & Privacy

### Data Protection
- **Document Encryption**: AES-256 encryption untuk stored documents
- **Secure Upload**: Validation dan sanitization file upload
- **Access Control**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive logging untuk compliance

### Privacy Features
```python
# Data retention policy
DOCUMENT_RETENTION_DAYS=30
AUTO_DELETE_PROCESSED=true
ANONYMIZE_USER_DATA=true

# Security headers
CORS_ORIGINS=["https://yourdomain.com"]
CSRF_PROTECTION=true
RATE_LIMITING=true
```

## 🌐 Deployment

### Docker Deployment
```bash
# Build image
docker build -t plagicheck:latest .

# Run container
docker run -d \
  --name plagicheck \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/plagicheck \
  -v uploads:/app/uploads \
  plagicheck:latest
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/plagicheck
    depends_on:
      - db
    volumes:
      - uploads:/app/uploads
      
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=plagicheck
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  uploads:
  postgres_data:
```

### Production Deployment
```bash
# Setup production environment
export ENVIRONMENT=production
export DEBUG=false
export DATABASE_URL=postgresql://user:pass@prod-db:5432/plagicheck

# Use production server
gunicorn app.api.routes:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --max-requests 1000 \
  --max-requests-jitter 100
```

## 📚 Documentation Links

- **📖 API Documentation**: `/docs` (Swagger UI)
- **📋 Alternative Docs**: `/redoc` (ReDoc)
- **🧪 Testing Guide**: [TESTING.md](TESTING.md)
- **📮 Postman Collection**: [postman/README.md](postman/README.md)
- **🔧 Configuration Guide**: [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

## 🤝 Contributing

### Development Setup
```bash
# Fork repository
git clone https://github.com/your-username/plagicheck.git
cd plagicheck

# Create feature branch
git checkout -b feature/new-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Style
```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/
mypy app/

# Type checking
mypy app/ --strict
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Transformers Library**: Hugging Face untuk IndoT5 model
- **spaCy**: Advanced NLP processing capabilities
- **FastAPI**: Modern Python web framework
- **PostgreSQL**: Robust database system
- **UDPipe**: Universal dependencies parsing

## 👥 Team

- **Lead Developer**: [Your Name](mailto:your.email@example.com)
- **NLP Engineer**: [NLP Expert](mailto:nlp@example.com)
- **System Architect**: [Architect](mailto:architect@example.com)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/if-unismuh/plagicheck/issues)
- **Discussions**: [GitHub Discussions](https://github.com/if-unismuh/plagicheck/discussions)
- **Email**: support@plagicheck.com

---

<div align="center">
<b>PlagiCheck - Empowering Academic Writing with AI</b><br>
🎓 Developed with ❤️ for the Academic Community
</div>

# File Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# API Keys (optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration
USE_GPU=true
MODEL_CACHE_DIR=models
```

## 🚀 Usage

### Start the Application
```bash
python main.py
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Basic API Endpoints

#### Document Processing
```http
POST /api/documents/upload
POST /api/v2/documents/upload-enhanced
GET /api/documents/{document_id}
POST /api/documents/{document_id}/paraphrase
```

#### Enhanced Features
```http
POST /api/v2/text/paraphrase-direct
POST /api/v2/text/analyze
POST /api/v2/text/quality-assessment
GET /api/v2/performance/stats
```

### Example Usage

**Upload and paraphrase document**:
```bash
curl -X POST "http://localhost:8000/api/v2/documents/upload-enhanced" \
  -F "file=@document.pdf" \
  -F "preserve_structure=true" \
  -F "extract_academic_terms=true"
```

**Direct text paraphrasing**:
```bash
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Penelitian ini menggunakan metode analisis kualitatif.",
    "method": "hybrid",
    "num_variants": 3
  }'
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test.py
```

Or run specific tests:
```bash
python -m pytest tests/
```

## 📊 Quality Metrics

The system provides comprehensive quality assessment:

- **Similarity Score**: Semantic similarity to original (0-1)
- **Grammar Score**: Grammar and fluency assessment (0-1)
- **Readability Score**: Text readability level (0-1)
- **Academic Tone**: Formal academic language score (0-1)
- **Overall Quality**: Combined quality metric (0-1)

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU only
```

**Model Download Failures**:
```bash
rm -rf ~/.cache/huggingface/  # Clear cache
```

**Database Issues**:
```bash
python migrate_db.py reset --force  # Reset database
```

## 📁 Project Structure

```
paraphrase-system/
├── app/                    # Main application package
│   ├── core/              # Core configuration and database
│   ├── models/            # SQLAlchemy models
│   ├── services/          # Business logic services
│   └── api/               # API routes
├── uploads/               # File upload directory
├── migrations/            # Database migrations
├── tests/                 # Test suite
├── setup.py              # Setup script
├── test.py               # Test runner
└── main.py               # Application entry point
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Wikidepia** for the IndoT5 model
- **spaCy** team for NLP framework
- **Hugging Face** for transformer models
- **NLTK** team for text processing tools
