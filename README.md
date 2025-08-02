# Auto-Paraphrasing System

A comprehensive document processing and paraphrasing system with advanced NLP capabilities for Indonesian and English texts.

## 🚀 Features

### Document Processing
- **Multi-format support**: PDF, DOCX, TXT with structure preservation
- **Text preprocessing**: Cleaning, normalization, sentence segmentation
- **Academic term detection**: Identifies and preserves academic terminology
- **Citation preservation**: Maintains reference formats during processing
- **Document reconstruction**: Converts processed text back to DOCX format

### Indonesian NLP Pipeline
- **Indonesian language support**: spaCy with UDPipe integration
- **Comprehensive analysis**: POS tagging, NER, dependency parsing
- **Quality metrics**: Readability scores, complexity analysis
- **Academic writing detection**: Identifies formal academic language patterns
- **Sentence prioritization**: Determines which sentences need paraphrasing

### Paraphrasing Engine
- **Enhanced IndoT5 Paraphraser**: GPU optimization, batch processing, multiple variants
- **Hybrid Rule-Based Paraphraser**: Lexical substitution, syntactic transformation
- **Quality filtering**: Semantic similarity and grammar scoring
- **Academic tone preservation**: Maintains formal academic language
- **Real-time processing**: Status tracking and progress monitoring

## 📋 Requirements

- Python 3.9+
- PostgreSQL 12+ (or SQLite for development)
- GPU support recommended for local model inference
- 8GB+ RAM recommended

## ⚡ Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd paraphrase-system

# Run automated setup
python setup.py

# Start the application
python main.py
```

## 🛠️ Installation

### Automated Setup (Recommended)
```bash
python setup.py
```

### Manual Setup

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Create .env file
DATABASE_URL=postgresql://user:password@localhost/paraphrase_db
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
USE_GPU=true
```

4. **Set up database**:
```bash
# Using PostgreSQL
createdb paraphrase_db
python migrate_db.py migrate

# Or using SQLite (development)
python migrate_db.py migrate
```

5. **Download models**:
```bash
python -c "import spacy; spacy.download('en_core_web_sm')"
python -c "import nltk; nltk.download('punkt')"
```

## 🔧 Configuration

Create a `.env` file with the following configuration:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/paraphrase_db

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
