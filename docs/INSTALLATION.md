# 🚀 Panduan Instalasi PlagiCheck

Dokumen ini menyediakan panduan lengkap untuk menginstal dan mengkonfigurasi PlagiCheck - Auto Paraphrasing System.

## 📋 Prasyarat Sistem

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), Windows 10+
- **Python**: 3.9 atau lebih tinggi
- **RAM**: 8GB (16GB direkomendasikan)
- **Storage**: 10GB free space untuk models dan data
- **Internet**: Koneksi stabil untuk download models

### Recommended Specifications
- **GPU**: NVIDIA CUDA-compatible (RTX 3060 atau setara)
- **VRAM**: 6GB+ untuk model inference optimal
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 atau lebih tinggi)
- **RAM**: 16GB+ untuk batch processing
- **SSD**: Recommended untuk performa I/O yang lebih baik

## 🔧 Metode Instalasi

### 1. Quick Setup (Automated)

```bash
# Clone repository
git clone https://github.com/if-unismuh/plagicheck.git
cd plagicheck

# Jalankan automated setup
python setup.py

# Verifikasi instalasi
python -c "from app.api.routes import app; print('✅ Installation successful!')"
```

### 2. Manual Installation

#### Step 1: Environment Setup
```bash
# Buat virtual environment
python -m venv plagicheck-env

# Aktivasi environment
# Linux/macOS:
source plagicheck-env/bin/activate
# Windows:
plagicheck-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 2: Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Verifikasi instalasi PyTorch (untuk GPU support)
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### Step 3: Download NLP Models
```bash
# NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
print('✅ NLTK data downloaded')
"

# spaCy models
python -c "
import spacy
spacy.download('en_core_web_sm')
print('✅ spaCy model downloaded')
"

# Download Indonesian UDPipe model (auto-download saat first run)
python -c "
from app.services.indonesian_nlp_pipeline import indonesian_nlp_pipeline
print('✅ Indonesian NLP pipeline ready')
"
```

#### Step 4: Database Setup

##### Option A: PostgreSQL (Production)
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
brew install postgresql                              # macOS

# Start PostgreSQL service
sudo systemctl start postgresql    # Linux
brew services start postgresql     # macOS

# Create database dan user
sudo -u postgres psql
CREATE DATABASE plagicheck_db;
CREATE USER plagicheck_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE plagicheck_db TO plagicheck_user;
\q

# Set environment variable
export DATABASE_URL="postgresql://plagicheck_user:your_password@localhost/plagicheck_db"
```

##### Option B: SQLite (Development)
```bash
# SQLite akan dibuat otomatis
export DATABASE_URL="sqlite:///./plagicheck.db"
```

#### Step 5: Environment Configuration
```bash
# Buat file .env
cat > .env << 'EOF'
# Application Settings
APP_NAME="PlagiCheck - Auto Paraphrasing System"
APP_VERSION="2.0.0"
DEBUG=true

# Database
DATABASE_URL=sqlite:///./plagicheck.db

# File Uploads
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# AI Models
USE_GPU=true
MAX_CONCURRENT_JOBS=5
SIMILARITY_THRESHOLD=0.7

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
EOF
```

#### Step 6: Database Migration
```bash
# Jalankan migrations
python migrate_db.py migrate

# Verifikasi database
python migrate_db.py status

# (Optional) Seed sample data
python migrate_db.py seed
```

#### Step 7: Verify Installation
```bash
# Test import semua modules
python -c "
from app.api.routes import app
from app.services.enhanced_indot5_paraphraser import get_enhanced_indot5_paraphraser
from app.services.indonesian_nlp_pipeline import indonesian_nlp_pipeline
print('✅ All modules imported successfully')
"

# Test database connection
python -c "
from app.core.database import engine
from sqlalchemy import text
result = engine.execute(text('SELECT 1')).scalar()
print(f'✅ Database connection: {result}')
"
```

## 🐳 Docker Installation

### Docker Compose (Recommended)
```bash
# Clone repository
git clone https://github.com/if-unismuh/plagicheck.git
cd plagicheck

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Manual Docker Build
```bash
# Build image
docker build -t plagicheck:latest .

# Run container
docker run -d \
  --name plagicheck \
  -p 8000:8000 \
  -e DATABASE_URL=sqlite:///./plagicheck.db \
  -v $(pwd)/uploads:/app/uploads \
  plagicheck:latest

# Check logs
docker logs -f plagicheck
```

## 🔧 Configuration Options

### Environment Variables

#### Application Settings
```env
APP_NAME="PlagiCheck - Auto Paraphrasing System"
APP_VERSION="2.0.0"
DEBUG=false                    # Set to false in production
SECRET_KEY=your-secret-key     # Generate secure key for production
```

#### Database Configuration
```env
# PostgreSQL (Production)
DATABASE_URL=postgresql://username:password@localhost/plagicheck_db
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=plagicheck_db
DATABASE_USER=username
DATABASE_PASSWORD=password

# SQLite (Development)
DATABASE_URL=sqlite:///./plagicheck.db
```

#### File Upload Settings
```env
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB             # Maximum file size
ALLOWED_EXTENSIONS=.pdf,.docx,.txt
```

#### AI Model Configuration
```env
USE_GPU=true                   # Enable GPU acceleration
CUDA_VISIBLE_DEVICES=0         # GPU device ID
MAX_CONCURRENT_JOBS=5          # Concurrent processing jobs
SIMILARITY_THRESHOLD=0.7       # Paraphrase similarity threshold
BATCH_SIZE=4                   # Batch size for model inference
```

#### Performance Settings
```env
# Caching
USE_REDIS=false               # Redis caching (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600                # Cache time-to-live (seconds)

# Monitoring
ENABLE_METRICS=true           # Performance metrics
LOG_LEVEL=INFO                # Logging level
ENABLE_PERFORMANCE_LOGGING=true
```

## 🚀 Starting the Application

### Development Mode
```bash
# Dengan auto-reload
uvicorn app.api.routes:app --reload --host 0.0.0.0 --port 8000

# Atau menggunakan main.py
python main.py
```

### Production Mode
```bash
# Menggunakan Gunicorn
gunicorn app.api.routes:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --preload

# Dengan systemd service
sudo systemctl start plagicheck
sudo systemctl enable plagicheck
```

## ✅ Post-Installation Verification

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. API Documentation
Buka browser: http://localhost:8000/docs

### 3. Test Upload
```bash
# Test file upload
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.txt" \
  -F "chapter=Test"
```

### 4. Test Paraphrasing
```bash
# Test direct paraphrasing
curl -X POST "http://localhost:8000/api/v2/text/paraphrase-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Penelitian ini menggunakan metode kualitatif.",
    "method": "rule_based",
    "num_variants": 2
  }'
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Error untuk torch/transformers
```bash
# Reinstall PyTorch
pip uninstall torch transformers
pip install torch transformers --upgrade

# Verifikasi CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. Database Connection Error
```bash
# PostgreSQL
sudo systemctl status postgresql
sudo systemctl start postgresql

# Test connection
psql -h localhost -U plagicheck_user -d plagicheck_db
```

#### 3. Model Download Issues
```bash
# Clear cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Download ulang
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Wikidepia/IndoT5-base-paraphrase')"
```

#### 4. Permission Issues
```bash
# Fix file permissions
chmod +x migrate_db.py setup.py main.py
chown -R $USER:$USER uploads/
```

#### 5. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.api.routes:app --port 8001
```

## 🔒 Security Configuration

### Production Security Settings
```env
# Security
DEBUG=false
SECRET_KEY=your-super-secret-key-min-32-chars
CORS_ORIGINS=["https://yourdomain.com"]

# Database
DATABASE_URL=postgresql://user:pass@secure-db-host/plagicheck_db

# SSL/TLS
USE_HTTPS=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Firewall Configuration
```bash
# Ubuntu UFW
sudo ufw allow 8000/tcp
sudo ufw enable

# CentOS/RHEL firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

## 📊 Performance Optimization

### GPU Optimization
```env
# GPU settings
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
```

### Memory Optimization
```env
# Memory settings
MAX_CONCURRENT_JOBS=3         # Reduce if limited memory
BATCH_SIZE=2                  # Smaller batch size
TORCH_NUM_THREADS=4           # CPU threads
```

### Database Optimization
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;
SELECT pg_reload_conf();
```

## 📞 Mendapatkan Bantuan

Jika mengalami masalah selama instalasi:

1. **Check Logs**: `tail -f logs/plagicheck.log`
2. **GitHub Issues**: [Create Issue](https://github.com/if-unismuh/plagicheck/issues)
3. **Documentation**: Baca dokumentasi lengkap di README.md
4. **Test Environment**: Jalankan `pytest tests/test_installation.py`

---

✅ **Instalasi Selesai!** 

Aplikasi PlagiCheck siap digunakan di: http://localhost:8000
