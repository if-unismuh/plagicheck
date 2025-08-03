# 🔧 Panduan Konfigurasi PlagiCheck

Dokumen ini menjelaskan semua opsi konfigurasi yang tersedia untuk PlagiCheck - Auto Paraphrasing System.

## 📋 Overview

PlagiCheck menggunakan file `.env` untuk konfigurasi aplikasi. Semua setting dapat dikonfigurasi melalui environment variables atau file konfigurasi.

## 🏗️ Struktur Konfigurasi

### Aplikasi Utama
```env
# Basic Application Settings
APP_NAME="PlagiCheck - Auto Paraphrasing System"
APP_VERSION="2.0.0"
DEBUG=false                    # Set true untuk development
SECRET_KEY=your-secret-key     # Min 32 characters untuk security
ENVIRONMENT=production         # development/staging/production
```

### Database Settings

#### PostgreSQL (Production)
```env
DATABASE_URL=postgresql://username:password@host:port/database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=plagicheck_db
DATABASE_USER=plagicheck_user
DATABASE_PASSWORD=secure_password
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

#### SQLite (Development)
```env
DATABASE_URL=sqlite:///./plagicheck.db
DATABASE_ECHO=false            # Set true untuk debug SQL queries
```

### File Upload Configuration
```env
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB             # Maximum file size
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.doc
TEMP_DIR=/tmp/plagicheck       # Temporary files directory
AUTO_CLEANUP_TEMP=true         # Auto cleanup temporary files
CLEANUP_INTERVAL_HOURS=24      # Cleanup interval
```

### AI Model Settings

#### IndoT5 Model Configuration
```env
INDOT5_MODEL_NAME=Wikidepia/IndoT5-base-paraphrase
INDOT5_MAX_LENGTH=512
INDOT5_NUM_BEAMS=5
INDOT5_NUM_RETURN_SEQUENCES=3
INDOT5_TEMPERATURE=0.7
INDOT5_TOP_P=0.9
INDOT5_TOP_K=50
INDOT5_REPETITION_PENALTY=1.2
INDOT5_BATCH_SIZE=4
```

#### GPU & Hardware Settings
```env
USE_GPU=true
CUDA_VISIBLE_DEVICES=0         # GPU device IDs (0,1,2)
TORCH_NUM_THREADS=4            # CPU threads for PyTorch
OMP_NUM_THREADS=4              # OpenMP threads
GPU_MEMORY_FRACTION=0.8        # GPU memory usage limit
ENABLE_MIXED_PRECISION=true    # FP16 untuk efficiency
```

#### Model Caching
```env
MODELS_CACHE_DIR=~/.cache/plagicheck
ENABLE_MODEL_CACHE=true
CACHE_MAX_SIZE_GB=10
AUTO_DOWNLOAD_MODELS=true
```

### Performance Settings

#### Concurrency & Processing
```env
MAX_CONCURRENT_JOBS=5          # Concurrent processing jobs
WORKER_TIMEOUT=300             # Worker timeout in seconds
MAX_QUEUE_SIZE=100             # Maximum job queue size
ENABLE_BACKGROUND_TASKS=true   # Background processing
BATCH_PROCESSING_SIZE=10       # Batch size untuk multiple docs
```

#### Memory Management
```env
MAX_MEMORY_USAGE_GB=8          # Maximum memory usage
ENABLE_MEMORY_MONITORING=true  # Monitor memory usage
MEMORY_CLEANUP_THRESHOLD=0.9   # Cleanup threshold (90%)
GARBAGE_COLLECTION_INTERVAL=300 # GC interval in seconds
```

### Caching Configuration

#### Redis (Production)
```env
USE_REDIS=true
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password
REDIS_POOL_SIZE=20
REDIS_SOCKET_TIMEOUT=5
CACHE_TTL=3600                 # Cache time-to-live (seconds)
CACHE_PREFIX=plagicheck        # Cache key prefix
```

#### Memory Cache (Development)
```env
USE_MEMORY_CACHE=true
MEMORY_CACHE_SIZE=1000         # Number of items
MEMORY_CACHE_TTL=1800          # TTL in seconds
```

### Security Settings

#### Authentication & Authorization
```env
SECRET_KEY=your-super-secret-key-minimum-32-characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
REQUIRE_AUTHENTICATION=false   # Set true untuk enable auth
```

#### CORS & Security Headers
```env
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
CORS_CREDENTIALS=true
CORS_METHODS=["GET","POST","PUT","DELETE"]
CORS_HEADERS=["*"]
ENABLE_HTTPS_REDIRECT=false    # Set true untuk production
```

#### Rate Limiting
```env
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60       # Requests per minute per IP
RATE_LIMIT_BURST=10            # Burst limit
RATE_LIMIT_STORAGE=memory      # memory/redis
```

### Logging & Monitoring

#### Logging Configuration
```env
LOG_LEVEL=INFO                 # DEBUG/INFO/WARNING/ERROR/CRITICAL
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=logs/plagicheck.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
ENABLE_JSON_LOGGING=false      # JSON format logs
```

#### Performance Monitoring
```env
ENABLE_METRICS=true
METRICS_ENDPOINT=/metrics
ENABLE_PERFORMANCE_LOGGING=true
PERFORMANCE_LOG_FILE=logs/performance.log
TRACK_REQUEST_DURATION=true
TRACK_MEMORY_USAGE=true
TRACK_MODEL_PERFORMANCE=true
```

#### Health Checks
```env
ENABLE_HEALTH_CHECKS=true
HEALTH_CHECK_INTERVAL=60       # Seconds
HEALTH_CHECK_TIMEOUT=10        # Seconds
DATABASE_HEALTH_CHECK=true
MODEL_HEALTH_CHECK=true
DISK_SPACE_HEALTH_CHECK=true
MEMORY_HEALTH_CHECK=true
```

### NLP Pipeline Settings

#### Indonesian NLP
```env
SPACY_MODEL=id_core_news_sm
UDPIPE_MODEL=indonesian-gsd
ENABLE_NER=true
ENABLE_POS_TAGGING=true
ENABLE_DEPENDENCY_PARSING=true
NLP_BATCH_SIZE=32
```

#### Text Analysis
```env
SIMILARITY_THRESHOLD=0.7       # Minimum similarity score
READABILITY_THRESHOLD=0.6      # Minimum readability score
ACADEMIC_TERMS_MIN_LENGTH=3    # Minimum term length
MAX_SENTENCE_LENGTH=200        # Maximum sentence length
ENABLE_ACADEMIC_DETECTION=true
PRESERVE_CITATIONS=true
```

### Quality Assessment

#### Scoring Configuration
```env
QUALITY_WEIGHT_SIMILARITY=0.3
QUALITY_WEIGHT_READABILITY=0.25
QUALITY_WEIGHT_GRAMMAR=0.25
QUALITY_WEIGHT_ACADEMIC_TONE=0.2
MIN_QUALITY_SCORE=0.6
ENABLE_QUALITY_FILTERING=true
```

#### Paraphrasing Options
```env
DEFAULT_PARAPHRASE_METHOD=hybrid
MAX_PARAPHRASE_VARIANTS=5
MIN_PARAPHRASE_VARIANTS=1
ENABLE_BATCH_PARAPHRASING=true
PARAPHRASE_TIMEOUT=300         # Seconds per document
```

### Notification Settings

#### Email (Optional)
```env
ENABLE_EMAIL_NOTIFICATIONS=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_FROM=noreply@plagicheck.com
EMAIL_ADMIN=admin@plagicheck.com
```

#### Webhooks (Optional)
```env
ENABLE_WEBHOOKS=false
WEBHOOK_URL=https://your-webhook-endpoint.com
WEBHOOK_SECRET=your_webhook_secret
WEBHOOK_EVENTS=document_processed,paraphrase_completed
```

## 🌍 Environment-Specific Configurations

### Development Environment
```env
# development.env
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///./plagicheck_dev.db
USE_GPU=false
MAX_CONCURRENT_JOBS=2
ENABLE_METRICS=false
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
```

### Staging Environment
```env
# staging.env
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@staging-db:5432/plagicheck_staging
USE_GPU=true
MAX_CONCURRENT_JOBS=3
ENABLE_METRICS=true
CORS_ORIGINS=["https://staging.plagicheck.com"]
```

### Production Environment
```env
# production.env
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:pass@prod-db:5432/plagicheck_prod
USE_GPU=true
MAX_CONCURRENT_JOBS=5
ENABLE_METRICS=true
ENABLE_RATE_LIMITING=true
REQUIRE_AUTHENTICATION=true
CORS_ORIGINS=["https://plagicheck.com"]
ENABLE_HTTPS_REDIRECT=true
```

## 🔒 Security Best Practices

### Production Security
```env
# Strong secret key
SECRET_KEY=$(openssl rand -hex 32)

# Secure database connection
DATABASE_URL=postgresql://user:$(openssl rand -hex 16)@secure-host:5432/db?sslmode=require

# Restrict CORS
CORS_ORIGINS=["https://yourdomain.com"]

# Enable security features
ENABLE_HTTPS_REDIRECT=true
REQUIRE_AUTHENTICATION=true
ENABLE_RATE_LIMITING=true
```

### SSL/TLS Configuration
```env
USE_HTTPS=true
SSL_CERT_PATH=/path/to/certificate.pem
SSL_KEY_PATH=/path/to/private-key.pem
SSL_CA_CERTS=/path/to/ca-bundle.pem
SSL_VERIFY_MODE=required
```

## 📊 Performance Tuning

### High-Performance Configuration
```env
# Optimize for high throughput
MAX_CONCURRENT_JOBS=10
BATCH_PROCESSING_SIZE=20
USE_GPU=true
ENABLE_MIXED_PRECISION=true
GPU_MEMORY_FRACTION=0.9

# Database optimization
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_POOL_RECYCLE=1800

# Caching
USE_REDIS=true
CACHE_TTL=7200
MEMORY_CACHE_SIZE=2000
```

### Low-Resource Configuration
```env
# Optimize for limited resources
MAX_CONCURRENT_JOBS=2
BATCH_PROCESSING_SIZE=5
USE_GPU=false
INDOT5_BATCH_SIZE=1

# Memory conservation
MAX_MEMORY_USAGE_GB=4
MEMORY_CLEANUP_THRESHOLD=0.7
ENABLE_MEMORY_MONITORING=true

# Minimal caching
USE_MEMORY_CACHE=true
MEMORY_CACHE_SIZE=100
```

## 🔧 Configuration Validation

### Validate Configuration
```bash
# Check configuration
python -c "
from app.core.config import settings
print('✅ Configuration loaded successfully')
print(f'App: {settings.app_name}')
print(f'Debug: {settings.debug}')
print(f'Database: {settings.database_url}')
"
```

### Common Configuration Issues

#### Database Connection
```bash
# Test database connection
python -c "
from app.core.database import engine
from sqlalchemy import text
try:
    result = engine.execute(text('SELECT 1')).scalar()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database error: {e}')
"
```

#### GPU Configuration
```bash
# Test GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
"
```

## 📝 Configuration Templates

### Complete .env Template
```env
# PlagiCheck Configuration Template
# Copy and modify according to your environment

# Application
APP_NAME="PlagiCheck - Auto Paraphrasing System"
APP_VERSION="2.0.0"
DEBUG=false
SECRET_KEY=your-secret-key-here
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://username:password@localhost/plagicheck_db

# File Upload
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# AI Models
USE_GPU=true
MAX_CONCURRENT_JOBS=5
SIMILARITY_THRESHOLD=0.7

# Security
CORS_ORIGINS=["https://yourdomain.com"]
ENABLE_RATE_LIMITING=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/plagicheck.log

# Performance
ENABLE_METRICS=true
USE_REDIS=false
```

---

📝 **Note**: Selalu backup file konfigurasi sebelum melakukan perubahan dan test konfigurasi di environment development terlebih dahulu.
