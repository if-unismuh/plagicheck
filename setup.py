#!/usr/bin/env python3
"""
PlagiCheck - Auto Paraphrasing System Setup Script
Automated installation and configuration script.
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def print_banner():
    """Print setup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    PlagiCheck Setup                         ║
║              Auto Paraphrasing System                       ║
║                                                              ║
║  🚀 Automated Installation & Configuration                  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_python_version():
    """Check Python version requirement."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required. Current version:", 
              f"{version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")


def check_system_requirements():
    """Check system requirements."""
    print("\n🔍 Checking system requirements...")
    
    # Check OS
    os_name = platform.system()
    print(f"📱 Operating System: {os_name}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available Memory: {memory_gb:.1f} GB")
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM. Performance may be limited.")
    except ImportError:
        print("⚠️  Cannot check memory (psutil not installed)")
    
    # Check disk space
    disk_space = shutil.disk_usage('.').free / (1024**3)
    print(f"💽 Free Disk Space: {disk_space:.1f} GB")
    if disk_space < 10:
        print("❌ Insufficient disk space. At least 10GB required.")
        sys.exit(1)


def run_command(command, description, check=True):
    """Run shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False


def setup_virtual_environment():
    """Create and activate virtual environment."""
    print("\n🏗️  Setting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    print("✅ Virtual environment created")
    return True


def get_pip_command():
    """Get pip command for current OS."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    else:
        return "venv/bin/pip"


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", 
                      "Installing Python packages"):
        return False
    
    print("✅ Dependencies installed successfully")
    return True


def download_nlp_models():
    """Download required NLP models."""
    print("\n🧠 Downloading NLP models...")
    
    pip_cmd = get_pip_command()
    python_cmd = pip_cmd.replace("pip", "python")
    
    # Download NLTK data
    nltk_script = """
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('✅ NLTK data downloaded')
except Exception as e:
    print(f'⚠️ NLTK download issue: {e}')
"""
    
    # Download spaCy model
    spacy_script = """
import spacy
import subprocess
try:
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], 
                   check=True, capture_output=True)
    print('✅ spaCy model downloaded')
except Exception as e:
    print(f'⚠️ spaCy download issue: {e}')
"""
    
    # Execute NLTK download
    print("🔄 Downloading NLTK data...")
    subprocess.run([python_cmd, "-c", nltk_script])
    
    # Execute spaCy download
    print("🔄 Downloading spaCy model...")
    subprocess.run([python_cmd, "-c", spacy_script])
    
    print("✅ NLP models setup complete")


def setup_database():
    """Setup database."""
    print("\n🗄️  Setting up database...")
    
    python_cmd = get_pip_command().replace("pip", "python")
    
    # Check if migrate_db.py exists
    if not Path("migrate_db.py").exists():
        print("⚠️  migrate_db.py not found, skipping database setup")
        return True
    
    # Run database migration
    if run_command(f"{python_cmd} migrate_db.py migrate", 
                  "Running database migrations"):
        print("✅ Database setup complete")
        return True
    else:
        print("⚠️  Database setup failed, but continuing...")
        return True


def create_env_file():
    """Create .env file if not exists."""
    print("\n⚙️  Setting up configuration...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create default .env file
    env_content = """# PlagiCheck Configuration
# Generated by setup.py

# Application Settings
APP_NAME="PlagiCheck - Auto Paraphrasing System"
APP_VERSION="2.0.0"
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production

# Database Configuration
DATABASE_URL=sqlite:///./plagicheck.db

# File Upload Settings
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# AI Model Configuration
USE_GPU=true
MAX_CONCURRENT_JOBS=5
SIMILARITY_THRESHOLD=0.7

# Security Settings
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/plagicheck.log

# Performance
ENABLE_METRICS=true
"""
    
    try:
        env_file.write_text(env_content)
        print("✅ Created .env configuration file")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def create_directories():
    """Create required directories."""
    print("\n📁 Creating directories...")
    
    directories = ["uploads", "logs", "temp"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created {directory} directory")
            except Exception as e:
                print(f"❌ Failed to create {directory}: {e}")
        else:
            print(f"✅ {directory} directory exists")


def verify_installation():
    """Verify installation by importing main modules."""
    print("\n✅ Verifying installation...")
    
    python_cmd = get_pip_command().replace("pip", "python")
    
    test_script = """
try:
    from app.api.routes import app
    print('✅ FastAPI app import - OK')
except Exception as e:
    print(f'❌ FastAPI app import failed: {e}')

try:
    from app.core.config import settings
    print('✅ Configuration import - OK')
except Exception as e:
    print(f'❌ Configuration import failed: {e}')

try:
    from app.core.database import engine
    print('✅ Database connection - OK')
except Exception as e:
    print(f'❌ Database connection failed: {e}')

try:
    import torch
    print(f'✅ PyTorch - OK (CUDA: {torch.cuda.is_available()})')
except Exception as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import transformers
    print('✅ Transformers - OK')
except Exception as e:
    print(f'❌ Transformers import failed: {e}')

try:
    import spacy
    print('✅ spaCy - OK')
except Exception as e:
    print(f'❌ spaCy import failed: {e}')
"""
    
    subprocess.run([python_cmd, "-c", test_script])


def print_completion_message():
    """Print completion message with next steps."""
    completion_msg = """
╔══════════════════════════════════════════════════════════════╗
║                  🎉 Setup Complete!                         ║
╚══════════════════════════════════════════════════════════════╝

📋 Next Steps:

1. 🚀 Start the application:
   python main.py

2. 🌐 Open your browser:
   http://localhost:8000/docs

3. 🧪 Test the API:
   curl http://localhost:8000/health

4. 📚 Read the documentation:
   - README.md - Project overview
   - docs/QUICKSTART.md - 5-minute guide
   - docs/API.md - API documentation

5. 🔧 Configuration:
   - Edit .env file for custom settings
   - docs/CONFIGURATION.md for options

🆘 Need help?
   - GitHub Issues: https://github.com/if-unismuh/plagicheck/issues
   - Documentation: docs/
   - Postman Collection: postman/

✨ Happy Paraphrasing!
"""
    print(completion_msg)


def main():
    """Main setup function."""
    print_banner()
    
    try:
        # System checks
        check_python_version()
        check_system_requirements()
        
        # Setup steps
        if not setup_virtual_environment():
            print("❌ Virtual environment setup failed")
            sys.exit(1)
        
        if not install_dependencies():
            print("❌ Dependency installation failed")
            sys.exit(1)
        
        download_nlp_models()
        setup_database()
        create_env_file()
        create_directories()
        
        # Verification
        verify_installation()
        
        # Completion
        print_completion_message()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
