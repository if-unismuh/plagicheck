#!/usr/bin/env python3
"""
Comprehensive Setup Script for Auto-Paraphrasing System
Combines environment setup, dependency installation, model downloads, and database setup.
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SystemSetup:
    """Comprehensive system setup orchestrator."""
    
    def __init__(self, skip_models: bool = False, skip_database: bool = False, auto_mode: bool = False):
        self.skip_models = skip_models
        self.skip_database = skip_database
        self.auto_mode = auto_mode
        
    def run_command(self, command: str, description: str, ignore_errors: bool = False) -> bool:
        """Run a command and handle errors."""
        logger.info(f"Running: {description}")
        try:
            if os.name == 'nt':  # Windows
                shell_command = f"cmd /c {command}"
            else:  # Linux/Mac
                shell_command = command
                
            result = subprocess.run(shell_command, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"✓ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            if ignore_errors:
                logger.warning(f"⚠ {description} failed but continuing: {e}")
                return True
            else:
                logger.error(f"✗ {description} failed: {e}")
                if e.stderr:
                    logger.error(f"Error output: {e.stderr}")
                return False

    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        logger.info("Checking Python version...")
        if sys.version_info < (3, 9):
            logger.error("Python 3.9 or higher is required")
            return False
        logger.info(f"✓ Python {sys.version.split()[0]} is compatible")
        return True

    def create_virtual_environment(self) -> bool:
        """Create virtual environment if it doesn't exist."""
        venv_path = Path("venv")
        if venv_path.exists():
            logger.info("✓ Virtual environment already exists")
            return True
        
        logger.info("Creating virtual environment...")
        return self.run_command("python -m venv venv", "Creating virtual environment")

    def get_python_cmd(self) -> str:
        """Get the correct Python command for the virtual environment."""
        if os.name == 'nt':  # Windows
            return "venv\\Scripts\\python"
        else:  # Linux/Mac
            return "venv/bin/python"

    def get_pip_cmd(self) -> str:
        """Get the correct pip command for the virtual environment."""
        if os.name == 'nt':  # Windows
            return "venv\\Scripts\\pip"
        else:  # Linux/Mac
            return "venv/bin/pip"

    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        logger.info("Installing Python dependencies...")
        
        pip_cmd = self.get_pip_cmd()
        
        commands = [
            (f"{pip_cmd} install --upgrade pip", "Upgrading pip"),
            (f"{pip_cmd} install -r requirements.txt", "Installing requirements.txt"),
        ]
        
        # Optional: Install PyTorch with CUDA support
        if not self.skip_models:
            commands.append((
                f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "Installing PyTorch with CUDA support"
            ))
        
        for command, description in commands:
            if not self.run_command(command, description, ignore_errors=(description == "Installing PyTorch with CUDA support")):
                if description != "Installing PyTorch with CUDA support":
                    return False
        
        return True

    def download_spacy_models(self) -> bool:
        """Download spaCy models."""
        if self.skip_models:
            logger.info("Skipping spaCy model downloads")
            return True
            
        logger.info("Downloading spaCy models...")
        python_cmd = self.get_python_cmd()
        
        commands = [
            (f"{python_cmd} -m spacy download en_core_web_sm", "Downloading English spaCy model (small)"),
            (f"{python_cmd} -m spacy download en_core_web_md", "Downloading English spaCy model (medium)"),
        ]
        
        success = True
        for command, description in commands:
            if not self.run_command(command, description, ignore_errors=True):
                success = False
        
        return success

    def download_nltk_data(self) -> bool:
        """Download NLTK data."""
        if self.skip_models:
            logger.info("Skipping NLTK data downloads")
            return True
            
        logger.info("Downloading NLTK data...")
        python_cmd = self.get_python_cmd()
        
        try:
            # Use subprocess to run NLTK downloads
            script = """
import nltk
import sys
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    sys.exit(1)
"""
            return self.run_command(f'{python_cmd} -c "{script}"', "Downloading NLTK data")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            return False

    def setup_indonesian_nlp(self) -> bool:
        """Setup Indonesian NLP models."""
        if self.skip_models:
            logger.info("Skipping Indonesian NLP setup")
            return True
            
        logger.info("Setting up Indonesian NLP models...")
        pip_cmd = self.get_pip_cmd()
        python_cmd = self.get_python_cmd()
        
        try:
            # Install spacy-udpipe
            if not self.run_command(f"{pip_cmd} install spacy-udpipe", "Installing spacy-udpipe"):
                return False
            
            # Download Indonesian UDPipe model
            script = """
import spacy_udpipe
import sys
try:
    spacy_udpipe.download("id")
    print("Indonesian UDPipe model downloaded successfully")
except Exception as e:
    print(f"Error downloading Indonesian model: {e}")
    sys.exit(1)
"""
            return self.run_command(f'{python_cmd} -c "{script}"', "Downloading Indonesian UDPipe model")
        except Exception as e:
            logger.error(f"Failed to setup Indonesian NLP: {e}")
            return False

    def test_models(self) -> bool:
        """Test if models can be loaded."""
        if self.skip_models:
            logger.info("Skipping model testing")
            return True
            
        logger.info("Testing model installations...")
        python_cmd = self.get_python_cmd()
        
        test_script = """
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    # Test PyTorch
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("! CUDA not available - will use CPU")
    
    # Test spaCy
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy English model loaded")
    
    # Test transformers
    from transformers import AutoTokenizer
    print("✓ Transformers library loaded")
    
    print("Model testing completed successfully")
except Exception as e:
    print(f"Model testing failed: {e}")
    sys.exit(1)
"""
        return self.run_command(f'{python_cmd} -c "{test_script}"', "Testing model installations")

    def create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Creating directories...")
        
        directories = ["uploads", "logs", "cache", "models"]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✓ Directory created: {directory}")
        
        return True

    def setup_environment(self) -> bool:
        """Setup environment configuration."""
        logger.info("Setting up environment...")
        
        env_file = Path(".env")
        if env_file.exists():
            logger.info("✓ .env file already exists")
            return True
        
        env_content = """# Database Configuration
DATABASE_URL=sqlite:///./paraphrase_db.sqlite

# File Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# API Keys (optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration
USE_GPU=true
MODEL_CACHE_DIR=models

# Application Settings
DEBUG=true
"""
        
        try:
            with open(env_file, "w") as f:
                f.write(env_content.strip())
            logger.info("✓ .env file created with default configuration")
            return True
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

    def setup_database(self) -> bool:
        """Setup database."""
        if self.skip_database:
            logger.info("Skipping database setup")
            return True
            
        logger.info("Setting up database...")
        python_cmd = self.get_python_cmd()
        
        # Run database migrations
        return self.run_command(f"{python_cmd} migrate_db.py migrate", "Running database migrations")

    def run_comprehensive_setup(self) -> bool:
        """Run the complete setup process."""
        logger.info("🚀 Starting Comprehensive Auto-Paraphrasing System Setup")
        logger.info("=" * 60)
        
        setup_steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Creating directories", self.create_directories),
            ("Setting up environment", self.setup_environment),
            ("Installing Python dependencies", self.install_dependencies),
            ("Downloading spaCy models", self.download_spacy_models),
            ("Downloading NLTK data", self.download_nltk_data),
            ("Setting up Indonesian NLP", self.setup_indonesian_nlp),
            ("Testing model installations", self.test_models),
            ("Setting up database", self.setup_database),
        ]
        
        success_count = 0
        total_steps = len(setup_steps)
        
        for step_name, step_function in setup_steps:
            logger.info(f"\n📦 {step_name}...")
            try:
                if step_function():
                    success_count += 1
                    logger.info(f"✓ {step_name} completed")
                else:
                    logger.warning(f"⚠ {step_name} completed with issues")
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Setup completed: {success_count}/{total_steps} steps successful")
        
        if success_count >= total_steps - 2:  # Allow some optional steps to fail
            logger.info("🎉 Setup completed successfully!")
            self.display_next_steps()
            return True
        else:
            logger.warning("⚠ Setup completed with some issues. Check the logs above.")
            return False

    def display_next_steps(self):
        """Display next steps for the user."""
        logger.info("\nNext steps:")
        logger.info("1. Review and configure .env file for your environment")
        
        if not self.skip_database:
            logger.info("2. Database is ready to use")
        else:
            logger.info("2. Set up database: python migrate_db.py migrate")
        
        logger.info("3. Start the application: python main.py")
        logger.info("4. Access the API documentation: http://localhost:8000/docs")
        
        if self.skip_models:
            logger.info("5. Run setup again without --skip-models to download ML models")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Comprehensive Setup Script for Auto-Paraphrasing System')
    parser.add_argument('--skip-models', action='store_true', help='Skip ML model downloads')
    parser.add_argument('--skip-database', action='store_true', help='Skip database setup')
    parser.add_argument('--auto', action='store_true', help='Run setup automatically without prompts')
    parser.add_argument('--basic', action='store_true', help='Basic setup only (equivalent to --skip-models)')
    
    args = parser.parse_args()
    
    # Basic setup means skip models
    if args.basic:
        args.skip_models = True
    
    # Create setup instance
    setup = SystemSetup(
        skip_models=args.skip_models,
        skip_database=args.skip_database,
        auto_mode=args.auto
    )
    
    # Run setup
    success = setup.run_comprehensive_setup()
    
    if success:
        logger.info("Setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
