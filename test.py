#!/usr/bin/env python3
"""
Comprehensive Test Script for Auto-Paraphrasing System
Tests system components, dependencies, and functionality.
"""
import os
import sys
import subprocess
import argparse
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SystemTester:
    """Comprehensive system testing orchestrator."""
    
    def __init__(self, skip_models: bool = False, skip_api: bool = False, verbose: bool = False):
        self.skip_models = skip_models
        self.skip_api = skip_api
        self.verbose = verbose
        self.test_results: Dict[str, bool] = {}
        self.start_time = time.time()
        
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result and store it."""
        self.test_results[test_name] = success
        if success:
            logger.info(f"✓ {test_name}: PASSED {message}")
        else:
            logger.error(f"✗ {test_name}: FAILED {message}")
    
    def run_command(self, command: str, description: str, timeout: int = 30) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            if self.verbose:
                logger.info(f"Running: {command}")
            
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f"Command failed: {e.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def test_python_environment(self) -> bool:
        """Test Python environment and dependencies."""
        logger.info("Testing Python environment...")
        
        # Test Python version
        if sys.version_info < (3, 9):
            self.log_test_result("Python Version", False, f"Python 3.9+ required, got {sys.version}")
            return False
        else:
            self.log_test_result("Python Version", True, f"Python {sys.version.split()[0]}")
        
        # Test virtual environment
        venv_path = Path("venv")
        if venv_path.exists():
            self.log_test_result("Virtual Environment", True, "venv directory exists")
        else:
            self.log_test_result("Virtual Environment", False, "venv directory not found")
        
        return True

    def test_dependencies(self) -> bool:
        """Test Python dependencies."""
        logger.info("Testing Python dependencies...")
        
        required_packages = [
            "fastapi",
            "uvicorn", 
            "sqlalchemy",
            "psycopg2",
            "transformers",
            "torch",
            "spacy",
            "nltk",
            "pydantic"
        ]
        
        all_success = True
        for package in required_packages:
            try:
                __import__(package)
                self.log_test_result(f"Package: {package}", True)
            except ImportError as e:
                self.log_test_result(f"Package: {package}", False, str(e))
                all_success = False
        
        return all_success

    def test_nltk_data(self) -> bool:
        """Test NLTK data availability."""
        if self.skip_models:
            logger.info("Skipping NLTK data tests")
            return True
            
        logger.info("Testing NLTK data...")
        
        try:
            import nltk
            
            # Test punkt tokenizer
            try:
                nltk.data.find('tokenizers/punkt')
                self.log_test_result("NLTK Punkt", True)
            except LookupError:
                self.log_test_result("NLTK Punkt", False, "Punkt tokenizer not found")
            
            # Test stopwords
            try:
                nltk.data.find('corpora/stopwords')
                self.log_test_result("NLTK Stopwords", True)
            except LookupError:
                self.log_test_result("NLTK Stopwords", False, "Stopwords not found")
            
            return True
        except Exception as e:
            self.log_test_result("NLTK Data", False, str(e))
            return False

    def test_spacy_models(self) -> bool:
        """Test spaCy models."""
        if self.skip_models:
            logger.info("Skipping spaCy model tests")
            return True
            
        logger.info("Testing spaCy models...")
        
        try:
            import spacy
            
            # Test English model
            try:
                nlp = spacy.load("en_core_web_sm")
                doc = nlp("This is a test sentence.")
                self.log_test_result("spaCy English Model", len(doc) > 0, f"Processed {len(doc)} tokens")
            except OSError:
                self.log_test_result("spaCy English Model", False, "en_core_web_sm not found")
            
            return True
        except Exception as e:
            self.log_test_result("spaCy Models", False, str(e))
            return False

    def test_pytorch(self) -> bool:
        """Test PyTorch installation."""
        if self.skip_models:
            logger.info("Skipping PyTorch tests")
            return True
            
        logger.info("Testing PyTorch...")
        
        try:
            import torch
            
            # Test basic tensor operations
            x = torch.tensor([1.0, 2.0, 3.0])
            y = torch.tensor([4.0, 5.0, 6.0])
            z = x + y
            
            self.log_test_result("PyTorch Basic Operations", True, f"PyTorch {torch.__version__}")
            
            # Test CUDA availability
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.log_test_result("CUDA Support", True, f"GPU: {device_name}")
            else:
                self.log_test_result("CUDA Support", True, "CPU only (GPU not available)")
            
            return True
        except Exception as e:
            self.log_test_result("PyTorch", False, str(e))
            return False

    def test_transformers(self) -> bool:
        """Test Transformers library."""
        if self.skip_models:
            logger.info("Skipping Transformers tests")
            return True
            
        logger.info("Testing Transformers library...")
        
        try:
            from transformers import AutoTokenizer
            
            # Test basic tokenizer loading
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            test_text = "Hello world"
            tokens = tokenizer.encode(test_text)
            
            self.log_test_result("Transformers Tokenizer", len(tokens) > 0, f"Tokenized: {len(tokens)} tokens")
            
            return True
        except Exception as e:
            self.log_test_result("Transformers", False, str(e))
            return False

    def test_database_connection(self) -> bool:
        """Test database connection."""
        logger.info("Testing database connection...")
        
        try:
            # Add project root to Python path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from app.core.database import engine
            from sqlalchemy import text
            
            # Test database connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.log_test_result("Database Connection", True)
            return True
        except Exception as e:
            self.log_test_result("Database Connection", False, str(e))
            return False

    def test_app_import(self) -> bool:
        """Test application imports."""
        logger.info("Testing application imports...")
        
        try:
            # Add project root to Python path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # Test core imports
            from app.core.config import settings
            self.log_test_result("App Config Import", True)
            
            from app.core.database import Base, SessionLocal
            self.log_test_result("App Database Import", True)
            
            from app.models.document import Document
            self.log_test_result("App Models Import", True)
            
            # Test API imports
            from app.api.routes import app
            self.log_test_result("App API Import", True)
            
            return True
        except Exception as e:
            self.log_test_result("App Imports", False, str(e))
            return False

    def test_services(self) -> bool:
        """Test service modules."""
        if self.skip_models:
            logger.info("Skipping service tests (models required)")
            return True
            
        logger.info("Testing service modules...")
        
        try:
            # Add project root to Python path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # Test document processor
            from app.services.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            self.log_test_result("Document Processor", True)
            
            # Test paraphraser
            from app.services.paraphraser import ParaphraseService
            paraphraser = ParaphraseService()
            self.log_test_result("Paraphrase Service", True)
            
            return True
        except Exception as e:
            self.log_test_result("Services", False, str(e))
            return False

    def test_api_server(self) -> bool:
        """Test API server startup."""
        if self.skip_api:
            logger.info("Skipping API server tests")
            return True
            
        logger.info("Testing API server startup...")
        
        try:
            # Test if main.py can be executed
            python_cmd = "python" if os.name != 'nt' else "python"
            success, output = self.run_command(f"{python_cmd} -c \"import main; print('Main module loaded')\"", "Main module test", timeout=10)
            
            if success:
                self.log_test_result("API Server Module", True)
                return True
            else:
                self.log_test_result("API Server Module", False, output)
                return False
        except Exception as e:
            self.log_test_result("API Server", False, str(e))
            return False

    def test_file_structure(self) -> bool:
        """Test file structure."""
        logger.info("Testing file structure...")
        
        required_files = [
            "main.py",
            "requirements.txt",
            "alembic.ini",
            "app/__init__.py",
            "app/core/__init__.py",
            "app/models/__init__.py",
            "app/services/__init__.py",
            "app/api/__init__.py"
        ]
        
        required_dirs = [
            "app",
            "app/core",
            "app/models", 
            "app/services",
            "app/api",
            "migrations",
            "tests"
        ]
        
        all_success = True
        
        # Check files
        for file_path in required_files:
            if Path(file_path).exists():
                self.log_test_result(f"File: {file_path}", True)
            else:
                self.log_test_result(f"File: {file_path}", False, "File not found")
                all_success = False
        
        # Check directories
        for dir_path in required_dirs:
            if Path(dir_path).is_dir():
                self.log_test_result(f"Directory: {dir_path}", True)
            else:
                self.log_test_result(f"Directory: {dir_path}", False, "Directory not found")
                all_success = False
        
        return all_success

    def test_environment_config(self) -> bool:
        """Test environment configuration."""
        logger.info("Testing environment configuration...")
        
        env_file = Path(".env")
        if env_file.exists():
            self.log_test_result("Environment File", True, ".env file exists")
            
            # Check for required environment variables
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from app.core.config import settings
                
                # Test database URL
                if hasattr(settings, 'database_url_sync') and settings.database_url_sync:
                    self.log_test_result("Database URL Config", True)
                else:
                    self.log_test_result("Database URL Config", False, "DATABASE_URL not configured")
                
                return True
            except Exception as e:
                self.log_test_result("Environment Config", False, str(e))
                return False
        else:
            self.log_test_result("Environment File", False, ".env file not found")
            return False

    def test_basic_functionality(self) -> bool:
        """Test basic functionality with sample data."""
        if self.skip_models:
            logger.info("Skipping functionality tests (models required)")
            return True
            
        logger.info("Testing basic functionality...")
        
        try:
            # Add project root to Python path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            # Test text processing
            test_text = "Ini adalah contoh teks untuk diuji."
            
            # Simple tokenization test
            import nltk
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(test_text)
            
            if len(sentences) > 0:
                self.log_test_result("Text Processing", True, f"Processed {len(sentences)} sentences")
                return True
            else:
                self.log_test_result("Text Processing", False, "No sentences processed")
                return False
                
        except Exception as e:
            self.log_test_result("Basic Functionality", False, str(e))
            return False

    def test_comprehensive_test_suite(self) -> bool:
        """Test that the comprehensive test suite can run."""
        logger.info("Testing comprehensive test suite...")
        
        try:
            # Check if pytest is available
            success, output = self.run_command("python -m pytest --version", "Pytest availability check", timeout=10)
            
            if success:
                self.log_test_result("Pytest Available", True)
                
                # Run a quick test to ensure test suite works
                success, output = self.run_command("python -m pytest tests/conftest.py --collect-only", "Test collection check", timeout=15)
                
                if success:
                    self.log_test_result("Test Suite Structure", True)
                    return True
                else:
                    self.log_test_result("Test Suite Structure", False, "Cannot collect tests")
                    return False
            else:
                self.log_test_result("Pytest Available", False, "Pytest not installed")
                return False
                
        except Exception as e:
            self.log_test_result("Comprehensive Test Suite", False, str(e))
            return False

    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("🧪 Starting Comprehensive System Tests")
        logger.info("=" * 60)
        
        test_suites = [
            ("Python Environment", self.test_python_environment),
            ("Dependencies", self.test_dependencies),
            ("File Structure", self.test_file_structure),
            ("Environment Config", self.test_environment_config),
            ("App Imports", self.test_app_import),
            ("Database Connection", self.test_database_connection),
            ("NLTK Data", self.test_nltk_data),
            ("spaCy Models", self.test_spacy_models),
            ("PyTorch", self.test_pytorch),
            ("Transformers", self.test_transformers),
            ("Services", self.test_services),
            ("API Server", self.test_api_server),
            ("Basic Functionality", self.test_basic_functionality),
            ("Comprehensive Test Suite", self.test_comprehensive_test_suite),
        ]
        
        passed_tests = 0
        total_tests = len(test_suites)
        
        for test_name, test_function in test_suites:
            logger.info(f"\n🔍 Running {test_name} tests...")
            try:
                if test_function():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test suite '{test_name}' crashed: {e}")
        
        # Generate summary
        elapsed_time = time.time() - self.start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"Test Summary: {passed_tests}/{total_tests} test suites passed")
        logger.info(f"Individual tests: {sum(self.test_results.values())}/{len(self.test_results)} passed")
        logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! System is ready to use.")
        elif passed_tests >= total_tests - 2:
            logger.warning("⚠️ Most tests passed. System should work with minor issues.")
        else:
            logger.error("❌ Multiple test failures. System may not work correctly.")
        
        # Print failed tests
        failed_tests = [name for name, result in self.test_results.items() if not result]
        if failed_tests:
            logger.info("\nFailed tests:")
            for test in failed_tests:
                logger.info(f"  - {test}")
        
        return self.test_results

    def generate_report(self) -> str:
        """Generate a detailed test report."""
        report = []
        report.append("# System Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        report.append(f"## Summary")
        report.append(f"- Total tests: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {total_tests - passed_tests}")
        report.append(f"- Success rate: {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            report.append(f"- {test_name}: {status}")
        
        return "\n".join(report)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Comprehensive Test Script for Auto-Paraphrasing System')
    parser.add_argument('--skip-models', action='store_true', help='Skip ML model tests')
    parser.add_argument('--skip-api', action='store_true', help='Skip API server tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', '-r', help='Generate report file')
    parser.add_argument('--quick', action='store_true', help='Quick tests only (equivalent to --skip-models --skip-api)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.skip_models = True
        args.skip_api = True
    
    # Create tester instance
    tester = SystemTester(
        skip_models=args.skip_models,
        skip_api=args.skip_api,
        verbose=args.verbose
    )
    
    # Run tests
    results = tester.run_comprehensive_tests()
    
    # Generate report if requested
    if args.report:
        report_content = tester.generate_report()
        with open(args.report, 'w') as f:
            f.write(report_content)
        logger.info(f"Report saved to: {args.report}")
    
    # Exit with appropriate code
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    elif passed_tests >= total_tests - 2:
        sys.exit(1)  # Most tests passed but some issues
    else:
        sys.exit(2)  # Major issues


if __name__ == "__main__":
    main()
