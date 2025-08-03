# Testing Documentation

This document provides comprehensive information about testing the Auto-Paraphrasing System, including unit tests, integration tests, performance tests, and API testing with Postman.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Postman Testing](#postman-testing)
- [Performance Testing](#performance-testing)
- [Continuous Integration](#continuous-integration)
- [Test Data Management](#test-data-management)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The testing suite is designed to ensure the reliability, performance, and correctness of the Auto-Paraphrasing System. It covers all major components including:

- API endpoints (basic and enhanced)
- Service layer functionality
- Database operations
- Error handling and edge cases
- Performance and scalability
- Integration workflows

## 🏗️ Test Structure

```
tests/
├── conftest.py                 # Test configuration and fixtures
├── test_api.py                # Basic API endpoint tests
├── test_enhanced_api.py       # Enhanced API endpoint tests
├── test_integration.py        # End-to-end integration tests
├── test_performance.py        # Performance and load tests
├── test_error_handling.py     # Error scenarios and edge cases
├── test_database.py           # Database operation tests
└── README.md                  # Testing instructions

postman/
├── paraphrase_system_collection.json  # Postman collection
└── environments/
    ├── development.json        # Development environment
    └── production.json         # Production environment

docs/
└── TESTING.md                 # This documentation
```

## 🚀 Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio pytest-cov
   ```

2. **Set Up Test Database**:
   ```bash
   # Tests use SQLite by default
   # No additional setup required
   ```

3. **Environment Setup**:
   ```bash
   # Create test environment file if needed
   cp .env.example .env.test
   ```

### Running All Tests

```bash
# Method 1: Use the test runner script (recommended)
python run_tests.py

# Method 2: Run pytest directly from project root
python -m pytest tests/

# Method 3: Run with coverage
python -m pytest tests/ --cov=app --cov-report=html

# Method 4: Run with verbose output
python -m pytest tests/ -v

# Method 5: Run specific test file
python -m pytest tests/test_api.py

# Method 6: Run specific test function
python -m pytest tests/test_api.py::test_health_check
```

### Running Test Categories

```bash
# Using the test runner script
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance
python run_tests.py --all

# Or using pytest directly
python -m pytest tests/test_api.py tests/test_enhanced_api.py
python -m pytest tests/test_integration.py
python -m pytest tests/ -m performance
python -m pytest tests/test_database.py
python -m pytest tests/test_error_handling.py
```

### Running Tests in Parallel

```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## 📊 Test Categories

### 1. Unit Tests

**Files**: `test_api.py`, `test_enhanced_api.py`

Test individual API endpoints and their responses:

```bash
# Run unit tests
pytest tests/test_api.py tests/test_enhanced_api.py -v
```

**Coverage**:
- All HTTP methods (GET, POST, DELETE)
- Request validation
- Response format validation
- Error status codes
- Authentication (if applicable)

### 2. Integration Tests

**File**: `test_integration.py`

Test complete workflows and service interactions:

```bash
# Run integration tests
pytest tests/test_integration.py -v
```

**Coverage**:
- Document upload → processing → paraphrasing → retrieval
- Multi-step workflows
- Service communication
- Data consistency across operations

### 3. Performance Tests

**File**: `test_performance.py`

Test system performance and scalability:

```bash
# Run performance tests
pytest tests/test_performance.py -v -m performance
```

**Coverage**:
- Response time benchmarks
- Concurrent request handling
- Memory usage patterns
- Database performance
- Cache efficiency

### 4. Error Handling Tests

**File**: `test_error_handling.py`

Test error scenarios and edge cases:

```bash
# Run error handling tests
pytest tests/test_error_handling.py -v
```

**Coverage**:
- Invalid input handling
- Service failure scenarios
- Database error conditions
- Security edge cases
- Resource exhaustion

### 5. Database Tests

**File**: `test_database.py`

Test database operations and data integrity:

```bash
# Run database tests
pytest tests/test_database.py -v
```

**Coverage**:
- CRUD operations
- Transaction handling
- Constraint validation
- Performance optimization
- Migration scenarios

## 🔧 Postman Testing

### Setup

1. **Import Collection**:
   - Open Postman
   - Import `postman/paraphrase_system_collection.json`

2. **Import Environment**:
   - Import `postman/environments/development.json`
   - Set as active environment

3. **Configure Variables**:
   ```
   base_url: http://localhost:8000
   document_id: (will be set automatically)
   session_id: (will be set automatically)
   ```

### Running Postman Tests

1. **Manual Testing**:
   - Execute requests individually
   - Verify responses manually
   - Use for exploratory testing

2. **Automated Collection Run**:
   ```bash
   # Install Newman (Postman CLI)
   npm install -g newman

   # Run collection
   newman run postman/paraphrase_system_collection.json \
     -e postman/environments/development.json \
     --reporters cli,html \
     --reporter-html-export test-results.html
   ```

3. **Continuous Integration**:
   ```yaml
   # GitHub Actions example
   - name: Run API Tests
     run: |
       newman run postman/paraphrase_system_collection.json \
         -e postman/environments/development.json \
         --reporters junit \
         --reporter-junit-export results.xml
   ```

### Test Scenarios

The Postman collection includes:

1. **Health & Status Checks**
2. **Document Management**
   - Upload (basic and enhanced)
   - Retrieve document details
   - List with filtering
   - Delete operations
3. **Text Analysis**
   - NLP analysis
   - Quality assessment
4. **Paraphrasing**
   - All methods (IndoT5, Rule-based, Hybrid)
   - Direct text paraphrasing
   - Document paraphrasing
5. **Session Management**
6. **Demo Endpoints**

## ⚡ Performance Testing

### Metrics Monitored

1. **Response Time**:
   - Average response time < 2s for simple operations
   - Average response time < 10s for complex paraphrasing
   - 95th percentile < 5s for all operations

2. **Throughput**:
   - Minimum 10 requests/second for basic endpoints
   - Minimum 2 requests/second for paraphrasing

3. **Concurrency**:
   - Support 50+ concurrent users
   - No degradation up to 20 concurrent requests

4. **Memory Usage**:
   - Memory increase < 500MB under normal load
   - No memory leaks during extended operation

### Load Testing

```bash
# Using pytest for load testing
pytest tests/test_performance.py::TestStressTest -v

# Using external tools (example with Apache Bench)
ab -n 1000 -c 10 http://localhost:8000/health

# Using wrk for more sophisticated load testing
wrk -t12 -c400 -d30s http://localhost:8000/health
```

### Performance Benchmarks

| Operation | Target Response Time | Concurrency | Success Rate |
|-----------|---------------------|-------------|--------------|
| Health Check | < 100ms | 100 | 99.9% |
| Document Upload | < 2s | 10 | 99% |
| Text Analysis | < 3s | 5 | 98% |
| Paraphrasing | < 10s | 3 | 95% |

## 🔄 Continuous Integration

### GitHub Actions

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml -n auto
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

### Test Configuration

Create `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=app
    --cov-report=term-missing
markers =
    performance: marks tests as performance tests
    integration: marks tests as integration tests
    slow: marks tests as slow running
```

## 📊 Test Data Management

### Test Fixtures

Located in `tests/conftest.py`:

```python
@pytest.fixture
def sample_text():
    return "Penelitian ini menggunakan metode kualitatif."

@pytest.fixture
def sample_document(temp_upload_dir):
    # Creates a temporary test document
    pass

@pytest.fixture
def mock_nlp_service():
    # Mocks NLP service responses
    pass
```

### Test Database

- Tests use SQLite in-memory database
- Fresh database for each test function
- Automatic cleanup after tests
- No impact on production data

### Sample Data

```python
# Academic text samples
SAMPLE_TEXTS = {
    "indonesian_academic": "Penelitian ini menggunakan metode analisis kualitatif...",
    "english_academic": "This research employs qualitative analysis methods...",
    "mixed_content": "Penelitian (research) ini menggunakan mixed-method approach..."
}

# File samples
SAMPLE_FILES = {
    "small_txt": "< 1KB text file",
    "large_txt": "~ 50KB text file", 
    "pdf_document": "Academic PDF with citations",
    "docx_document": "Word document with formatting"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Test Database Errors**:
   ```bash
   # Clear test database
   rm test.db
   
   # Regenerate tables
   python migrate_db.py migrate
   ```

2. **Import Errors**:
   ```bash
   # Method 1: Use the test runner script (recommended)
   python run_tests.py
   
   # Method 2: Add project to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   python -m pytest tests/
   
   # Method 3: Install in development mode
   pip install -e .
   python -m pytest tests/
   
   # Method 4: Set PYTHONPATH for Windows
   set PYTHONPATH=%PYTHONPATH%;.
   python -m pytest tests/
   ```

3. **Mock Service Failures**:
   ```python
   # Ensure all external services are mocked
   @patch('app.services.external_service.method')
   def test_function(mock_service):
       mock_service.return_value = expected_result
   ```

4. **Performance Test Variability**:
   ```bash
   # Run performance tests multiple times
   pytest tests/test_performance.py --count=5
   
   # Use more lenient thresholds in CI
   pytest tests/test_performance.py --benchmark-disable
   ```

5. **Postman Collection Issues**:
   - Verify environment variables are set
   - Check base_url is correct
   - Ensure API server is running
   - Validate JSON syntax in requests

### Debug Mode

```bash
# Run tests with debug output
pytest -v -s --tb=long

# Run single test with debugging
pytest tests/test_api.py::test_health_check -v -s --pdb
```

### Logging During Tests

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_with_logging():
    logger = logging.getLogger(__name__)
    logger.debug("Debug message")
```

## 📈 Test Coverage

### Coverage Goals

- **Overall**: > 85%
- **Critical paths**: > 95%
- **API endpoints**: 100%
- **Service layer**: > 90%
- **Database layer**: > 80%

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# View report
open htmlcov/index.html

# Generate XML for CI
pytest --cov=app --cov-report=xml
```

### Coverage Configuration

Create `.coveragerc`:

```ini
[run]
source = app
omit = 
    */tests/*
    */venv/*
    */migrations/*
    app/__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 🎯 Best Practices

### Test Writing

1. **Test Naming**: Use descriptive names
   ```python
   def test_upload_document_with_valid_pdf_returns_success()
   def test_paraphrase_with_invalid_method_returns_400()
   ```

2. **Test Structure**: Follow AAA pattern
   ```python
   def test_function():
       # Arrange
       setup_test_data()
       
       # Act
       result = function_under_test()
       
       # Assert
       assert result.status_code == 200
   ```

3. **Test Isolation**: Each test should be independent
   ```python
   @pytest.fixture(autouse=True)
   def clean_database():
       # Clean up before each test
       pass
   ```

4. **Mock External Dependencies**:
   ```python
   @patch('app.services.external_api.call')
   def test_with_mocked_service(mock_api):
       mock_api.return_value = {"status": "success"}
   ```

### API Testing

1. **Test All HTTP Methods**
2. **Validate Response Structure**
3. **Test Error Conditions**
4. **Check Status Codes**
5. **Verify Content Types**

### Performance Testing

1. **Set Realistic Benchmarks**
2. **Test Under Load**
3. **Monitor Resource Usage**
4. **Test Scalability Limits**
5. **Identify Bottlenecks**

## 📞 Support

For testing issues or questions:

1. Check this documentation
2. Review test logs and error messages
3. Check GitHub issues for similar problems
4. Create new issue with test details
5. Contact development team

Remember to always run the full test suite before deploying changes to ensure system reliability and stability.
