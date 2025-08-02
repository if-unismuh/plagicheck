# Test Suite

This directory contains comprehensive tests for the Auto-Paraphrasing System.

## 📋 Test Files

| File | Description | Coverage |
|------|-------------|----------|
| `conftest.py` | Test configuration and fixtures | Test setup |
| `test_api.py` | Basic API endpoint tests | Core API functionality |
| `test_enhanced_api.py` | Enhanced API endpoint tests | Advanced API features |
| `test_integration.py` | End-to-end workflow tests | Complete user journeys |
| `test_performance.py` | Performance and load tests | System scalability |
| `test_error_handling.py` | Error scenarios and edge cases | System resilience |
| `test_database.py` | Database operation tests | Data persistence |

## 🚀 Quick Start

### Running All Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py
```

### Running by Category
```bash
# Unit tests
pytest -m unit

# Integration tests  
pytest -m integration

# Performance tests
pytest -m performance

# Database tests
pytest -m database
```

## 🧪 Test Categories

### Unit Tests (`test_api.py`, `test_enhanced_api.py`)
- Individual endpoint testing
- Request/response validation
- Input validation
- Error handling

### Integration Tests (`test_integration.py`)
- Complete workflow testing
- Service interaction testing
- Data consistency verification
- Multi-step operations

### Performance Tests (`test_performance.py`)
- Response time benchmarks
- Concurrent request handling
- Memory usage monitoring
- Scalability testing

### Error Handling Tests (`test_error_handling.py`)
- Invalid input scenarios
- Service failure simulation
- Security edge cases
- Resource exhaustion

### Database Tests (`test_database.py`)
- CRUD operations
- Transaction integrity
- Constraint validation
- Performance optimization

## 🔧 Configuration

### Test Database
- Uses SQLite in-memory database
- Fresh database for each test
- No impact on production data

### Test Environment
- Mocked external services
- Isolated test execution
- Configurable test parameters

### Fixtures
Common test fixtures available:
- `client`: FastAPI test client
- `test_db`: Test database session
- `temp_upload_dir`: Temporary file directory
- `sample_text_file`: Sample test files

## 📊 Test Coverage

Target coverage goals:
- **Overall**: > 85%
- **API endpoints**: 100%
- **Service layer**: > 90%
- **Database layer**: > 80%

Generate coverage report:
```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Database Errors**:
   ```bash
   rm test.db  # Clear test database
   ```

3. **Mock Failures**:
   - Ensure all external services are properly mocked
   - Check mock return values match expected format

### Debug Mode
```bash
# Run with verbose output
pytest -v -s

# Run single test with debugging
pytest tests/test_api.py::test_health_check -v -s --pdb
```

## 📈 Performance Benchmarks

| Endpoint | Target Response Time | Success Rate |
|----------|---------------------|--------------|
| Health Check | < 100ms | 99.9% |
| Document Upload | < 2s | 99% |
| Text Analysis | < 3s | 98% |
| Paraphrasing | < 10s | 95% |

## 🎯 Best Practices

### Writing Tests
1. Use descriptive test names
2. Follow AAA pattern (Arrange, Act, Assert)
3. Keep tests independent and isolated
4. Mock external dependencies
5. Test both success and failure scenarios

### Test Data
1. Use realistic test data
2. Include edge cases
3. Test with different text lengths
4. Include various file formats
5. Test special characters and Unicode

### Performance Testing
1. Set realistic benchmarks
2. Test under various load conditions
3. Monitor resource usage
4. Identify performance bottlenecks
5. Test scalability limits

## 📚 Additional Resources

- [Testing Documentation](../docs/TESTING.md)
- [Postman Collection](../postman/)
- [API Documentation](http://localhost:8000/docs)
- [Development Setup](../README.md)

For questions or issues with tests, please check the main testing documentation or create an issue in the project repository.
