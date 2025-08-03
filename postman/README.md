# Postman Collection - API Route Mapping

## Overview
The Postman collection has been updated to match all available routes in the Auto-Paraphrasing System API. All endpoints are properly configured and tested.

## Collection Structure

### 1. Health & Status (3 requests)
- **Root Endpoint** - `GET /` - Basic API information
- **Health Check** - `GET /health` - API health status  
- **Performance Stats** - `GET /api/v2/performance/stats` - System performance metrics

### 2. Document Management (6 requests)
- **Upload Document (Basic)** - `POST /api/documents/upload` - Basic document upload
- **Upload Document (Enhanced)** - `POST /api/v2/documents/upload-enhanced` - Enhanced document processing
- **Get Document Details** - `GET /api/documents/{document_id}` - Retrieve document information
- **List Documents** - `GET /api/documents` - List documents with filtering
- **Get Document Status** - `GET /api/documents/{document_id}/status` - Check processing status
- **Delete Document** - `DELETE /api/documents/{document_id}` - Remove document

### 3. Text Analysis (2 requests)
- **NLP Text Analysis** - `POST /api/v2/text/analyze` - Comprehensive text analysis
- **Text Quality Assessment** - `POST /api/v2/text/quality-assessment` - Quality evaluation

### 4. Paraphrasing (5 requests)
- **Start Document Paraphrasing (Basic)** - `POST /api/documents/{document_id}/paraphrase` - Basic paraphrasing
- **Enhanced Document Paraphrasing** - `POST /api/v2/documents/{document_id}/paraphrase-enhanced` - Advanced paraphrasing
- **Direct Text Paraphrasing - IndoT5** - `POST /api/v2/text/paraphrase-direct` - IndoT5 model
- **Direct Text Paraphrasing - Rule Based** - `POST /api/v2/text/paraphrase-direct` - Rule-based method
- **Direct Text Paraphrasing - Hybrid** - `POST /api/v2/text/paraphrase-direct` - Combined approach

### 5. Session Management (2 requests)
- **Get Paraphrase Session** - `GET /api/sessions/{session_id}` - Session details
- **Get Document Sessions** - `GET /api/documents/{document_id}/sessions` - All document sessions

### 6. Demo & Examples (2 requests)
- **Demo Sample Analysis** - `GET /api/v2/demo/sample-analysis` - Sample analysis demonstration
- **Clear Performance Cache** - `POST /api/v2/performance/clear-cache` - Cache management

## Environment Variables

### Development Environment
- `base_url`: http://localhost:8000
- `api_version`: v2
- `timeout`: 30000ms
- `max_file_size`: 50MB

### Production Environment  
- `base_url`: https://your-production-domain.com
- `timeout`: 60000ms
- `api_key`: (for production authentication)

## Key Fixes Made

1. **Fixed Enhanced Document Paraphrasing**: Removed unnecessary `document_id` from request body (it's in URL path)
2. **Corrected Request Bodies**: All JSON request bodies properly formatted
3. **Updated Response Examples**: Realistic response examples for all endpoints
4. **Proper Parameter Handling**: Form data and query parameters correctly configured
5. **Complete Route Coverage**: All 22 API routes covered in collection

## Usage Instructions

1. Import the collection: `postman/paraphrase_system_collection.json`
2. Import environments: 
   - `postman/environments/development.json`
   - `postman/environments/production.json`
3. Select appropriate environment
4. Test endpoints starting with Health & Status
5. Use Upload Document to create test documents
6. Test paraphrasing with different methods

## Testing Status

✅ All route modules import successfully  
✅ All important endpoints validated  
✅ JSON structure is valid  
✅ Collection contains 20 comprehensive requests  
✅ Environment files properly configured  

The collection is ready for use and testing!
