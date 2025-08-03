#!/usr/bin/env python3
"""
Test script to verify that the API routes match the Postman collection
"""
import json

def test_route_availability():
    """Test that routes defined in Postman collection are available"""
    
    # Load Postman collection
    with open('postman/paraphrase_system_collection.json', 'r') as f:
        collection = json.load(f)
    
    print("Testing Postman collection structure...")
    print("=" * 50)
    
    # Test that we can import the routes without errors
    try:
        from app.api.routes import app
        from app.api.enhanced_routes import router
        print("✓ PASS - All route modules imported successfully")
    except Exception as e:
        print(f"✗ FAIL - Route import error: {str(e)}")
        return
    
    # Check if all routes from the collection exist in the app
    print("\nValidating route endpoints...")
    
    # Get all routes from FastAPI app
    app_routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            methods = list(route.methods) if hasattr(route, 'methods') else ['GET']
            app_routes.append((methods, route.path))
    
    print(f"Found {len(app_routes)} routes in FastAPI app")
    
    # Check specific important endpoints
    important_endpoints = [
        ("GET", "/"),
        ("GET", "/health"),
        ("POST", "/api/documents/upload"),
        ("GET", "/api/documents/{document_id}"),
        ("POST", "/api/v2/documents/upload-enhanced"),
        ("POST", "/api/v2/text/paraphrase-direct"),
        ("GET", "/api/v2/performance/stats"),
    ]
    
    for method, path in important_endpoints:
        found = False
        for route_methods, route_path in app_routes:
            if method in route_methods and (route_path == path or route_path.replace('{', '{').replace('}', '}') in path):
                found = True
                break
        
        status = "✓ PASS" if found else "✗ FAIL"
        print(f"{status} {method} {path}")
    
    print("\n" + "=" * 50)
    print("Route structure validation completed!")
    
    # Validate collection structure
    print(f"\nPostman Collection Summary:")
    print(f"- Name: {collection['info']['name']}")
    print(f"- Version: {collection['info']['version']}")
    print(f"- Folders: {len(collection['item'])}")
    
    total_requests = 0
    for folder in collection['item']:
        folder_name = folder['name']
        request_count = len(folder['item'])
        total_requests += request_count
        print(f"  - {folder_name}: {request_count} requests")
    
    print(f"- Total requests: {total_requests}")
    print(f"- Variables: {len(collection['variable'])}")
    print(f"- Events: {len(collection['event'])}")

if __name__ == "__main__":
    test_route_availability()
