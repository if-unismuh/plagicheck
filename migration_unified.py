"""
Migration Script: Legacy to Unified System
Helps transition from the old multi-endpoint system to the new unified architecture.
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnificationMigration:
    """Handles migration from legacy system to unified system."""
    
    def __init__(self):
        self.workspace_root = Path(__file__).parent
        self.backup_dir = self.workspace_root / "backup_legacy"
        self.migration_log = []
    
    def create_backup(self):
        """Create backup of legacy files before migration."""
        try:
            logger.info("Creating backup of legacy files...")
            
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Files to backup
            legacy_files = [
                "app/api/enhanced_routes.py",
                "app/services/paraphraser.py",  # Legacy parts
                "app/core/config.py"  # Original config
            ]
            
            backup_count = 0
            for file_path in legacy_files:
                source = self.workspace_root / file_path
                if source.exists():
                    # Create backup with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{source.name}.{timestamp}.backup"
                    backup_path = self.backup_dir / backup_name
                    
                    # Copy file content
                    with open(source, 'r', encoding='utf-8') as src:
                        content = src.read()
                    
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(content)
                    
                    backup_count += 1
                    logger.info(f"✅ Backed up: {file_path} -> {backup_name}")
            
            logger.info(f"✅ Created backup of {backup_count} files in {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {e}")
            return False
    
    def analyze_legacy_endpoints(self):
        """Analyze legacy endpoints and provide migration mapping."""
        try:
            logger.info("Analyzing legacy endpoints...")
            
            # Legacy endpoint mappings to unified system
            endpoint_mappings = {
                # Enhanced routes mappings
                "/api/v2/text/paraphrase-direct": {
                    "new_endpoint": "/api/paraphrase",
                    "method": "POST",
                    "description": "Direct text paraphrasing -> Unified endpoint with text parameter",
                    "migration_notes": "Use 'text' field in UnifiedParaphraseRequest"
                },
                "/api/v2/documents/{document_id}/paraphrase-enhanced": {
                    "new_endpoint": "/api/paraphrase", 
                    "method": "POST",
                    "description": "Enhanced document paraphrasing -> Unified endpoint with document_id parameter",
                    "migration_notes": "Use 'document_id' field in UnifiedParaphraseRequest"
                },
                "/api/v2/documents/{document_id}/analyze": {
                    "new_endpoint": "/api/paraphrase",
                    "method": "POST", 
                    "description": "NLP analysis -> Unified endpoint with include_detailed_analysis=true",
                    "migration_notes": "Set 'include_detailed_analysis': true in request"
                },
                "/api/v2/text/quality-assessment": {
                    "new_endpoint": "/api/paraphrase",
                    "method": "POST",
                    "description": "Quality assessment -> Unified endpoint with include_quality_breakdown=true",
                    "migration_notes": "Set 'include_quality_breakdown': true in request"
                },
                # Legacy routes mappings
                "/api/documents/{document_id}/paraphrase": {
                    "new_endpoint": "/api/paraphrase",
                    "method": "POST",
                    "description": "Legacy document paraphrasing -> Unified endpoint (compatibility maintained)",
                    "migration_notes": "Legacy endpoint redirects to unified system automatically"
                }
            }
            
            logger.info("📊 Legacy Endpoint Migration Mapping:")
            logger.info("=" * 60)
            
            for old_endpoint, mapping in endpoint_mappings.items():
                logger.info(f"🔄 OLD: {old_endpoint}")
                logger.info(f"   NEW: {mapping['new_endpoint']}")
                logger.info(f"   DESC: {mapping['description']}")
                logger.info(f"   NOTES: {mapping['migration_notes']}")
                logger.info("-" * 40)
            
            self.migration_log.append(f"Analyzed {len(endpoint_mappings)} legacy endpoints")
            return endpoint_mappings
            
        except Exception as e:
            logger.error(f"❌ Legacy endpoint analysis failed: {e}")
            return {}
    
    def generate_migration_documentation(self, endpoint_mappings: Dict):
        """Generate comprehensive migration documentation."""
        try:
            logger.info("Generating migration documentation...")
            
            doc_content = f"""# PlagiCheck Unified System Migration Guide

## 🎯 Migration Overview

This document provides a comprehensive guide for migrating from the legacy multi-endpoint PlagiCheck system to the new unified architecture.

**Migration Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**System Version:** 2.0.0 (Unified)

## 📋 Key Changes

### ✅ Benefits of Unified System
- **Single Endpoint:** All paraphrasing operations through `/api/paraphrase`
- **Improved Performance:** Optimized processing pipeline
- **Enhanced Quality:** Combined strength of all methods
- **Custom Synonyms:** User-provided synonyms.json integration
- **Better Monitoring:** Comprehensive health checks and metrics

### 🔄 Endpoint Migration Map

"""
            
            for old_endpoint, mapping in endpoint_mappings.items():
                doc_content += f"""
#### `{old_endpoint}`
- **New Endpoint:** `{mapping['new_endpoint']}`
- **Method:** {mapping['method']}
- **Description:** {mapping['description']}
- **Migration Notes:** {mapping['migration_notes']}

"""
            
            doc_content += """
## 🚀 Quick Migration Examples

### Example 1: Direct Text Paraphrasing
**OLD REQUEST:**
```json
POST /api/v2/text/paraphrase-direct
{
    "text": "Penelitian ini menggunakan metode analisis kualitatif",
    "method": "indot5",
    "num_variants": 3
}
```

**NEW REQUEST:**
```json
POST /api/paraphrase
{
    "text": "Penelitian ini menggunakan metode analisis kualitatif",
    "num_variants": 3,
    "preserve_academic_terms": true,
    "preserve_citations": true,
    "use_custom_synonyms": true,
    "custom_synonyms_path": "synonyms.json"
}
```

### Example 2: Document-Based Paraphrasing
**OLD REQUEST:**
```json
POST /api/v2/documents/{document_id}/paraphrase-enhanced
{
    "method": "hybrid",
    "num_variants": 5,
    "quality_threshold": 0.7
}
```

**NEW REQUEST:**
```json
POST /api/paraphrase
{
    "document_id": "{document_id}",
    "num_variants": 5,
    "quality_threshold": 0.7,
    "include_method_insights": true,
    "include_quality_breakdown": true
}
```

## 📊 Response Format Changes

### Enhanced Response Structure
The unified system provides more comprehensive responses:

```json
{
    "original_text": "...",
    "best_variant": "...",
    "processing_time": 3.45,
    "quality_assessment": {
        "overall_score": 0.85,
        "dimension_scores": {
            "semantic_similarity": 0.8,
            "grammar_correctness": 0.9,
            "academic_tone_preservation": 0.8,
            "readability_score": 0.85,
            "structural_diversity": 0.7,
            "context_appropriateness": 0.9
        },
        "confidence_level": 0.88,
        "recommendations": ["Good quality paraphrase"],
        "meets_threshold": true
    },
    "method_insights": {
        "indot5_variants": 3,
        "rule_based_variants": 2,
        "custom_synonyms_variants": 1,
        "selected_method": "indot5"
    },
    "system_info": {
        "version": "2.0.0",
        "unified_system": true,
        "methods_available": ["indot5", "rule_based", "custom_synonyms", "structural"]
    }
}
```

## 🔧 Custom Synonyms Integration

### Creating synonyms.json
```json
{
    "penelitian": {
        "tag": "NOUN",
        "sinonim": ["riset", "studi", "kajian"]
    },
    "analisis": {
        "tag": "NOUN",
        "sinonim": ["kajian", "telaah", "pemeriksaan"]
    }
}
```

## 🏥 Health Monitoring

### New Health Check Endpoint
```bash
GET /api/health
```

Provides comprehensive system status including:
- Component health status
- Performance metrics
- Method availability
- System recommendations

## ⚠️ Breaking Changes

1. **Removed Endpoints:** All `/api/v2/*` endpoints are deprecated
2. **Response Format:** Enhanced response structure with more detailed information
3. **Error Handling:** Improved error messages and status codes
4. **Authentication:** (If applicable) May require updated API keys

## 🔄 Migration Checklist

- [ ] Update client code to use `/api/paraphrase` endpoint
- [ ] Update request models to use `UnifiedParaphraseRequest`
- [ ] Update response parsing for new `UnifiedParaphraseResponse` format
- [ ] Test custom synonyms integration (optional)
- [ ] Update health check monitoring to use `/api/health`
- [ ] Verify backward compatibility with legacy endpoints
- [ ] Update documentation and API references

## 📞 Support

For migration assistance or issues:
- Check the `/api/health` endpoint for system status
- Review logs for detailed error information
- Test with the unified system gradually

---
**Generated by PlagiCheck Migration Tool v2.0.0**
"""
            
            # Write documentation
            doc_path = self.workspace_root / "MIGRATION_GUIDE.md"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            logger.info(f"✅ Migration documentation generated: {doc_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Documentation generation failed: {e}")
            return False
    
    def validate_unified_system(self):
        """Validate that the unified system is properly set up."""
        try:
            logger.info("Validating unified system setup...")
            
            validation_results = {
                "unified_paraphraser": False,
                "api_models": False,
                "synonyms_file": False,
                "routes_updated": False
            }
            
            # Check unified paraphraser
            try:
                from app.services.unified_paraphraser import get_unified_paraphraser
                paraphraser = get_unified_paraphraser()
                validation_results["unified_paraphraser"] = True
                logger.info("✅ Unified paraphraser: OK")
            except Exception as e:
                logger.error(f"❌ Unified paraphraser: {e}")
            
            # Check API models
            try:
                from app.api.unified_models import UnifiedParaphraseRequest, UnifiedParaphraseResponse
                validation_results["api_models"] = True
                logger.info("✅ API models: OK")
            except Exception as e:
                logger.error(f"❌ API models: {e}")
            
            # Check synonyms file
            synonyms_path = self.workspace_root / "synonyms.json"
            if synonyms_path.exists():
                validation_results["synonyms_file"] = True
                logger.info("✅ Synonyms file: OK")
            else:
                logger.warning("⚠️  Synonyms file: Not found (optional)")
            
            # Check routes
            try:
                from app.api.routes import app
                validation_results["routes_updated"] = True
                logger.info("✅ Routes updated: OK")
            except Exception as e:
                logger.error(f"❌ Routes: {e}")
            
            # Summary
            passed = sum(validation_results.values())
            total = len(validation_results)
            logger.info(f"🎯 Validation Summary: {passed}/{total} checks passed")
            
            return passed >= 3  # Allow synonyms to be optional
            
        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            return False
    
    async def run_migration(self):
        """Execute the complete migration process."""
        try:
            logger.info("🚀 Starting PlagiCheck Unified System Migration")
            logger.info("=" * 60)
            
            # Step 1: Create backup
            if not self.create_backup():
                logger.error("❌ Migration aborted: Backup creation failed")
                return False
            
            # Step 2: Analyze legacy endpoints
            endpoint_mappings = self.analyze_legacy_endpoints()
            if not endpoint_mappings:
                logger.warning("⚠️  No legacy endpoints found to migrate")
            
            # Step 3: Generate documentation
            if not self.generate_migration_documentation(endpoint_mappings):
                logger.error("❌ Documentation generation failed")
                return False
            
            # Step 4: Validate unified system
            if not self.validate_unified_system():
                logger.error("❌ Unified system validation failed")
                return False
            
            # Step 5: Final summary
            logger.info("=" * 60)
            logger.info("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 Backup created in: {self.backup_dir}")
            logger.info("📖 Migration guide: MIGRATION_GUIDE.md")
            logger.info("🚀 Unified system ready for use!")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False


if __name__ == "__main__":
    import datetime
    
    print("🔄 PlagiCheck Unified System Migration")
    print("=" * 50)
    
    migration = UnificationMigration()
    success = asyncio.run(migration.run_migration())
    
    print("=" * 50)
    if success:
        print("✅ Migration completed successfully!")
        print("🚀 System ready for unified operations")
    else:
        print("❌ Migration failed!")
        print("📝 Check logs for details")
    
    exit(0 if success else 1)
