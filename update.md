# PlagiCheck Unified System Refactoring - Cursor AI Agent Prompt

## 🎯 **OBJECTIVE: Complete System Unification**

Transform the existing multi-method PlagiCheck paraphrasing system into a single, unified, high-performance solution that consolidates all existing approaches (Enhanced IndoT5, Rule-based, Hybrid, Indonesian NLP Pipeline) into one optimized endpoint with custom synonyms integration.

## 📋 **SYSTEM ANALYSIS - Current Architecture**

### **Existing Components to Consolidate:**
1. **Enhanced IndoT5 Paraphraser** (`app/services/enhanced_indot5_paraphraser.py`)
   - GPU-optimized neural model with CUDA support
   - Batch processing capabilities with quality filtering
   - Sentence transformer integration for semantic similarity
   - Performance caching and memory management

2. **Rule-Based Paraphraser** (`app/services/rule_based_paraphraser.py`)
   - Indonesian synonym dictionary with POS tagging
   - Syntactic transformation engine (active-passive, clause reordering)
   - Academic term preservation with regex patterns
   - Grammar validation using LanguageTool integration

3. **Indonesian NLP Pipeline** (`app/services/indonesian_nlp_pipeline.py`)
   - spaCy + UDPipe integration for comprehensive analysis
   - Academic terminology detection and extraction
   - Named Entity Recognition with context awareness
   - Sentence complexity and readability assessment

4. **Enhanced Document Processor** (`app/services/enhanced_document_processor.py`)
   - Multi-format document parsing (PDF, DOCX, TXT)
   - Document structure preservation and reconstruction
   - Citation format detection and protection
   - Academic formatting maintenance

5. **Multiple API Endpoints** (`app/api/routes.py`, `app/api/enhanced_routes.py`)
   - 15+ scattered endpoints with method-specific routing
   - Inconsistent response formats across methods
   - Redundant validation and error handling logic

## 🚀 **TARGET ARCHITECTURE - Unified System**

### **Core Unification Requirements:**

#### **1. Single Unified Service Class**
Create `app/services/unified_paraphraser.py` with the following specifications:

```python
class UnifiedParaphraser:
    """
    Consolidated paraphrasing engine integrating all existing methodologies
    with custom synonyms dictionary integration and comprehensive quality assessment.
    """
    
    def __init__(self, custom_synonyms_path: str):
        # Initialize ALL existing components
        self.indot5_engine = EnhancedIndoT5Paraphraser()
        self.rule_based_engine = HybridRuleBasedParaphraser()
        self.nlp_pipeline = IndonesianNLPPipeline()
        self.document_processor = EnhancedDocumentProcessor()
        
        # Load and integrate custom synonyms dictionary
        self.custom_synonyms = self._load_custom_synonyms(custom_synonyms_path)
        self._merge_synonym_databases()
        
        # Initialize unified quality assessment framework
        self.quality_assessor = UnifiedQualityAssessment()
        
    async def unified_paraphrase(self, input_data: UnifiedInput) -> UnifiedResult:
        """
        Master paraphrasing method consolidating all existing approaches
        """
        # PHASE 1: Comprehensive Text Analysis
        analysis_result = await self._comprehensive_analysis(input_data.text)
        
        # PHASE 2: Multi-Layer Protection Strategy
        protected_content = await self._apply_comprehensive_protection(
            input_data.text, analysis_result
        )
        
        # PHASE 3: Multi-Method Variant Generation
        variant_candidates = await self._generate_unified_variants(
            protected_content, analysis_result, input_data.options
        )
        
        # PHASE 4: Advanced Quality Assessment & Selection
        optimized_result = await self._select_optimal_variant(
            variant_candidates, analysis_result, input_data.quality_criteria
        )
        
        # PHASE 5: Content Restoration & Formatting
        final_output = await self._restore_and_format(
            optimized_result, protected_content.protection_map
        )
        
        return final_output
```

#### **2. Custom Synonyms Integration System**
Implement seamless integration for user-provided synonyms.json with the following structure:

```json
{
    "word": {
        "tag": "pos_tag",
        "sinonim": ["synonym1", "synonym2", "synonym3"]
    }
}
```

**Integration Requirements:**
- Load custom synonyms with high priority scoring (confidence: 0.95)
- Merge with existing Indonesian synonym databases from rule-based system
- Implement context-aware synonym selection using NLP analysis
- Maintain grammatical consistency through POS tag validation
- Preserve academic terminology integrity during substitution

#### **3. Unified API Endpoint Architecture**
Completely restructure API to single endpoint:

**Target Endpoint Structure:**
```python
@router.post("/api/paraphrase", response_model=UnifiedParaphraseResponse)
async def unified_paraphrase_endpoint(
    request: UnifiedParaphraseRequest,
    db: Session = Depends(get_db)
):
    """
    Single comprehensive endpoint handling all paraphrasing scenarios:
    - Direct text input processing
    - Document-based paraphrasing
    - Automatic method selection and optimization
    - Comprehensive quality assessment and reporting
    """
```

**Request Model Specifications:**
```python
class UnifiedParaphraseRequest(BaseModel):
    # Input Methods (mutually exclusive)
    text: Optional[str] = Field(None, description="Direct text input for paraphrasing")
    document_id: Optional[UUID] = Field(None, description="Previously uploaded document ID")
    
    # Processing Configuration
    num_variants: int = Field(3, ge=1, le=5, description="Number of paraphrase variants to generate")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum acceptable quality score")
    preserve_academic_terms: bool = Field(True, description="Maintain academic terminology integrity")
    preserve_citations: bool = Field(True, description="Protect citation formats and references")
    
    # Output Customization
    include_detailed_analysis: bool = Field(False, description="Include comprehensive NLP analysis")
    include_quality_breakdown: bool = Field(False, description="Provide detailed quality metrics")
    include_method_insights: bool = Field(False, description="Show which methods contributed to result")
    
    # Advanced Options
    formality_level: str = Field("academic", enum=["academic", "formal", "neutral"])
    target_complexity: str = Field("maintain", enum=["simplify", "maintain", "enhance"])
```

## 🔧 **IMPLEMENTATION ROADMAP**

### **Phase 1: Core Service Unification**

#### **Task 1.1: Create Unified Paraphraser Service**
- Create `app/services/unified_paraphraser.py`
- Implement comprehensive initialization loading all existing engines
- Design unified input/output data structures
- Implement master paraphrasing orchestration logic

#### **Task 1.2: Synonym Dictionary Integration**
- Implement custom synonyms.json loader with validation
- Create synonym database merger for existing + custom dictionaries
- Develop context-aware synonym selection algorithm
- Implement POS-based grammatical consistency validation

#### **Task 1.3: Multi-Method Variant Generation**
```python
async def _generate_unified_variants(self, protected_text, analysis, options):
    """
    Generate variants using ALL existing methods in parallel:
    """
    variant_pool = []
    
    # Method 1: Enhanced IndoT5 Neural Processing
    indot5_variants = await self.indot5_engine.paraphrase_single(
        protected_text, options.num_variants
    )
    variant_pool.extend(self._format_indot5_variants(indot5_variants))
    
    # Method 2: Advanced Rule-Based Transformation
    rule_based_variants = await self.rule_based_engine.paraphrase(
        protected_text, 
        preserve_academic_terms=options.preserve_academic_terms,
        preserve_citations=options.preserve_citations,
        num_variants=options.num_variants
    )
    variant_pool.extend(self._format_rule_based_variants(rule_based_variants))
    
    # Method 3: Custom Synonyms Application
    custom_synonym_variants = await self._apply_custom_synonyms(
        protected_text, analysis, options
    )
    variant_pool.extend(custom_synonym_variants)
    
    # Method 4: Hybrid Structural Transformation
    structural_variants = await self._apply_structural_transformations(
        protected_text, analysis
    )
    variant_pool.extend(structural_variants)
    
    return variant_pool
```

### **Phase 2: Quality Assessment Consolidation**

#### **Task 2.1: Unified Quality Framework**
Implement comprehensive quality assessment combining all existing metrics:

```python
class UnifiedQualityAssessment:
    def assess_comprehensive_quality(self, original, variant, analysis):
        """
        Multi-dimensional quality assessment consolidating:
        - Semantic similarity (from Enhanced IndoT5)
        - Grammar correctness (from Rule-based LanguageTool)
        - Academic tone preservation (from Rule-based academic assessor)
        - Readability maintenance (from NLP Pipeline)
        - Structural diversity (new metric)
        - Context appropriateness (new metric)
        """
        quality_dimensions = {
            'semantic_similarity': self._calculate_semantic_similarity(original, variant),
            'grammar_correctness': self._assess_grammar_quality(variant),
            'academic_tone_preservation': self._evaluate_academic_tone(variant),
            'readability_score': self._calculate_readability(variant),
            'structural_diversity': self._measure_structural_changes(original, variant),
            'context_appropriateness': self._evaluate_context_fit(variant, analysis)
        }
        
        # Weighted comprehensive scoring
        weights = {
            'semantic_similarity': 0.25,
            'grammar_correctness': 0.20,
            'academic_tone_preservation': 0.20,
            'readability_score': 0.15,
            'structural_diversity': 0.10,
            'context_appropriateness': 0.10
        }
        
        overall_score = sum(
            quality_dimensions[dimension] * weights[dimension]
            for dimension in quality_dimensions
        )
        
        return QualityAssessmentResult(
            overall_score=overall_score,
            dimension_scores=quality_dimensions,
            confidence_level=self._calculate_confidence(quality_dimensions),
            recommendations=self._generate_quality_recommendations(quality_dimensions)
        )
```

### **Phase 3: API Endpoint Consolidation**

#### **Task 3.1: Single Endpoint Implementation**
- Remove all existing paraphrasing endpoints from `app/api/enhanced_routes.py`
- Implement unified endpoint in `app/api/routes.py`
- Design flexible request/response models
- Implement comprehensive error handling and validation

#### **Task 3.2: Backward Compatibility Handler**
```python
# Optional: Maintain backward compatibility with deprecated endpoints
@router.post("/api/documents/{document_id}/paraphrase", deprecated=True)
async def legacy_document_paraphrase(document_id: UUID, legacy_request):
    """
    Legacy endpoint redirecting to unified system
    """
    unified_request = UnifiedParaphraseRequest(
        document_id=document_id,
        # Map legacy parameters to unified format
    )
    return await unified_paraphrase_endpoint(unified_request)
```

### **Phase 4: System Optimization & Cleanup**

#### **Task 4.1: Performance Optimization**
- Implement intelligent caching strategy across all methods
- Optimize memory usage through selective model loading
- Implement parallel processing for variant generation
- Add comprehensive performance monitoring

#### **Task 4.2: Code Cleanup & Consolidation**
**Files to Remove/Consolidate:**
- `app/api/enhanced_routes.py` → Merge essential functionality into `app/api/routes.py`
- Redundant endpoint handlers in `app/api/routes.py`
- Method-specific configuration options in `app/core/config.py`
- Duplicate utility functions across service files

**Configuration Simplification:**
```python
# Simplified configuration focusing on unified system
UNIFIED_PARAPHRASE_CONFIG = {
    'custom_synonyms_path': 'synonyms.json',
    'quality_threshold_default': 0.7,
    'max_variants_per_request': 5,
    'enable_gpu_acceleration': True,
    'enable_performance_caching': True,
    'academic_focus_mode': True,
    'preserve_formatting': True
}
```

## 📊 **QUALITY ASSURANCE & TESTING**

### **Validation Requirements:**
1. **Functional Testing**: Ensure unified system produces quality equal to or better than existing methods
2. **Performance Benchmarking**: Verify response times meet or exceed current performance
3. **Integration Testing**: Validate custom synonyms.json integration works correctly
4. **Regression Testing**: Ensure no loss of existing functionality
5. **Load Testing**: Confirm system handles concurrent requests efficiently

### **Success Metrics:**
- **API Simplification**: Reduction from 15+ endpoints to 3 core endpoints
- **Code Reduction**: Minimum 40% reduction in total codebase complexity
- **Performance Improvement**: Maintain or improve current response times
- **Quality Enhancement**: Improved paraphrase quality through method combination
- **Maintainability**: Simplified architecture for easier future enhancements

## 🔄 **MIGRATION STRATEGY**

### **Implementation Phases:**
1. **Phase 1**: Create unified service alongside existing services (parallel development)
2. **Phase 2**: Implement unified API endpoint with comprehensive testing
3. **Phase 3**: Validate unified system performance and quality
4. **Phase 4**: Deprecate old endpoints and remove redundant code
5. **Phase 5**: Update documentation and deployment configurations

### **Risk Mitigation:**
- Maintain existing endpoints during transition period
- Implement feature flags for gradual rollout
- Comprehensive logging for debugging during migration
- Rollback plan to previous system if critical issues arise

## 🎯 **EXPECTED OUTCOMES**

### **Technical Benefits:**
- **Simplified Architecture**: Single paraphrasing service instead of multiple scattered services
- **Enhanced Performance**: Optimized processing pipeline with intelligent method selection
- **Improved Quality**: Combined strength of all existing methods plus custom synonyms
- **Reduced Maintenance**: Consolidated codebase with clear separation of concerns
- **Better Scalability**: Unified caching and optimization strategies

### **User Experience Benefits:**
- **API Simplicity**: One endpoint for all paraphrasing needs
- **Consistent Responses**: Standardized output format across all scenarios
- **Enhanced Customization**: Flexible options for different use cases
- **Improved Reliability**: Robust error handling and fallback mechanisms

### **Development Benefits:**
- **Code Clarity**: Clear, well-documented unified service architecture
- **Testing Simplification**: Centralized testing strategy for all paraphrasing functionality
- **Feature Development**: Easier to add new features to consolidated system
- **Performance Monitoring**: Unified metrics and monitoring across all methods

## 🚀 **IMPLEMENTATION COMMANDS**

Execute this comprehensive refactoring to transform PlagiCheck into a powerful, unified paraphrasing system that leverages the best of all existing methods while providing a clean, simple API interface. The result will be a more maintainable, performant, and user-friendly system that preserves all current functionality while dramatically simplifying the architecture.
