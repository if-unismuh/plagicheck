"""
Unified API Models
Request and response models for the unified paraphrasing system.
"""
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class FormalityLevel(str, Enum):
    """Formality levels for paraphrasing."""
    ACADEMIC = "academic"
    FORMAL = "formal" 
    NEUTRAL = "neutral"


class ComplexityTarget(str, Enum):
    """Target complexity levels."""
    SIMPLIFY = "simplify"
    MAINTAIN = "maintain"
    ENHANCE = "enhance"


class UnifiedParaphraseRequest(BaseModel):
    """Request model for unified paraphrasing endpoint."""
    
    # Input Methods (mutually exclusive)
    text: Optional[str] = Field(None, description="Direct text input for paraphrasing")
    document_id: Optional[uuid.UUID] = Field(None, description="Previously uploaded document ID")
    
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
    formality_level: FormalityLevel = Field(FormalityLevel.ACADEMIC, description="Target formality level")
    target_complexity: ComplexityTarget = Field(ComplexityTarget.MAINTAIN, description="Target complexity level")
    
    # Method Selection (optional fine-tuning)
    use_indot5: bool = Field(True, description="Enable IndoT5 neural model")
    use_rule_based: bool = Field(True, description="Enable rule-based transformations")
    use_custom_synonyms: bool = Field(True, description="Apply custom synonyms if available")
    custom_synonyms_path: Optional[str] = Field(None, description="Path to custom synonyms.json file")
    
    def validate_input(self):
        """Validate that exactly one input method is provided."""
        if not self.text and not self.document_id:
            raise ValueError("Either 'text' or 'document_id' must be provided")
        if self.text and self.document_id:
            raise ValueError("Only one of 'text' or 'document_id' should be provided")


class QualityDimensions(BaseModel):
    """Quality assessment dimensions."""
    semantic_similarity: float = Field(..., description="Semantic similarity to original")
    grammar_correctness: float = Field(..., description="Grammar and syntax quality")
    academic_tone_preservation: float = Field(..., description="Academic tone maintenance")
    readability_score: float = Field(..., description="Text readability")
    structural_diversity: float = Field(..., description="Structural variation from original")
    context_appropriateness: float = Field(..., description="Context fit assessment")


class QualityAssessment(BaseModel):
    """Comprehensive quality assessment result."""
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    dimension_scores: QualityDimensions = Field(..., description="Individual dimension scores")
    confidence_level: float = Field(..., description="Confidence in assessment (0-1)")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    meets_threshold: bool = Field(..., description="Whether quality meets specified threshold")


class VariantInfo(BaseModel):
    """Information about a paraphrase variant."""
    text: str = Field(..., description="The paraphrased text")
    source_method: str = Field(..., description="Method used to generate this variant")
    confidence: float = Field(..., description="Method-specific confidence score")
    quality_scores: Dict[str, float] = Field(..., description="Quality metrics for this variant")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional variant metadata")


class NLPAnalysisSummary(BaseModel):
    """Summary of NLP analysis results."""
    total_sentences: int = Field(..., description="Total number of sentences")
    overall_readability: float = Field(..., description="Overall readability score")
    overall_complexity: float = Field(..., description="Overall complexity score")
    academic_terms_count: int = Field(..., description="Number of academic terms identified")
    named_entities_count: int = Field(..., description="Number of named entities found")
    high_priority_sentences: int = Field(..., description="Sentences prioritized for paraphrasing")


class MethodContributions(BaseModel):
    """Statistics on method contributions to final result."""
    indot5_variants: int = Field(default=0, description="Variants generated by IndoT5")
    rule_based_variants: int = Field(default=0, description="Variants generated by rule-based")
    custom_synonyms_variants: int = Field(default=0, description="Variants using custom synonyms")
    structural_variants: int = Field(default=0, description="Variants from structural transformations")
    selected_method: str = Field(..., description="Method that generated the selected variant")


class UnifiedParaphraseResponse(BaseModel):
    """Response model for unified paraphrasing endpoint."""
    
    # Core Results
    original_text: str = Field(..., description="Original input text")
    best_variant: str = Field(..., description="Best paraphrase variant selected")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    # Quality Assessment
    quality_assessment: QualityAssessment = Field(..., description="Comprehensive quality evaluation")
    
    # Optional Detailed Information
    all_variants: Optional[List[VariantInfo]] = Field(None, description="All generated variants (if requested)")
    nlp_analysis: Optional[NLPAnalysisSummary] = Field(None, description="NLP analysis summary (if requested)")
    method_insights: Optional[MethodContributions] = Field(None, description="Method contribution details (if requested)")
    
    # Processing Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")
    
    # System Information
    system_info: Dict[str, Any] = Field(
        default_factory=lambda: {
            "version": "2.0.0",
            "unified_system": True,
            "methods_available": ["indot5", "rule_based", "custom_synonyms", "structural"]
        },
        description="System information and capabilities"
    )


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: uuid.UUID = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_preview: str = Field(..., description="Preview of extracted content")
    processing_status: str = Field(..., description="Current processing status")
    upload_timestamp: str = Field(..., description="Upload timestamp")


class ProcessingStatusResponse(BaseModel):
    """Response for processing status check."""
    document_id: uuid.UUID = Field(..., description="Document identifier")
    status: str = Field(..., description="Current processing status")
    progress: float = Field(..., description="Processing progress (0-1)")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class PerformanceStats(BaseModel):
    """System performance statistics."""
    total_requests: int = Field(..., description="Total requests processed")
    average_processing_time: float = Field(..., description="Average processing time in seconds")
    method_usage: Dict[str, int] = Field(..., description="Usage statistics per method")
    success_rate: float = Field(..., description="Success rate percentage")
    current_load: float = Field(..., description="Current system load (0-1)")


class HealthCheckResponse(BaseModel):
    """System health check response."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Individual component status")
    performance: PerformanceStats = Field(..., description="Performance metrics")
    recommendations: List[str] = Field(default_factory=list, description="System recommendations")


# Legacy compatibility models (deprecated but maintained for transition)
class LegacyParaphraseRequest(BaseModel):
    """Legacy request model for backward compatibility."""
    text: str = Field(..., description="Text to paraphrase")
    method: str = Field("unified", description="Paraphrasing method")
    num_variants: int = Field(3, ge=1, le=5)
    
    def to_unified_request(self) -> UnifiedParaphraseRequest:
        """Convert to unified request format."""
        return UnifiedParaphraseRequest(
            text=self.text,
            num_variants=self.num_variants,
            preserve_academic_terms=True,
            preserve_citations=True
        )


class LegacyParaphraseResponse(BaseModel):
    """Legacy response model for backward compatibility."""
    original_text: str
    paraphrased_text: str
    similarity_score: float
    processing_time: float
    method_used: str = "unified"
    
    @classmethod
    def from_unified_response(cls, unified_response: UnifiedParaphraseResponse) -> 'LegacyParaphraseResponse':
        """Convert from unified response format."""
        return cls(
            original_text=unified_response.original_text,
            paraphrased_text=unified_response.best_variant,
            similarity_score=unified_response.quality_assessment.dimension_scores.semantic_similarity,
            processing_time=unified_response.processing_time,
            method_used="unified"
        )
