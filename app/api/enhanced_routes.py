"""
Enhanced API Routes
Comprehensive endpoints for enhanced document processing and paraphrasing.
"""
import uuid
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import torch

from app.core.database import get_db
from app.models.document import ParaphraseMethod, DocumentStatus
from app.services.enhanced_document_processor import enhanced_document_processor
from app.services.indonesian_nlp_pipeline import indonesian_nlp_pipeline
from app.services.enhanced_indot5_paraphraser import get_enhanced_indot5_paraphraser
from app.services.rule_based_paraphraser import rule_based_paraphraser
from app.services.paraphraser import enhanced_paraphrasing_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Enhanced Processing"])


# Request/Response Models
class EnhancedProcessingRequest(BaseModel):
    """Request model for enhanced document processing."""
    preserve_structure: bool = Field(True, description="Whether to preserve document structure")
    extract_academic_terms: bool = Field(True, description="Whether to extract academic terminology")


class NLPAnalysisRequest(BaseModel):
    """Request model for NLP analysis."""
    text: str = Field(..., description="Text to analyze")
    extract_academic_terms: bool = Field(True, description="Whether to extract academic terms")


class EnhancedParaphraseRequest(BaseModel):
    """Request model for enhanced paraphrasing."""
    document_id: uuid.UUID = Field(..., description="Document ID to paraphrase")
    method: ParaphraseMethod = Field(..., description="Paraphrasing method")
    use_nlp_analysis: bool = Field(True, description="Whether to use NLP analysis")
    preserve_academic_terms: bool = Field(True, description="Whether to preserve academic terms")
    preserve_citations: bool = Field(True, description="Whether to preserve citations")
    num_variants: int = Field(3, ge=1, le=10, description="Number of variants to generate")
    quality_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Quality threshold")


class DirectParaphraseRequest(BaseModel):
    """Request model for direct text paraphrasing."""
    text: str = Field(..., description="Text to paraphrase")
    method: str = Field("indot5", description="Paraphrasing method (indot5, rule_based, hybrid)")
    preserve_academic_terms: bool = Field(True, description="Whether to preserve academic terms")
    preserve_citations: bool = Field(True, description="Whether to preserve citations")
    num_variants: int = Field(3, ge=1, le=5, description="Number of variants to generate")


class TextQualityResponse(BaseModel):
    """Response model for text quality assessment."""
    readability_score: float
    complexity_score: float
    academic_tone_score: float
    grammar_score: float
    overall_quality: float
    issues: List[str]
    recommendations: List[str]


class NLPAnalysisResponse(BaseModel):
    """Response model for NLP analysis."""
    total_sentences: int
    overall_readability: float
    overall_complexity: float
    academic_terms_count: int
    named_entities_count: int
    quality_metrics: Dict[str, Any]
    paraphrasing_priorities: List[int]
    high_priority_sentences: List[Dict[str, Any]]


class ParaphraseVariant(BaseModel):
    """Individual paraphrase variant."""
    text: str
    similarity_score: float
    quality_score: float
    method_used: str


class DirectParaphraseResponse(BaseModel):
    """Response model for direct paraphrasing."""
    original_text: str
    variants: List[ParaphraseVariant]
    best_variant: str
    processing_time: float
    metadata: Dict[str, Any]


# Enhanced Document Processing Endpoints

@router.post("/documents/upload-enhanced")
async def upload_document_enhanced(
    file: UploadFile = File(...),
    chapter: Optional[str] = Form(None),
    preserve_structure: bool = Form(True),
    extract_academic_terms: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Upload and process document with enhanced capabilities.
    
    - **file**: Document file (PDF, DOCX, TXT)
    - **chapter**: Optional chapter designation
    - **preserve_structure**: Whether to preserve document structure
    - **extract_academic_terms**: Whether to extract academic terminology
    """
    try:
        document = await enhanced_document_processor.process_document_enhanced(
            file.file,
            file.filename,
            db,
            chapter=chapter,
            preserve_structure=preserve_structure,
            extract_academic_terms=extract_academic_terms
        )
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "metadata": document.document_metadata
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enhanced document upload failed: {e}")
        raise HTTPException(status_code=500, detail="Document processing failed")


@router.post("/text/analyze", response_model=NLPAnalysisResponse)
async def analyze_text_nlp(request: NLPAnalysisRequest):
    """
    Perform comprehensive NLP analysis on text.
    
    Analyzes text for:
    - Sentence complexity and readability
    - Academic terminology
    - Named entities
    - Paraphrasing priorities
    """
    try:
        analysis = await indonesian_nlp_pipeline.analyze_document(request.text)
        
        # Extract high priority sentences for response
        high_priority = indonesian_nlp_pipeline.extract_sentences_for_paraphrasing(
            analysis, min_priority=0.6, max_sentences=5
        )
        
        high_priority_sentences = [
            {
                "index": idx,
                "text": sentence,
                "priority": analysis.sentences[idx].priority_for_paraphrasing,
                "complexity": analysis.sentences[idx].complexity_score,
                "readability": analysis.sentences[idx].readability_score
            }
            for idx, sentence in high_priority
        ]
        
        return NLPAnalysisResponse(
            total_sentences=len(analysis.sentences),
            overall_readability=analysis.overall_readability,
            overall_complexity=analysis.overall_complexity,
            academic_terms_count=len(analysis.academic_terms),
            named_entities_count=len(analysis.named_entities),
            quality_metrics=analysis.quality_metrics,
            paraphrasing_priorities=analysis.paraphrasing_priorities[:10],
            high_priority_sentences=high_priority_sentences
        )
        
    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        raise HTTPException(status_code=500, detail="NLP analysis failed")


@router.post("/text/quality-assessment", response_model=TextQualityResponse)
async def assess_text_quality(text: str):
    """
    Assess text quality across multiple dimensions.
    
    Evaluates:
    - Readability and complexity
    - Academic tone
    - Grammar quality
    - Overall quality score
    """
    try:
        # Use rule-based paraphraser for quality assessment
        quality_results = await rule_based_paraphraser.paraphrase(
            text, preserve_academic_terms=True, preserve_citations=True, num_variants=1
        )
        
        if quality_results:
            quality = quality_results[0][1]  # Get quality from first result
            
            recommendations = []
            if quality.readability_score < 0.5:
                recommendations.append("Consider simplifying sentence structure")
            if quality.grammar_score < 0.7:
                recommendations.append("Review grammar and spelling")
            if quality.academic_tone_score < 0.5:
                recommendations.append("Use more formal academic language")
            if quality.similarity_score > 0.95:
                recommendations.append("Text may benefit from paraphrasing")
            
            return TextQualityResponse(
                readability_score=quality.readability_score,
                complexity_score=0.5,  # Placeholder
                academic_tone_score=quality.academic_tone_score,
                grammar_score=quality.grammar_score,
                overall_quality=quality.overall_score,
                issues=quality.issues,
                recommendations=recommendations
            )
        else:
            # Fallback quality assessment
            return TextQualityResponse(
                readability_score=0.5,
                complexity_score=0.5,
                academic_tone_score=0.5,
                grammar_score=0.5,
                overall_quality=0.5,
                issues=["Unable to perform detailed assessment"],
                recommendations=["Try enhanced paraphrasing for quality improvement"]
            )
            
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail="Quality assessment failed")


# Enhanced Paraphrasing Endpoints

@router.post("/documents/{document_id}/paraphrase-enhanced")
async def paraphrase_document_enhanced(
    document_id: uuid.UUID,
    request: EnhancedParaphraseRequest,
    db: Session = Depends(get_db)
):
    """
    Paraphrase document using enhanced methods with comprehensive options.
    
    Features:
    - NLP-based sentence prioritization
    - Academic term preservation
    - Quality threshold filtering
    - Multiple variant generation
    """
    try:
        session = await enhanced_paraphrasing_service.paraphrase_document_enhanced(
            document_id=document_id,
            method=request.method,
            db=db,
            use_nlp_analysis=request.use_nlp_analysis,
            preserve_academic_terms=request.preserve_academic_terms,
            preserve_citations=request.preserve_citations,
            num_variants=request.num_variants,
            quality_threshold=request.quality_threshold
        )
        
        if not session:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "message": "Enhanced paraphrasing completed",
            "session_id": session.id,
            "document_id": document_id,
            "method_used": session.method_used,
            "similarity_score": session.similarity_score,
            "processing_time": session.processing_time,
            "enhanced_metadata": session.token_usage
        }
        
    except Exception as e:
        logger.error(f"Enhanced paraphrasing failed: {e}")
        raise HTTPException(status_code=500, detail="Enhanced paraphrasing failed")


@router.post("/text/paraphrase-direct", response_model=DirectParaphraseResponse)
async def paraphrase_text_direct(request: DirectParaphraseRequest):
    """
    Directly paraphrase text without document upload.
    
    Supports multiple methods:
    - **indot5**: Enhanced IndoT5 model with quality filtering
    - **rule_based**: Rule-based paraphrasing with lexical/syntactic transformation
    - **hybrid**: Combination of both methods
    """
    try:
        start_time = time.time()
        variants = []
        metadata = {}
        
        if request.method == "indot5":
            enhanced_indot5_paraphraser = get_enhanced_indot5_paraphraser()
            result = await enhanced_indot5_paraphraser.paraphrase_single(
                request.text, request.num_variants
            )
            
            for i, variant in enumerate(result.paraphrased_variants):
                variants.append(ParaphraseVariant(
                    text=variant,
                    similarity_score=result.similarity_scores[i] if i < len(result.similarity_scores) else 0.0,
                    quality_score=result.quality_scores[i] if i < len(result.quality_scores) else 0.0,
                    method_used="enhanced_indot5"
                ))
            
            best_variant = result.best_variant
            metadata = result.metadata
            
        elif request.method == "rule_based":
            results = await rule_based_paraphraser.paraphrase(
                request.text,
                preserve_academic_terms=request.preserve_academic_terms,
                preserve_citations=request.preserve_citations,
                num_variants=request.num_variants
            )
            
            for variant_text, quality in results:
                variants.append(ParaphraseVariant(
                    text=variant_text,
                    similarity_score=quality.similarity_score,
                    quality_score=quality.overall_score,
                    method_used="rule_based"
                ))
            
            best_variant = results[0][0] if results else request.text
            metadata = {"rule_based_variants": len(results)}
            
        elif request.method == "hybrid":
            # Use both methods and compare
            enhanced_indot5_paraphraser = get_enhanced_indot5_paraphraser()
            indot5_result = await enhanced_indot5_paraphraser.paraphrase_single(
                request.text, request.num_variants
            )
            
            rule_results = await rule_based_paraphraser.paraphrase(
                request.text,
                preserve_academic_terms=request.preserve_academic_terms,
                preserve_citations=request.preserve_citations,
                num_variants=request.num_variants
            )
            
            # Combine results
            for i, variant in enumerate(indot5_result.paraphrased_variants):
                variants.append(ParaphraseVariant(
                    text=variant,
                    similarity_score=indot5_result.similarity_scores[i] if i < len(indot5_result.similarity_scores) else 0.0,
                    quality_score=indot5_result.quality_scores[i] if i < len(indot5_result.quality_scores) else 0.0,
                    method_used="indot5"
                ))
            
            for variant_text, quality in rule_results:
                variants.append(ParaphraseVariant(
                    text=variant_text,
                    similarity_score=quality.similarity_score,
                    quality_score=quality.overall_score,
                    method_used="rule_based"
                ))
            
            # Select best variant
            best_variant = max(variants, key=lambda x: x.quality_score).text if variants else request.text
            metadata = {
                "hybrid_approach": True,
                "indot5_variants": len(indot5_result.paraphrased_variants),
                "rule_based_variants": len(rule_results)
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid paraphrasing method")
        
        processing_time = time.time() - start_time
        
        return DirectParaphraseResponse(
            original_text=request.text,
            variants=variants,
            best_variant=best_variant,
            processing_time=processing_time,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Direct paraphrasing failed: {e}")
        raise HTTPException(status_code=500, detail="Direct paraphrasing failed")


# Performance and Status Endpoints

@router.get("/performance/stats")
async def get_performance_stats():
    """Get performance statistics for all enhanced services."""
    try:
        enhanced_indot5_paraphraser = get_enhanced_indot5_paraphraser()
        stats = {
            "enhanced_indot5": enhanced_indot5_paraphraser.get_performance_stats(),
            "rule_based": rule_based_paraphraser.get_transformation_stats(),
            "system_status": {
                "services_loaded": True,
                "gpu_available": torch.cuda.is_available() if 'torch' in globals() else False,
                "models_ready": {
                    "indot5": enhanced_indot5_paraphraser.model_loaded,
                    "nlp_pipeline": indonesian_nlp_pipeline.nlp_id is not None,
                    "rule_based": True
                }
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance statistics")


@router.post("/performance/clear-cache")
async def clear_performance_cache():
    """Clear all performance caches to free memory."""
    try:
        enhanced_indot5_paraphraser = get_enhanced_indot5_paraphraser()
        enhanced_indot5_paraphraser.clear_cache()
        
        return {
            "message": "Performance caches cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/demo/sample-analysis")
async def demo_sample_analysis():
    """
    Demo endpoint showing sample text analysis capabilities.
    """
    sample_text = """
    Penelitian ini menggunakan metode analisis kualitatif untuk mengkaji dampak teknologi digital 
    terhadap pembelajaran mahasiswa. Data dikumpulkan melalui wawancara mendalam dengan 30 responden 
    yang dipilih secara purposive sampling. Hasil penelitian menunjukkan bahwa teknologi digital 
    memberikan pengaruh signifikan terhadap efektivitas pembelajaran, dengan tingkat kepuasan 
    mahasiswa mencapai 85%. Temuan ini konsisten dengan penelitian sebelumnya (Smith, 2023) 
    yang menyatakan bahwa integrasi teknologi dalam pendidikan meningkatkan engagement dan outcomes.
    """
    
    try:
        # Perform comprehensive analysis
        analysis = await indonesian_nlp_pipeline.analyze_document(sample_text)
        
        # Generate paraphrases with different methods
        indot5_result = await enhanced_indot5_paraphraser.paraphrase_single(sample_text, 2)
        rule_based_results = await rule_based_paraphraser.paraphrase(sample_text, num_variants=2)
        
        return {
            "sample_text": sample_text,
            "nlp_analysis": {
                "total_sentences": len(analysis.sentences),
                "readability_score": analysis.overall_readability,
                "complexity_score": analysis.overall_complexity,
                "academic_terms": list(analysis.academic_terms),
                "named_entities": list(analysis.named_entities)
            },
            "paraphrasing_results": {
                "indot5": {
                    "best_variant": indot5_result.best_variant,
                    "similarity_scores": indot5_result.similarity_scores,
                    "quality_scores": indot5_result.quality_scores
                },
                "rule_based": [
                    {
                        "text": text,
                        "quality_score": quality.overall_score,
                        "similarity_score": quality.similarity_score
                    }
                    for text, quality in rule_based_results
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Demo analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Demo analysis failed")
