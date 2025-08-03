"""
FastAPI Routes and Application Setup - Unified System
"""
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db, create_tables
from app.models.document import Document, ParaphraseSession, DocumentStatus, ParaphraseMethod
from app.services.document_processor import document_processor
from app.services.paraphraser import paraphrasing_service
from app.services.unified_paraphraser import get_unified_paraphraser, UnifiedInput, UnifiedOptions, QualityCriteria
from app.api.unified_models import (
    UnifiedParaphraseRequest, UnifiedParaphraseResponse, 
    DocumentUploadResponse, ProcessingStatusResponse,
    HealthCheckResponse, PerformanceStats,
    LegacyParaphraseRequest, LegacyParaphraseResponse,
    QualityAssessment, QualityDimensions, VariantInfo,
    NLPAnalysisSummary, MethodContributions
)

logger = logging.getLogger(__name__)

# Legacy Pydantic models for API (maintained for backward compatibility)
class DocumentResponse(BaseModel):
    """Response model for document data."""
    id: uuid.UUID
    filename: str
    chapter: Optional[str]
    status: DocumentStatus
    upload_date: datetime
    processed_date: Optional[datetime]
    metadata: Optional[dict]
    
    class Config:
        from_attributes = True


class DocumentDetailResponse(DocumentResponse):
    """Detailed response model including content."""
    original_content: str
    paraphrased_content: Optional[str]


class ParaphraseSessionResponse(BaseModel):
    """Response model for paraphrase session data."""
    id: uuid.UUID
    document_id: uuid.UUID
    method_used: ParaphraseMethod
    similarity_score: Optional[float]
    processing_time: Optional[int]
    token_usage: Optional[dict]
    
    class Config:
        from_attributes = True


# Create FastAPI application
app = FastAPI(
    title="PlagiCheck - Unified Paraphrasing System", 
    description="Advanced Indonesian academic text paraphrasing with unified methodology",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Legacy models for backward compatibility
class ParaphraseRequest(BaseModel):
    """Request model for paraphrasing."""
    method: ParaphraseMethod


class StatusResponse(BaseModel):
    """Response model for status checks.""" 
    status: DocumentStatus
    message: str
    progress: Optional[float] = None

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables and required directories."""
    try:
        # Import models to ensure they are registered with SQLAlchemy
        from app.models.document import Document, ParaphraseSession  # noqa: F401
        
        # Create database tables
        create_tables()
        
        # Create upload directory
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        
        logger.info("Application startup completed successfully")
        logger.info(f"Upload directory: {upload_dir}")
        logger.info("Unified paraphrasing system ready")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


# =====================================================
# UNIFIED PARAPHRASING ENDPOINTS - NEW ARCHITECTURE
# =====================================================

@app.post("/api/paraphrase", response_model=UnifiedParaphraseResponse)
async def unified_paraphrase_endpoint(
    request: UnifiedParaphraseRequest,
    db: Session = Depends(get_db)
):
    """
    🚀 **UNIFIED PARAPHRASING ENDPOINT**
    
    Single comprehensive endpoint handling all paraphrasing scenarios:
    - Direct text input processing
    - Document-based paraphrasing  
    - Automatic method selection and optimization
    - Comprehensive quality assessment and reporting
    
    **Features:**
    - ✅ Consolidates all existing paraphrasing methods
    - ✅ Custom synonyms integration
    - ✅ Advanced quality assessment
    - ✅ Intelligent method selection
    - ✅ Academic content preservation
    - ✅ Comprehensive error handling
    """
    try:
        # Validate input
        request.validate_input()
        
        # Get or initialize unified paraphraser
        unified_paraphraser = get_unified_paraphraser(request.custom_synonyms_path)
        
        # Prepare input data
        input_text = ""
        
        if request.text:
            # Direct text input
            input_text = request.text
        elif request.document_id:
            # Document-based input
            document = document_processor.get_document(request.document_id, db)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            if not document.original_content:
                raise HTTPException(status_code=400, detail="Document has no content")
            input_text = document.original_content
        
        # Configure processing options
        options = UnifiedOptions(
            num_variants=request.num_variants,
            preserve_academic_terms=request.preserve_academic_terms,
            preserve_citations=request.preserve_citations,
            formality_level=request.formality_level,
            target_complexity=request.target_complexity,
            use_indot5=request.use_indot5,
            use_rule_based=request.use_rule_based,
            use_custom_synonyms=request.use_custom_synonyms
        )
        
        # Configure quality criteria
        quality_criteria = QualityCriteria(
            quality_threshold=request.quality_threshold,
            require_academic_tone=request.preserve_academic_terms
        )
        
        # Create unified input
        unified_input = UnifiedInput(
            text=input_text,
            options=options,
            quality_criteria=quality_criteria
        )
        
        # Execute unified paraphrasing
        logger.info(f"Processing unified paraphrase request for {len(input_text)} characters")
        result = await unified_paraphraser.unified_paraphrase(unified_input)
        
        # Build response
        response_data = {
            "original_text": input_text,
            "best_variant": result.best_variant,
            "processing_time": result.processing_time,
            "quality_assessment": QualityAssessment(
                overall_score=result.quality_assessment.overall_score,
                dimension_scores=QualityDimensions(**result.quality_assessment.dimension_scores),
                confidence_level=result.quality_assessment.confidence_level,
                recommendations=result.quality_assessment.recommendations,
                meets_threshold=result.quality_assessment.meets_threshold
            ),
            "metadata": result.metadata
        }
        
        # Add optional detailed information if requested
        if request.include_detailed_analysis and hasattr(result, 'nlp_analysis'):
            response_data["nlp_analysis"] = NLPAnalysisSummary(
                total_sentences=len(result.nlp_analysis.sentences) if result.nlp_analysis else 0,
                overall_readability=result.nlp_analysis.overall_readability if result.nlp_analysis else 0.5,
                overall_complexity=result.nlp_analysis.overall_complexity if result.nlp_analysis else 0.5,
                academic_terms_count=len(result.nlp_analysis.academic_terms) if result.nlp_analysis else 0,
                named_entities_count=len(result.nlp_analysis.named_entities) if result.nlp_analysis else 0,
                high_priority_sentences=0  # Would be calculated from analysis
            )
        
        if request.include_method_insights:
            response_data["method_insights"] = MethodContributions(
                indot5_variants=result.method_contributions.get('indot5', 0),
                rule_based_variants=result.method_contributions.get('rule_based', 0),
                custom_synonyms_variants=result.method_contributions.get('custom_synonyms', 0),
                structural_variants=result.method_contributions.get('structural', 0),
                selected_method=result.all_variants[0].source_method if result.all_variants else "fallback"
            )
        
        if request.include_quality_breakdown:
            response_data["all_variants"] = [
                VariantInfo(
                    text=variant.text,
                    source_method=variant.source_method,
                    confidence=variant.confidence,
                    quality_scores=variant.quality_scores,
                    metadata=variant.metadata
                )
                for variant in result.all_variants
            ]
        
        logger.info(f"Unified paraphrasing completed in {result.processing_time:.2f}s")
        return UnifiedParaphraseResponse(**response_data)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unified paraphrasing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """
    System health check and performance monitoring.
    """
    try:
        # Get unified paraphraser stats
        unified_paraphraser = get_unified_paraphraser()
        stats = unified_paraphraser.get_performance_stats()
        
        # Check component health
        components = {
            "database": "healthy",
            "unified_paraphraser": "healthy",
            "nlp_pipeline": "healthy",
            "document_processor": "healthy"
        }
        
        # Check IndoT5 availability
        try:
            indot5_engine = unified_paraphraser._get_indot5_engine()
            components["indot5_model"] = "healthy" if indot5_engine else "unavailable"
        except Exception:
            components["indot5_model"] = "error"
        
        # Check rule-based engine
        try:
            rule_engine = unified_paraphraser._get_rule_based_engine()
            components["rule_based_engine"] = "healthy" if rule_engine else "unavailable"
        except Exception:
            components["rule_based_engine"] = "error"
        
        # Overall status
        status = "healthy"
        if any(comp == "error" for comp in components.values()):
            status = "degraded"
        elif any(comp == "unavailable" for comp in components.values()):
            status = "partial"
        
        # Performance stats
        performance = PerformanceStats(
            total_requests=stats.get('total_requests', 0),
            average_processing_time=stats.get('average_processing_time', 0.0),
            method_usage=stats.get('method_usage', {}),
            success_rate=0.95,  # Would be calculated from actual metrics
            current_load=0.1    # Would be calculated from system metrics
        )
        
        # Recommendations
        recommendations = []
        if performance.average_processing_time > 10.0:
            recommendations.append("Consider optimizing processing pipeline")
        if performance.current_load > 0.8:
            recommendations.append("High system load detected")
        
        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            components=components,
            performance=performance,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            components={"system": "error"},
            performance=PerformanceStats(
                total_requests=0,
                average_processing_time=0.0,
                method_usage={},
                success_rate=0.0,
                current_load=1.0
            ),
            recommendations=["System experiencing errors"]
        )


# =====================================================
# LEGACY ENDPOINTS - BACKWARD COMPATIBILITY
# =====================================================

@app.post("/api/documents/{document_id}/paraphrase", deprecated=True)
async def legacy_document_paraphrase(
    document_id: uuid.UUID,
    legacy_request: LegacyParaphraseRequest,
    db: Session = Depends(get_db)
):
    """
    🔄 **LEGACY ENDPOINT** - Redirects to unified system
    
    **⚠️ DEPRECATED:** Use `/api/paraphrase` instead
    """
    logger.warning(f"Legacy endpoint called: /api/documents/{document_id}/paraphrase")
    
    # Convert to unified request
    unified_request = UnifiedParaphraseRequest(
        document_id=document_id,
        num_variants=legacy_request.num_variants,
        preserve_academic_terms=True,
        preserve_citations=True
    )
    
    # Process with unified endpoint
    unified_response = await unified_paraphrase_endpoint(unified_request, db)
    
    # Convert back to legacy format
    return LegacyParaphraseResponse.from_unified_response(unified_response)


# =====================================================
# DOCUMENT MANAGEMENT ENDPOINTS (Preserved)
# =====================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PlagiCheck - Unified Paraphrasing System",
        "version": "2.0.0",
        "status": "running",
        "docs_url": "/docs",
        "unified_endpoint": "/api/paraphrase"
    }


# Document management endpoints
@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    chapter: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a document for processing.
    
    Args:
        file: Document file (PDF, DOCX, or TXT)
        chapter: Chapter designation (e.g., "BAB 1")
        db: Database session
        
    Returns:
        DocumentResponse: Created document information
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (basic check)
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Process document
        document = await document_processor.upload_document(
            file.file,
            file.filename,
            db,
            chapter=chapter
        )
        
        return DocumentResponse.from_orm(document)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get document details by ID.
    
    Args:
        document_id: Document UUID
        db: Database session
        
    Returns:
        DocumentDetailResponse: Document details including content
    """
    document = document_processor.get_document(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentDetailResponse.from_orm(document)


@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(
    status: Optional[DocumentStatus] = None,
    chapter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List documents with optional filtering.
    
    Args:
        status: Filter by document status
        chapter: Filter by chapter
        limit: Maximum number of results
        offset: Number of results to skip
        db: Database session
        
    Returns:
        List[DocumentResponse]: List of documents
    """
    query = db.query(Document)
    
    if status:
        query = query.filter(Document.status == status)
    
    if chapter:
        query = query.filter(Document.chapter == chapter)
    
    documents = query.order_by(Document.upload_date.desc()).offset(offset).limit(limit).all()
    
    return [DocumentResponse.from_orm(doc) for doc in documents]


@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its associated file.
    
    Args:
        document_id: Document UUID
        db: Database session
        
    Returns:
        dict: Success message
    """
    success = document_processor.delete_document(document_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


# Paraphrasing endpoints
@app.post("/api/documents/{document_id}/paraphrase", response_model=ParaphraseSessionResponse)
async def start_paraphrasing(
    document_id: uuid.UUID,
    request: ParaphraseRequest,
    db: Session = Depends(get_db)
):
    """
    Start paraphrasing a document.
    
    Args:
        document_id: Document UUID
        request: Paraphrasing configuration
        db: Database session
        
    Returns:
        ParaphraseSessionResponse: Created session information
    """
    # Check if document exists
    document = document_processor.get_document(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Document is already being processed")
    
    try:
        # Start paraphrasing (this should be async in production)
        session = await paraphrasing_service.paraphrase_document(
            document_id,
            request.method,
            db
        )
        
        if not session:
            raise HTTPException(status_code=500, detail="Failed to start paraphrasing")
        
        return ParaphraseSessionResponse.from_orm(session)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paraphrasing failed: {str(e)}")


@app.get("/api/documents/{document_id}/status", response_model=StatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get current processing status of a document.
    
    Args:
        document_id: Document UUID
        db: Database session
        
    Returns:
        StatusResponse: Current status information
    """
    document = document_processor.get_document(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    status_messages = {
        DocumentStatus.PENDING: "Document uploaded, waiting for processing",
        DocumentStatus.PROCESSING: "Document is being paraphrased",
        DocumentStatus.COMPLETED: "Paraphrasing completed successfully",
        DocumentStatus.FAILED: "Paraphrasing failed"
    }
    
    return StatusResponse(
        status=document.status,
        message=status_messages.get(document.status, "Unknown status")
    )


@app.get("/api/sessions/{session_id}", response_model=ParaphraseSessionResponse)
async def get_paraphrase_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get paraphrase session details.
    
    Args:
        session_id: Session UUID
        db: Database session
        
    Returns:
        ParaphraseSessionResponse: Session details
    """
    session = paraphrasing_service.get_session(session_id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ParaphraseSessionResponse.from_orm(session)


@app.get("/api/documents/{document_id}/sessions", response_model=List[ParaphraseSessionResponse])
async def get_document_sessions(
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get all paraphrase sessions for a document.
    
    Args:
        document_id: Document UUID
        db: Database session
        
    Returns:
        List[ParaphraseSessionResponse]: List of sessions
    """
    # Check if document exists
    document = document_processor.get_document(document_id, db)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    sessions = paraphrasing_service.get_document_sessions(document_id, db)
    
    return [ParaphraseSessionResponse.from_orm(session) for session in sessions]


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(IOError)
async def io_error_handler(request, exc):
    """Handle IOError exceptions.""" 
    return JSONResponse(
        status_code=500,
        content={"detail": f"File processing error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
