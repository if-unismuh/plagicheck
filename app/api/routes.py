"""
FastAPI Routes and Application Setup
"""
import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db, create_tables
from app.models.document import Document, ParaphraseSession, DocumentStatus, ParaphraseMethod
from app.services.document_processor import document_processor
from app.services.paraphraser import paraphrasing_service

# Pydantic models for API
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
    created_at: datetime
    
    class Config:
        from_attributes = True


class ParaphraseRequest(BaseModel):
    """Request model for paraphrasing."""
    method: ParaphraseMethod


class StatusResponse(BaseModel):
    """Response model for status checks."""
    status: DocumentStatus
    message: str
    progress: Optional[float] = None


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Academic document paraphrasing system with multiple AI models",
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

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables and required directories."""
    try:
        # Import models to ensure they are registered with SQLAlchemy
        from app.models.document import Document, ParaphraseSession  # noqa: F401
        
        # Create database tables
        create_tables()
        print("Database tables created successfully")
        
        # Create upload directory
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(exist_ok=True)
        print(f"Upload directory ready: {upload_dir}")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        # Don't raise the error to prevent app from failing to start
        # Log it instead for debugging


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources during app shutdown."""
    try:
        print("Shutting down gracefully...")
        # Add any cleanup tasks here if needed
        print("Shutdown complete")
    except Exception as e:
        print(f"Error during shutdown: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


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
