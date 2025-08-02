"""
Document and ParaphraseSession Models
"""
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, 
    ForeignKey, JSON, Index, Enum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class DocumentStatus(PyEnum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ParaphraseMethod(PyEnum):
    """Available paraphrasing methods."""
    HYBRID = "hybrid"
    INDOT5 = "indot5"
    GEMINI = "gemini"


class Document(Base):
    """
    Document model for storing uploaded documents and their processing status.
    """
    __tablename__ = "documents"

    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        index=True
    )
    filename = Column(String(255), nullable=False, index=True)
    chapter = Column(String(50), nullable=True, index=True)  # e.g., "BAB 1", "BAB 2"
    original_content = Column(Text, nullable=False)
    paraphrased_content = Column(Text, nullable=True)
    status = Column(
        Enum(DocumentStatus, native_enum=False), 
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True
    )
    upload_date = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        index=True
    )
    processed_date = Column(DateTime(timezone=True), nullable=True)
    file_path = Column(String(500), nullable=False)
    document_metadata = Column(JSON, nullable=True)  # Additional document metadata
    
    # Relationship to paraphrase sessions
    paraphrase_sessions = relationship(
        "ParaphraseSession", 
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"


class ParaphraseSession(Base):
    """
    ParaphraseSession model for tracking individual paraphrasing attempts.
    """
    __tablename__ = "paraphrase_sessions"

    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        index=True
    )
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    method_used = Column(
        Enum(ParaphraseMethod, native_enum=False),
        nullable=False,
        index=True
    )
    similarity_score = Column(Float, nullable=True)  # Similarity to original
    processing_time = Column(Integer, nullable=True)  # Processing time in seconds
    token_usage = Column(JSON, nullable=True)  # For tracking API costs
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    # Relationship to document
    document = relationship("Document", back_populates="paraphrase_sessions")
    
    def __repr__(self):
        return (
            f"<ParaphraseSession(id={self.id}, method={self.method_used}, "
            f"similarity={self.similarity_score})>"
        )


# Create indexes for better query performance
Index('idx_documents_status_upload', Document.status, Document.upload_date)
Index('idx_documents_chapter_status', Document.chapter, Document.status)
Index('idx_sessions_document_created', ParaphraseSession.document_id, ParaphraseSession.created_at)
Index('idx_sessions_method_created', ParaphraseSession.method_used, ParaphraseSession.created_at)

# Full-text search index for document content (PostgreSQL specific)
# Only create this index when using PostgreSQL
from sqlalchemy import event
from app.core.config import settings

@event.listens_for(Document.__table__, 'after_create')
def create_search_index(target, connection, **kw):
    """Create full-text search index only for PostgreSQL."""
    if connection.dialect.name == 'postgresql':
        try:
            # Create the gin_trgm_ops index for PostgreSQL
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_search "
                "ON documents USING gin(original_content gin_trgm_ops)"
            )
        except Exception as e:
            # Log the error but don't fail startup
            print(f"Warning: Could not create full-text search index: {e}")
