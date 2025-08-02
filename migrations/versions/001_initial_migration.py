"""Initial migration - create documents and paraphrase_sessions tables

Revision ID: 001
Revises: 
Create Date: 2025-08-01 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create documents table
    op.create_table('documents',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('filename', sa.String(length=255), nullable=False),
    sa.Column('chapter', sa.String(length=50), nullable=True),
    sa.Column('original_content', sa.Text(), nullable=False),
    sa.Column('paraphrased_content', sa.Text(), nullable=True),
    sa.Column('status', sa.Enum('pending', 'processing', 'completed', 'failed', name='documentstatus'), nullable=False),
    sa.Column('upload_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('processed_date', sa.DateTime(timezone=True), nullable=True),
    sa.Column('file_path', sa.String(length=500), nullable=False),
    sa.Column('document_metadata', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    
    # Create paraphrase_sessions table
    op.create_table('paraphrase_sessions',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('method_used', sa.Enum('hybrid', 'indot5', 'gemini', name='paraphrasemethod'), nullable=False),
    sa.Column('similarity_score', sa.Float(), nullable=True),
    sa.Column('processing_time', sa.Integer(), nullable=True),
    sa.Column('token_usage', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better performance
    op.create_index('idx_documents_chapter_status', 'documents', ['chapter', 'status'], unique=False)
    op.create_index('idx_documents_status_upload', 'documents', ['status', 'upload_date'], unique=False)
    op.create_index('idx_sessions_document_created', 'paraphrase_sessions', ['document_id', 'created_at'], unique=False)
    op.create_index('idx_sessions_method_created', 'paraphrase_sessions', ['method_used', 'created_at'], unique=False)
    
    # Create basic indexes
    op.create_index(op.f('ix_documents_chapter'), 'documents', ['chapter'], unique=False)
    op.create_index(op.f('ix_documents_filename'), 'documents', ['filename'], unique=False)
    op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
    op.create_index(op.f('ix_documents_status'), 'documents', ['status'], unique=False)
    op.create_index(op.f('ix_documents_upload_date'), 'documents', ['upload_date'], unique=False)
    op.create_index(op.f('ix_paraphrase_sessions_created_at'), 'paraphrase_sessions', ['created_at'], unique=False)
    op.create_index(op.f('ix_paraphrase_sessions_document_id'), 'paraphrase_sessions', ['document_id'], unique=False)
    op.create_index(op.f('ix_paraphrase_sessions_id'), 'paraphrase_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_paraphrase_sessions_method_used'), 'paraphrase_sessions', ['method_used'], unique=False)
    
    # Enable PostgreSQL extensions for full-text search
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')
    
    # Create full-text search index for document content
    op.create_index(
        'idx_documents_content_search',
        'documents',
        ['original_content'],
        postgresql_using='gin',
        postgresql_ops={'original_content': 'gin_trgm_ops'}
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_documents_content_search', table_name='documents')
    op.drop_index(op.f('ix_paraphrase_sessions_method_used'), table_name='paraphrase_sessions')
    op.drop_index(op.f('ix_paraphrase_sessions_id'), table_name='paraphrase_sessions')
    op.drop_index(op.f('ix_paraphrase_sessions_document_id'), table_name='paraphrase_sessions')
    op.drop_index(op.f('ix_paraphrase_sessions_created_at'), table_name='paraphrase_sessions')
    op.drop_index(op.f('ix_documents_upload_date'), table_name='documents')
    op.drop_index(op.f('ix_documents_status'), table_name='documents')
    op.drop_index(op.f('ix_documents_id'), table_name='documents')
    op.drop_index(op.f('ix_documents_filename'), table_name='documents')
    op.drop_index(op.f('ix_documents_chapter'), table_name='documents')
    op.drop_index('idx_sessions_method_created', table_name='paraphrase_sessions')
    op.drop_index('idx_sessions_document_created', table_name='paraphrase_sessions')
    op.drop_index('idx_documents_status_upload', table_name='documents')
    op.drop_index('idx_documents_chapter_status', table_name='documents')
    
    # Drop tables
    op.drop_table('paraphrase_sessions')
    op.drop_table('documents')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS paraphrasemethod;')
    op.execute('DROP TYPE IF EXISTS documentstatus;')
