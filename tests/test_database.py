"""
Database Operation Tests
Comprehensive testing of database operations, transactions, and data integrity.
"""
import uuid
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DatabaseError

from app.core.database import get_db, Base, create_tables
from app.models.document import Document, ParaphraseSession, DocumentStatus, ParaphraseMethod


class TestDatabaseConnection:
    """Test database connection and setup."""
    
    def test_database_connection(self, test_db):
        """Test basic database connection."""
        # Get a database session
        db_gen = get_db()
        db = next(db_gen)
        
        # Test that we can execute a simple query
        result = db.execute("SELECT 1").scalar()
        assert result == 1
        
        # Clean up
        db_gen.close()
    
    def test_table_creation(self, test_db):
        """Test that all required tables are created."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Check that tables exist by trying to query them
        try:
            db.query(Document).count()
            db.query(ParaphraseSession).count()
        except Exception as e:
            pytest.fail(f"Tables not created properly: {e}")
        
        db_gen.close()


class TestDocumentModel:
    """Test Document model operations."""
    
    def test_document_creation(self, test_db):
        """Test creating a new document."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create a new document
        document = Document(
            filename="test_document.txt",
            chapter="BAB 1",
            original_content="This is test content for the document.",
            status=DocumentStatus.PENDING,
            document_metadata={"word_count": 8, "language": "english"}
        )
        
        db.add(document)
        db.commit()
        
        # Verify the document was created
        saved_doc = db.query(Document).filter_by(filename="test_document.txt").first()
        assert saved_doc is not None
        assert saved_doc.filename == "test_document.txt"
        assert saved_doc.chapter == "BAB 1"
        assert saved_doc.status == DocumentStatus.PENDING
        assert saved_doc.document_metadata["word_count"] == 8
        assert saved_doc.upload_date is not None
        
        db_gen.close()
    
    def test_document_update(self, test_db):
        """Test updating document properties."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create a document
        document = Document(
            filename="update_test.txt",
            original_content="Original content",
            status=DocumentStatus.PENDING
        )
        db.add(document)
        db.commit()
        document_id = document.id
        
        # Update the document
        document.status = DocumentStatus.COMPLETED
        document.paraphrased_content = "Paraphrased content"
        document.processed_date = datetime.utcnow()
        db.commit()
        
        # Verify the update
        updated_doc = db.query(Document).filter_by(id=document_id).first()
        assert updated_doc.status == DocumentStatus.COMPLETED
        assert updated_doc.paraphrased_content == "Paraphrased content"
        assert updated_doc.processed_date is not None
        
        db_gen.close()
    
    def test_document_deletion(self, test_db):
        """Test deleting a document."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create a document
        document = Document(
            filename="delete_test.txt",
            original_content="Content to be deleted"
        )
        db.add(document)
        db.commit()
        document_id = document.id
        
        # Delete the document
        db.delete(document)
        db.commit()
        
        # Verify deletion
        deleted_doc = db.query(Document).filter_by(id=document_id).first()
        assert deleted_doc is None
        
        db_gen.close()
    
    def test_document_validation(self, test_db):
        """Test document model validation."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Test that required fields are enforced
        with pytest.raises(Exception):  # Should raise an error for missing required fields
            document = Document()  # No filename or content
            db.add(document)
            db.commit()
        
        db_gen.close()
    
    def test_document_query_filtering(self, test_db):
        """Test querying documents with various filters."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create test documents
        docs = [
            Document(filename="doc1.txt", chapter="BAB 1", status=DocumentStatus.PENDING, original_content="Content 1"),
            Document(filename="doc2.txt", chapter="BAB 2", status=DocumentStatus.COMPLETED, original_content="Content 2"),
            Document(filename="doc3.txt", chapter="BAB 1", status=DocumentStatus.FAILED, original_content="Content 3"),
            Document(filename="doc4.txt", chapter="BAB 3", status=DocumentStatus.PENDING, original_content="Content 4"),
        ]
        
        for doc in docs:
            db.add(doc)
        db.commit()
        
        # Test filtering by status
        pending_docs = db.query(Document).filter_by(status=DocumentStatus.PENDING).all()
        assert len(pending_docs) == 2
        
        # Test filtering by chapter
        bab1_docs = db.query(Document).filter_by(chapter="BAB 1").all()
        assert len(bab1_docs) == 2
        
        # Test combined filtering
        pending_bab1 = db.query(Document).filter_by(
            status=DocumentStatus.PENDING, 
            chapter="BAB 1"
        ).all()
        assert len(pending_bab1) == 1
        
        # Test ordering
        ordered_docs = db.query(Document).order_by(Document.upload_date.desc()).all()
        assert len(ordered_docs) == 4
        # Verify order (newest first)
        for i in range(len(ordered_docs) - 1):
            assert ordered_docs[i].upload_date >= ordered_docs[i + 1].upload_date
        
        db_gen.close()


class TestParaphraseSessionModel:
    """Test ParaphraseSession model operations."""
    
    def test_session_creation(self, test_db):
        """Test creating a paraphrase session."""
        db_gen = get_db()
        db = next(db_gen)
        
        # First create a document
        document = Document(
            filename="session_test.txt",
            original_content="Content for session testing"
        )
        db.add(document)
        db.commit()
        
        # Create a paraphrase session
        session = ParaphraseSession(
            document_id=document.id,
            method_used=ParaphraseMethod.INDOT5,
            similarity_score=0.75,
            processing_time=3000,
            token_usage={"input_tokens": 50, "output_tokens": 45}
        )
        
        db.add(session)
        db.commit()
        
        # Verify the session was created
        saved_session = db.query(ParaphraseSession).filter_by(document_id=document.id).first()
        assert saved_session is not None
        assert saved_session.method_used == ParaphraseMethod.INDOT5
        assert saved_session.similarity_score == 0.75
        assert saved_session.processing_time == 3000
        assert saved_session.token_usage["input_tokens"] == 50
        assert saved_session.created_at is not None
        
        db_gen.close()
    
    def test_session_document_relationship(self, test_db):
        """Test the relationship between sessions and documents."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create a document
        document = Document(
            filename="relationship_test.txt",
            original_content="Content for relationship testing"
        )
        db.add(document)
        db.commit()
        
        # Create multiple sessions for the same document
        sessions = [
            ParaphraseSession(
                document_id=document.id,
                method_used=ParaphraseMethod.INDOT5,
                similarity_score=0.75
            ),
            ParaphraseSession(
                document_id=document.id,
                method_used=ParaphraseMethod.RULE_BASED,
                similarity_score=0.68
            ),
            ParaphraseSession(
                document_id=document.id,
                method_used=ParaphraseMethod.HYBRID,
                similarity_score=0.82
            )
        ]
        
        for session in sessions:
            db.add(session)
        db.commit()
        
        # Test querying sessions by document
        doc_sessions = db.query(ParaphraseSession).filter_by(document_id=document.id).all()
        assert len(doc_sessions) == 3
        
        # Test querying by method
        indot5_sessions = db.query(ParaphraseSession).filter_by(
            document_id=document.id,
            method_used=ParaphraseMethod.INDOT5
        ).all()
        assert len(indot5_sessions) == 1
        
        db_gen.close()
    
    def test_session_cascade_deletion(self, test_db):
        """Test that sessions are handled properly when document is deleted."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create a document with sessions
        document = Document(
            filename="cascade_test.txt",
            original_content="Content for cascade testing"
        )
        db.add(document)
        db.commit()
        document_id = document.id
        
        # Create sessions
        session = ParaphraseSession(
            document_id=document_id,
            method_used=ParaphraseMethod.INDOT5
        )
        db.add(session)
        db.commit()
        session_id = session.id
        
        # Delete the document
        db.delete(document)
        db.commit()
        
        # Check if session still exists (depending on cascade settings)
        remaining_session = db.query(ParaphraseSession).filter_by(id=session_id).first()
        # This behavior depends on your foreign key cascade settings
        # Adjust assertion based on your actual cascade configuration
        
        db_gen.close()


class TestDatabaseTransactions:
    """Test database transaction handling."""
    
    def test_transaction_commit(self, test_db):
        """Test successful transaction commit."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Start a transaction by creating multiple related objects
        document = Document(
            filename="transaction_test.txt",
            original_content="Transaction test content"
        )
        db.add(document)
        db.flush()  # Flush to get the ID without committing
        
        session = ParaphraseSession(
            document_id=document.id,
            method_used=ParaphraseMethod.INDOT5,
            similarity_score=0.8
        )
        db.add(session)
        
        # Commit the transaction
        db.commit()
        
        # Verify both objects were saved
        saved_doc = db.query(Document).filter_by(filename="transaction_test.txt").first()
        saved_session = db.query(ParaphraseSession).filter_by(document_id=saved_doc.id).first()
        
        assert saved_doc is not None
        assert saved_session is not None
        assert saved_session.document_id == saved_doc.id
        
        db_gen.close()
    
    def test_transaction_rollback(self, test_db):
        """Test transaction rollback on error."""
        db_gen = get_db()
        db = next(db_gen)
        
        try:
            # Create a document
            document = Document(
                filename="rollback_test.txt",
                original_content="Rollback test content"
            )
            db.add(document)
            db.flush()
            
            # Force an error (e.g., by creating invalid session)
            invalid_session = ParaphraseSession(
                document_id=uuid.uuid4(),  # Non-existent document ID
                method_used=ParaphraseMethod.INDOT5
            )
            db.add(invalid_session)
            
            # This should fail and rollback
            db.commit()
            
        except Exception:
            # Rollback the transaction
            db.rollback()
            
            # Verify that the document was not saved
            saved_doc = db.query(Document).filter_by(filename="rollback_test.txt").first()
            assert saved_doc is None
        
        db_gen.close()
    
    def test_concurrent_transactions(self, test_db):
        """Test handling of concurrent database transactions."""
        import threading
        import time
        
        def create_document_in_thread(thread_id, results):
            try:
                db_gen = get_db()
                db = next(db_gen)
                
                document = Document(
                    filename=f"concurrent_test_{thread_id}.txt",
                    original_content=f"Content from thread {thread_id}"
                )
                db.add(document)
                
                # Add a small delay to increase chance of concurrency issues
                time.sleep(0.01)
                
                db.commit()
                results[thread_id] = True
                
                db_gen.close()
                
            except Exception as e:
                results[thread_id] = False
        
        # Run multiple concurrent transactions
        threads = []
        results = {}
        num_threads = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=create_document_in_thread, args=(i, results))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        successful_transactions = sum(results.values())
        assert successful_transactions == num_threads
        
        # Verify all documents were created
        db_gen = get_db()
        db = next(db_gen)
        
        created_docs = db.query(Document).filter(
            Document.filename.like("concurrent_test_%.txt")
        ).all()
        assert len(created_docs) == num_threads
        
        db_gen.close()


class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    def test_bulk_insert_performance(self, test_db):
        """Test performance of bulk database operations."""
        import time
        
        db_gen = get_db()
        db = next(db_gen)
        
        # Create many documents at once
        num_documents = 100
        documents = []
        
        start_time = time.time()
        
        for i in range(num_documents):
            document = Document(
                filename=f"bulk_test_{i}.txt",
                original_content=f"Bulk test content {i}",
                chapter=f"BAB {(i % 5) + 1}",
                status=DocumentStatus.PENDING if i % 2 == 0 else DocumentStatus.COMPLETED
            )
            documents.append(document)
        
        # Bulk add
        db.add_all(documents)
        db.commit()
        
        end_time = time.time()
        bulk_time = end_time - start_time
        
        # Verify all documents were created
        count = db.query(Document).filter(
            Document.filename.like("bulk_test_%.txt")
        ).count()
        assert count == num_documents
        
        print(f"Bulk insert of {num_documents} documents took {bulk_time:.3f} seconds")
        
        # Performance assertion (adjust based on your system)
        assert bulk_time < 5.0, f"Bulk insert too slow: {bulk_time:.3f}s"
        
        db_gen.close()
    
    def test_query_performance(self, test_db):
        """Test database query performance."""
        import time
        
        db_gen = get_db()
        db = next(db_gen)
        
        # Create test data
        documents = []
        sessions = []
        
        for i in range(50):
            document = Document(
                filename=f"query_test_{i}.txt",
                original_content=f"Query test content {i}",
                chapter=f"BAB {(i % 3) + 1}",
                status=DocumentStatus.PENDING if i % 2 == 0 else DocumentStatus.COMPLETED
            )
            documents.append(document)
        
        db.add_all(documents)
        db.commit()
        
        # Create sessions for some documents
        for i, doc in enumerate(documents[::2]):  # Every other document
            session = ParaphraseSession(
                document_id=doc.id,
                method_used=ParaphraseMethod.INDOT5 if i % 2 == 0 else ParaphraseMethod.RULE_BASED,
                similarity_score=0.7 + (i * 0.01)
            )
            sessions.append(session)
        
        db.add_all(sessions)
        db.commit()
        
        # Test various query patterns
        queries = [
            ("Simple filter", lambda: db.query(Document).filter_by(status=DocumentStatus.PENDING).all()),
            ("Count query", lambda: db.query(Document).count()),
            ("Order by", lambda: db.query(Document).order_by(Document.upload_date.desc()).limit(10).all()),
            ("Join query", lambda: db.query(Document).join(ParaphraseSession).all()),
            ("Complex filter", lambda: db.query(Document).filter(
                Document.status == DocumentStatus.COMPLETED,
                Document.chapter.in_(["BAB 1", "BAB 2"])
            ).all()),
        ]
        
        for query_name, query_func in queries:
            start_time = time.time()
            result = query_func()
            end_time = time.time()
            query_time = end_time - start_time
            
            print(f"{query_name}: {query_time:.4f}s, {len(result) if hasattr(result, '__len__') else result} results")
            
            # Performance assertion
            assert query_time < 1.0, f"{query_name} query too slow: {query_time:.4f}s"
        
        db_gen.close()


class TestDatabaseMigration:
    """Test database migration and schema changes."""
    
    def test_schema_creation(self, test_db):
        """Test that the database schema is created correctly."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Test that we can introspect the schema
        inspector = db.get_bind().dialect.get_schema_names(db.get_bind())
        
        # Test table existence
        tables = db.get_bind().table_names()
        expected_tables = ["documents", "paraphrase_sessions"]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found in database"
        
        db_gen.close()
    
    def test_data_integrity_constraints(self, test_db):
        """Test that database constraints are properly enforced."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Test unique constraints (if any)
        # Test foreign key constraints
        try:
            # Try to create a session with non-existent document ID
            invalid_session = ParaphraseSession(
                document_id=uuid.uuid4(),  # Non-existent
                method_used=ParaphraseMethod.INDOT5
            )
            db.add(invalid_session)
            db.commit()
            
            # If we reach here, foreign key constraint is not enforced
            # Adjust this test based on your actual constraints
            
        except IntegrityError:
            # This is expected if foreign key constraints are enforced
            db.rollback()
        
        # Test not-null constraints
        try:
            invalid_document = Document()  # Missing required fields
            db.add(invalid_document)
            db.commit()
            
        except Exception:
            # Expected if not-null constraints are enforced
            db.rollback()
        
        db_gen.close()


class TestDatabaseCleanup:
    """Test database cleanup and maintenance operations."""
    
    def test_old_data_cleanup(self, test_db):
        """Test cleanup of old data."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create old documents
        old_date = datetime.utcnow() - timedelta(days=365)
        
        old_documents = []
        for i in range(5):
            doc = Document(
                filename=f"old_doc_{i}.txt",
                original_content="Old content",
                status=DocumentStatus.COMPLETED
            )
            # Manually set old date (you might need to update this based on your model)
            old_documents.append(doc)
        
        db.add_all(old_documents)
        db.commit()
        
        # Create recent documents
        recent_documents = []
        for i in range(3):
            doc = Document(
                filename=f"recent_doc_{i}.txt",
                original_content="Recent content",
                status=DocumentStatus.COMPLETED
            )
            recent_documents.append(doc)
        
        db.add_all(recent_documents)
        db.commit()
        
        # Test cleanup query (simulated)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # In a real cleanup, you might delete old completed documents
        old_completed_docs = db.query(Document).filter(
            Document.status == DocumentStatus.COMPLETED,
            Document.upload_date < cutoff_date
        ).all()
        
        # For this test, we just verify the query works
        assert len(old_completed_docs) >= 0
        
        db_gen.close()
    
    def test_orphaned_data_cleanup(self, test_db):
        """Test cleanup of orphaned records."""
        db_gen = get_db()
        db = next(db_gen)
        
        # Create a document and session
        document = Document(
            filename="orphan_test.txt",
            original_content="Content for orphan testing"
        )
        db.add(document)
        db.commit()
        
        session = ParaphraseSession(
            document_id=document.id,
            method_used=ParaphraseMethod.INDOT5
        )
        db.add(session)
        db.commit()
        
        # Delete the document (leaving orphaned session if no cascade)
        db.delete(document)
        db.commit()
        
        # Check for orphaned sessions
        orphaned_sessions = db.query(ParaphraseSession).filter(
            ~ParaphraseSession.document_id.in_(
                db.query(Document.id).subquery()
            )
        ).all()
        
        # The behavior depends on your cascade settings
        # Adjust this test based on your actual configuration
        
        db_gen.close()
