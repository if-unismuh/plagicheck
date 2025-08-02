#!/usr/bin/env python3
"""
Database Migration and Management Script

This script provides comprehensive database management functionality including:
- Database creation
- Running migrations
- Database reset
- Schema validation
- Database seeding

Usage:
    python migrate_db.py create      # Create database
    python migrate_db.py migrate     # Run migrations
    python migrate_db.py reset       # Reset database
    python migrate_db.py status      # Check migration status
    python migrate_db.py seed        # Seed database with sample data
    python migrate_db.py validate    # Validate database connection
"""

import sys
import os
import asyncio
import argparse
import logging
from typing import Optional
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from app.core.config import settings
from app.core.database import Base, SessionLocal, engine
from app.models.document import Document, ParaphraseSession, DocumentStatus, ParaphraseMethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database management operations."""
    
    def __init__(self):
        self.settings = settings
        self.alembic_cfg = Config("alembic.ini")
        
    def create_database(self) -> bool:
        """
        Create the database if it doesn't exist.
        
        Returns:
            bool: True if database was created or already exists, False otherwise
        """
        try:
            # Parse database URL to get connection parameters
            db_params = self._parse_database_url()
            
            # Connect to PostgreSQL server (not specific database)
            server_url = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/postgres"
            
            logger.info(f"Connecting to PostgreSQL server at {db_params['host']}:{db_params['port']}")
            
            # Create connection to PostgreSQL server
            conn = psycopg2.connect(
                host=db_params['host'],
                port=db_params['port'],
                user=db_params['user'],
                password=db_params['password'],
                database='postgres'  # Connect to default postgres database
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (db_params['database'],)
            )
            exists = cursor.fetchone()
            
            if exists:
                logger.info(f"Database '{db_params['database']}' already exists")
                return True
            
            # Create database
            logger.info(f"Creating database '{db_params['database']}'...")
            cursor.execute(f'CREATE DATABASE "{db_params["database"]}"')
            logger.info(f"Database '{db_params['database']}' created successfully")
            
            cursor.close()
            conn.close()
            
            return True
            
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating database: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Testing database connection...")
            
            # Test connection
            test_engine = create_engine(self.settings.database_url_sync)
            with test_engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info("Database connection successful")
            return True
            
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing connection: {e}")
            return False
    
    def run_migrations(self) -> bool:
        """
        Run database migrations using Alembic.
        
        Returns:
            bool: True if migrations successful, False otherwise
        """
        try:
            logger.info("Running database migrations...")
            
            # Run migrations
            command.upgrade(self.alembic_cfg, "head")
            
            logger.info("Database migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def check_migration_status(self) -> dict:
        """
        Check current migration status.
        
        Returns:
            dict: Migration status information
        """
        try:
            # Get current revision
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
            
            # Get script directory
            script = ScriptDirectory.from_config(self.alembic_cfg)
            head_rev = script.get_current_head()
            
            # Get all revisions
            revisions = list(script.walk_revisions())
            
            status = {
                'current_revision': current_rev,
                'head_revision': head_rev,
                'is_up_to_date': current_rev == head_rev,
                'total_revisions': len(revisions),
                'pending_upgrades': []
            }
            
            # Check for pending upgrades
            if current_rev != head_rev:
                for rev in script.iterate_revisions(head_rev, current_rev):
                    if rev.revision != current_rev:
                        status['pending_upgrades'].append({
                            'revision': rev.revision,
                            'message': rev.doc
                        })
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking migration status: {e}")
            return {'error': str(e)}
    
    def reset_database(self) -> bool:
        """
        Reset database by dropping and recreating all tables.
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            logger.warning("Resetting database - this will delete ALL data!")
            
            # Drop all tables
            logger.info("Dropping all tables...")
            Base.metadata.drop_all(bind=engine)
            
            # Run migrations to recreate tables
            logger.info("Recreating tables with migrations...")
            return self.run_migrations()
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            return False
    
    def validate_schema(self) -> bool:
        """
        Validate database schema against models.
        
        Returns:
            bool: True if schema is valid, False otherwise
        """
        try:
            logger.info("Validating database schema...")
            
            inspector = inspect(engine)
            
            # Check if required tables exist
            required_tables = ['documents', 'paraphrase_sessions']
            existing_tables = inspector.get_table_names()
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False
            
            # Validate documents table structure
            doc_columns = {col['name']: col for col in inspector.get_columns('documents')}
            required_doc_columns = ['id', 'filename', 'chapter', 'original_content', 'status']
            
            missing_doc_columns = [col for col in required_doc_columns if col not in doc_columns]
            if missing_doc_columns:
                logger.error(f"Missing columns in documents table: {missing_doc_columns}")
                return False
            
            # Validate paraphrase_sessions table structure
            session_columns = {col['name']: col for col in inspector.get_columns('paraphrase_sessions')}
            required_session_columns = ['id', 'document_id', 'method_used', 'created_at']
            
            missing_session_columns = [col for col in required_session_columns if col not in session_columns]
            if missing_session_columns:
                logger.error(f"Missing columns in paraphrase_sessions table: {missing_session_columns}")
                return False
            
            logger.info("Database schema validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def seed_database(self) -> bool:
        """
        Seed database with sample data for testing.
        
        Returns:
            bool: True if seeding successful, False otherwise
        """
        try:
            logger.info("Seeding database with sample data...")
            
            db = SessionLocal()
            
            # Create sample documents
            sample_docs = [
                {
                    'filename': 'sample_chapter1.txt',
                    'chapter': 'BAB 1',
                    'original_content': 'Ini adalah contoh konten dari BAB 1 yang akan diparafrasekan oleh sistem.',
                    'status': DocumentStatus.COMPLETED,
                    'file_path': '/uploads/sample_chapter1.txt',
                    'metadata': {'word_count': 12, 'language': 'id'}
                },
                {
                    'filename': 'sample_chapter2.txt',
                    'chapter': 'BAB 2',
                    'original_content': 'Konten BAB 2 berisi pembahasan lebih mendalam tentang parafrase otomatis.',
                    'status': DocumentStatus.PENDING,
                    'file_path': '/uploads/sample_chapter2.txt',
                    'metadata': {'word_count': 10, 'language': 'id'}
                }
            ]
            
            # Insert sample documents
            for doc_data in sample_docs:
                existing_doc = db.query(Document).filter(
                    Document.filename == doc_data['filename']
                ).first()
                
                if not existing_doc:
                    document = Document(**doc_data)
                    db.add(document)
                    db.commit()
                    db.refresh(document)
                    
                    # Create sample paraphrase session for completed document
                    if doc_data['status'] == DocumentStatus.COMPLETED:
                        session = ParaphraseSession(
                            document_id=document.id,
                            method_used=ParaphraseMethod.HYBRID,
                            similarity_score=0.85,
                            processing_time=1500,  # milliseconds
                            token_usage={'input_tokens': 25, 'output_tokens': 30}
                        )
                        db.add(session)
                        db.commit()
            
            db.close()
            logger.info("Database seeded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database seeding failed: {e}")
            return False
    
    def _parse_database_url(self) -> dict:
        """Parse database URL into components."""
        url = self.settings.database_url_sync
        
        # Simple URL parsing for PostgreSQL
        if url.startswith('postgresql://'):
            url = url[13:]  # Remove postgresql://
            
            # Split user:password@host:port/database
            auth_and_host = url.split('/')
            database = auth_and_host[1] if len(auth_and_host) > 1 else 'postgres'
            
            auth_host = auth_and_host[0]
            auth, host_port = auth_host.split('@')
            user, password = auth.split(':')
            
            if ':' in host_port:
                host, port = host_port.split(':')
                port = int(port)
            else:
                host = host_port
                port = 5432
            
            return {
                'user': user,
                'password': password,
                'host': host,
                'port': port,
                'database': database
            }
        else:
            raise ValueError(f"Unsupported database URL format: {url}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Database Migration and Management Tool')
    parser.add_argument(
        'command',
        choices=['create', 'migrate', 'reset', 'status', 'seed', 'validate'],
        help='Command to execute'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force operation without confirmation (for reset command)'
    )
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    
    # Print configuration info
    logger.info(f"Database URL: {settings.database_url_sync}")
    logger.info(f"Command: {args.command}")
    
    if args.command == 'create':
        logger.info("=== Creating Database ===")
        success = db_manager.create_database()
        if success:
            logger.info("Database creation completed successfully")
        else:
            logger.error("Database creation failed")
            sys.exit(1)
    
    elif args.command == 'migrate':
        logger.info("=== Running Migrations ===")
        
        # First ensure database exists
        if not db_manager.test_connection():
            logger.info("Database doesn't exist, creating it first...")
            if not db_manager.create_database():
                logger.error("Failed to create database")
                sys.exit(1)
        
        success = db_manager.run_migrations()
        if success:
            logger.info("Migrations completed successfully")
        else:
            logger.error("Migrations failed")
            sys.exit(1)
    
    elif args.command == 'reset':
        if not args.force:
            confirm = input("Are you sure you want to reset the database? This will delete ALL data! (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Database reset cancelled")
                sys.exit(0)
        
        logger.info("=== Resetting Database ===")
        success = db_manager.reset_database()
        if success:
            logger.info("Database reset completed successfully")
        else:
            logger.error("Database reset failed")
            sys.exit(1)
    
    elif args.command == 'status':
        logger.info("=== Checking Migration Status ===")
        status = db_manager.check_migration_status()
        
        if 'error' in status:
            logger.error(f"Error checking status: {status['error']}")
            sys.exit(1)
        
        logger.info(f"Current revision: {status['current_revision']}")
        logger.info(f"Head revision: {status['head_revision']}")
        logger.info(f"Up to date: {status['is_up_to_date']}")
        logger.info(f"Total revisions: {status['total_revisions']}")
        
        if status['pending_upgrades']:
            logger.warning("Pending upgrades:")
            for upgrade in status['pending_upgrades']:
                logger.warning(f"  - {upgrade['revision']}: {upgrade['message']}")
        else:
            logger.info("No pending upgrades")
    
    elif args.command == 'seed':
        logger.info("=== Seeding Database ===")
        success = db_manager.seed_database()
        if success:
            logger.info("Database seeding completed successfully")
        else:
            logger.error("Database seeding failed")
            sys.exit(1)
    
    elif args.command == 'validate':
        logger.info("=== Validating Database Schema ===")
        success = db_manager.validate_schema()
        if success:
            logger.info("Database schema validation passed")
        else:
            logger.error("Database schema validation failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
