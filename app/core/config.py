"""
Application Configuration Settings
"""
import os
from typing import Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App Configuration
    app_name: str = Field(default="Auto-Paraphrasing System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # Database Configuration
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_host: str = Field(default="localhost", env="DATABASE_HOST")
    database_port: int = Field(default=5432, env="DATABASE_PORT")
    database_name: str = Field(default="paraphrase_db", env="DATABASE_NAME")
    database_user: Optional[str] = Field(default=None, env="DATABASE_USER")
    database_password: Optional[str] = Field(default=None, env="DATABASE_PASSWORD")
    
    # AI Model Configuration
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # File Upload Configuration
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    max_file_size: str = Field(default="50MB", env="MAX_FILE_SIZE")
    allowed_extensions: Union[str, list[str]] = Field(
        default=[".pdf", ".docx", ".txt"], 
        env="ALLOWED_EXTENSIONS"
    )
    
    # Processing Configuration
    max_concurrent_jobs: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Unified Paraphrase Configuration
    unified_paraphrase_config: dict = Field(
        default={
            'custom_synonyms_path': 'synonyms.json',
            'quality_threshold_default': 0.7,
            'max_variants_per_request': 5,
            'enable_gpu_acceleration': True,
            'enable_performance_caching': True,
            'academic_focus_mode': True,
            'preserve_formatting': True
        },
        env="UNIFIED_PARAPHRASE_CONFIG"
    )
    
    @field_validator('allowed_extensions', mode='before')
    @classmethod
    def validate_allowed_extensions(cls, v):
        """Convert string to list for allowed_extensions if needed."""
        if isinstance(v, str):
            if not v.strip():  # Empty string
                return [".pdf", ".docx", ".txt"]  # Default value
            # Try to parse as JSON first
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                # If not JSON, split by comma
                return [ext.strip() for ext in v.split(',') if ext.strip()]
        return v
    
    @property
    def database_url_sync(self) -> str:
        """Synchronous database URL for SQLAlchemy."""
        if hasattr(self, '_database_url_sync'):
            return self._database_url_sync
        
        if self.database_url:
            return self.database_url
        
        # Default to SQLite for development if PostgreSQL credentials not provided
        if not self.database_user or not self.database_password:
            return "sqlite:///./paraphrase_db.sqlite"
        
        return (
            f"postgresql://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )
    
    @property
    def database_url_async(self) -> str:
        """Asynchronous database URL for SQLAlchemy."""
        sync_url = self.database_url_sync
        if sync_url.startswith("sqlite:"):
            # SQLite doesn't support async in the same way, return sync URL
            return sync_url
        return sync_url.replace("postgresql://", "postgresql+asyncpg://")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
