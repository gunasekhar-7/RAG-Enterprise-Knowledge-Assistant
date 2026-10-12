
from functools import lru_cache
from pathlib import Path
from typing import Literal
import urllib.parse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate root directory containing .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE = BASE_DIR / '.env'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else '.env',
        env_file_encoding='utf-8',
        extra='forbid',
    )

    # --- Application ---
    ENVIRONMENT: str = Field(default='development')
    LOG_LEVEL: str = Field(default='INFO')
    APP_NAME: str = Field(default='RAG Enterprise Knowledge Assistant')
    API_V1_STR: str = Field(default='/api/v1')
    CORS_ORIGINS: list[str] = Field(default=['http://localhost:5173', 'http://127.0.0.1:5173'])

    # --- Security & Auth ---
    JWT_SECRET: SecretStr = Field(default=SecretStr('insecure-dev-secret-key-change-me'))
    JWT_ALGORITHM: Literal['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512'] = Field(default='HS256')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, ge=1, le=1440)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=90)
    JWT_ISSUER: str = Field(default='rag-assistant')
    JWT_AUDIENCE: str = Field(default='rag-assistant-api')

    # --- PostgreSQL & pgvector ---
    POSTGRES_USER: str = Field(default='postgres')
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr('postgres'))
    POSTGRES_DB: str = Field(default='rag_knowledge_db')
    POSTGRES_HOST: str = Field(default='localhost')
    POSTGRES_PORT: int = Field(default=5432)
    DATABASE_URL: str | None = Field(default=None)

    # --- Redis ---
    REDIS_HOST: str = Field(default='localhost')
    REDIS_PORT: int = Field(default=6379)
    REDIS_URL: str = Field(default='redis://localhost:6379/0')

    # --- S3 / MinIO ---
    MINIO_ROOT_USER: str = Field(default='minioadmin')
    MINIO_ROOT_PASSWORD: SecretStr = Field(default=SecretStr('minioadmin'))
    MINIO_PORT: int = Field(default=9000)
    MINIO_CONSOLE_PORT: int = Field(default=9001)
    S3_ENDPOINT_URL: str = Field(default='http://localhost:9000')
    S3_BUCKET_NAME: str = Field(default='rag-documents')
    S3_REGION: str = Field(default='us-east-1')
    S3_ACCESS_KEY_ID: str = Field(default='minioadmin')
    S3_SECRET_ACCESS_KEY: SecretStr = Field(default=SecretStr('minioadmin'))

    # --- OpenAI & AI Providers ---
    OPENAI_API_KEY: SecretStr = Field(default=SecretStr('mock-key'))
    EMBEDDING_MODEL: str = Field(default='text-embedding-3-small')
    LLM_MODEL: str = Field(default='gpt-4o-mini')
    LLM_TEMPERATURE: float = Field(default=0.1)
    LLM_MAX_TOKENS: int = Field(default=1024)

    # --- RAG Parameters ---
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)
    RETRIEVAL_TOP_K: int = Field(default=20)
    RERANK_TOP_K: int = Field(default=5)
    CONFIDENCE_THRESHOLD: float = Field(default=0.40)
    VECTOR_DIM: int = Field(default=1536)
    MAX_MESSAGE_LENGTH: int = Field(default=8000)

    # --- Celery ---
    CELERY_BROKER_URL: str = Field(default='redis://localhost:6379/1')
    CELERY_RESULT_BACKEND: str = Field(default='redis://localhost:6379/2')

    # --- Rate Limiting ---
    RATE_LIMIT_BACKEND: str = Field(default='redis')
    RATE_LIMIT_BACKEND_URL: str = Field(default='redis://localhost:6379/0')
    RATE_LIMIT_FAIL_MODE: str = Field(default='closed')
    RATE_LIMIT_REFILL_INTERVAL: int = Field(default=60)
    RATE_LIMIT_BURST_MULTIPLIER: float = Field(default=1.5)
    RATE_LIMIT_TRUSTED_PROXIES: str = Field(default='10.0.0.0/8,172.16.0.0/12')
    RATE_LIMIT_PROXY_HEADERS: str = Field(default='X-Forwarded-For,X-Real-IP')

    # Chat message rate limits
    RATE_LIMIT_CHAT_MESSAGE_USER: int = Field(default=10)
    RATE_LIMIT_CHAT_MESSAGE_TENANT: int = Field(default=100)
    RATE_LIMIT_CHAT_MESSAGE_IP: int = Field(default=50)

    # Search rate limits
    RATE_LIMIT_SEARCH_USER: int = Field(default=20)
    RATE_LIMIT_SEARCH_TENANT: int = Field(default=300)
    RATE_LIMIT_SEARCH_IP: int = Field(default=100)

    # Document upload rate limits
    RATE_LIMIT_UPLOAD_USER: int = Field(default=5)
    RATE_LIMIT_UPLOAD_TENANT: int = Field(default=50)
    RATE_LIMIT_UPLOAD_IP: int = Field(default=20)

    # Auth login rate limits
    RATE_LIMIT_AUTH_IP: int = Field(default=10)

    @field_validator('JWT_ALGORITHM')
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']
        if v not in allowed:
            raise ValueError(f'JWT_ALGORITHM must be one of {allowed}')
        return v

    @model_validator(mode='after')
    def _build_database_url(self) -> 'Settings':
        '''Build DATABASE_URL from POSTGRES_* fields if not explicitly provided.'''
        if self.DATABASE_URL is not None:
            # If DATABASE_URL is explicitly provided, use it as-is
            return self
        
        # Check if we have all required POSTGRES fields
        if not all([
            self.POSTGRES_USER,
            self.POSTGRES_PASSWORD.get_secret_value() if self.POSTGRES_PASSWORD else None,
            self.POSTGRES_DB,
            self.POSTGRES_HOST,
            self.POSTGRES_PORT
        ]):
            # If we don't have all POSTGRES fields and no DATABASE_URL, that's an error
            # But we'll let the individual field validators handle missing required fields
            return self
        
        # Construct DATABASE_URL from POSTGRES_* fields
        # URL encode the credentials to handle special characters
        user = urllib.parse.quote_plus(self.POSTGRES_USER)
        password = urllib.parse.quote_plus(self.POSTGRES_PASSWORD.get_secret_value())
        host = self.POSTGRES_HOST
        port = str(self.POSTGRES_PORT)
        db = self.POSTGRES_DB
        
        # Use asyncpg dialect for SQLAlchemy
        self.DATABASE_URL = f'postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}'
        
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

