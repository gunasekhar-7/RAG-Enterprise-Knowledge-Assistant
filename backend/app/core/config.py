from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate root directory containing .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    APP_NAME: str = Field(default="RAG Enterprise Knowledge Assistant")
    API_V1_STR: str = Field(default="/api/v1")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # --- Security & Auth ---
    JWT_SECRET: SecretStr = Field(default=SecretStr("insecure-dev-secret-key-change-me"))
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)

    # --- PostgreSQL & pgvector ---
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="rag_knowledge_db")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/rag_knowledge_db"
    )

    # --- Redis ---
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # --- S3 / MinIO ---
    S3_ENDPOINT_URL: str = Field(default="http://localhost:9000")
    S3_BUCKET_NAME: str = Field(default="rag-documents")
    S3_REGION: str = Field(default="us-east-1")
    S3_ACCESS_KEY_ID: str = Field(default="minioadmin")
    S3_SECRET_ACCESS_KEY: SecretStr = Field(default=SecretStr("minioadmin"))

    # --- OpenAI & AI Providers ---
    OPENAI_API_KEY: SecretStr = Field(default=SecretStr("mock-key"))
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    LLM_MODEL: str = Field(default="gpt-4o-mini")
    LLM_TEMPERATURE: float = Field(default=0.1)
    LLM_MAX_TOKENS: int = Field(default=1024)

    # --- RAG Parameters ---
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)
    RETRIEVAL_TOP_K: int = Field(default=20)
    RERANK_TOP_K: int = Field(default=5)
    CONFIDENCE_THRESHOLD: float = Field(default=0.40)

    # --- Celery ---
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
