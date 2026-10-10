from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.config import settings

# Create async engine for PostgreSQL
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True, pool_pre_ping=True)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

Base = declarative_base()


async def get_db_session():  # type: ignore
    """Dependency for getting async DB sessions."""
    async with AsyncSessionLocal() as session:
        yield session
