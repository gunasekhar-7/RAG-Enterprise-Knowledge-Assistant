from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    setup_logging()
    logger.info(
        "application_starting",
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
    )
    yield
    # Shutdown
    logger.info("application_shutdown", app_name=settings.APP_NAME)


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Production-grade RAG Enterprise Knowledge Assistant API.",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root & Health Check Endpoints
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "status": "online",
            "docs_url": "/docs",
        }

    @app.get("/health", tags=["Health"])
    @app.get("/health/live", tags=["Health"])
    async def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "version": "0.1.0",
        }

    return app


app = create_application()
