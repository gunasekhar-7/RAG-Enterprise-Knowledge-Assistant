import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger, setup_logging, request_id_ctx_var
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from app.core.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    setup_logging()
    from app.rag.retrieval.reranker import FlashRankReranker

    logger.info("Loading FlashRankReranker...")
    reranker = FlashRankReranker()
    await reranker.initialize()
    _app.state.reranker = reranker
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

    # Register exception handlers
    register_exception_handlers(app)

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID Middleware
    @app.middleware("http")
    async def add_request_id(request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        request_id_ctx_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

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

    from app.api.v1.api import api_router

    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_application()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)