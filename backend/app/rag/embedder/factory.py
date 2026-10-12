import asyncio
import structlog

from app.core.config import settings
from app.rag.embedder.base import BaseEmbedder
from app.rag.embedder.local import LocalEmbedder
from app.rag.embedder.openai_embedder import OpenAIEmbedder

logger = structlog.get_logger(__name__)


class EmbedderFactory:
    """Factory to instantiate the appropriate embedder based on environment and config."""

    _embedder: BaseEmbedder | None = None

    @classmethod
    async def get_embedder(cls) -> BaseEmbedder:
        """
        Returns the configured embedder instance.
        If ENVIRONMENT is 'development' and EMBEDDING_MODEL is local, it loads LocalEmbedder.
        Otherwise, it loads OpenAIEmbedder.
        """
        if cls._embedder is not None:
            return cls._embedder

        model_name = settings.EMBEDDING_MODEL
        env = settings.ENVIRONMENT

        logger.info("Instantiating embedder", model=model_name, environment=env)

        if "bge" in model_name.lower() or "sentence-transformers" in model_name.lower():
            # Local embedding model
            embedder = LocalEmbedder(model_name=model_name)
            await embedder.initialize()
        elif "text-embedding" in model_name.lower():
            # OpenAI embedding model
            embedder = OpenAIEmbedder(model_name=model_name, dimensions=settings.VECTOR_DIM)
        else:
            # Fallback to OpenAI if not explicitly recognized as local
            logger.warning(
                f"Unrecognized embedding model '{model_name}', defaulting to OpenAIEmbedder."
            )
            embedder = OpenAIEmbedder(model_name=model_name, dimensions=settings.VECTOR_DIM)

        cls._embedder = embedder
        return embedder

    @classmethod
    def reset(cls) -> None:
        """Reset the cached embedder (for testing)."""
        cls._embedder = None
