import structlog

from app.core.config import settings
from app.rag.embedder.base import BaseEmbedder
from app.rag.embedder.local import LocalEmbedder
from app.rag.embedder.openai_embedder import OpenAIEmbedder

logger = structlog.get_logger(__name__)


class EmbedderFactory:
    """Factory to instantiate the appropriate embedder based on environment and config."""

    @staticmethod
    def get_embedder() -> BaseEmbedder:
        """
        Returns the configured embedder instance.
        If ENVIRONMENT is 'development' and EMBEDDING_MODEL is local, it loads LocalEmbedder.
        Otherwise, it loads OpenAIEmbedder.
        """
        model_name = settings.EMBEDDING_MODEL
        env = settings.ENVIRONMENT

        logger.info("Instantiating embedder", model=model_name, environment=env)

        if "bge" in model_name.lower() or "sentence-transformers" in model_name.lower():
            # Local embedding model
            return LocalEmbedder(model_name=model_name)
        elif "text-embedding" in model_name.lower():
            # OpenAI embedding model
            # Note: dimension can be configured, default to 1536 for text-embedding-3-small
            dimensions = 1536
            if "large" in model_name.lower():
                dimensions = 3072
            return OpenAIEmbedder(model_name=model_name, dimensions=dimensions)
        else:
            # Fallback to OpenAI if not explicitly recognized as local
            logger.warning(
                f"Unrecognized embedding model '{model_name}', defaulting to OpenAIEmbedder."
            )
            return OpenAIEmbedder(model_name=model_name)
