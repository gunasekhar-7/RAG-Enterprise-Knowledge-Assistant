import asyncio
import litellm
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.rag.embedder.base import BaseEmbedder

logger = structlog.get_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding model via LiteLLM.
    Suitable for production and cloud environments.
    Supports automatic retries on rate limits and network errors.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", dimensions: int = 1536):
        """
        Initialize the OpenAI embedder.

        Args:
            model_name (str): Model identifier (e.g. text-embedding-3-small).
            dimensions (int): Vector dimension size.
        """
        self._model_name = model_name
        self._dimension = dimensions
        logger.info("Initializing OpenAIEmbedder", model=model_name, dimension=dimensions)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(litellm.exceptions.RateLimitError),
        reraise=True,
    )
    def _embed_text_sync(self, text: str) -> list[float]:
        """Synchronous embedding for single text (for use in thread pool)."""
        try:
            response = litellm.embedding(
                model=self._model_name,
                input=text,
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
                dimensions=self._dimension,
            )
            return response.data[0]["embedding"]  # type: ignore
        except Exception as e:
            logger.error("Failed to embed text", error=str(e), model=self._model_name)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(litellm.exceptions.RateLimitError),
        reraise=True,
    )
    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding for batch (for use in thread pool)."""
        if not texts:
            return []

        try:
            response = litellm.embedding(
                model=self._model_name,
                input=texts,
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
                dimensions=self._dimension,
            )
            # The response.data list is guaranteed to be in the same order as the input
            embeddings = [item["embedding"] for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(
                "Failed to embed batch", error=str(e), model=self._model_name, batch_size=len(texts)
            )
            raise

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string asynchronously."""
        return await asyncio.to_thread(self._embed_text_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings asynchronously."""
        return await asyncio.to_thread(self._embed_batch_sync, texts)