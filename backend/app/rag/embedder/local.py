import asyncio
import structlog
from sentence_transformers import SentenceTransformer

from app.rag.embedder.base import BaseEmbedder

logger = structlog.get_logger(__name__)


class LocalEmbedder(BaseEmbedder):
    """
    Local embedding model using SentenceTransformers.
    Suitable for local development and offline environments.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", normalize_embeddings: bool = True
    ):
        """
        Initialize the local embedder.

        Args:
            model_name (str): HuggingFace model identifier.
            normalize_embeddings (bool): If True, L2-normalizes the vectors so dot product == cosine similarity.
        """
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._initialized = False
        logger.info("LocalEmbedder created", model=model_name)

    async def initialize(self) -> None:
        """Initialize the SentenceTransformer model asynchronously."""
        if self._initialized:
            return
        logger.info("Initializing LocalEmbedder model", model=self._model_name)
        # Offload model loading to thread pool to avoid blocking the event loop
        self._model = await asyncio.to_thread(SentenceTransformer, self._model_name)

        # To determine dimension, we embed a dummy text (also in thread pool)
        dummy_embedding = await asyncio.to_thread(
            self._model.encode,
            ["dummy"],
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        self._dimension = len(dummy_embedding[0])
        self._initialized = True
        logger.info("LocalEmbedder initialized", dimension=self._dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("LocalEmbedder not initialized. Call initialize() first.")
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string asynchronously."""
        if not self._initialized:
            await self.initialize()

        embedding = await asyncio.to_thread(
            self._model.encode,
            [text],
            show_progress_bar=False,
            normalize_embeddings=self._normalize_embeddings,
        )
        # Convert numpy array to list of floats for pgvector compatibility
        return embedding[0].tolist()  # type: ignore

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings efficiently asynchronously."""
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            show_progress_bar=False,
            normalize_embeddings=self._normalize_embeddings,
        )
        return embeddings.tolist()  # type: ignore
