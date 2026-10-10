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
        logger.info("Initializing LocalEmbedder", model=model_name)

        # This will automatically download the model if not present and use GPU if available
        self._model = SentenceTransformer(self._model_name)

        # To determine dimension, we embed a dummy text
        dummy_embedding = self._model.encode(
            ["dummy"], normalize_embeddings=self._normalize_embeddings
        )[0]
        self._dimension = len(dummy_embedding)
        logger.info("LocalEmbedder initialized", dimension=self._dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        embedding = self._model.encode(
            [text], show_progress_bar=False, normalize_embeddings=self._normalize_embeddings
        )[0]
        # Convert numpy array to list of floats for pgvector compatibility
        return embedding.tolist()  # type: ignore

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings efficiently."""
        if not texts:
            return []

        embeddings = self._model.encode(
            texts, show_progress_bar=False, normalize_embeddings=self._normalize_embeddings
        )
        return embeddings.tolist()  # type: ignore
