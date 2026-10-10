from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from app.rag.models import DocumentChunk


class SearchResult:
    """Wrapper for a vector search result."""

    def __init__(self, chunk: DocumentChunk, score: float):
        self.chunk = chunk
        self.score = score


class BaseVectorStore(ABC):
    """
    Abstract repository for Vector Database operations.
    Ensures the RAG pipeline is decoupled from the underlying storage technology.
    """

    @abstractmethod
    async def insert(self, chunk: DocumentChunk) -> None:
        """Insert a single document chunk into the vector store."""
        pass

    @abstractmethod
    async def batch_insert(self, chunks: list[DocumentChunk]) -> None:
        """Insert a batch of document chunks efficiently."""
        pass

    @abstractmethod
    async def similarity_search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Search for most similar chunks.

        Args:
            query_embedding: The vector representation of the search query.
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g., {"document_id": "123"}).

        Returns:
            List of SearchResult objects containing chunks and their similarity scores.
        """
        pass

    @abstractmethod
    async def delete(self, document_id: UUID) -> None:
        """Delete all chunks associated with a specific document ID."""
        pass

    @abstractmethod
    async def update(self, document_id: UUID, chunks: list[DocumentChunk]) -> None:
        """
        Update chunks for a document.
        Typically implemented as a delete followed by batch_insert.
        """
        pass

    @abstractmethod
    async def create_collection(self) -> None:
        """Initialize the collection/index if it does not exist."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connection to the vector store."""
        pass
