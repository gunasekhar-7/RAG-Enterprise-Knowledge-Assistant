from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from app.rag.models import DocumentChunk
from app.rag.vectorstore.base import SearchResult


class RetrievedContext(BaseModel):
    """Structured representation of a single retrieved chunk."""

    chunk_id: str
    text: str
    score: float
    citation: str
    metadata: dict[str, Any]


class RankedContext(BaseModel):
    """The complete structured output from the retriever."""

    query: str
    results: list[RetrievedContext]
    total_found: int


class BaseRetriever(ABC):
    """Abstract interface for any retrieval strategy."""

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> RankedContext:
        pass


class BaseKeywordStore(ABC):
    """Abstract interface for sparse/keyword search (BM25)."""

    @abstractmethod
    async def add_documents(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the keyword index."""
        pass

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Perform exact keyword search and return scored SearchResults."""
        pass


class BaseReranker(ABC):
    """Abstract interface for cross-encoder rerankers."""

    @abstractmethod
    async def rerank(
        self, query: str, results: list[RetrievedContext], top_k: int = 5
    ) -> list[RetrievedContext]:
        """Re-score and re-sort a list of candidate contexts against the query."""
        pass
