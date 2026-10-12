from typing import Any

import structlog

from app.core.exceptions import EmbeddingError, RetrievalError
from app.rag.embedder.base import BaseEmbedder
from app.rag.retrieval.base import BaseRetriever, RankedContext, RetrievedContext
from app.rag.vectorstore.base import BaseVectorStore

logger = structlog.get_logger(__name__)


class VectorRetriever(BaseRetriever):
    """Semantic dense retriever utilizing embeddings and cosine similarity."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        default_top_k: int = 5,
        confidence_threshold: float = 0.0,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.confidence_threshold = confidence_threshold

    async def retrieve(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> RankedContext:
        k = top_k or self.default_top_k
        if not query.strip():
            return RankedContext(query=query, results=[], total_found=0)

        try:
            query_embedding = await self.embedder.embed_text(query)
        except Exception as e:
            raise EmbeddingError(message="Failed to generate query embedding", details={"error": str(e)})

        fetch_k = k + 3

        try:
            raw_results = await self.vector_store.similarity_search(
                query_embedding=query_embedding, top_k=fetch_k, filters=filters
            )
        except Exception as e:
            raise RetrievalError(message="Vector search failed", retriever_type="dense", details={"error": str(e)})

        seen_chunk_ids = set()
        structured_results = []

        for result in raw_results:
            if result.chunk.chunk_id in seen_chunk_ids or result.score < self.confidence_threshold:
                continue

            seen_chunk_ids.add(result.chunk.chunk_id)
            structured_results.append(
                RetrievedContext(
                    chunk_id=result.chunk.chunk_id,
                    text=result.chunk.text,
                    score=result.score,
                    citation=result.chunk.citation_source,
                    metadata=result.chunk.metadata.model_dump(),
                )
            )
            if len(structured_results) == k:
                break

        return RankedContext(
            query=query, results=structured_results, total_found=len(structured_results)
        )
