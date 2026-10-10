import asyncio
from typing import Any

import structlog

from app.core.exceptions import RetrievalError, RerankerError
from app.rag.retrieval.base import BaseReranker, BaseRetriever, RankedContext
from app.rag.retrieval.fusion import reciprocal_rank_fusion

logger = structlog.get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Advanced Retriever coordinating Dense + Sparse parallel retrieval,
    Reciprocal Rank Fusion, and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        reranker: BaseReranker | None = None,
        default_top_k: int = 5,
        candidate_pool_size: int = 20,
    ):
        """
        Args:
            dense_retriever: Retriever handling vector search.
            sparse_retriever: Retriever handling keyword search.
            reranker: Optional cross-encoder for final scoring.
            default_top_k: Final number of chunks to return.
            candidate_pool_size: How many chunks to pull from each retriever for fusion.
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.candidate_pool_size = candidate_pool_size

    async def retrieve(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> RankedContext:
        k = top_k or self.default_top_k
        if not query.strip():
            return RankedContext(query=query, results=[], total_found=0)

        logger.info("Starting hybrid retrieval pipeline", query=query)

        # 1. Execute both retrievers in parallel to build the candidate pool
        try:
            dense_task = self.dense_retriever.retrieve(
                query=query, top_k=self.candidate_pool_size, filters=filters
            )
            sparse_task = self.sparse_retriever.retrieve(
                query=query, top_k=self.candidate_pool_size, filters=filters
            )

            dense_ctx, sparse_ctx = await asyncio.gather(dense_task, sparse_task)
        except Exception as e:
            raise RetrievalError(message="Parallel retrieval failed", retriever_type="hybrid", details={"error": str(e)})

        logger.debug(
            "Parallel retrieval complete",
            dense_found=dense_ctx.total_found,
            sparse_found=sparse_ctx.total_found,
        )

        # 2. Fuse candidates using RRF
        fused_results = reciprocal_rank_fusion([dense_ctx.results, sparse_ctx.results], k=60)

        if not fused_results:
            logger.info("No candidates found in either retriever")
            return RankedContext(query=query, results=[], total_found=0)

        # 3. Rerank if a reranker is provided
        if self.reranker:
            try:
                final_results = await self.reranker.rerank(query=query, results=fused_results, top_k=k)
            except Exception as e:
                raise RerankerError(message="Reranking failed", details={"error": str(e)})
        else:
            # If no reranker, just truncate the fused list
            final_results = fused_results[:k]

        logger.info("Hybrid retrieval pipeline complete", final_count=len(final_results))

        return RankedContext(query=query, results=final_results, total_found=len(final_results))
