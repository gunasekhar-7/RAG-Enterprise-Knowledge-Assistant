import structlog
from flashrank import Ranker, RerankRequest

from app.rag.retrieval.base import BaseReranker, RetrievedContext

logger = structlog.get_logger(__name__)


class FlashRankReranker(BaseReranker):
    """
    Cross-Encoder reranker using FlashRank.
    Extremely fast, memory-efficient CPU/GPU reranking.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        logger.info("Loading FlashRank model", model=self.model_name)
        # Note: model will download on first run
        self.ranker = Ranker(model_name=self.model_name)
        logger.info("FlashRank model loaded")

    async def rerank(
        self, query: str, results: list[RetrievedContext], top_k: int = 5
    ) -> list[RetrievedContext]:
        if not results:
            return []

        logger.info("Reranking candidates", candidate_count=len(results), top_k=top_k)

        # Convert to FlashRank input format
        # List of dictionaries with 'id' and 'text'
        passages = [
            {"id": str(i), "text": res.text, "_original_context": res}
            for i, res in enumerate(results)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)

        # Execute rerank
        reranked_raw = self.ranker.rerank(rerank_request)

        reranked_results = []
        for item in reranked_raw[:top_k]:
            # Recover the original object
            original_ctx: RetrievedContext = item["_original_context"]
            # Overwrite score with cross-encoder score
            original_ctx.score = float(item["score"])
            reranked_results.append(original_ctx)

        logger.debug("Reranking complete")
        return reranked_results
