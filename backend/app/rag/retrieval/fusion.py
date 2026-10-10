import structlog

from app.rag.retrieval.base import RetrievedContext

logger = structlog.get_logger(__name__)


def reciprocal_rank_fusion(
    results_lists: list[list[RetrievedContext]], k: int = 60
) -> list[RetrievedContext]:
    """
    Fuses multiple ranked lists of results using Reciprocal Rank Fusion (RRF).

    Args:
        results_lists: A list containing lists of RetrievedContext.
        k: The RRF constant (default 60 is standard in literature).

    Returns:
        A single fused and sorted list of RetrievedContext.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedContext] = {}

    for results in results_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk_id

            # Store the chunk if we haven't seen it
            if chunk_id not in chunk_map:
                # We copy it because we will mutate the score to be the RRF score
                chunk_map[chunk_id] = result.model_copy()
                rrf_scores[chunk_id] = 0.0

            # Add the RRF component
            rrf_scores[chunk_id] += 1.0 / (k + rank)

    # Apply the new scores
    fused_results = []
    for chunk_id, score in rrf_scores.items():
        chunk = chunk_map[chunk_id]
        chunk.score = score
        fused_results.append(chunk)

    # Sort descending by RRF score
    fused_results.sort(key=lambda x: x.score, reverse=True)

    return fused_results
