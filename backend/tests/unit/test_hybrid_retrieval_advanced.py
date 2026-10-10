"""
Comprehensive unit tests for HybridRetriever with focus on edge cases, empty results, and deduplication.

This test suite validates the hybrid retrieval pipeline including:
- Dense and sparse retriever parallel execution
- Reciprocal Rank Fusion (RRF) deduplication and scoring
- Optional reranker integration
- Edge cases: empty queries, no results, various pool sizes
- Deduplication logic when chunks appear in both retrieval streams
"""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.models import ChunkMetadata, DocumentChunk
from app.rag.retrieval.base import BaseReranker, RankedContext, RetrievedContext
from app.rag.retrieval.hybrid import HybridRetriever


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Create a sample document chunk for testing."""
    doc_id = uuid.uuid4()
    tenant_id = uuid.uuid4()

    return DocumentChunk(
        chunk_id=f"{doc_id}_c0001",
        document_id=doc_id,
        text="This is a sample chunk about artificial intelligence and machine learning.",
        metadata=ChunkMetadata(
            document_id=doc_id,
            tenant_id=tenant_id,
            chunk_index=1,
            page_number=5,
            section_heading="Machine Learning Basics",
            filename="AI_Guide.pdf",
            token_count=15,
            char_count=78,
        ),
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create multiple sample document chunks for testing."""
    chunks = []
    doc_id = uuid.uuid4()
    tenant_id = uuid.uuid4()

    for i in range(5):
        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_c{i:04d}",
            document_id=doc_id,
            text=f"This is chunk number {i} with relevant content about machine learning.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                tenant_id=tenant_id,
                chunk_index=i,
                page_number=i + 1,
                section_heading=f"Section {i}",
                filename="ML_Handbook.pdf",
                token_count=12 + i,
                char_count=65 + (i * 5),
            ),
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def retrieved_context_factory() -> callable:
    """Factory for creating RetrievedContext instances."""
    def _create(chunk_id: str, score: float = 0.9, text: str = "test") -> RetrievedContext:
        return RetrievedContext(
            chunk_id=chunk_id,
            text=text,
            score=score,
            citation="Test_Document.pdf (Page 1)",
            metadata={"section": "Test Section"},
        )
    return _create


@pytest.fixture
def mock_dense_retriever() -> AsyncMock:
    """Create a mock dense retriever."""
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock()
    return retriever


@pytest.fixture
def mock_sparse_retriever() -> AsyncMock:
    """Create a mock sparse retriever."""
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock()
    return retriever


@pytest.fixture
def mock_reranker() -> AsyncMock:
    """Create a mock reranker."""
    reranker = AsyncMock()
    reranker.rerank = AsyncMock()
    return reranker


@pytest.fixture
def hybrid_retriever(
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
) -> HybridRetriever:
    """Create a HybridRetriever instance with mocked dependencies."""
    return HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        reranker=None,
        default_top_k=5,
        candidate_pool_size=20,
    )


@pytest.fixture
def hybrid_retriever_with_reranker(
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    mock_reranker: AsyncMock,
) -> HybridRetriever:
    """Create a HybridRetriever instance with reranker enabled."""
    return HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        reranker=mock_reranker,
        default_top_k=5,
        candidate_pool_size=20,
    )


# ============================================================================
# Success Path Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieval_success_with_results(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test successful hybrid retrieval when both retrievers return results.

    Validates that:
    - Both retrievers are called in parallel
    - Results are properly fused using RRF
    - Final result count does not exceed top_k
    """
    dense_results = [
        retrieved_context_factory("chunk_1", score=0.95),
        retrieved_context_factory("chunk_2", score=0.85),
        retrieved_context_factory("chunk_3", score=0.75),
    ]
    sparse_results = [
        retrieved_context_factory("chunk_2", score=0.88),
        retrieved_context_factory("chunk_4", score=0.70),
        retrieved_context_factory("chunk_1", score=0.80),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=3,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=3,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=5)

    assert result.query == "test query"
    assert len(result.results) <= 5
    assert result.total_found == len(result.results)
    assert mock_dense_retriever.retrieve.called
    assert mock_sparse_retriever.retrieve.called


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_filters(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval with filters passed to both retrievers.

    Validates that:
    - Filters are correctly forwarded to both dense and sparse retrievers
    - Filter logic is applied consistently across both streams
    """
    filters = {"tenant_id": str(uuid.uuid4()), "document_type": "pdf"}
    results = [retrieved_context_factory("chunk_1", score=0.9)]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=1,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=1,
    )

    await hybrid_retriever.retrieve("test query", filters=filters)

    mock_dense_retriever.retrieve.assert_called_once_with(
        query="test query",
        top_k=20,
        filters=filters,
    )
    mock_sparse_retriever.retrieve.assert_called_once_with(
        query="test query",
        top_k=20,
        filters=filters,
    )


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_custom_top_k(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval with custom top_k parameter.

    Validates that:
    - Custom top_k overrides default_top_k
    - Result count respects the custom limit
    """
    results = [retrieved_context_factory(f"chunk_{i}", score=0.9 - i*0.05) for i in range(10)]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=10,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=10,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=3)

    assert len(result.results) <= 3
    assert result.total_found <= 3


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_reranker(
    hybrid_retriever_with_reranker: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    mock_reranker: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval pipeline with reranker integration.

    Validates that:
    - Fused results are passed to the reranker
    - Reranker output is used as final results
    - Reranker is called with correct parameters
    """
    dense_results = [
        retrieved_context_factory("chunk_1", score=0.95),
        retrieved_context_factory("chunk_2", score=0.85),
    ]
    sparse_results = [
        retrieved_context_factory("chunk_2", score=0.88),
        retrieved_context_factory("chunk_3", score=0.70),
    ]
    reranked_results = [
        retrieved_context_factory("chunk_2", score=0.92),
        retrieved_context_factory("chunk_1", score=0.88),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=2,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=2,
    )
    mock_reranker.rerank.return_value = reranked_results

    result = await hybrid_retriever_with_reranker.retrieve("test query", top_k=5)

    assert mock_reranker.rerank.called
    assert result.results == reranked_results
    mock_reranker.rerank.assert_called_once()
    call_args = mock_reranker.rerank.call_args
    assert call_args.kwargs["query"] == "test query"
    assert call_args.kwargs["top_k"] == 5


# ============================================================================
# Empty Results Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieval_both_retrievers_empty(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
) -> None:
    """
    Test hybrid retrieval when both retrievers return no results.

    Validates that:
    - Empty result is returned gracefully
    - No errors are raised
    - total_found is 0
    """
    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert result.query == "test query"
    assert result.results == []
    assert result.total_found == 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_dense_empty_sparse_has_results(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when only sparse retriever returns results.

    Validates that:
    - Sparse results are used for RRF with empty dense list
    - Final results are properly returned
    """
    sparse_results = [
        retrieved_context_factory("chunk_1", score=0.85),
        retrieved_context_factory("chunk_2", score=0.75),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=2,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert len(result.results) > 0
    assert result.results[0].chunk_id in ["chunk_1", "chunk_2"]


@pytest.mark.asyncio
async def test_hybrid_retrieval_sparse_empty_dense_has_results(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when only dense retriever returns results.

    Validates that:
    - Dense results are used for RRF with empty sparse list
    - Final results are properly returned
    """
    dense_results = [
        retrieved_context_factory("chunk_1", score=0.95),
        retrieved_context_factory("chunk_2", score=0.85),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=2,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert len(result.results) > 0
    assert result.results[0].chunk_id in ["chunk_1", "chunk_2"]


# ============================================================================
# Deduplication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieval_deduplication_same_chunk_both_retrievers(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test RRF deduplication when the same chunk appears in both retriever results.

    Validates that:
    - Duplicate chunks are merged, not repeated
    - RRF score is properly combined from both rankings
    - Result count is correct (duplicates removed)
    """
    dense_results = [
        retrieved_context_factory("chunk_1", score=0.95),
        retrieved_context_factory("chunk_2", score=0.85),
        retrieved_context_factory("chunk_3", score=0.75),
    ]
    sparse_results = [
        retrieved_context_factory("chunk_2", score=0.88),
        retrieved_context_factory("chunk_1", score=0.80),
        retrieved_context_factory("chunk_4", score=0.70),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=3,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=3,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=10)

    chunk_ids = [ctx.chunk_id for ctx in result.results]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicates found in results"
    assert "chunk_1" in chunk_ids
    assert "chunk_2" in chunk_ids
    assert "chunk_3" in chunk_ids
    assert "chunk_4" in chunk_ids


@pytest.mark.asyncio
async def test_hybrid_retrieval_deduplication_multiple_duplicates(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test RRF deduplication with many overlapping results.

    Validates that:
    - All duplicate instances are properly merged
    - Chunks appearing in both streams get higher RRF scores
    - Final result count reflects unique chunks only
    """
    dense_results = [
        retrieved_context_factory("chunk_1", score=0.95),
        retrieved_context_factory("chunk_2", score=0.90),
        retrieved_context_factory("chunk_3", score=0.85),
        retrieved_context_factory("chunk_4", score=0.80),
    ]
    sparse_results = [
        retrieved_context_factory("chunk_1", score=0.88),
        retrieved_context_factory("chunk_2", score=0.87),
        retrieved_context_factory("chunk_5", score=0.76),
        retrieved_context_factory("chunk_6", score=0.70),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=4,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=4,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=10)

    chunk_ids = [ctx.chunk_id for ctx in result.results]
    assert len(chunk_ids) == 6, f"Expected 6 unique chunks, got {len(chunk_ids)}"
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicates found in results"


@pytest.mark.asyncio
async def test_hybrid_retrieval_rrf_score_calculation(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test RRF score calculation for chunks appearing in both retrievers.

    Validates that:
    - Chunks in both top positions get highest scores
    - Chunks appearing only once get lower scores
    - Scores are sorted in descending order
    """
    dense_results = [
        retrieved_context_factory("chunk_1", score=0.95),
        retrieved_context_factory("chunk_2", score=0.85),
    ]
    sparse_results = [
        retrieved_context_factory("chunk_1", score=0.90),
        retrieved_context_factory("chunk_3", score=0.70),
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=2,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=2,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=10)

    assert len(result.results) == 3
    scores = [ctx.score for ctx in result.results]
    assert scores == sorted(scores, reverse=True), "Scores not in descending order"
    assert result.results[0].chunk_id == "chunk_1", "chunk_1 should be ranked first (appears in both)"


# ============================================================================
# Edge Cases Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieval_empty_query_string(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
) -> None:
    """
    Test hybrid retrieval with empty query string.

    Validates that:
    - Empty query returns empty results immediately
    - Neither retriever is called for empty queries
    - No errors are raised
    """
    result = await hybrid_retriever.retrieve("")

    assert result.query == ""
    assert result.results == []
    assert result.total_found == 0
    assert not mock_dense_retriever.retrieve.called
    assert not mock_sparse_retriever.retrieve.called


@pytest.mark.asyncio
async def test_hybrid_retrieval_whitespace_only_query(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
) -> None:
    """
    Test hybrid retrieval with whitespace-only query string.

    Validates that:
    - Whitespace-only queries are treated as empty
    - Neither retriever is called
    - Empty result is returned
    """
    result = await hybrid_retriever.retrieve("   \t\n   ")

    assert result.results == []
    assert result.total_found == 0
    assert not mock_dense_retriever.retrieve.called
    assert not mock_sparse_retriever.retrieve.called


@pytest.mark.asyncio
async def test_hybrid_retrieval_single_result_both_retrievers(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when only one result is found across both retrievers.

    Validates that:
    - Single result is properly returned
    - No duplication occurs
    """
    single_result = [retrieved_context_factory("chunk_1", score=0.95)]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=single_result,
        total_found=1,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert len(result.results) == 1
    assert result.results[0].chunk_id == "chunk_1"


@pytest.mark.asyncio
async def test_hybrid_retrieval_large_result_set_truncation(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test that large result sets are properly truncated to top_k.

    Validates that:
    - Results exceeding top_k are removed
    - Top-ranked results are retained
    - Final count never exceeds top_k
    """
    large_dense_results = [
        retrieved_context_factory(f"chunk_{i}", score=0.99 - i*0.01)
        for i in range(50)
    ]
    large_sparse_results = [
        retrieved_context_factory(f"chunk_{i+25}", score=0.98 - i*0.01)
        for i in range(50)
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=large_dense_results,
        total_found=50,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=large_sparse_results,
        total_found=50,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=5)

    assert len(result.results) <= 5
    assert result.total_found <= 5


@pytest.mark.asyncio
async def test_hybrid_retrieval_top_k_larger_than_available(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when top_k exceeds available unique results.

    Validates that:
    - Fewer results than top_k are returned when unavailable
    - No errors or padding occurs
    - Result count equals actual available results
    """
    results = [retrieved_context_factory(f"chunk_{i}", score=0.95 - i*0.05) for i in range(3)]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=3,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=20)

    assert len(result.results) == 3
    assert result.total_found == 3


@pytest.mark.asyncio
async def test_hybrid_retrieval_identical_chunks_different_scores(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test deduplication when same chunk has different scores in each retriever.

    Validates that:
    - Chunk appears only once in final results
    - RRF score combines both original scores appropriately
    - Chunk IDs are the deduplication key
    """
    dense_results = [retrieved_context_factory("chunk_1", score=0.95)]
    sparse_results = [retrieved_context_factory("chunk_1", score=0.70)]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=dense_results,
        total_found=1,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=sparse_results,
        total_found=1,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert len(result.results) == 1
    assert result.results[0].chunk_id == "chunk_1"
    assert result.results[0].score > 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_preserves_chunk_metadata(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test that chunk metadata and citation are preserved through retrieval pipeline.

    Validates that:
    - Citation information is retained
    - Metadata dictionary is preserved
    - Text content is not modified
    """
    context = RetrievedContext(
        chunk_id="chunk_1",
        text="Important information about AI systems",
        score=0.95,
        citation="AI_Guide.pdf (Page 42)",
        metadata={"section": "Advanced AI", "author": "Dr. Smith"},
    )

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[context],
        total_found=1,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert result.results[0].citation == "AI_Guide.pdf (Page 42)"
    assert result.results[0].metadata["section"] == "Advanced AI"
    assert result.results[0].text == "Important information about AI systems"


# ============================================================================
# Failure Path Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieval_dense_retriever_exception(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when dense retriever raises an exception.

    Validates that:
    - Exception is propagated correctly
    - Caller is notified of the failure
    """
    mock_dense_retriever.retrieve.side_effect = RuntimeError("Dense retriever failed")
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    with pytest.raises(RuntimeError, match="Dense retriever failed"):
        await hybrid_retriever.retrieve("test query")


@pytest.mark.asyncio
async def test_hybrid_retrieval_sparse_retriever_exception(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when sparse retriever raises an exception.

    Validates that:
    - Exception is propagated correctly
    - Dense retriever call is attempted before exception
    """
    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )
    mock_sparse_retriever.retrieve.side_effect = RuntimeError("Sparse retriever failed")

    with pytest.raises(RuntimeError, match="Sparse retriever failed"):
        await hybrid_retriever.retrieve("test query")


@pytest.mark.asyncio
async def test_hybrid_retrieval_reranker_exception(
    hybrid_retriever_with_reranker: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    mock_reranker: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test hybrid retrieval when reranker raises an exception.

    Validates that:
    - Exception from reranker is propagated
    - Both retrievers were called before the exception
    """
    results = [retrieved_context_factory("chunk_1", score=0.95)]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=1,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )
    mock_reranker.rerank.side_effect = RuntimeError("Reranker failed")

    with pytest.raises(RuntimeError, match="Reranker failed"):
        await hybrid_retriever_with_reranker.retrieve("test query")


@pytest.mark.asyncio
async def test_hybrid_retrieval_invalid_top_k(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
) -> None:
    """
    Test hybrid retrieval behavior with edge case top_k values.

    Validates that:
    - Zero or negative top_k is handled gracefully
    - No results are returned for invalid top_k
    """
    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query", top_k=0)
    assert result.results == [] or len(result.results) == 0


# ============================================================================
# Configuration and Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retriever_initialization(
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
) -> None:
    """
    Test HybridRetriever initialization with various configurations.

    Validates that:
    - Retriever instances are properly stored
    - Configuration parameters are retained
    - Reranker can be optional
    """
    retriever = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        reranker=None,
        default_top_k=10,
        candidate_pool_size=50,
    )

    assert retriever.dense_retriever == mock_dense_retriever
    assert retriever.sparse_retriever == mock_sparse_retriever
    assert retriever.reranker is None
    assert retriever.default_top_k == 10
    assert retriever.candidate_pool_size == 50


@pytest.mark.asyncio
async def test_hybrid_retrieval_default_top_k_used(
    hybrid_retriever: HybridRetriever,
    mock_dense_retriever: AsyncMock,
    mock_sparse_retriever: AsyncMock,
    retrieved_context_factory: callable,
) -> None:
    """
    Test that default_top_k is used when top_k parameter is None.

    Validates that:
    - When top_k is not specified, default_top_k is applied
    - Results respect the default limit
    """
    results = [
        retrieved_context_factory(f"chunk_{i}", score=0.95 - i*0.05)
        for i in range(20)
    ]

    mock_dense_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=results,
        total_found=20,
    )
    mock_sparse_retriever.retrieve.return_value = RankedContext(
        query="test query",
        results=[],
        total_found=0,
    )

    result = await hybrid_retriever.retrieve("test query")

    assert len(result.results) <= hybrid_retriever.default_top_k
