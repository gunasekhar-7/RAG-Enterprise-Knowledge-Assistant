from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.rag.models import ChunkMetadata, DocumentChunk
from app.rag.retrieval.base import RetrievedContext
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.sparse import KeywordRetriever


@pytest.fixture
def sample_chunks():
    doc_id = uuid4()
    return [
        DocumentChunk(
            chunk_id=f"{doc_id}_c0001",
            document_id=doc_id,
            text="The secret error code is ERR-592-A.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                tenant_id=doc_id,
                chunk_index=1,
                page_number=1,
                filename="test.pdf",
                token_count=10,
                char_count=50,
            ),
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_c0002",
            document_id=doc_id,
            text="Standard error handling procedures for the enterprise.",
            metadata=ChunkMetadata(
                document_id=doc_id,
                tenant_id=doc_id,
                chunk_index=2,
                page_number=1,
                filename="test.pdf",
                token_count=10,
                char_count=50,
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_reciprocal_rank_fusion():
    doc1 = RetrievedContext(chunk_id="1", text="A", score=0.9, citation="doc", metadata={})
    doc2 = RetrievedContext(chunk_id="2", text="B", score=0.8, citation="doc", metadata={})
    doc3 = RetrievedContext(chunk_id="3", text="C", score=0.7, citation="doc", metadata={})

    # List 1: doc1, doc2
    # List 2: doc2, doc3
    fused = reciprocal_rank_fusion([[doc1, doc2], [doc2, doc3]], k=60)

    # doc2 is rank 2 in list 1 (1/62) and rank 1 in list 2 (1/61)
    # doc1 is rank 1 in list 1 (1/61)
    # RRF(doc2) = 1/62 + 1/61 = 0.0325
    # RRF(doc1) = 1/61 = 0.0163
    # Therefore doc2 should win the fused ranking.
    assert fused[0].chunk_id == "2"
    assert fused[1].chunk_id == "1"
    assert fused[2].chunk_id == "3"


@pytest.mark.asyncio
async def test_hybrid_retriever_flow():
    dense_mock = AsyncMock()
    sparse_mock = AsyncMock()
    reranker_mock = AsyncMock()

    doc_a = RetrievedContext(
        chunk_id="A", text="Semantic Match", score=0.9, citation="x", metadata={}
    )
    doc_b = RetrievedContext(
        chunk_id="B", text="Keyword Match", score=0.9, citation="x", metadata={}
    )

    # Setup mocks to return RankedContext objects
    from app.rag.retrieval.base import RankedContext

    dense_mock.retrieve.return_value = RankedContext(query="test", results=[doc_a], total_found=1)
    sparse_mock.retrieve.return_value = RankedContext(query="test", results=[doc_b], total_found=1)

    reranker_mock.rerank.return_value = [doc_b, doc_a]  # Simulated reranker swapped them

    retriever = HybridRetriever(
        dense_retriever=dense_mock,
        sparse_retriever=sparse_mock,
        reranker=reranker_mock,
        default_top_k=2,
    )

    res = await retriever.retrieve("test")

    dense_mock.retrieve.assert_awaited_once()
    sparse_mock.retrieve.assert_awaited_once()
    reranker_mock.rerank.assert_awaited_once()

    assert res.total_found == 2
    assert res.results[0].chunk_id == "B"  # Reranker winner


@pytest.mark.asyncio
async def test_keyword_retriever_flow(sample_chunks):
    store = AsyncMock()
    from app.rag.vectorstore.base import SearchResult

    store.search.return_value = [SearchResult(chunk=sample_chunks[0], score=1.0)]

    retriever = KeywordRetriever(keyword_store=store)
    res = await retriever.retrieve("ERR-592-A")

    store.search.assert_awaited_once_with(query="ERR-592-A", top_k=5, filters=None)
    assert res.total_found == 1
    assert "secret error code" in res.results[0].text
