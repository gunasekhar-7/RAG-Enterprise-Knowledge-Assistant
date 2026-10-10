from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.models import ChunkMetadata, DocumentChunk
from app.rag.vectorstore.base import SearchResult
from app.rag.vectorstore.pgvector_store import ChunkORM, PGVectorStore


@pytest.fixture
def mock_session():
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def pg_store(mock_session):
    return PGVectorStore(session=mock_session, dimensions=1536)


@pytest.fixture
def sample_chunk():
    doc_id = uuid4()
    return DocumentChunk(
        chunk_id=f"{doc_id}_c0001",
        document_id=doc_id,
        text="This is a test chunk about company policy.",
        embedding=[0.1] * 1536,
        metadata=ChunkMetadata(
            document_id=doc_id,
            tenant_id=doc_id,
            chunk_index=1,
            page_number=5,
            filename="HR_Policy.pdf",
            token_count=10,
            char_count=42,
            custom_metadata={"department": "HR"},
        ),
    )


@pytest.mark.asyncio
async def test_insert_chunk(pg_store, mock_session, sample_chunk):
    await pg_store.insert(sample_chunk)

    mock_session.add.assert_called_once()
    added_orm = mock_session.add.call_args[0][0]

    assert isinstance(added_orm, ChunkORM)
    assert added_orm.chunk_id == sample_chunk.chunk_id
    assert added_orm.document_id == sample_chunk.document_id
    assert added_orm.text == sample_chunk.text

    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_insert(pg_store, mock_session, sample_chunk):
    chunks = [sample_chunk, sample_chunk]
    await pg_store.batch_insert(chunks)

    mock_session.add_all.assert_called_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete(pg_store, mock_session):
    doc_id = uuid4()
    await pg_store.delete(doc_id)

    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_similarity_search(pg_store, mock_session, sample_chunk):
    # Mock the execute result
    mock_result = MagicMock()

    # SQLAlchemy rows are returned as tuples. The query returns (ChunkORM, distance)
    orm_obj = pg_store._to_orm(sample_chunk)
    mock_result.all.return_value = [(orm_obj, 0.15)]

    mock_session.execute.return_value = mock_result

    query_vector = [0.1] * 1536
    filters = {"document_id": sample_chunk.document_id, "filename": "HR_Policy.pdf"}

    results = await pg_store.similarity_search(query_vector, top_k=5, filters=filters)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].score == 0.85  # 1.0 - 0.15
    assert results[0].chunk.chunk_id == sample_chunk.chunk_id

    mock_session.execute.assert_awaited_once()
