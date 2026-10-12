from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.rag.embedder.base import BaseEmbedder
from app.rag.embedder.factory import EmbedderFactory
from app.rag.embedder.local import LocalEmbedder
from app.rag.embedder.openai_embedder import OpenAIEmbedder
from app.rag.embedder.pipeline import EmbeddingPipeline
from app.rag.models import ChunkMetadata, DocumentChunk


class MockEmbedder(BaseEmbedder):
    @property
    def model_name(self) -> str:
        return "mock-embedder"

    @property
    def dimension(self) -> int:
        return 3

    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def dummy_chunks() -> list[DocumentChunk]:
    doc_id = uuid4()
    return [
        DocumentChunk(
            chunk_id=f"{doc_id}_c0000",
            document_id=doc_id,
            text="First test chunk",
            metadata=ChunkMetadata(
                document_id=doc_id,
                tenant_id=doc_id,
                chunk_index=0,
                page_number=1,
                filename="test.pdf",
                token_count=10,
                char_count=50,
            ),
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_c0001",
            document_id=doc_id,
            text="Second test chunk",
            metadata=ChunkMetadata(
                document_id=doc_id,
                tenant_id=doc_id,
                chunk_index=1,
                page_number=1,
                filename="test.pdf",
                token_count=15,
                char_count=60,
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_embedding_pipeline_process_chunks(dummy_chunks):
    """Test that pipeline successfully embeds chunks and uses cache."""
    with patch(
        "app.rag.embedder.factory.EmbedderFactory.get_embedder", new_callable=AsyncMock, return_value=MockEmbedder()
    ):
        pipeline = EmbeddingPipeline(batch_size=2)

        # Process first time (cache miss)
        processed = await pipeline.process_chunks(dummy_chunks)

        assert len(processed) == 2
        assert processed[0].embedding == [0.1, 0.2, 0.3]
        assert processed[0].metadata.embedding_model == "mock-embedder"
        assert processed[1].embedding == [0.1, 0.2, 0.3]

        # Verify cache was populated
        assert len(pipeline._cache) == 2

        # Process again with identical text (cache hit)
        # We can mock the embedder's embed_batch to throw an error if called again,
        # proving it hits the cache.
        with patch.object(
            pipeline._embedder, "embed_batch", side_effect=Exception("Should not be called!")
        ):
            processed_again = await pipeline.process_chunks(dummy_chunks)
            assert processed_again[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_factory_returns_correct_embedder():
    with patch("app.core.config.settings.EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"):  # noqa: SIM117
        with patch("app.rag.embedder.local.SentenceTransformer"):
            # Mock SentenceTransformer so it doesn't download weights during unit tests
            EmbedderFactory.reset()
            embedder = await EmbedderFactory.get_embedder()
            assert isinstance(embedder, LocalEmbedder)
            assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    with patch("app.core.config.settings.EMBEDDING_MODEL", "text-embedding-3-small"):  # noqa: SIM117
        EmbedderFactory.reset()
        embedder = await EmbedderFactory.get_embedder()
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.dimension == 1536


@pytest.mark.asyncio
async def test_openai_embedder_embed_text():
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small", dimensions=1536)

    mock_response = MagicMock()
    mock_response.data = [{"embedding": [0.5, 0.5]}]

    with patch("litellm.embedding", return_value=mock_response) as mock_embedding:
        res = await embedder.embed_text("hello world")
        assert res == [0.5, 0.5]
        mock_embedding.assert_called_once()
        kwargs = mock_embedding.call_args.kwargs
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == "hello world"
        assert kwargs["dimensions"] == 1536
