import hashlib

import structlog

from app.rag.embedder.factory import EmbedderFactory
from app.rag.models import DocumentChunk

logger = structlog.get_logger(__name__)


class EmbeddingPipeline:
    """
    Pipeline for taking DocumentChunk objects, batching their texts,
    and attaching the resulting embeddings and metadata back to the chunks.
    Includes an in-memory cache to prevent re-embedding identical text.
    """

    def __init__(self, batch_size: int = 100):
        self._batch_size = batch_size
        # Simple in-memory cache mapping md5(text) -> embedding
        self._cache: dict[str, list[float]] = {}
        self._embedder = None
        self._initialized = False

    async def _get_embedder(self):
        if self._embedder is None:
            self._embedder = await EmbedderFactory.get_embedder()
        return self._embedder

    def _get_hash(self, text: str) -> str:
        """Return MD5 hash of the text for cache keying."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def process_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Embeds a list of DocumentChunks in batches.
        Mutates the chunks in-place to add .embedding and .metadata.embedding_model,
        then returns them.
        """
        if not chunks:
            return chunks

        embedder = await self._get_embedder()
        model_name = embedder.model_name
        logger.info("Initialized EmbeddingPipeline", model=model_name, batch_size=self._batch_size)

        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]

            # Texts that need to be embedded (cache miss)
            texts_to_embed = []
            chunk_to_hash = []

            for chunk in batch:
                text_hash = self._get_hash(chunk.text)
                chunk_to_hash.append((chunk, text_hash))
                if text_hash not in self._cache:
                    texts_to_embed.append(chunk.text)

            # Embed the cache misses
            if texts_to_embed:
                logger.debug(f"Embedding batch of {len(texts_to_embed)} chunks (cache misses)...")
                embeddings = await embedder.embed_batch(texts_to_embed)

                # Update cache
                for text, emb in zip(texts_to_embed, embeddings, strict=False):
                    self._cache[self._get_hash(text)] = emb
            else:
                logger.debug("All chunks in batch found in cache.")

            # Assign embeddings back to all chunks in the batch
            for chunk, text_hash in chunk_to_hash:
                chunk.embedding = self._cache[text_hash]
                chunk.metadata.embedding_model = model_name

        logger.info("Finished embedding chunks", total=len(chunks), model=model_name)
        return chunks
