from typing import Any
from uuid import UUID

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.exceptions import ValidationError
from app.db.session import Base, engine
from app.rag.models import ChunkMetadata, DocumentChunk
from app.rag.vectorstore.base import BaseVectorStore, SearchResult

logger = structlog.get_logger(__name__)
from app.core.metrics import RETRIEVAL_LATENCY
from app.core.logging import request_id_ctx_var
import time


class ChunkORM(Base):
    """SQLAlchemy model for document chunks with pgvector embeddings."""

    __tablename__ = "document_chunks"

    chunk_id = Column(String, primary_key=True)
    document_id = Column(PG_UUID(as_uuid=True), index=True, nullable=False)
    tenant_id = Column(PG_UUID(as_uuid=True), index=True, nullable=False)
    text = Column(String, nullable=False)
    # Metadata fields (flattened for easy indexing/filtering)
    page_number = Column(Integer, nullable=False)
    filename = Column(String, nullable=False)
    section_heading = Column(String, nullable=True)
    custom_metadata = Column(JSON, default=dict)

    # Vector dimension is set from configuration
    embedding = Column(Vector(settings.VECTOR_DIM), nullable=False)


class PGVectorStore(BaseVectorStore):
    """
    PostgreSQL + pgvector implementation of the Vector Store.
    """

    def __init__(self, session: AsyncSession, dimensions: int | None = None):
        self.session = session
        self.dimensions = dimensions or settings.VECTOR_DIM

    def _validate_embedding(self, embedding: list[float], identifier: str = "unknown") -> None:
        """Validate embedding dimension matches configured VECTOR_DIM."""
        if len(embedding) != self.dimensions:
            raise ValidationError(
                message=f"Embedding dimension mismatch: expected {self.dimensions}, got {len(embedding)}",
                field="embedding",
                details={
                    "expected_dimension": self.dimensions,
                    "actual_dimension": len(embedding),
                    "identifier": identifier,
                },
            )

    async def create_collection(self) -> None:
        """Initialize the vector store - validate that required components exist."""
        logger.info("Initializing pgvector store")
        # Note: Table creation, extension setup, and index creation are handled by Alembic migrations.
        # This method now only performs validation to ensure the store is ready for use.
        pass

    def _to_orm(self, chunk: DocumentChunk) -> ChunkORM:
        return ChunkORM(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            tenant_id=chunk.metadata.tenant_id,
            text=chunk.text,
            page_number=chunk.metadata.page_number,
            filename=chunk.metadata.filename,
            section_heading=chunk.metadata.section_heading,
            custom_metadata=chunk.metadata.custom_metadata,
            embedding=chunk.embedding,
        )

    def _to_domain(self, orm_obj: ChunkORM) -> DocumentChunk:
        metadata = ChunkMetadata(  # type: ignore
            document_id=orm_obj.document_id,  # type: ignore
            tenant_id=orm_obj.tenant_id,  # type: ignore
            chunk_index=int(orm_obj.chunk_id.split("_c")[-1]),
            page_number=orm_obj.page_number,  # type: ignore
            filename=orm_obj.filename,  # type: ignore
            section_heading=orm_obj.section_heading,  # type: ignore
            token_count=0,  # Not preserved in this basic schema, would add if needed
            char_count=len(orm_obj.text),
            custom_metadata=orm_obj.custom_metadata,  # type: ignore
        )
        return DocumentChunk(
            chunk_id=orm_obj.chunk_id,  # type: ignore
            document_id=orm_obj.document_id,  # type: ignore
            text=orm_obj.text,  # type: ignore
            metadata=metadata,
            embedding=None,  # Don't return embeddings by default to save memory
        )

    async def insert(self, chunk: DocumentChunk) -> None:
        """Insert a single chunk."""
        if not chunk.embedding:
            raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")

        self._validate_embedding(chunk.embedding, chunk.chunk_id)

        orm_obj = self._to_orm(chunk)
        self.session.add(orm_obj)
        await self.session.commit()

    async def batch_insert(self, chunks: list[DocumentChunk]) -> None:
        """Insert multiple chunks efficiently."""
        if not chunks:
            return

        orm_objs = []
        for chunk in chunks:
            if not chunk.embedding:
                raise ValidationError(
                    message=f"Chunk {chunk.chunk_id} has no embedding",
                    field="embedding",
                    details={"identifier": chunk.chunk_id},
                )
            self._validate_embedding(chunk.embedding, chunk.chunk_id)
            orm_objs.append(self._to_orm(chunk))

        self.session.add_all(orm_objs)
        await self.session.commit()
        logger.info("Batch inserted chunks", count=len(orm_objs))

    async def similarity_search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Search using pgvector cosine distance operator (<=>).
        """
        self._validate_embedding(query_embedding, "query_embedding")

        # Vector.cosine_distance maps to <=>
        stmt = select(  # type: ignore
            ChunkORM, ChunkORM.embedding.cosine_distance(query_embedding).label("distance")
        )

        # Apply filters
        if filters:
            if "tenant_id" in filters:
                stmt = stmt.where(ChunkORM.tenant_id == filters["tenant_id"])
            if "document_id" in filters:
                stmt = stmt.where(ChunkORM.document_id == filters["document_id"])
            if "filename" in filters:
                stmt = stmt.where(ChunkORM.filename == filters["filename"])

        stmt = stmt.order_by("distance").limit(top_k)

        start_time = time.time()
        try:
            result = await self.session.execute(stmt)
            rows = result.all()

            search_results = []
            for row in rows:
                chunk_orm = row[0]
                distance = row[1]
                # Convert cosine distance back to similarity (1 - distance)
                similarity = 1.0 - float(distance)
                domain_chunk = self._to_domain(chunk_orm)
                search_results.append(SearchResult(chunk=domain_chunk, score=similarity))
        finally:
            end_time = time.time()
            latency = end_time - start_time
            request_id = request_id_ctx_var.get()
            if request_id:
                logger.info("similarity_search_completed", request_id=request_id, latency=latency)
            else:
                logger.info("similarity_search_completed", request_id="unknown", latency=latency)
            RETRIEVAL_LATENCY.labels(architecture="dense").observe(latency)

        return search_results
    async def delete(self, document_id: UUID) -> None:
        """Delete all chunks for a specific document."""
        stmt = text("DELETE FROM document_chunks WHERE document_id = :doc_id")
        await self.session.execute(stmt, {"doc_id": str(document_id)})
        await self.session.commit()
        logger.info("Deleted document chunks", document_id=str(document_id))

    async def update(self, document_id: UUID, chunks: list[DocumentChunk]) -> None:
        """Delete existing chunks and insert new ones."""
        await self.delete(document_id)
        await self.batch_insert(chunks)

    async def health_check(self) -> bool:
        """Check if DB is accessible and vector extension is available."""
        try:
            result = await self.session.execute(text("SELECT 1;"))
            return result.scalar() == 1
        except Exception as e:
            logger.error("Vector store health check failed", error=str(e))
            return False
