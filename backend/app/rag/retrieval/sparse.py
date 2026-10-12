from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.exceptions import RetrievalError
from app.core.metrics import RETRIEVAL_LATENCY
from app.core.logging import request_id_ctx_var
import time

from app.rag.models import DocumentChunk
from app.rag.retrieval.base import BaseKeywordStore, BaseRetriever, RankedContext, RetrievedContext
from app.rag.vectorstore.base import SearchResult

logger = structlog.get_logger(__name__)


class PostgresFTSStore(BaseKeywordStore):
    """
    PostgreSQL Full-Text Search (FTS) implementation for sparse keyword search.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_documents(self, chunks: list[DocumentChunk]) -> None:
        # Document chunks are already added to DB by PGVectorStore.
        pass

    async def search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        from app.rag.vectorstore.pgvector_store import ChunkORM

        stmt = select(  # type: ignore
            ChunkORM,
            text("ts_rank_cd(search_vector, plainto_tsquery('english', :q))").label("rank"),
        ).where(text("search_vector @@ plainto_tsquery('english', :q)"))

        params = {"q": query}

        if filters:
            if "tenant_id" in filters:
                stmt = stmt.where(ChunkORM.tenant_id == filters["tenant_id"])
            if "document_id" in filters:
                stmt = stmt.where(ChunkORM.document_id == filters["document_id"])
            if "filename" in filters:
                stmt = stmt.where(ChunkORM.filename == filters["filename"])

        stmt = stmt.order_by(text("rank DESC")).limit(top_k)

        start_time = time.time()
        try:
            result = await self.session.execute(stmt, params)
        except Exception as e:
            raise RetrievalError(message="Keyword search failed", retriever_type="sparse", details={"error": str(e)})
        finally:
            end_time = time.time()
            latency = end_time - start_time
            request_id = request_id_ctx_var.get()
            if request_id:
                logger.info("sparse_search_completed", request_id=request_id, latency=latency)
            else:
                logger.info("sparse_search_completed", request_id="unknown", latency=latency)
            RETRIEVAL_LATENCY.labels(architecture="sparse").observe(latency)

        rows = result.all()

        search_results = []
        for row in rows:
            chunk_orm = row[0]
            rank = row[1]

            from app.rag.models import ChunkMetadata

            metadata = ChunkMetadata(  # type: ignore
                document_id=chunk_orm.document_id,
                tenant_id=chunk_orm.tenant_id,
                chunk_index=int(chunk_orm.chunk_id.split("_c")[-1]),
                page_number=chunk_orm.page_number,
                filename=chunk_orm.filename,
                section_heading=chunk_orm.section_heading,
                token_count=0,
                char_count=len(chunk_orm.text),
                custom_metadata=chunk_orm.custom_metadata,
            )
            domain_chunk = DocumentChunk(
                chunk_id=chunk_orm.chunk_id,
                document_id=chunk_orm.document_id,
                text=chunk_orm.text,
                metadata=metadata,
                embedding=None,
            )
            search_results.append(SearchResult(chunk=domain_chunk, score=float(rank)))

        if search_results:
            max_s = search_results[0].score
            min_s = search_results[-1].score
            if max_s > min_s:
                denom = max_s - min_s
                for r in search_results:
                    r.score = (r.score - min_s) / denom

        return search_results


class KeywordRetriever(BaseRetriever):
    """Sparse retriever utilizing BM25 exact keyword matching."""

    def __init__(self, keyword_store: BaseKeywordStore, default_top_k: int = 5):
        self.keyword_store = keyword_store
        self.default_top_k = default_top_k

    async def retrieve(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> RankedContext:
        k = top_k or self.default_top_k
        if not query.strip():
            return RankedContext(query=query, results=[], total_found=0)

        raw_results = await self.keyword_store.search(query=query, top_k=k, filters=filters)

        structured_results = []
        for result in raw_results:
            structured_results.append(
                RetrievedContext(
                    chunk_id=result.chunk.chunk_id,
                    text=result.chunk.text,
                    score=result.score,
                    citation=result.chunk.citation_source,
                    metadata=result.chunk.metadata.model_dump(),
                )
            )

        return RankedContext(
            query=query, results=structured_results, total_found=len(structured_results)
        )
