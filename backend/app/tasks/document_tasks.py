import asyncio
import os
import uuid

import structlog
from sqlalchemy.future import select

from app.core.celery_app import celery_app
from app.db.session import AsyncSessionLocal
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.models.db import DocumentModel
from app.rag.chunker import DocumentChunker
from app.rag.embedder.factory import EmbedderFactory
from app.rag.retrieval.sparse import PostgresFTSStore
from app.rag.vectorstore.pgvector_store import PGVectorStore

logger = structlog.get_logger(__name__)

async def _process_document_async(document_id: str, tenant_id: str, file_path: str, filename: str):
    logger.info(f"Starting async processing for document {document_id}")

    with open(file_path, "rb") as f:
        content = f.read()

    try:
        pipeline = DocumentIngestionPipeline()
        parsed_doc = await pipeline.process(
            filename=filename,
            content=content,
            tenant_id=tenant_id,
        )

        chunker = DocumentChunker()
        chunks = chunker.chunk(parsed_doc)

        embedder = EmbedderFactory.get_embedder()
        embeddings = embedder.embed_batch([c.text for c in chunks])
        for c, emb in zip(chunks, embeddings, strict=False):
            c.embedding = emb

        async with AsyncSessionLocal() as db:
            store = PGVectorStore(db)
            fts_store = PostgresFTSStore(db)

            await store.batch_insert(chunks)
            await fts_store.add_documents(chunks)

            # Update Document status
            stmt = select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
            result = await db.execute(stmt)
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "completed"
                doc.token_count = sum(c.metadata.token_count for c in chunks) if chunks else 0
                doc.file_size_bytes = parsed_doc.metadata.file_size_bytes
                doc.mime_type = parsed_doc.metadata.mime_type
                doc.sha256_hash = parsed_doc.metadata.sha256_hash
                doc.storage_key = parsed_doc.storage_key
                await db.commit()
                logger.info(f"Successfully processed document {document_id}")
            else:
                logger.error(f"Document {document_id} not found in DB to update status.")

    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        async with AsyncSessionLocal() as db:
            stmt = select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
            result = await db.execute(stmt)
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "failed"
                await db.commit()
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@celery_app.task
def process_document_task(document_id: str, tenant_id: str, file_path: str, filename: str):
    logger.info(f"Celery received task for document {document_id}")
    asyncio.run(_process_document_async(document_id, tenant_id, file_path, filename))
