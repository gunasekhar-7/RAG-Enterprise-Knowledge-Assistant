import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

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

# Dedicated temporary directory for uploads (configured at module load)
_TEMP_UPLOAD_DIR: Optional[Path] = None


def _get_temp_upload_dir() -> Path:
    """Get or create the dedicated temporary upload directory."""
    global _TEMP_UPLOAD_DIR
    if _TEMP_UPLOAD_DIR is None:
        # Use system temp with a fixed subdirectory for our app
        _TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
        _TEMP_UPLOAD_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    return _TEMP_UPLOAD_DIR


def _safe_upload_path(filename: str) -> Path:
    """
    Generate a safe unique upload path within the temp directory.

    Uses UUID for uniqueness and validates the final path is within the temp directory.
    """
    temp_dir = _get_temp_upload_dir()
    safe_name = f"{uuid.uuid4()}_{filename}"
    file_path = temp_dir / safe_name

    # Ensure resolved path is within temp directory (prevents traversal)
    try:
        file_path.resolve().relative_to(temp_dir.resolve())
    except ValueError:
        raise ValueError(f"Generated path escapes temp directory: {file_path}")

    return file_path


async def _process_document_async(
    document_id: str,
    tenant_id: str,
    file_path: Path,
    filename: str,
) -> None:
    """
    Process document asynchronously.

    Args:
        document_id: UUID of the document record
        tenant_id: Tenant UUID
        file_path: Path to the temporary uploaded file (guaranteed within temp dir)
        filename: Original sanitized filename for metadata
    """
    logger.info("async_processing_started", document_id=document_id, temp_path=str(file_path))

    try:
        # Read file content asynchronously
        import aiofiles
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        pipeline = DocumentIngestionPipeline()
        parsed_doc = await pipeline.process(
            filename=filename,
            content=content,
            tenant_id=tenant_id,
        )

        chunker = DocumentChunker()
        chunks = chunker.chunk(parsed_doc)

        embedder = await EmbedderFactory.get_embedder()
        embeddings = await embedder.embed_batch([c.text for c in chunks])
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
                logger.info("async_processing_completed", document_id=document_id)
            else:
                logger.error("document_not_found_for_status_update", document_id=document_id)

    except Exception as e:
        logger.error("async_processing_failed", document_id=document_id, error=str(e))
        async with AsyncSessionLocal() as db:
            stmt = select(DocumentModel).where(DocumentModel.id == uuid.UUID(document_id))
            result = await db.execute(stmt)
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "failed"
                await db.commit()
        # Re-raise to signal failure to caller
        raise
    finally:
        # Cleanup temp file - idempotent, does not mask original exception
        await _safe_remove_file(file_path)


async def _safe_remove_file(file_path: Path) -> None:
    """Remove file idempotently, logging but not raising errors."""
    try:
        if file_path and file_path.exists():
            # Verify path is still within our temp directory before deletion
            temp_dir = _get_temp_upload_dir()
            try:
                file_path.resolve().relative_to(temp_dir.resolve())
                file_path.unlink()
                logger.debug("temp_file_cleaned_up", path=str(file_path))
            except ValueError:
                logger.warning("temp_file_cleanup_skipped_path_traversal", path=str(file_path))
    except OSError as e:
        logger.warning("temp_file_cleanup_failed", path=str(file_path), error=str(e))
    except Exception as e:
        logger.error("temp_file_cleanup_unexpected_error", path=str(file_path), error=str(e))


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document_task(
    self,
    document_id: str,
    tenant_id: str,
    file_path: str,
    filename: str,
) -> None:
    """
    Celery task to process a document.

    Uses asyncio.run() with a fresh event loop for each task execution.
    This is safe for Celery's prefork worker model as each task gets its own loop.
    """
    logger.info("celery_task_received", document_id=document_id, task_id=self.request.id)

    # Validate file path is within our temp directory before processing
    try:
        path = Path(file_path)
        temp_dir = _get_temp_upload_dir()
        path.resolve().relative_to(temp_dir.resolve())
    except (ValueError, OSError) as e:
        logger.error("celery_task_rejected_unsafe_path", document_id=document_id, file_path=file_path, error=str(e))
        raise

    try:
        # Create new event loop for this task execution
        # This is safe in prefork workers as each task runs in its own process
        asyncio.run(_process_document_async(document_id, tenant_id, Path(file_path), filename))
    except Exception as e:
        logger.error("celery_task_failed", document_id=document_id, error=str(e))
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))