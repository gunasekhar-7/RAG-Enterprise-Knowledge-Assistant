import os
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.dependencies.auth import RequireRole, get_current_user
from app.api.dependencies.rate_limit_checker import check_document_upload_limit
from app.core.exceptions import (
    EmptyDocumentError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
    DocumentParsingError,
    DatabaseError,
    AuthorizationError,
)
from app.db.session import get_db_session
from app.models.db import DocumentModel
from app.models.user import Role, UserResponse
from app.rag.vectorstore.pgvector_store import PGVectorStore
from app.tasks.document_tasks import _get_temp_upload_dir, _safe_upload_path, process_document_task

router = APIRouter()


class DocumentResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: str
    created_at: datetime
    token_count: int


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(  # type: ignore
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(RequireRole(Role.MANAGER)),  # Only managers+ can upload
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(check_document_upload_limit),
):
    """
    Upload a document for processing.
    The document will be chunked, embedded, and stored asynchronously under the user's tenant.
    """
    content = await file.read()
    if not content:
        raise EmptyDocumentError(filename=file.filename)

    max_size = 25 * 1024 * 1024
    if len(content) > max_size:
        raise DocumentTooLargeError(size_bytes=len(content), max_bytes=max_size, filename=file.filename)

    # Validate file type
    from app.ingestion.validator import FileValidator
    validator = FileValidator(max_file_size_bytes=max_size)
    try:
        sanitized_filename, doc_type, mime_type, sha256_hash = validator.validate(file.filename or "unknown", content)
    except Exception as e:
        raise UnsupportedDocumentTypeError(
            detected_type=file.content_type or "unknown",
            filename=file.filename,
            allowed_types=["PDF", "DOCX", "TXT", "MD"],
        )

    try:
        # Create a temp file to pass to celery worker
        # Use dedicated temp directory with safe unique path
        file_path = _safe_upload_path(sanitized_filename)

        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)

    except Exception as e:
        raise DocumentParsingError(filename=file.filename, reason=f"Failed to save file: {e}")

    # Create initial document record with status "processing"
    doc_id = uuid.uuid4()
    doc = DocumentModel(
        id=doc_id,
        tenant_id=current_user.tenant_id,
        filename=sanitized_filename,
        file_size_bytes=len(content),
        mime_type=mime_type,
        sha256_hash=sha256_hash,
        status="processing",
        token_count=0,
        storage_key=f"{current_user.tenant_id}/{doc_id}"
    )
    db.add(doc)
    try:
        await db.commit()
        await db.refresh(doc)
    except Exception as e:
        raise DatabaseError(message="Failed to create document record", operation="commit", details={"error": str(e)})

    # Queue the celery task
    process_document_task.delay(str(doc.id), str(current_user.tenant_id), file_path, sanitized_filename)

    return DocumentResponse(
        document_id=doc.id,  # type: ignore
        filename=doc.filename,  # type: ignore
        status=doc.status,  # type: ignore
        created_at=doc.created_at,  # type: ignore
        token_count=doc.token_count,  # type: ignore
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(  # type: ignore
    skip: int = 0,
    limit: int = 100,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List documents owned by or accessible to the current user's tenant.
    """
    stmt = (
        select(DocumentModel)
        .where(DocumentModel.tenant_id == current_user.tenant_id)
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    docs = result.scalars().all()
    return [
        DocumentResponse(
            document_id=d.id,  # type: ignore
            filename=d.filename,  # type: ignore
            status=d.status,  # type: ignore
            created_at=d.created_at,  # type: ignore
            token_count=d.token_count,  # type: ignore
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(  # type: ignore
    document_id: uuid.UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Check the status and metadata of a specific document.
    """
    stmt = select(DocumentModel).where(
        DocumentModel.id == document_id, DocumentModel.tenant_id == current_user.tenant_id
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentResponse(
        document_id=doc.id,  # type: ignore
        filename=doc.filename,  # type: ignore
        status=doc.status,  # type: ignore
        created_at=doc.created_at,  # type: ignore
        token_count=doc.token_count,  # type: ignore
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(  # type: ignore
    document_id: uuid.UUID,
    current_user: UserResponse = Depends(RequireRole(Role.ADMINISTRATOR)),  # Only admins can delete
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete a document and purge all associated vector chunks from the database.
    """
    stmt = select(DocumentModel).where(
        DocumentModel.id == document_id, DocumentModel.tenant_id == current_user.tenant_id
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        store = PGVectorStore(db)
        await store.delete(document_id)
    except Exception as e:
        raise DatabaseError(message="Failed to delete document vectors", operation="delete_vectors", details={"error": str(e)})

    try:
        await db.delete(doc)
        await db.commit()
    except Exception as e:
        raise DatabaseError(message="Failed to delete document record", operation="commit", details={"error": str(e)})
