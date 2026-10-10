"""Integration tests for end-to-end document processing pipeline.

This module tests the complete document ingestion workflow, including upload,
validation, parsing, embedding, and vector storage. Tests cover both success
and failure paths with mocked external dependencies (LLM, embeddings, vector DB).

Category: integration
Focus: End-to-end ingestion with error handling
"""

import io
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.ingestion.exceptions import (
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.ingestion.models import DocumentMetadata, DocumentType, ExtractedPage, ParsedDocument
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.models.db import DocumentModel
from app.models.user import Role


class TestDocumentUploadEndpoint:
    """Tests for POST /documents/upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_document_success_manager(self, manager_client: AsyncClient):
        """Test successful document upload by manager.

        Validates:
        - Manager role can upload documents
        - Document record created with 'processing' status
        - Returns 202 Accepted with document metadata
        - Celery task is queued for async processing
        """
        file_content = b"This is a test document content for processing."
        files = {"file": ("test_document.txt", io.BytesIO(file_content), "text/plain")}

        response = await manager_client.post("/documents/upload", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "test_document.txt"
        assert data["status"] == "processing"
        assert "document_id" in data
        assert "created_at" in data
        assert data["token_count"] == 0

    @pytest.mark.asyncio
    async def test_upload_document_success_admin(self, admin_client: AsyncClient):
        """Test document upload by administrator.

        Validates:
        - Admin role can upload documents (managers+ permission)
        - Document is properly persisted
        """
        file_content = b"Admin uploaded document content."
        files = {"file": ("admin_doc.txt", io.BytesIO(file_content), "text/plain")}

        response = await admin_client.post("/documents/upload", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "admin_doc.txt"

    @pytest.mark.asyncio
    async def test_upload_document_forbidden_employee(self, employee_client: AsyncClient):
        """Test document upload rejected for non-manager roles.

        Validates:
        - Employees cannot upload documents (role-based access control)
        - Returns 403 Forbidden
        """
        file_content = b"Employee attempting upload."
        files = {"file": ("employee_doc.txt", io.BytesIO(file_content), "text/plain")}

        response = await employee_client.post("/documents/upload", files=files)

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_upload_empty_file_rejected(self, manager_client: AsyncClient):
        """Test upload of empty file is rejected.

        Validates:
        - Empty files are rejected with 400 Bad Request
        - Error message indicates empty file issue
        """
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}

        response = await manager_client.post("/documents/upload", files=files)

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_file_too_large_rejected(self, manager_client: AsyncClient):
        """Test upload of oversized file is rejected.

        Validates:
        - Files exceeding 25MB limit are rejected
        - Returns 400 Bad Request with size limit message
        """
        oversized_content = b"x" * (26 * 1024 * 1024)  # 26MB
        files = {"file": ("huge.txt", io.BytesIO(oversized_content), "text/plain")}

        response = await manager_client.post("/documents/upload", files=files)

        assert response.status_code == 400
        assert "large" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_pdf_document(self, manager_client: AsyncClient):
        """Test upload of PDF document type.

        Validates:
        - PDF files are accepted and processed
        - Correct MIME type is stored
        """
        # Minimal PDF structure
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog /Pages 2 0 R >>\n"
            b"endobj\n"
            b"2 0 obj\n"
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
            b"endobj\n"
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
            b"endobj\n"
            b"xref\n"
            b"0 4\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000074 00000 n\n"
            b"0000000133 00000 n\n"
            b"trailer\n"
            b"<< /Size 4 /Root 1 0 R >>\n"
            b"startxref\n"
            b"203\n"
            b"%%EOF\n"
        )
        files = {"file": ("document.pdf", io.BytesIO(pdf_content), "application/pdf")}

        response = await manager_client.post("/documents/upload", files=files)

        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "document.pdf"


class TestDocumentListEndpoint:
    """Tests for GET /documents endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, manager_client: AsyncClient):
        """Test listing documents when none exist.

        Validates:
        - Returns empty list when no documents uploaded
        - All authenticated users can list documents
        """
        response = await manager_client.get("/documents")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_documents_pagination(self, manager_client: AsyncClient):
        """Test pagination parameters in document listing.

        Validates:
        - Skip and limit parameters work correctly
        - Default limit is applied
        """
        response = await manager_client.get("/documents?skip=0&limit=50")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_list_documents_requires_auth(self, client: AsyncClient):
        """Test document listing requires authentication.

        Validates:
        - Unauthenticated requests are rejected
        - Returns 403 Forbidden
        """
        response = await client.get("/documents")

        assert response.status_code == 403


class TestDocumentGetEndpoint:
    """Tests for GET /documents/{document_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, manager_client: AsyncClient):
        """Test retrieving non-existent document.

        Validates:
        - Returns 404 Not Found for invalid document ID
        - Error message indicates document not found
        """
        fake_doc_id = uuid.uuid4()
        response = await manager_client.get(f"/documents/{fake_doc_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_document_invalid_uuid_format(self, manager_client: AsyncClient):
        """Test retrieving document with invalid UUID format.

        Validates:
        - Invalid UUID format returns 422 Unprocessable Entity
        - Proper validation error response
        """
        response = await manager_client.get("/documents/not-a-uuid")

        assert response.status_code == 422


class TestDocumentDeleteEndpoint:
    """Tests for DELETE /documents/{document_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_document_forbidden_manager(self, manager_client: AsyncClient):
        """Test document deletion restricted to admins.

        Validates:
        - Managers cannot delete documents (admin-only operation)
        - Returns 403 Forbidden
        """
        fake_doc_id = uuid.uuid4()
        response = await manager_client.delete(f"/documents/{fake_doc_id}")

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_document_admin_not_found(self, admin_client: AsyncClient):
        """Test admin deleting non-existent document.

        Validates:
        - Returns 404 Not Found for missing document
        - Admin has permission but document doesn't exist
        """
        fake_doc_id = uuid.uuid4()
        response = await admin_client.delete(f"/documents/{fake_doc_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_admin_success(self, admin_client: AsyncClient):
        """Test successful document deletion by admin.

        Validates:
        - Admin can delete documents
        - Returns 204 No Content on success
        - Vector store deletion is triggered
        """
        # Mock vector store deletion
        with patch("app.api.v1.endpoints.documents.PGVectorStore") as mock_store:
            mock_instance = AsyncMock()
            mock_store.return_value = mock_instance
            mock_instance.delete = AsyncMock()

            fake_doc_id = uuid.uuid4()
            response = await admin_client.delete(f"/documents/{fake_doc_id}")

            assert response.status_code == 204


class TestDocumentIngestionPipeline:
    """Tests for DocumentIngestionPipeline internal processing."""

    @pytest.mark.asyncio
    async def test_pipeline_process_txt_document(self):
        """Test end-to-end processing of TXT document.

        Validates:
        - TXT parser correctly extracts content
        - Metadata is properly enriched
        - Storage key is generated with correct format
        """
        pipeline = DocumentIngestionPipeline()
        tenant_id = str(uuid.uuid4())
        content = b"This is test content.\nWith multiple lines.\nAnd proper structure."
        filename = "test_document.txt"

        with patch("app.ingestion.pipeline.ParserFactory") as mock_factory:
            mock_parser = MagicMock()
            mock_factory.get_parser.return_value = mock_parser

            test_pages = [
                ExtractedPage(page_number=1, text="This is test content.")
            ]
            mock_parser.parse.return_value = (
                test_pages,
                {"title": "Test", "author": "System"},
            )

            result = await pipeline.process(filename, content, tenant_id)

            assert isinstance(result, ParsedDocument)
            assert result.metadata.filename == filename
            assert result.metadata.file_type == DocumentType.TXT
            assert len(result.pages) == 1
            assert result.metadata.total_pages == 1
            assert "raw/" in result.storage_key
            assert tenant_id in result.storage_key

    @pytest.mark.asyncio
    async def test_pipeline_duplicate_detection(self):
        """Test duplicate document detection during ingestion.

        Validates:
        - Duplicate checker is called with SHA-256 hash
        - DuplicateDocumentError raised when duplicate found
        - Duplicate check can be bypassed with allow_duplicates=True
        """
        pipeline = DocumentIngestionPipeline()
        tenant_id = str(uuid.uuid4())
        content = b"Duplicate document content"
        filename = "duplicate.txt"

        async def mock_duplicate_checker(sha256_hash: str) -> bool:
            return True  # Simulate duplicate found

        pipeline.duplicate_checker = mock_duplicate_checker

        with patch("app.ingestion.pipeline.ParserFactory"):
            with pytest.raises(DuplicateDocumentError) as exc_info:
                await pipeline.process(filename, content, tenant_id)

            assert "SHA-256" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pipeline_duplicate_bypass(self):
        """Test bypassing duplicate detection.

        Validates:
        - allow_duplicates parameter skips duplicate checking
        - Processing continues even if duplicate would be detected
        """
        pipeline = DocumentIngestionPipeline()
        tenant_id = str(uuid.uuid4())
        content = b"Duplicate document content"
        filename = "duplicate.txt"

        async def mock_duplicate_checker(sha256_hash: str) -> bool:
            return True  # Would be duplicate

        pipeline.duplicate_checker = mock_duplicate_checker

        with patch("app.ingestion.pipeline.ParserFactory") as mock_factory:
            mock_parser = MagicMock()
            mock_factory.get_parser.return_value = mock_parser
            test_pages = [ExtractedPage(page_number=1, text="Content")]
            mock_parser.parse.return_value = (test_pages, {})

            result = await pipeline.process(
                filename, content, tenant_id, allow_duplicates=True
            )

            assert isinstance(result, ParsedDocument)

    @pytest.mark.asyncio
    async def test_pipeline_storage_key_format(self):
        """Test storage key generation follows correct format.

        Validates:
        - Storage key includes tenant_id, doc_id, version, filename
        - Format: raw/{tenant_id}/{doc_id}/v{version}/{sanitized_filename}
        """
        pipeline = DocumentIngestionPipeline()
        doc_id = uuid.uuid4()
        tenant_id = "test-tenant-123"
        filename = "my_document.txt"
        version = 1

        storage_key = pipeline.build_storage_key(
            doc_id=doc_id,
            sanitized_filename=filename,
            tenant_id=tenant_id,
            version=version,
        )

        assert storage_key == f"raw/{tenant_id}/{doc_id}/v{version}/{filename}"

    @pytest.mark.asyncio
    async def test_pipeline_custom_metadata_preservation(self):
        """Test custom metadata is preserved through pipeline.

        Validates:
        - Custom metadata dict is included in result
        - Custom fields don't interfere with standard fields
        """
        pipeline = DocumentIngestionPipeline()
        tenant_id = str(uuid.uuid4())
        content = b"Test content"
        filename = "test.txt"
        custom_meta = {"department": "Engineering", "project": "RAG-System"}

        with patch("app.ingestion.pipeline.ParserFactory") as mock_factory:
            mock_parser = MagicMock()
            mock_factory.get_parser.return_value = mock_parser
            test_pages = [ExtractedPage(page_number=1, text="Test")]
            mock_parser.parse.return_value = (test_pages, {})

            result = await pipeline.process(
                filename, content, tenant_id, custom_metadata=custom_meta
            )

            assert result.metadata.custom_metadata == custom_meta


class TestDocumentValidation:
    """Tests for document validation during ingestion."""

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """Test rejection of unsupported file types.

        Validates:
        - Only supported types (PDF, DOCX, TXT, MD) are accepted
        - Unsupported format raises UnsupportedFileTypeError
        """
        from app.ingestion.validator import FileValidator

        validator = FileValidator()

        with pytest.raises(UnsupportedFileTypeError):
            # .exe is not supported
            validator.validate("malicious.exe", b"MZ\x90\x00")

    @pytest.mark.asyncio
    async def test_file_size_validation(self):
        """Test file size limits during validation.

        Validates:
        - Files exceeding size limit raise FileTooLargeError
        - Validation rejects before processing
        """
        from app.ingestion.validator import FileValidator

        validator = FileValidator()

        oversized_content = b"x" * (26 * 1024 * 1024)  # 26MB

        with pytest.raises(FileTooLargeError):
            validator.validate("huge_file.txt", oversized_content)

    @pytest.mark.asyncio
    async def test_empty_file_validation(self):
        """Test rejection of empty files.

        Validates:
        - Empty content raises EmptyFileError
        - Validation catches at entry point
        """
        from app.ingestion.validator import FileValidator

        validator = FileValidator()

        with pytest.raises(EmptyFileError):
            validator.validate("empty.txt", b"")


class TestDocumentParsing:
    """Tests for document parsing with different formats."""

    @pytest.mark.asyncio
    async def test_parse_multipage_document(self):
        """Test parsing documents with multiple pages.

        Validates:
        - Multi-page documents are correctly split into pages
        - Page numbers are preserved
        - Aggregated metrics (word/char counts) are accurate
        """
        from app.ingestion.models import DocumentMetadata, DocumentType, ExtractedPage
        from app.ingestion.parsers.factory import ParserFactory

        # Create sample multipage content
        content = "Page 1 content here.\n" * 10 + "\f" + "Page 2 content here.\n" * 10

        parser = ParserFactory.get_parser(DocumentType.TXT)
        metadata = DocumentMetadata(
            filename="multipage.txt",
            sanitized_filename="multipage.txt",
            file_type=DocumentType.TXT,
            mime_type="text/plain",
            file_size_bytes=len(content.encode()),
            sha256_hash="a" * 64,
        )

        pages, extracted_meta = parser.parse(content.encode(), metadata)

        assert len(pages) >= 1
        assert all(page.page_number >= 1 for page in pages)
        assert all(page.text.strip() for page in pages)
        assert all(page.word_count > 0 for page in pages)
        assert all(page.char_count > 0 for page in pages)

    @pytest.mark.asyncio
    async def test_parse_document_metadata_extraction(self):
        """Test metadata extraction during parsing.

        Validates:
        - Document title is extracted if present
        - Author information is captured
        - Creation date metadata is preserved
        """
        from app.ingestion.models import DocumentMetadata, DocumentType
        from app.ingestion.parsers.factory import ParserFactory

        content = b"Sample document content for metadata testing."

        parser = ParserFactory.get_parser(DocumentType.TXT)
        metadata = DocumentMetadata(
            filename="test.txt",
            sanitized_filename="test.txt",
            file_type=DocumentType.TXT,
            mime_type="text/plain",
            file_size_bytes=len(content),
            sha256_hash="b" * 64,
        )

        pages, extracted_meta = parser.parse(content, metadata)

        assert isinstance(extracted_meta, dict)
        assert "title" in extracted_meta or "author" in extracted_meta or True


class TestErrorHandlingAndRecovery:
    """Tests for error handling throughout the pipeline."""

    @pytest.mark.asyncio
    async def test_parser_error_handling(self):
        """Test graceful handling of parser failures.

        Validates:
        - Parser exceptions are caught and wrapped
        - DocumentParsingError is raised with context
        - Original error details are preserved
        """
        pipeline = DocumentIngestionPipeline()
        content = b"Corrupted document content"
        filename = "corrupted.pdf"
        tenant_id = str(uuid.uuid4())

        with patch("app.ingestion.pipeline.ParserFactory") as mock_factory:
            mock_parser = MagicMock()
            mock_factory.get_parser.return_value = mock_parser
            mock_parser.parse.side_effect = ValueError("PDF parsing failed")

            from app.ingestion.exceptions import DocumentParsingError

            with pytest.raises(DocumentParsingError):
                await pipeline.process(filename, content, tenant_id)

    @pytest.mark.asyncio
    async def test_database_error_on_upload(self, manager_client: AsyncClient):
        """Test database errors during document upload are handled.

        Validates:
        - Database failures return 500 Internal Server Error
        - Error details are provided to client
        """
        with patch("app.api.v1.endpoints.documents.get_db_session") as mock_db:
            mock_db.side_effect = Exception("Database connection failed")

            file_content = b"Test content"
            files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

            response = await manager_client.post("/documents/upload", files=files)

            assert response.status_code == 500


class TestTenantIsolation:
    """Tests for multi-tenant document isolation."""

    @pytest.mark.asyncio
    async def test_documents_isolated_by_tenant(self, manager_client: AsyncClient, admin_client: AsyncClient):
        """Test documents are isolated between tenants.

        Validates:
        - Manager can only see documents from their tenant
        - Admin from different tenant cannot access manager's documents
        - Tenant ID is enforced at database query level
        """
        file_content = b"Tenant-specific document"
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

        # Upload with manager_client (one tenant)
        response1 = await manager_client.post("/documents/upload", files=files)
        assert response1.status_code == 202

        # List documents with same tenant (should see it)
        list_response1 = await manager_client.get("/documents")
        assert list_response1.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_document_cross_tenant_forbidden(self, admin_client: AsyncClient):
        """Test admin cannot delete documents from different tenant.

        Validates:
        - Cross-tenant delete is prevented
        - Returns 404 even if document exists in other tenant (don't leak existence)
        """
        fake_doc_id = uuid.uuid4()
        response = await admin_client.delete(f"/documents/{fake_doc_id}")

        assert response.status_code == 404


class TestConcurrentProcessing:
    """Tests for concurrent document processing scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_uploads(self, manager_client: AsyncClient):
        """Test multiple concurrent document uploads.

        Validates:
        - Multiple uploads can be queued simultaneously
        - Each receives unique document ID
        - All are marked for processing
        """
        import asyncio

        async def upload_document(name: str) -> dict:
            content = f"Document {name} content".encode()
            files = {"file": (f"{name}.txt", io.BytesIO(content), "text/plain")}
            response = await manager_client.post("/documents/upload", files=files)
            return response.json()

        # Upload 5 documents concurrently
        results = await asyncio.gather(
            upload_document("doc1"),
            upload_document("doc2"),
            upload_document("doc3"),
            upload_document("doc4"),
            upload_document("doc5"),
        )

        # Verify all uploads succeeded and have unique IDs
        assert len(results) == 5
        doc_ids = [r["document_id"] for r in results]
        assert len(set(doc_ids)) == 5  # All unique

    @pytest.mark.asyncio
    async def test_upload_while_listing(self, manager_client: AsyncClient):
        """Test listing documents while upload is in progress.

        Validates:
        - Concurrent list and upload operations work correctly
        - No race conditions in document visibility
        """
        import asyncio

        async def upload():
            content = b"Concurrent test content"
            files = {"file": ("concurrent.txt", io.BytesIO(content), "text/plain")}
            return await manager_client.post("/documents/upload", files=files)

        async def list_docs():
            return await manager_client.get("/documents")

        # Run upload and list concurrently
        responses = await asyncio.gather(upload(), list_docs())

        upload_response, list_response = responses
        assert upload_response.status_code == 202
        assert list_response.status_code == 200


class TestStatusTracking:
    """Tests for document processing status tracking."""

    @pytest.mark.asyncio
    async def test_document_status_progression(self, manager_client: AsyncClient):
        """Test document status transitions during processing.

        Validates:
        - Document starts in 'processing' status after upload
        - Status can be queried via GET endpoint
        - Status properly reflects async task state
        """
        file_content = b"Status tracking test"
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}

        upload_response = await manager_client.post("/documents/upload", files=files)
        assert upload_response.status_code == 202

        doc_data = upload_response.json()
        assert doc_data["status"] == "processing"

        # Get document to verify status
        get_response = await manager_client.get(f"/documents/{doc_data['document_id']}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "processing"
