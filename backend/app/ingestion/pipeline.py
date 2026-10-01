"""Master Document Ingestion Pipeline coordinating validation, deduplication, parsing, and metadata extraction."""

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID, uuid4

from app.core.logging import logger
from app.ingestion.exceptions import DuplicateDocumentError
from app.ingestion.models import DocumentMetadata, ParsedDocument
from app.ingestion.parsers.factory import ParserFactory
from app.ingestion.validator import FileValidator

DuplicateCheckFn = Callable[[str], Awaitable[bool]]


class DocumentIngestionPipeline:
    """Orchestrates secure validation, duplicate detection, parsing, and storage key generation for uploaded files."""

    def __init__(
        self,
        validator: FileValidator | None = None,
        duplicate_checker: DuplicateCheckFn | None = None,
    ) -> None:
        self.validator = validator or FileValidator()
        self.duplicate_checker = duplicate_checker

    def build_storage_key(
        self,
        doc_id: UUID,
        sanitized_filename: str,
        tenant_id: str = "default",
        version: int = 1,
    ) -> str:
        """Constructs canonical S3/MinIO object storage path:

        Format: raw/{tenant_id}/{doc_id}/v{version}/{sanitized_filename}
        """
        return f"raw/{tenant_id}/{doc_id}/v{version}/{sanitized_filename}"

    async def process(
        self,
        filename: str,
        content: bytes,
        tenant_id: str = "default",
        custom_metadata: dict[str, Any] | None = None,
        allow_duplicates: bool = False,
    ) -> ParsedDocument:
        """Processes raw file bytes through the complete ingestion lifecycle.

        Steps:
            1. Validate size, sanitize filename, sniff magic bytes, calculate SHA-256.
            2. Check for duplicate documents via content hash.
            3. Retrieve specialized parser from ParserFactory.
            4. Extract pages (preserving page numbers) and embedded metadata.
            5. Aggregate counts (words, chars, pages).
            6. Generate deterministic/UUID storage paths.
            7. Return structured ParsedDocument.
        """
        logger.info("ingestion_pipeline_started", filename=filename, size=len(content))

        # 1. Validation & Security Checks
        sanitized_filename, doc_type, mime_type, sha256_hash = self.validator.validate(
            filename, content
        )

        # 2. Duplicate Detection
        if not allow_duplicates and self.duplicate_checker:
            is_duplicate = await self.duplicate_checker(sha256_hash)
            if is_duplicate:
                logger.warn(
                    "duplicate_document_detected",
                    filename=filename,
                    sha256=sha256_hash,
                )
                raise DuplicateDocumentError(
                    f"A document with matching SHA-256 content checksum ({sha256_hash[:12]}...) has already been ingested."
                )

        # 3. Document ID & Storage Key Generation
        doc_id = uuid4()
        storage_key = self.build_storage_key(
            doc_id=doc_id,
            sanitized_filename=sanitized_filename,
            tenant_id=tenant_id,
            version=1,
        )

        # 4. Preliminary Metadata Construct
        initial_meta = DocumentMetadata(
            filename=filename,
            sanitized_filename=sanitized_filename,
            file_type=doc_type,
            mime_type=mime_type,
            file_size_bytes=len(content),
            sha256_hash=sha256_hash,
            custom_metadata=custom_metadata or {},
        )

        # 5. Parser Dispatch
        parser = ParserFactory.get_parser(doc_type)
        extracted_pages, extracted_meta = parser.parse(content, initial_meta)

        # 6. Aggregate Metrics & Enrich Metadata
        total_words = sum(page.word_count for page in extracted_pages)
        total_chars = sum(page.char_count for page in extracted_pages)

        enriched_meta = initial_meta.model_copy(
            update={
                "title": extracted_meta.get("title"),
                "author": extracted_meta.get("author"),
                "created_at_meta": extracted_meta.get("created_at_meta"),
                "total_pages": len(extracted_pages),
                "total_words": total_words,
                "total_chars": total_chars,
            }
        )

        # 7. Construct Final Normalized Output
        parsed_document = ParsedDocument(
            id=doc_id,
            metadata=enriched_meta,
            pages=extracted_pages,
            storage_key=storage_key,
        )

        logger.info(
            "ingestion_pipeline_completed",
            doc_id=str(doc_id),
            filename=sanitized_filename,
            file_type=doc_type.value,
            pages=len(extracted_pages),
            words=total_words,
            storage_key=storage_key,
        )

        return parsed_document
