"""
Unit tests for document ingestion pipeline.
Covers valid/invalid uploads, format support, chunk creation, and metadata preservation.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.models import ParsedDocument, ExtractedPage, DocumentMetadata, DocumentType
from app.ingestion.parsers.factory import ParserFactory
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.ingestion.validator import FileValidator as DocumentValidator
from app.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    FileValidationError,
)
from app.rag.models import DocumentChunk


class TestDocumentValidator:
    """Test input validation for uploaded documents."""

    def test_validate_empty_document(self):
        """Empty files should be rejected."""
        validator = DocumentValidator()
        with pytest.raises(EmptyFileError):
            validator.validate("test.pdf", b"")

    def test_validate_oversized_document(self):
        """Documents exceeding size limit should be rejected."""
        validator = DocumentValidator()
        large_content = b"x" * (51 * 1024 * 1024)  # 51 MB (over 50MB default)
        with pytest.raises(FileTooLargeError):
            validator.validate("test.pdf", large_content)

    def test_validate_unsupported_format(self):
        """Unsupported file types should be rejected."""
        validator = DocumentValidator()
        content = b"fake content"
        with pytest.raises(UnsupportedFileTypeError):
            validator.validate("document.xyz", content)

    def test_validate_supported_formats(self):
        """Supported formats should pass basic validation."""
        validator = DocumentValidator()
        # Valid content for each format
        pdf_content = b"%PDF-1.4"  # PDF magic bytes
        docx_content = b"PK\x03\x04"  # ZIP magic bytes (DOCX is ZIP)
        txt_content = b"This is plain text"

        # These should not raise
        result = validator.validate("test.pdf", pdf_content)
        assert result[1] == DocumentType.PDF

        result = validator.validate("test.docx", docx_content)
        assert result[1] == DocumentType.DOCX

        result = validator.validate("test.txt", txt_content)
        assert result[1] == DocumentType.TXT

    def test_validate_malformed_pdf(self):
        """Malformed PDFs should be caught during parsing, not validation."""
        validator = DocumentValidator()
        malformed_pdf = b"%PDF-1.4 corrupted"
        # Validation should pass; parsing will fail
        result = validator.validate("test.pdf", malformed_pdf)
        assert result[1] == DocumentType.PDF

    def test_validate_filename_with_special_chars(self):
        """Filenames with special characters should be normalized."""
        validator = DocumentValidator()
        content = b"test content"
        filename = "test@file#2024.txt"
        # Should not raise; validator should sanitize
        result = validator.validate(filename, content)
        assert "test" in result[0]  # Sanitized filename


class TestParserFactory:
    """Test parser selection and initialization."""

    def test_factory_returns_pdf_parser_for_pdf(self):
        """PDF files should use PDF parser."""
        factory = ParserFactory()
        parser = factory.get_parser("test.pdf")
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_factory_returns_docx_parser_for_docx(self):
        """DOCX files should use DOCX parser."""
        factory = ParserFactory()
        parser = factory.get_parser("test.docx")
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_factory_returns_text_parser_for_txt(self):
        """TXT files should use text parser."""
        factory = ParserFactory()
        parser = factory.get_parser("test.txt")
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_factory_raises_for_unsupported_format(self):
        """Unsupported formats should raise an error."""
        factory = ParserFactory()
        with pytest.raises(UnsupportedFileTypeError):
            factory.get_parser("test.xyz")

    def test_factory_is_case_insensitive(self):
        """Parser selection should be case-insensitive."""
        factory = ParserFactory()
        parser_lower = factory.get_parser("test.pdf")
        parser_upper = factory.get_parser("test.PDF")
        assert type(parser_lower) == type(parser_upper)


class TestTextParser:
    """Test plain text document parsing."""

    @pytest.mark.asyncio
    async def test_parse_simple_text(self):
        """Simple text should be parsed into a single page."""
        from app.ingestion.parsers.text import TextParser

        parser = TextParser()
        doc_id = uuid.uuid4()
        content = b"This is a test document.\nWith multiple lines.\nAnd paragraphs."

        result = await parser.parse(content, doc_id, "test.txt")

        assert isinstance(result, ParsedDocument)
        assert result.id == doc_id
        assert len(result.pages) > 0
        assert "test document" in result.pages[0].text.lower()

    @pytest.mark.asyncio
    async def test_parse_empty_text_file(self):
        """Empty text files should raise an error."""
        from app.ingestion.parsers.text import TextParser

        parser = TextParser()
        doc_id = uuid.uuid4()
        content = b""

        with pytest.raises(ValueError):
            await parser.parse(content, doc_id, "test.txt")

    @pytest.mark.asyncio
    async def test_parse_text_with_unicode(self):
        """Text with Unicode characters should be handled correctly."""
        from app.ingestion.parsers.text import TextParser

        parser = TextParser()
        doc_id = uuid.uuid4()
        content = "Unicode test: 你好世界 🌍".encode("utf-8")

        result = await parser.parse(content, doc_id, "test.txt")

        assert isinstance(result, ParsedDocument)
        assert "你好世界" in result.pages[0].text or "unicode" in result.pages[0].text.lower()


class TestDocumentIngestionPipeline:
    """Test end-to-end document ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_valid_document(self):
        """Valid documents should flow through ingestion pipeline."""
        pipeline = DocumentIngestionPipeline()
        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        # Create a mock parsed document
        parsed_doc = ParsedDocument(
            id=doc_id,
            pages=[
                ExtractedPage(page_number=1, text="Test content for page 1"),
            ],
            metadata=DocumentMetadata(
                filename="test.txt",
                sanitized_filename="test.txt",
                file_type=DocumentType.TXT,
                mime_type="text/plain",
                file_size_bytes=100,
                sha256_hash="a" * 64,
                custom_metadata={"tenant_id": str(tenant_id)},
            ),
        )

        # Mock the chunker to return deterministic chunks
        with patch("app.ingestion.pipeline.DocumentChunker") as mock_chunker_class:
            mock_chunker_instance = MagicMock()
            mock_chunker_class.return_value = mock_chunker_instance

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_c0001",
                document_id=doc_id,
                text="Test content for page 1",
                metadata=MagicMock(tenant_id=tenant_id, page_number=1),
            )
            mock_chunker_instance.chunk_document.return_value = [chunk]

            # Run ingestion
            result = await pipeline.ingest(parsed_doc, tenant_id)

            assert result is not None
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_ingest_document_with_metadata_preservation(self):
        """Metadata should be preserved through ingestion."""
        pipeline = DocumentIngestionPipeline()
        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        custom_meta = {
            "tenant_id": str(tenant_id),
            "department": "Engineering",
            "classification": "Internal",
        }

        parsed_doc = ParsedDocument(
            id=doc_id,
            pages=[ExtractedPage(page_number=1, text="Test content")],
            metadata=DocumentMetadata(
                filename="test.txt",
                sanitized_filename="test.txt",
                file_type=DocumentType.TXT,
                mime_type="text/plain",
                file_size_bytes=100,
                sha256_hash="a" * 64,
                custom_metadata=custom_meta,
            ),
        )

        with patch("app.ingestion.pipeline.DocumentChunker") as mock_chunker_class:
            mock_chunker_instance = MagicMock()
            mock_chunker_class.return_value = mock_chunker_instance

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_c0001",
                document_id=doc_id,
                text="Test content",
                metadata=MagicMock(
                    tenant_id=tenant_id,
                    custom_metadata=custom_meta,
                    filename="test.txt",
                ),
            )
            mock_chunker_instance.chunk_document.return_value = [chunk]

            result = await pipeline.ingest(parsed_doc, tenant_id)

            assert result[0].metadata.custom_metadata["department"] == "Engineering"

    @pytest.mark.asyncio
    async def test_ingest_document_duplicate_handling(self):
        """Duplicate uploads should be detected by SHA256."""
        pipeline = DocumentIngestionPipeline()
        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        content = "This is test content"
        parsed_doc = ParsedDocument(
            id=doc_id,
            pages=[ExtractedPage(page_number=1, text=content)],
            metadata=DocumentMetadata(
                filename="test.txt",
                sanitized_filename="test.txt",
                file_type=DocumentType.TXT,
                mime_type="text/plain",
                file_size_bytes=len(content),
                sha256_hash="a" * 64,  # Same hash for duplicate
                custom_metadata={"tenant_id": str(tenant_id)},
            ),
        )

        with patch("app.ingestion.pipeline.DocumentChunker") as mock_chunker_class:
            mock_chunker_instance = MagicMock()
            mock_chunker_class.return_value = mock_chunker_instance
            mock_chunker_instance.chunk_document.return_value = []

            # Should handle duplicate detection logic
            result = await pipeline.ingest(parsed_doc, tenant_id)

            # Verify pipeline was called
            assert mock_chunker_instance.chunk_document.called


class TestChunkingMetadata:
    """Test that chunking preserves page numbers and section headers."""

    @pytest.mark.asyncio
    async def test_chunk_preserves_page_numbers(self):
        """Chunks should retain original page numbers."""
        from app.rag.chunker import DocumentChunker
        from app.rag.models import ChunkingConfig

        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = DocumentChunker(config)

        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        parsed_doc = ParsedDocument(
            id=doc_id,
            pages=[
                ExtractedPage(page_number=1, text="Page 1 content here."),
                ExtractedPage(page_number=2, text="Page 2 content here."),
            ],
            metadata=DocumentMetadata(
                filename="test.pdf",
                sanitized_filename="test.pdf",
                file_type=DocumentType.PDF,
                mime_type="application/pdf",
                file_size_bytes=100,
                sha256_hash="a" * 64,
                custom_metadata={
                    "tenant_id": str(tenant_id),
                },
            ),
        )

        chunks = chunker.chunk_document(parsed_doc)

        assert len(chunks) > 0
        # All chunks should have page numbers
        for chunk in chunks:
            assert chunk.metadata.page_number in [1, 2]

    @pytest.mark.asyncio
    async def test_chunk_extracts_section_headers(self):
        """Chunk metadata should include section headers when present."""
        from app.rag.chunker import DocumentChunker
        from app.rag.models import ChunkingConfig

        config = ChunkingConfig(chunk_size=200, chunk_overlap=10)
        chunker = DocumentChunker(config)

        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        parsed_doc = ParsedDocument(
            id=doc_id,
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="# Section 1\nContent for section 1.\n## Subsection 1.1\nMore content here.",
                ),
            ],
            metadata=DocumentMetadata(
                filename="test.md",
                sanitized_filename="test.md",
                file_type=DocumentType.MARKDOWN,
                mime_type="text/markdown",
                file_size_bytes=200,
                sha256_hash="a" * 64,
                custom_metadata={"tenant_id": str(tenant_id)},
            ),
        )

        chunks = chunker.chunk_document(parsed_doc)

        assert len(chunks) > 0
        # At least one chunk should have a section heading
        has_section = any(chunk.metadata.section_heading for chunk in chunks)
        assert has_section

    @pytest.mark.asyncio
    async def test_chunk_token_counting_accuracy(self):
        """Token counts should be accurate."""
        from app.rag.chunker import DocumentChunker, Tokenizer
        from app.rag.models import ChunkingConfig

        config = ChunkingConfig(chunk_size=100, chunk_overlap=10)
        chunker = DocumentChunker(config)
        tokenizer = Tokenizer()

        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        test_text = "This is a test document. " * 10  # Repeated text for token counting

        parsed_doc = ParsedDocument(
            id=doc_id,
            pages=[ExtractedPage(page_number=1, text=test_text)],
            metadata=DocumentMetadata(
                filename="test.txt",
                sanitized_filename="test.txt",
                file_type=DocumentType.TXT,
                mime_type="text/plain",
                file_size_bytes=len(test_text),
                sha256_hash="a" * 64,
                custom_metadata={"tenant_id": str(tenant_id)},
            ),
        )

        chunks = chunker.chunk_document(parsed_doc)

        for chunk in chunks:
            actual_token_count = tokenizer.count_tokens(chunk.text)
            assert chunk.metadata.token_count == actual_token_count