"""Data models for document ingestion, parsed pages, and metadata."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class DocumentType(StrEnum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"

    @classmethod
    def from_mime(cls, mime: str) -> "DocumentType":
        mime_map = {
            "application/pdf": cls.PDF,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": cls.DOCX,
            "text/plain": cls.TXT,
            "text/markdown": cls.MARKDOWN,
            "text/x-markdown": cls.MARKDOWN,
        }
        if mime in mime_map:
            return mime_map[mime]
        raise ValueError(f"Unsupported MIME type: {mime}")


class ExtractedPage(BaseModel):
    """Represents text extracted from a single document page or section."""

    page_number: int = Field(..., ge=1, description="1-indexed page or section number")
    text: str = Field(..., description="Cleaned, extracted text content of the page")
    char_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Page-level metadata")

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())


class DocumentMetadata(BaseModel):
    """Normalized metadata for an ingested organizational document."""

    filename: str = Field(..., description="Original uploaded filename")
    sanitized_filename: str = Field(..., description="Path-safe sanitized filename")
    file_type: DocumentType = Field(...)
    mime_type: str = Field(...)
    file_size_bytes: int = Field(..., ge=1)
    sha256_hash: str = Field(..., min_length=64, max_length=64)
    title: str | None = None
    author: str | None = None
    created_at_meta: str | None = None
    total_pages: int = Field(default=1, ge=1)
    total_words: int = Field(default=0, ge=0)
    total_chars: int = Field(default=0, ge=0)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Complete, normalized output of the ingestion and parsing pipeline."""

    id: UUID = Field(default_factory=uuid4, description="Unique document ID (UUIDv4)")
    metadata: DocumentMetadata
    pages: list[ExtractedPage] = Field(..., min_length=1)
    storage_key: str = Field(..., description="Object storage (S3/MinIO) path")
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_text(self) -> str:
        """Full concatenated text across all pages with page markers."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    def get_page_text(self, page_number: int) -> str | None:
        """Retrieve text for a specific 1-indexed page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        return None
