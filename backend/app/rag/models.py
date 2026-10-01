"""Data models for document chunks, chunk metadata, and chunking configurations."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Rich metadata preserved with every document chunk for retrieval and citations."""

    document_id: UUID = Field(..., description="ID of parent document")
    chunk_index: int = Field(..., ge=0, description="Sequential 0-indexed position within document")
    page_number: int = Field(..., ge=1, description="Primary starting page number for citations")
    page_end: int | None = Field(
        None, ge=1, description="Ending page number if chunk spans pages"
    )
    section_heading: str | None = Field(None, description="Active section or header context")
    filename: str = Field(..., description="Original source document filename")
    token_count: int = Field(..., ge=0, description="Exact token count of chunk text")
    char_count: int = Field(..., ge=0, description="Character count of chunk text")
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom tenant or domain metadata"
    )


class DocumentChunk(BaseModel):
    """An individual indexed passage ready for embedding and hybrid retrieval."""

    chunk_id: str = Field(
        ..., description="Deterministic unique ID: {document_id}_c{chunk_index:04d}"
    )
    document_id: UUID = Field(...)
    text: str = Field(..., min_length=1, description="Passage text content")
    metadata: ChunkMetadata

    @property
    def citation_source(self) -> str:
        """Human-readable citation label, e.g. 'Employee_Handbook.pdf (Page 3)'."""
        if self.metadata.page_end and self.metadata.page_end != self.metadata.page_number:
            return f"{self.metadata.filename} (Pages {self.metadata.page_number}-{self.metadata.page_end})"
        return f"{self.metadata.filename} (Page {self.metadata.page_number})"


class ChunkingConfig(BaseModel):
    """Fine-grained configuration for experimental tuning of the chunking engine."""

    chunk_size: int = Field(
        default=512,
        ge=16,
        le=4096,
        description="Target maximum chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Number of overlapping tokens between consecutive chunks",
    )
    tokenizer_encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding used for token measurement (e.g. cl100k_base for OpenAI)",
    )
    min_chunk_size_chars: int = Field(
        default=25,
        ge=1,
        description="Discard noise chunks smaller than this character threshold",
    )
    preserve_page_boundaries: bool = Field(
        default=True,
        description="If True, chunks never cross page boundaries to ensure 100% exact page citations",
    )
    extract_section_headers: bool = Field(
        default=True,
        description="If True, detect Markdown/text section headers and bind them to chunk metadata",
    )
    separators: list[str] = Field(
        default_factory=lambda: [
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentences
            "? ",  # Questions
            "! ",  # Exclamations
            "; ",  # Clauses
            " ",  # Words
            "",  # Character fallback
        ]
    )
