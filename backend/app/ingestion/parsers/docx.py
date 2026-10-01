"""Microsoft Word (.docx) document parser using python-docx."""

import io
from typing import Any

import docx

from app.core.logging import logger
from app.ingestion.exceptions import CorruptDocumentError, DocumentParsingError
from app.ingestion.models import DocumentMetadata, ExtractedPage
from app.ingestion.parsers.base import BaseParser


class DocxParser(BaseParser):
    """Parses DOCX documents into structured pages/sections, extracting paragraphs, tables, and document metadata."""

    # Approximate word count per simulated page when physical page breaks are omitted
    WORDS_PER_PAGE_FALLBACK = 450

    def parse(
        self, content: bytes, metadata: DocumentMetadata
    ) -> tuple[list[ExtractedPage], dict[str, Any]]:
        extracted_pages: list[ExtractedPage] = []
        extracted_meta: dict[str, Any] = {}

        try:
            doc = docx.Document(io.BytesIO(content))
        except Exception as err:
            logger.error("docx_open_failed", error=str(err), filename=metadata.filename)
            raise CorruptDocumentError(
                f"Failed to open DOCX document '{metadata.filename}'. The file may be corrupt: {err}"
            ) from err

        try:
            # Extract document core properties
            props = doc.core_properties
            extracted_meta = {
                "title": props.title or None,
                "author": props.author or None,
                "created_at_meta": str(props.created) if props.created else None,
                "category": props.category or None,
            }

            # Collect all text segments (paragraphs + tables)
            text_blocks: list[str] = []

            for paragraph in doc.paragraphs:
                cleaned = self.clean_text(paragraph.text)
                if cleaned:
                    text_blocks.append(cleaned)

            for table in doc.tables:
                table_lines = []
                for row in table.rows:
                    row_cells = [self.clean_text(cell.text) for cell in row.cells]
                    # Filter out blank rows
                    if any(row_cells):
                        table_lines.append(" | ".join(row_cells))
                if table_lines:
                    text_blocks.append("\n".join(table_lines))

            # Combine into pages (partitioning by natural size or single page if small)
            full_text = "\n\n".join(text_blocks)
            words = full_text.split()

            if not words:
                # Still emit a single empty page for valid empty documents
                extracted_pages.append(
                    ExtractedPage(page_number=1, text="", metadata={"type": "docx_empty"})
                )
            else:
                # Segment words into simulated pages of ~450 words
                chunk_words: list[str] = []
                current_page = 1

                for word in words:
                    chunk_words.append(word)
                    if len(chunk_words) >= self.WORDS_PER_PAGE_FALLBACK:
                        page_text = " ".join(chunk_words)
                        extracted_pages.append(
                            ExtractedPage(
                                page_number=current_page,
                                text=self.clean_text(page_text),
                                metadata={"type": "docx_section"},
                            )
                        )
                        chunk_words = []
                        current_page += 1

                # Append remaining words
                if chunk_words:
                    page_text = " ".join(chunk_words)
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=current_page,
                            text=self.clean_text(page_text),
                            metadata={"type": "docx_section"},
                        )
                    )

            logger.info(
                "docx_parsed_successfully",
                filename=metadata.filename,
                total_sections=len(extracted_pages),
            )

        except Exception as err:
            logger.error("docx_parse_error", error=str(err), filename=metadata.filename)
            raise DocumentParsingError(
                f"Unexpected error parsing DOCX '{metadata.filename}': {err}"
            ) from err

        return extracted_pages, extracted_meta
