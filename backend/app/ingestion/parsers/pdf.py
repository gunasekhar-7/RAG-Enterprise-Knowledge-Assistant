"""High-performance PDF parser using PyMuPDF (fitz), preserving page numbers for citations."""

from typing import Any

import fitz  # PyMuPDF

from app.core.logging import logger
from app.ingestion.exceptions import CorruptDocumentError, DocumentParsingError
from app.ingestion.models import DocumentMetadata, ExtractedPage
from app.ingestion.parsers.base import BaseParser


class PDFParser(BaseParser):
    """Parses PDF documents into per-page extracted segments while preserving exact 1-indexed page numbers."""

    def parse(
        self, content: bytes, metadata: DocumentMetadata
    ) -> tuple[list[ExtractedPage], dict[str, Any]]:
        extracted_pages: list[ExtractedPage] = []
        extracted_meta: dict[str, Any] = {}

        try:
            # Open PDF directly from memory stream (no disk writes required)
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as err:
            logger.error("pdf_open_failed", error=str(err), filename=metadata.filename)
            raise CorruptDocumentError(
                f"Failed to open PDF document '{metadata.filename}'. The file may be corrupt or encrypted: {err}"
            ) from err

        try:
            # Check for encryption / password requirement
            if doc.is_encrypted:
                raise CorruptDocumentError(
                    f"PDF document '{metadata.filename}' is password-protected or encrypted."
                )

            total_pages = doc.page_count
            if total_pages == 0:
                raise CorruptDocumentError(f"PDF document '{metadata.filename}' contains 0 pages.")

            # Extract document-level PDF metadata
            doc_meta = doc.metadata or {}
            extracted_meta = {
                "title": doc_meta.get("title") or None,
                "author": doc_meta.get("author") or None,
                "created_at_meta": doc_meta.get("creationDate") or None,
                "producer": doc_meta.get("producer") or None,
                "format": doc_meta.get("format") or "PDF",
            }

            # Iterate over all pages (1-indexed page numbering)
            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text")
                clean_page_text = self.clean_text(raw_text)

                # Page dimensions and rotation metadata
                page_meta = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                }

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_idx + 1,  # 1-indexed for human-readable citations
                        text=clean_page_text,
                        metadata=page_meta,
                    )
                )

            logger.info(
                "pdf_parsed_successfully",
                filename=metadata.filename,
                total_pages=len(extracted_pages),
            )

        except CorruptDocumentError:
            raise
        except Exception as err:
            logger.error("pdf_parse_error", error=str(err), filename=metadata.filename)
            raise DocumentParsingError(
                f"Unexpected error parsing PDF '{metadata.filename}': {err}"
            ) from err
        finally:
            doc.close()

        return extracted_pages, extracted_meta
