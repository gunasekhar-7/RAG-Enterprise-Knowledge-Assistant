"""Parser for Plain Text (.txt) and Markdown (.md) documents."""

import re
from typing import Any

from app.core.logging import logger
from app.ingestion.exceptions import DocumentParsingError
from app.ingestion.models import DocumentMetadata, DocumentType, ExtractedPage
from app.ingestion.parsers.base import BaseParser


class TextParser(BaseParser):
    """Parses plain text and Markdown documents, segmenting long texts into logical pages/sections."""

    WORDS_PER_PAGE = 500

    def parse(
        self, content: bytes, metadata: DocumentMetadata
    ) -> tuple[list[ExtractedPage], dict[str, Any]]:
        extracted_pages: list[ExtractedPage] = []
        extracted_meta: dict[str, Any] = {}

        try:
            # Decode text with UTF-8 primary, fallback to Latin-1
            try:
                raw_text = content.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = content.decode("latin-1")

            cleaned_text = self.clean_text(raw_text)

            # Metadata extraction: extract title from Markdown '# Title' if present
            title = None
            if metadata.file_type == DocumentType.MARKDOWN:
                title_match = re.search(r"^#\s+(.+)$", cleaned_text, re.MULTILINE)
                if title_match:
                    title = title_match.group(1).strip()

            extracted_meta = {
                "title": title or metadata.sanitized_filename,
                "format": metadata.file_type.value,
            }

            # Split into sections if Markdown has headers, or by word thresholds
            words = cleaned_text.split()
            if not words:
                extracted_pages.append(
                    ExtractedPage(page_number=1, text="", metadata={"type": "empty_text"})
                )
            else:
                chunk_words: list[str] = []
                current_page = 1

                for word in words:
                    chunk_words.append(word)
                    if len(chunk_words) >= self.WORDS_PER_PAGE:
                        extracted_pages.append(
                            ExtractedPage(
                                page_number=current_page,
                                text=" ".join(chunk_words),
                                metadata={"type": "text_section"},
                            )
                        )
                        chunk_words = []
                        current_page += 1

                if chunk_words:
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=current_page,
                            text=" ".join(chunk_words),
                            metadata={"type": "text_section"},
                        )
                    )

            logger.info(
                "text_document_parsed",
                filename=metadata.filename,
                total_sections=len(extracted_pages),
            )

        except Exception as err:
            logger.error("text_parse_error", error=str(err), filename=metadata.filename)
            raise DocumentParsingError(
                f"Failed to parse text document '{metadata.filename}': {err}"
            ) from err

        return extracted_pages, extracted_meta
