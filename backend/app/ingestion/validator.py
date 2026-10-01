"""Secure file validation, MIME sniffing, sanitization, and checksum hashing."""

import hashlib
import re
from pathlib import PurePath

from app.core.logging import logger
from app.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from app.ingestion.models import DocumentType

# Default maximum upload size: 50MB
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Magic byte signatures
MAGIC_PDF = b"%PDF-"
MAGIC_ZIP_PK = b"PK\x03\x04"

# MIME mapping for supported types
MIME_MAP = {
    DocumentType.PDF: "application/pdf",
    DocumentType.DOCX: ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    DocumentType.TXT: "text/plain",
    DocumentType.MARKDOWN: "text/markdown",
}


class FileValidator:
    """Validates uploaded files against security rules, size limits, and magic byte signatures."""

    def __init__(self, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def sanitize_filename(self, filename: str) -> str:
        """Sanitizes filename to prevent directory traversal and null-byte injection.

        Extracts only the basename, removes path separators, and replaces
        unsupported characters with underscores.
        """
        if not filename or not filename.strip():
            raise FileValidationError("Filename cannot be empty")

        # Strip null bytes and directory traversal components
        clean = filename.replace("\x00", "")
        basename = PurePath(clean).name

        # Replace non-alphanumeric (except dot, underscore, dash) with underscore
        sanitized = re.sub(r"[^\w\.-]", "_", basename)

        # Avoid hidden files or leading dot
        if sanitized.startswith("."):
            sanitized = f"doc_{sanitized.lstrip('.')}"

        if not sanitized or sanitized == "_":
            raise FileValidationError(f"Invalid filename after sanitization: '{filename}'")

        return sanitized

    def calculate_sha256(self, content: bytes) -> str:
        """Computes deterministic SHA-256 hexadecimal hash of byte content."""
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()

    def detect_file_type(self, filename: str, content: bytes) -> tuple[DocumentType, str]:
        """Detects document type and MIME type using magic numbers and extension verification.

        Guards against extension spoofing (e.g. an executable renamed to .pdf).
        """
        ext = PurePath(filename).suffix.lower()

        # 1. Check PDF Magic Bytes
        if content.startswith(MAGIC_PDF):
            if ext in [".pdf", ""]:
                return DocumentType.PDF, MIME_MAP[DocumentType.PDF]
            logger.warning(
                "extension_mismatch",
                detected="pdf",
                provided_extension=ext,
                filename=filename,
            )
            return DocumentType.PDF, MIME_MAP[DocumentType.PDF]

        # 2. Check DOCX (ZIP container with Word XML structure)
        if content.startswith(MAGIC_ZIP_PK):
            if ext == ".docx":
                return DocumentType.DOCX, MIME_MAP[DocumentType.DOCX]
            raise UnsupportedFileTypeError(
                f"Binary ZIP archive detected, but expected '.docx' extension (got '{ext}')"
            )

        # 3. Check Text / Markdown
        if ext in [".txt", ".md", ".markdown"]:
            # Verify file does not contain null bytes (which indicate binary payload)
            if b"\x00" in content[:4096]:
                raise UnsupportedFileTypeError(
                    f"File '{filename}' contains binary null bytes and is not valid text"
                )

            # Verify UTF-8 decode capability
            try:
                content[:4096].decode("utf-8")
            except UnicodeDecodeError as err:
                raise UnsupportedFileTypeError(
                    f"Text file '{filename}' is not valid UTF-8 encoded text: {err}"
                ) from err

            if ext in [".md", ".markdown"]:
                return DocumentType.MARKDOWN, MIME_MAP[DocumentType.MARKDOWN]
            return DocumentType.TXT, MIME_MAP[DocumentType.TXT]

        raise UnsupportedFileTypeError(
            f"Unsupported file format for '{filename}'. Allowed formats: PDF, DOCX, TXT, MD"
        )

    def validate(self, filename: str, content: bytes) -> tuple[str, DocumentType, str, str]:
        """Runs the complete validation pipeline on a file.

        Returns:
            Tuple of (sanitized_filename, document_type, mime_type, sha256_hash)

        Raises:
            EmptyFileError: if content is 0 bytes.
            FileTooLargeError: if content exceeds maximum byte size.
            UnsupportedFileTypeError: if file format is unsupported or spoofed.
            FileValidationError: if filename or data violates security rules.
        """
        # 1. Size Validation
        size = len(content)
        if size == 0:
            raise EmptyFileError(f"Uploaded file '{filename}' is empty (0 bytes).")

        if size > self.max_file_size_bytes:
            raise FileTooLargeError(
                f"File '{filename}' ({size} bytes) exceeds maximum limit of "
                f"{self.max_file_size_bytes} bytes."
            )

        # 2. Path Sanitization
        sanitized_filename = self.sanitize_filename(filename)

        # 3. Type & Magic Detection
        doc_type, mime_type = self.detect_file_type(sanitized_filename, content)

        # 4. Hash Calculation
        sha256 = self.calculate_sha256(content)

        logger.info(
            "file_validated_successfully",
            original_filename=filename,
            sanitized_filename=sanitized_filename,
            file_type=doc_type.value,
            size_bytes=size,
            sha256=sha256[:12] + "...",
        )

        return sanitized_filename, doc_type, mime_type, sha256
