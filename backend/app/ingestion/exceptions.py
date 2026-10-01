"""Custom exceptions for document ingestion and parsing."""


class IngestionError(Exception):
    """Base exception for all document ingestion errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class FileValidationError(IngestionError):
    """Raised when file validation fails (size, magic bytes, naming)."""

    pass


class UnsupportedFileTypeError(FileValidationError):
    """Raised when the uploaded file type is not supported."""

    pass


class FileTooLargeError(FileValidationError):
    """Raised when file exceeds configured size limit."""

    pass


class EmptyFileError(FileValidationError):
    """Raised when file is 0 bytes."""

    pass


class DuplicateDocumentError(IngestionError):
    """Raised when a document with identical SHA-256 hash already exists."""

    pass


class CorruptDocumentError(IngestionError):
    """Raised when file contents cannot be parsed or are corrupted."""

    pass


class DocumentParsingError(IngestionError):
    """Raised when an unrecoverable error occurs during text extraction."""

    pass
