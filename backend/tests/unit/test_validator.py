import pytest

from app.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from app.ingestion.models import DocumentType
from app.ingestion.validator import FileValidator


def test_sanitize_filename() -> None:
    validator = FileValidator()

    # Path traversal attack
    assert validator.sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert validator.sanitize_filename("..\\..\\windows\\system32.pdf") == "system32.pdf"

    # Spaces and special characters
    assert (
        validator.sanitize_filename("Employee Handbook (2026) [FINAL]!.pdf")
        == "Employee_Handbook__2026___FINAL__.pdf"
    )

    # Leading dots (hidden files)
    assert validator.sanitize_filename(".hidden_policy.pdf") == "doc_hidden_policy.pdf"

    # Empty filename
    with pytest.raises(FileValidationError):
        validator.sanitize_filename("   ")


def test_empty_file_validation() -> None:
    validator = FileValidator()
    with pytest.raises(EmptyFileError):
        validator.validate("test.pdf", b"")


def test_file_too_large_validation() -> None:
    validator = FileValidator(max_file_size_bytes=100)
    oversized = b"%PDF-" + b"x" * 200
    with pytest.raises(FileTooLargeError):
        validator.validate("large.pdf", oversized)


def test_detect_pdf_type() -> None:
    validator = FileValidator()
    pdf_bytes = b"%PDF-1.7 mock pdf content"
    sanitized, doc_type, mime, sha256 = validator.validate("my_policy.pdf", pdf_bytes)

    assert sanitized == "my_policy.pdf"
    assert doc_type == DocumentType.PDF
    assert mime == "application/pdf"
    assert len(sha256) == 64


def test_detect_docx_type() -> None:
    validator = FileValidator()
    docx_bytes = b"PK\x03\x04 mock docx zip stream"
    sanitized, doc_type, mime, sha256 = validator.validate("sop.docx", docx_bytes)

    assert sanitized == "sop.docx"
    assert doc_type == DocumentType.DOCX
    assert "wordprocessingml" in mime
    assert len(sha256) == 64


def test_detect_text_and_markdown() -> None:
    validator = FileValidator()

    # TXT
    txt_bytes = b"Standard operating procedure text content."
    _, doc_type, mime, _ = validator.validate("guide.txt", txt_bytes)
    assert doc_type == DocumentType.TXT
    assert mime == "text/plain"

    # Markdown
    md_bytes = b"# Security Policy\n\nAll employees must use MFA."
    _, doc_type, mime, _ = validator.validate("policy.md", md_bytes)
    assert doc_type == DocumentType.MARKDOWN
    assert mime == "text/markdown"


def test_reject_binary_content_in_text_file() -> None:
    validator = FileValidator()
    # Null bytes inside text file
    corrupt_text = b"Normal header\x00\x01\x02binary payload"
    with pytest.raises(UnsupportedFileTypeError):
        validator.validate("malicious.txt", corrupt_text)


def test_unsupported_file_extension() -> None:
    validator = FileValidator()
    with pytest.raises(UnsupportedFileTypeError):
        validator.validate("script.exe", b"MZ\x90\x00executable")
