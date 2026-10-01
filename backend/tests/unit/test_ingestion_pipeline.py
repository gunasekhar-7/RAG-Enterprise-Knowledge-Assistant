import fitz
import pytest

from app.ingestion.exceptions import DuplicateDocumentError
from app.ingestion.models import DocumentType
from app.ingestion.pipeline import DocumentIngestionPipeline


def create_test_pdf() -> bytes:
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Confidential SOP Document Content.")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.asyncio
async def test_pipeline_pdf_success() -> None:
    pipeline = DocumentIngestionPipeline()
    pdf_bytes = create_test_pdf()

    parsed_doc = await pipeline.process(
        filename="../../policies/Confidential SOP.pdf",
        content=pdf_bytes,
        tenant_id="acme_corp",
        custom_metadata={"department": "IT_Security"},
    )

    # Verify ID and metadata
    assert parsed_doc.id is not None
    assert parsed_doc.metadata.sanitized_filename == "Confidential_SOP.pdf"
    assert parsed_doc.metadata.file_type == DocumentType.PDF
    assert parsed_doc.metadata.total_pages == 1
    assert parsed_doc.metadata.custom_metadata["department"] == "IT_Security"

    # Verify canonical S3 storage key
    expected_prefix = f"raw/acme_corp/{parsed_doc.id}/v1/Confidential_SOP.pdf"
    assert parsed_doc.storage_key == expected_prefix

    # Verify text extraction & page numbering
    assert len(parsed_doc.pages) == 1
    assert parsed_doc.pages[0].page_number == 1
    assert "Confidential SOP Document Content." in parsed_doc.pages[0].text
    assert "Confidential SOP Document Content." in parsed_doc.full_text


@pytest.mark.asyncio
async def test_pipeline_duplicate_detection() -> None:
    # Mock duplicate checker that flags the document as existing
    async def mock_duplicate_checker(_sha256_hash: str) -> bool:
        return True

    pipeline = DocumentIngestionPipeline(duplicate_checker=mock_duplicate_checker)
    pdf_bytes = create_test_pdf()

    with pytest.raises(DuplicateDocumentError):
        await pipeline.process(
            filename="existing_doc.pdf",
            content=pdf_bytes,
            allow_duplicates=False,
        )

    # When allow_duplicates=True, duplicate check is bypassed
    doc = await pipeline.process(
        filename="existing_doc.pdf",
        content=pdf_bytes,
        allow_duplicates=True,
    )
    assert doc is not None
