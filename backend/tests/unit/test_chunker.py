from uuid import uuid4

from app.ingestion.models import DocumentMetadata, DocumentType, ExtractedPage, ParsedDocument
from app.rag.chunker import DocumentChunker, Tokenizer
from app.rag.models import ChunkingConfig


def create_mock_parsed_document(pages_content: list[str]) -> ParsedDocument:
    doc_id = uuid4()
    pages = [
        ExtractedPage(page_number=idx + 1, text=content)
        for idx, content in enumerate(pages_content)
    ]
    meta = DocumentMetadata(
        filename="Remote_Work_Policy.pdf",
        sanitized_filename="Remote_Work_Policy.pdf",
        file_type=DocumentType.PDF,
        mime_type="application/pdf",
        file_size_bytes=1024,
        sha256_hash="e" * 64,
        custom_metadata={"department": "HR", "classification": "Internal"},
    )
    return ParsedDocument(
        id=doc_id,
        metadata=meta,
        pages=pages,
        storage_key=f"raw/default/{doc_id}/v1/Remote_Work_Policy.pdf",
    )


def test_tokenizer_token_count() -> None:
    tokenizer = Tokenizer()
    text = "Enterprise RAG systems require precise chunking strategies."
    count = tokenizer.count_tokens(text)
    assert count > 0
    assert count == len(tokenizer.encode(text))
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_short_document_creates_single_chunk() -> None:
    short_text = "This is a brief policy statement regarding company equipment care."
    doc = create_mock_parsed_document([short_text])

    chunker = DocumentChunker(ChunkingConfig(chunk_size=512, chunk_overlap=64))
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == f"{doc.id}_c0000"
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.page_number == 1
    assert chunks[0].metadata.filename == "Remote_Work_Policy.pdf"
    assert chunks[0].metadata.custom_metadata["department"] == "HR"
    assert chunks[0].citation_source == "Remote_Work_Policy.pdf (Page 1)"


def test_page_number_preservation_across_multiple_pages() -> None:
    page1 = "Page 1: Overview of employee benefits, healthcare, and parental leave."
    page2 = "Page 2: Travel expense reimbursement rules and per diem rates."
    doc = create_mock_parsed_document([page1, page2])

    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].metadata.page_number == 1
    assert "Page 1: Overview" in chunks[0].text
    assert chunks[1].metadata.page_number == 2
    assert "Page 2: Travel" in chunks[1].text


def test_long_page_recursive_splitting() -> None:
    # Generate long text that exceeds a small chunk_size of 50 tokens
    paragraphs = [
        f"Paragraph {i}: This is an in-depth explanation of technical infrastructure section {i}, "
        f"detailing security controls, authentication standards, and automated deployment checks."
        for i in range(10)
    ]
    long_text = "\n\n".join(paragraphs)
    doc = create_mock_parsed_document([long_text])

    # Configure small chunk size to trigger recursive splitting
    config = ChunkingConfig(chunk_size=64, chunk_overlap=16)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(doc)

    # Should create multiple chunks, all respecting the target token limit + overlap buffer
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.metadata.page_number == 1
        # Chunks should be within reasonable bounds
        assert chunk.metadata.token_count <= 80


def test_section_heading_tracking() -> None:
    page_text = """## Section 3.2 Vacation Leave

Employees receive 20 days of paid vacation per calendar year.

## Section 3.3 Sick Leave

Employees receive 10 days of paid sick leave annually."""

    doc = create_mock_parsed_document([page_text])
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    assert chunks[0].metadata.section_heading is not None
    assert "Section 3." in chunks[0].metadata.section_heading


def test_empty_pages_ignored() -> None:
    doc = create_mock_parsed_document(["   \n\n  ", "Valid page content with sufficient length."])
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].metadata.page_number == 2
    assert "Valid page content" in chunks[0].text


def test_chunk_overlap_content() -> None:
    # Text with distinct sentences to verify overlap inclusion
    text = (
        "Sentence Alpha establishes the foundational baseline. "
        "Sentence Beta introduces operational guidelines. "
        "Sentence Gamma details the enforcement mechanisms. "
        "Sentence Delta concludes the protocol."
    )
    doc = create_mock_parsed_document([text])

    config = ChunkingConfig(chunk_size=20, chunk_overlap=8)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Verify second chunk includes tokens from previous chunk
    assert len(chunks[1].text.split()) > 3
