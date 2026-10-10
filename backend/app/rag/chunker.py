"""Production-quality document chunking engine with token bounds, page preservation, and section tracking."""

import contextlib
import re

import tiktoken

from app.core.logging import logger
from app.ingestion.models import ParsedDocument
from app.rag.models import ChunkingConfig, ChunkMetadata, DocumentChunk


class Tokenizer:
    """Thread-safe tokenizer wrapper for accurate token measurement and text slicing."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning("fallback_tokenizer_used", requested=encoding_name)
            self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens for a given text string."""
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        """Encodes text to a list of token integers."""
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        """Decodes token integers back to text."""
        return self._encoding.decode(tokens)


class DocumentChunker:
    """Splits structured documents into semantic passages while preserving page numbers and section headers."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = Tokenizer(self.config.tokenizer_encoding)

    def extract_header(self, text: str) -> str | None:
        """Extracts prominent Markdown or uppercase section headings from text."""
        # Check Markdown headers (# Header or ## Header)
        md_match = re.search(r"^(#{1,4}\s+[^\n]+)", text, re.MULTILINE)
        if md_match:
            return md_match.group(1).strip()

        # Check numbered section headers (e.g. "Section 4.1 Security Controls")
        sec_match = re.search(r"^([0-9]+\.[0-9]*\s+[^\n]+)", text, re.MULTILINE)
        if sec_match:
            return sec_match.group(1).strip()

        return None

    def _split_text_recursive(self, text: str, separator_idx: int = 0) -> list[str]:
        """Recursively decomposes text using a hierarchy of separators until all segments fit within chunk_size."""
        text = text.strip()
        if not text:
            return []

        # Base case: text already fits within chunk size
        if self.tokenizer.count_tokens(text) <= self.config.chunk_size:
            return [text]

        # Exhausted all separators: hard-slice by token length
        if separator_idx >= len(self.config.separators):
            tokens = self.tokenizer.encode(text)
            hard_splits = []
            for i in range(0, len(tokens), self.config.chunk_size):
                sub_slice = tokens[i : i + self.config.chunk_size]
                hard_splits.append(self.tokenizer.decode(sub_slice))
            return hard_splits

        sep = self.config.separators[separator_idx]

        # If separator is empty string, fall back to character slicing
        if sep == "":
            return self._split_text_recursive(text, separator_idx + 1)

        # Split on the current separator
        splits = text.split(sep)
        if len(splits) == 1:
            # Separator not found in text; advance to the next separator
            return self._split_text_recursive(text, separator_idx + 1)

        # Merge splits greedily up to chunk_size
        final_chunks: list[str] = []
        current_chunk_parts: list[str] = []
        current_token_count = 0

        for part in splits:
            part_str = part.strip()
            if not part_str:
                continue

            part_tokens = self.tokenizer.count_tokens(part_str)

            # If an individual part exceeds chunk_size, recursively split it further
            if part_tokens > self.config.chunk_size:
                # Flush existing buffer
                if current_chunk_parts:
                    final_chunks.append(sep.join(current_chunk_parts))
                    current_chunk_parts = []
                    current_token_count = 0
                sub_parts = self._split_text_recursive(part_str, separator_idx + 1)
                final_chunks.extend(sub_parts)
                continue

            # Check if adding part exceeds limit
            if current_token_count + part_tokens > self.config.chunk_size:
                if current_chunk_parts:
                    final_chunks.append(sep.join(current_chunk_parts))
                current_chunk_parts = [part_str]
                current_token_count = part_tokens
            else:
                current_chunk_parts.append(part_str)
                current_token_count += part_tokens

        # Flush remaining parts
        if current_chunk_parts:
            final_chunks.append(sep.join(current_chunk_parts))

        return final_chunks

    def _apply_overlap(self, raw_chunks: list[str]) -> list[str]:
        """Applies configured token overlap between consecutive chunks to prevent boundary semantic loss."""
        if self.config.chunk_overlap <= 0 or len(raw_chunks) <= 1:
            return raw_chunks

        overlapped_chunks: list[str] = []

        for idx, chunk in enumerate(raw_chunks):
            if idx == 0:
                overlapped_chunks.append(chunk)
                continue

            prev_chunk = raw_chunks[idx - 1]
            prev_tokens = self.tokenizer.encode(prev_chunk)

            # Take the trailing tokens of previous chunk
            overlap_token_slice = prev_tokens[-self.config.chunk_overlap :]
            overlap_prefix = self.tokenizer.decode(overlap_token_slice)

            # Strip any incomplete leading word from the overlap prefix
            overlap_words = overlap_prefix.split()
            if len(overlap_words) > 1:
                clean_overlap = " ".join(overlap_words[1:])
            else:
                clean_overlap = overlap_prefix

            combined = f"{clean_overlap} {chunk}".strip()
            overlapped_chunks.append(combined)

        return overlapped_chunks

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Transforms a ParsedDocument into a sequence of DocumentChunk objects.

        Preserves:
            - Exact 1-indexed page numbers for downstream citations
            - Section headings
            - Document IDs and custom metadata
        """
        chunks: list[DocumentChunk] = []
        global_chunk_idx = 0
        active_section_header: str | None = None

        logger.info(
            "chunking_started",
            document_id=str(document.id),
            total_pages=len(document.pages),
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

        global_text = ""
        char_to_page = []  # type: ignore

        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            if global_text:
                sep = "\n\n"
                global_text += sep
                char_to_page.extend([char_to_page[-1]] * len(sep))

            global_text += page_text
            char_to_page.extend([page.page_number] * len(page_text))

        if not global_text:
            return chunks

        raw_chunks = self._split_text_recursive(global_text)

        raw_positions = []
        search_idx = 0
        for rc in raw_chunks:
            idx = global_text.find(rc, search_idx)
            if idx == -1:
                idx = search_idx
            raw_positions.append((idx, idx + len(rc)))
            search_idx = idx + len(rc)

        overlapped_chunks = self._apply_overlap(raw_chunks)

        import uuid

        tenant_id = document.id
        tenant_id_str = document.metadata.custom_metadata.get("tenant_id")
        if not tenant_id_str and document.storage_key:
            parts = document.storage_key.split("/")
            if len(parts) >= 2:
                tenant_id_str = parts[0]

        if tenant_id_str:
            with contextlib.suppress(ValueError):
                tenant_id = uuid.UUID(str(tenant_id_str))

        for i, chunk_text in enumerate(overlapped_chunks):
            clean_chunk_text = chunk_text.strip()
            if len(clean_chunk_text) < self.config.min_chunk_size_chars:
                continue

            start_char_idx = raw_positions[i][0]
            if i > 0 and self.config.chunk_overlap > 0:
                prefix = clean_chunk_text[:20]
                idx = global_text.find(prefix, raw_positions[i - 1][0])
                if idx != -1 and idx <= raw_positions[i][0]:
                    start_char_idx = idx
                else:
                    start_char_idx = raw_positions[i - 1][0]

            end_char_idx = raw_positions[i][1] - 1
            if end_char_idx < 0:
                end_char_idx = 0

            page_start = char_to_page[start_char_idx] if start_char_idx < len(char_to_page) else 1
            page_end = (
                char_to_page[min(end_char_idx, len(char_to_page) - 1)]
                if end_char_idx >= 0
                else page_start
            )

            header = self.extract_header(clean_chunk_text)
            if header:
                active_section_header = header

            chunk_token_count = self.tokenizer.count_tokens(clean_chunk_text)
            chunk_id = f"{document.id}_c{global_chunk_idx:04d}"

            chunk_meta = ChunkMetadata(  # type: ignore
                document_id=document.id,
                tenant_id=tenant_id,
                chunk_index=global_chunk_idx,
                page_number=page_start,
                page_end=page_end,
                section_heading=active_section_header,
                filename=document.metadata.filename,
                token_count=chunk_token_count,
                char_count=len(clean_chunk_text),
                custom_metadata=document.metadata.custom_metadata,
            )

            chunks.append(
                DocumentChunk(  # type: ignore
                    chunk_id=chunk_id,
                    document_id=document.id,
                    text=clean_chunk_text,
                    metadata=chunk_meta,
                )
            )
            global_chunk_idx += 1

        logger.info(
            "chunking_completed",
            document_id=str(document.id),
            total_chunks_created=len(chunks),
        )
        return chunks
