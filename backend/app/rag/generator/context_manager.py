import structlog
import tiktoken

from app.rag.retrieval.base import RetrievedContext

logger = structlog.get_logger(__name__)


class ContextFormatter:
    """Handles parsing, formatting, and token-aware trimming of retrieved context."""

    def __init__(self, max_context_tokens: int = 4000, encoding_name: str = "cl100k_base"):
        self.max_context_tokens = max_context_tokens
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(
                f"Could not load tiktoken encoding '{encoding_name}'. Falling back to char approximation."
            )
            self.encoding = None  # type: ignore

    def _count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text) // 4  # Very rough approximation

    def build_context_string(self, chunks: list[RetrievedContext]) -> str:
        """
        Takes a ranked list of chunks and builds a formatted string suitable for the LLM.
        Stops appending chunks if the token limit is reached.
        """
        if not chunks:
            return "No documents were retrieved for this query."

        formatted_blocks = []
        current_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            sanitized_text = chunk.text.replace("<context>", "[context]").replace(
                "</context>", "[/context]"
            )
            block = (
                f"--- Document [{i}] ---\n"
                f"Source: {chunk.citation}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Content: <![CDATA[{sanitized_text}]]>\n"
            )

            block_tokens = self._count_tokens(block)

            if current_tokens + block_tokens > self.max_context_tokens:
                logger.info(
                    "Context token limit reached, truncating further chunks",
                    included=i - 1,
                    total=len(chunks),
                )
                break

            formatted_blocks.append(block)
            current_tokens += block_tokens

        return "\n".join(formatted_blocks)
