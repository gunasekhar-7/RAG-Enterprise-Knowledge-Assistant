import structlog

from app.core.exceptions import LLMError
from app.rag.generator.base import BaseLLM, GenerationResult
from app.rag.generator.context_manager import ContextFormatter
from app.rag.generator.prompt import PromptBuilder
from app.rag.retrieval.base import RetrievedContext

logger = structlog.get_logger(__name__)


class GenerationPipeline:
    """
    Coordinates the final generation step of the RAG pipeline.
    Takes the user query and the retrieved context, formats it safely,
    constructs the robust system prompt, and yields the structured result.
    """

    def __init__(
        self, llm: BaseLLM, context_formatter: ContextFormatter, prompt_builder: PromptBuilder
    ):
        self.llm = llm
        self.context_formatter = context_formatter
        self.prompt_builder = prompt_builder

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[RetrievedContext],
        conversation_history: list[dict] = None,  # type: ignore
    ) -> GenerationResult:
        """
        Generates the grounded answer.

        Args:
            query: The user's question.
            context_chunks: Ranked chunks retrieved from vector store.
            conversation_history: Prior chat messages (role/content format).

        Returns:
            GenerationResult containing answer, citations, and status.
        """
        logger.info("Starting generation pipeline", chunks_provided=len(context_chunks))

        # 1. Format context respecting token limits
        formatted_context = self.context_formatter.build_context_string(context_chunks)

        # 2. Build system prompt enforcing rules
        system_prompt = self.prompt_builder.build_system_prompt(formatted_context)

        # 3. Assemble messages
        messages = conversation_history or []
        messages.append({"role": "user", "content": query})

        # 4. Generate structured response
        try:
            result = await self.llm.generate_structured(system_prompt=system_prompt, messages=messages)
        except Exception as e:
            # Re-raise LLM errors as our custom exception type
            if isinstance(e, LLMError):
                raise
            raise LLMError(
                message="Failed to generate answer",
                model=getattr(self.llm, 'model_name', 'unknown'),
                retryable=True,
                details={"error": str(e)}
            )

        return result
