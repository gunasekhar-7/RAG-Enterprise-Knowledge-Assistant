from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a source reference for a generated answer."""

    chunk_id: str = Field(..., description="The ID of the exact document chunk used.")
    source_name: str = Field(..., description="Human-readable name of the document or page.")
    quote: str | None = Field(
        None, description="Exact text quote from the document supporting this claim."
    )


class GenerationResult(BaseModel):
    """Structured output returned from the LLM generator."""

    answer: str = Field(..., description="The grounded text answer provided to the user.")
    citations: list[Citation] = Field(
        default_factory=list, description="List of specific document chunks cited."
    )
    insufficient_information: bool = Field(
        default=False,
        description="True if the context lacked the necessary information to answer fully.",
    )


class BaseLLM:
    """Abstract interface for Language Models."""

    async def generate_structured(
        self, system_prompt: str, messages: list[dict]
    ) -> GenerationResult:
        """Generate a structured response based on instructions."""
        raise NotImplementedError
