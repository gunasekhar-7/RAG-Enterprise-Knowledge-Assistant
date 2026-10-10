import json
import litellm
import structlog
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import LLMError
from app.models.chat import Message

logger = structlog.get_logger(__name__)


class RewrittenQuery(BaseModel):
    standalone_query: str = Field(
        ..., description="The self-contained, rewritten query suitable for vector search."
    )


class QueryRewriter:
    """
    Analyzes conversation history and a new ambiguous follow-up query,
    transforming it into a fully contextualized, standalone search query.
    """

    SYSTEM_PROMPT = """You are a search query reformulation expert.
Given a conversation history and a follow-up user query, rewrite the follow-up query to be a standalone question that can be understood without the conversation history.

RULES:
1. If the query contains pronouns (it, they, them, he, she), replace them with the explicit nouns from the history.
2. If the query is already standalone, do not change it significantly.
3. Keep the query concise and optimized for semantic vector search.
4. Do not answer the question. Only output the rewritten query.
"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def rewrite(self, history: list[Message], current_query: str) -> str:
        """
        Takes the recent history and the new query. Returns the standalone query.
        """
        if not history:
            return current_query

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add a formatted transcript of history
        transcript = ""
        for msg in history:
            transcript += f"{msg.role.upper()}: {msg.content}\n"

        user_content = f"HISTORY:\n{transcript}\n\nFOLLOW-UP QUERY: {current_query}\n\nRewrite the follow-up query:"
        messages.append({"role": "user", "content": user_content})

        try:
            logger.debug("Rewriting query based on history", history_turns=len(history))

            # Using structured output to guarantee we just get the string
            schema = RewrittenQuery.model_json_schema()

            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "RewrittenQuery", "schema": schema, "strict": True},
                },
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )

            content = response.choices[0].message.content
            result = RewrittenQuery(**json.loads(content))

            logger.info(
                "Query rewritten", original=current_query, rewritten=result.standalone_query
            )
            return result.standalone_query

        except litellm.exceptions.RateLimitError as e:
            raise LLMError(message="Query rewrite rate limited", model=self.model_name, retryable=True, details={"error": str(e)})
        except litellm.exceptions.ServiceUnavailableError as e:
            raise LLMError(message="Query rewrite service unavailable", model=self.model_name, retryable=True, details={"error": str(e)})
        except Exception as e:
            logger.error("Query rewrite failed, falling back to original query", error=str(e))
            return current_query
