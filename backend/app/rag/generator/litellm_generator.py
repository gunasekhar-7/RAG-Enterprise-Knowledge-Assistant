import json

import litellm
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import AsyncIterator

from app.core.config import settings
from app.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from app.rag.generator.base import BaseLLM, GenerationResult

logger = structlog.get_logger(__name__)


class LiteLLMGenerator(BaseLLM):
    """
    Implements LLM generation using LiteLLM to support multiple providers (OpenAI, Anthropic, etc).
    Handles automatic retries for rate limits and ensures structured Pydantic output.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        logger.info("Initialized LiteLLMGenerator", model=model_name, temp=temperature)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (litellm.exceptions.RateLimitError, litellm.exceptions.ServiceUnavailableError)
        ),
        reraise=True,
    )
    async def generate_structured(
        self, system_prompt: str, messages: list[dict]
    ) -> GenerationResult:
        """
        Executes the LLM call enforcing the output to match the GenerationResult schema.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            # Pydantic v2 schema generation
            schema = GenerationResult.model_json_schema()

            logger.debug("Calling LLM", model=self.model_name)

            # Use LiteLLM's structured output feature (which wraps OpenAI function calling / JSON mode)
            response = await litellm.acompletion(
                model=self.model_name,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=1500,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "GenerationResult", "schema": schema, "strict": True},
                },
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )

            # Parse the JSON string back into the Pydantic model
            response_content = response.choices[0].message.content

            if not response_content:
                raise ValueError("LLM returned empty content.")

            data = json.loads(response_content)
            result = GenerationResult(**data)

            logger.info(
                "LLM generation successful",
                citations=len(result.citations),
                insufficient=result.insufficient_information,
            )
            return result

        except litellm.exceptions.RateLimitError as e:
            logger.error("LLM rate limit exceeded", error=str(e))
            raise LLMRateLimitError(model=self.model_name) from e
        except litellm.exceptions.ServiceUnavailableError as e:
            logger.error("LLM service unavailable", error=str(e))
            raise LLMError(message="LLM service unavailable", model=self.model_name, retryable=True, details={"error": str(e)}) from e
        except litellm.exceptions.TimeoutError as e:
            logger.error("LLM request timed out", error=str(e))
            raise LLMTimeoutError(model=self.model_name, timeout_seconds=1500) from e
        except litellm.exceptions.AuthenticationError as e:
            logger.error("LLM authentication failed", error=str(e))
            raise LLMError(message="LLM authentication failed", model=self.model_name, retryable=False, details={"error": str(e)}) from e
        except json.JSONDecodeError as e:
            logger.error("LLM returned invalid JSON", error=str(e))
            raise LLMError(message="LLM returned invalid response format", model=self.model_name, retryable=False, details={"error": str(e)}) from e
        except Exception as e:
            logger.error("LLM Generation failed", error=str(e))
            raise LLMError(message="LLM generation failed", model=self.model_name, retryable=False, details={"error": str(e)}) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (litellm.exceptions.RateLimitError, litellm.exceptions.ServiceUnavailableError)
        ),
        reraise=True,
    )
    async def generate_text_stream(
        self, system_prompt: str, messages: list[dict]
    ) -> AsyncIterator[str]:
        """
        Streams the LLM response as text deltas.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=1500,
                stream=True,
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
            )

            async for chunk in response:
                if chunk.choices[0].delta.get('content'):
                    yield chunk.choices[0].delta.content

        except litellm.exceptions.RateLimitError as e:
            logger.error("LLM rate limit exceeded", error=str(e))
            raise LLMRateLimitError(model=self.model_name) from e
        except litellm.exceptions.ServiceUnavailableError as e:
            logger.error("LLM service unavailable", error=str(e))
            raise LLMError(message="LLM service unavailable", model=self.model_name, retryable=True, details={"error": str(e)}) from e
        except litellm.exceptions.TimeoutError as e:
            logger.error("LLM request timed out", error=str(e))
            raise LLMTimeoutError(model=self.model_name, timeout_seconds=1500) from e
        except litellm.exceptions.AuthenticationError as e:
            logger.error("LLM authentication failed", error=str(e))
            raise LLMError(message="LLM authentication failed", model=self.model_name, retryable=False, details={"error": str(e)}) from e
        except Exception as e:
            logger.error("LLM Generation failed", error=str(e))
            raise LLMError(message="LLM generation failed", model=self.model_name, retryable=False, details={"error": str(e)}) from e
