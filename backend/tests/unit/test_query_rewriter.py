import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.models.chat import Message
from app.rag.generator.query_rewriter import QueryRewriter


@pytest.fixture
def mock_litellm():
    with patch(
        "app.rag.generator.query_rewriter.litellm.acompletion", new_callable=AsyncMock
    ) as mock_acomp:
        yield mock_acomp


@pytest.mark.asyncio
async def test_query_rewriter_no_history():
    rewriter = QueryRewriter()
    history = []

    query = "What is the annual leave policy?"
    standalone = await rewriter.rewrite(history, query)

    # If no history, it should just return the exact query without calling LLM
    assert standalone == query


@pytest.mark.asyncio
async def test_query_rewriter_with_history(mock_litellm):
    # Mocking LiteLLM's response format
    class MockMessage:
        content = '{"standalone_query": "Can annual leave days be carried forward according to the company policy?"}'

    class MockChoice:
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    mock_litellm.return_value = MockResponse()

    conversation_id = uuid.uuid4()
    history = [
        Message(
            conversation_id=conversation_id, role="user", content="What is the annual leave policy?"
        ),
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content="Employees receive 20 days per year.",
        ),
    ]

    rewriter = QueryRewriter()
    query = "Can I carry it forward?"

    standalone = await rewriter.rewrite(history, query)

    # Assert LLM was called
    mock_litellm.assert_called_once()

    # Assert correct response parsing
    assert standalone == "Can annual leave days be carried forward according to the company policy?"

    # Assert the history was passed into the prompt correctly
    call_args = mock_litellm.call_args[1]["messages"]
    assert len(call_args) == 2
    assert "HISTORY" in call_args[1]["content"]
    assert "USER: What is the annual leave policy?" in call_args[1]["content"]
    assert "FOLLOW-UP QUERY: Can I carry it forward?" in call_args[1]["content"]


@pytest.mark.asyncio
async def test_query_rewriter_fallback_on_error(mock_litellm):
    mock_litellm.side_effect = Exception("OpenAI API Down")

    conversation_id = uuid.uuid4()
    history = [
        Message(
            conversation_id=conversation_id, role="user", content="What is the annual leave policy?"
        ),
    ]

    rewriter = QueryRewriter()
    query = "Can I carry it forward?"

    # Should safely catch the error and fallback to original query
    standalone = await rewriter.rewrite(history, query)

    assert standalone == "Can I carry it forward?"
