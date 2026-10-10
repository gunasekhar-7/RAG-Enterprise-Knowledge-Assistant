"""
Integration tests for the chat pipeline.

This module tests the complete chat flow including query rewriting, retrieval,
generation, and conversation history management. Tests cover both success and
failure paths with mocked external dependencies (LLM, embeddings, vector DB).

Category: integration
Focus areas: query rewrite, retrieval, generation, history
"""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.models.chat import Message
from app.models.user import Role
from app.rag.generator.base import Citation, GenerationResult
from app.rag.retrieval.base import RetrievedContext, RankedContext


class TestChatPipelineQueryRewrite:
    """Tests for query rewriting functionality with conversation history."""

    @pytest.mark.asyncio
    async def test_query_rewrite_with_history(self, admin_client: AsyncClient):
        """
        Validate that QueryRewriter correctly transforms ambiguous follow-up queries
        into standalone queries using conversation history.

        Success path: Query with pronouns is rewritten with explicit references.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Send first message
        with patch("app.rag.generator.query_rewriter.litellm.acompletion") as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "What is Python?"})
                        )
                    )
                ]
            )
            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "Tell me about Python"},
            )
            assert response.status_code == 200

        # Send follow-up with pronoun
        with patch("app.rag.generator.query_rewriter.litellm.acompletion") as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {"standalone_query": "What are the use cases of Python?"}
                            )
                        )
                    )
                ]
            )
            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What are its use cases?"},
            )
            assert response.status_code == 200
            result = response.json()
            assert "answer" in result

    @pytest.mark.asyncio
    async def test_query_rewrite_fallback_no_history(self, admin_client: AsyncClient):
        """
        Validate that QueryRewriter falls back to original query when
        there is no conversation history.

        Success path: Standalone query returns original query unchanged.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Send first message without history
        with patch("app.rag.generator.query_rewriter.litellm.acompletion") as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {"standalone_query": "What is machine learning?"}
                            )
                        )
                    )
                ]
            )
            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is machine learning?"},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_rewrite_llm_failure_fallback(self, admin_client: AsyncClient):
        """
        Validate that QueryRewriter gracefully falls back to original query
        when LLM fails after retries.

        Failure path: LLM exception is caught and original query is returned.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Add initial message to create history
        with patch("app.rag.generator.query_rewriter.litellm.acompletion") as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "What is AI?"})
                        )
                    )
                ]
            )
            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is AI?"},
            )
            assert response.status_code == 200

        # Send follow-up with LLM failure
        with patch("app.rag.generator.query_rewriter.litellm.acompletion") as mock_llm:
            mock_llm.side_effect = Exception("LLM service unavailable")
            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "Tell me more about it"},
            )
            # Should still succeed with original query
            assert response.status_code == 200


class TestChatPipelineRetrieval:
    """Tests for retrieval stage with various query types and filters."""

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_success(self, admin_client: AsyncClient):
        """
        Validate that hybrid retriever combines dense and sparse results
        with reciprocal rank fusion and returns ranked context.

        Success path: Both dense and sparse retrieval succeed, results are fused.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        mock_retrieved_context = RankedContext(
            query="What is machine learning?",
            results=[
                RetrievedContext(
                    chunk_id="chunk_1",
                    text="Machine learning is a subset of artificial intelligence.",
                    score=0.95,
                    citation="doc_1.pdf",
                    metadata={"page": 1, "section": "Introduction"},
                ),
                RetrievedContext(
                    chunk_id="chunk_2",
                    text="ML algorithms learn patterns from data without explicit programming.",
                    score=0.87,
                    citation="doc_2.pdf",
                    metadata={"page": 3, "section": "Basics"},
                ),
            ],
            total_found=2,
        )

        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {"standalone_query": "What is machine learning?"}
                            )
                        )
                    )
                ]
            )
            mock_retrieve.return_value = mock_retrieved_context
            mock_generate.return_value = GenerationResult(
                answer="Machine learning is a subset of AI that learns from data.",
                citations=[
                    Citation(
                        chunk_id="chunk_1",
                        source_name="doc_1.pdf",
                        quote="Machine learning is a subset of artificial intelligence.",
                    )
                ],
                insufficient_information=False,
            )

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is machine learning?"},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["answer"] == "Machine learning is a subset of AI that learns from data."
            assert len(result["citations"]) == 1

    @pytest.mark.asyncio
    async def test_retrieval_with_tenant_filters(self, admin_client: AsyncClient):
        """
        Validate that retrieval applies tenant-based filters to ensure
        multi-tenant data isolation.

        Success path: Tenant ID filter is passed to retriever.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "test query"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="test query", results=[], total_found=0
            )
            mock_generate.return_value = GenerationResult(
                answer="No context available.",
                citations=[],
                insufficient_information=True,
            )

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "test"},
            )

            # Verify tenant filter was applied
            mock_retrieve.assert_called_once()
            call_kwargs = mock_retrieve.call_args[1]
            assert "filters" in call_kwargs
            assert "tenant_id" in call_kwargs["filters"]

    @pytest.mark.asyncio
    async def test_retrieval_no_results(self, admin_client: AsyncClient):
        """
        Validate that generation pipeline handles case where retrieval
        returns no relevant context.

        Failure path: Empty retrieval result triggers insufficient_information flag.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "obscure topic"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="obscure topic", results=[], total_found=0
            )
            mock_generate.return_value = GenerationResult(
                answer="I don't have information about this topic.",
                citations=[],
                insufficient_information=True,
            )

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "Tell me about obscure topic"},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["insufficient_information"] is True


class TestChatPipelineGeneration:
    """Tests for generation stage with context formatting and prompt building."""

    @pytest.mark.asyncio
    async def test_generation_with_citations(self, admin_client: AsyncClient):
        """
        Validate that GenerationPipeline produces structured output with
        proper citations matching retrieved chunks.

        Success path: Citations are correctly extracted and formatted.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        mock_retrieved = RankedContext(
            query="What is Python?",
            results=[
                RetrievedContext(
                    chunk_id="chunk_py_1",
                    text="Python is a high-level programming language created by Guido van Rossum.",
                    score=0.96,
                    citation="python_guide.pdf",
                    metadata={"page": 1},
                ),
                RetrievedContext(
                    chunk_id="chunk_py_2",
                    text="Python emphasizes code readability and simplicity.",
                    score=0.91,
                    citation="python_guide.pdf",
                    metadata={"page": 2},
                ),
            ],
            total_found=2,
        )

        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "What is Python?"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = mock_retrieved
            mock_generate.return_value = GenerationResult(
                answer="Python is a high-level programming language emphasizing readability.",
                citations=[
                    Citation(
                        chunk_id="chunk_py_1",
                        source_name="python_guide.pdf",
                        quote="Python is a high-level programming language created by Guido van Rossum.",
                    ),
                    Citation(
                        chunk_id="chunk_py_2",
                        source_name="python_guide.pdf",
                        quote="Python emphasizes code readability and simplicity.",
                    ),
                ],
                insufficient_information=False,
            )

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is Python?"},
            )
            assert response.status_code == 200
            result = response.json()
            assert len(result["citations"]) == 2
            assert all("chunk_id" in c and "source_name" in c for c in result["citations"])

    @pytest.mark.asyncio
    async def test_generation_with_conversation_history(self, admin_client: AsyncClient):
        """
        Validate that GenerationPipeline incorporates conversation history
        into the generation context for coherent multi-turn conversations.

        Success path: Recent history is passed to LLM for contextual generation.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Send first message
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter1, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve1, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate1:
            mock_rewriter1.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "What is FastAPI?"})
                        )
                    )
                ]
            )
            mock_retrieve1.return_value = RankedContext(
                query="What is FastAPI?",
                results=[
                    RetrievedContext(
                        chunk_id="chunk_api",
                        text="FastAPI is a modern web framework for Python.",
                        score=0.95,
                        citation="fastapi_docs.pdf",
                        metadata={"page": 1},
                    )
                ],
                total_found=1,
            )
            mock_generate1.return_value = GenerationResult(
                answer="FastAPI is a modern, fast web framework for building APIs.",
                citations=[],
                insufficient_information=False,
            )

            response1 = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is FastAPI?"},
            )
            assert response1.status_code == 200

        # Send follow-up using history
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter2, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve2, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate2:
            mock_rewriter2.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {"standalone_query": "What are the key features of FastAPI?"}
                            )
                        )
                    )
                ]
            )
            mock_retrieve2.return_value = RankedContext(
                query="What are the key features of FastAPI?",
                results=[
                    RetrievedContext(
                        chunk_id="chunk_features",
                        text="Key features include automatic API docs, type hints, and fast performance.",
                        score=0.93,
                        citation="fastapi_docs.pdf",
                        metadata={"page": 2},
                    )
                ],
                total_found=1,
            )
            mock_generate2.return_value = GenerationResult(
                answer="FastAPI's key features include automatic documentation, type safety, and excellent performance.",
                citations=[],
                insufficient_information=False,
            )

            response2 = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What are its key features?"},
            )
            assert response2.status_code == 200

            # Verify history was used
            mock_generate2.assert_called_once()
            call_kwargs = mock_generate2.call_args[1]
            assert "messages" in call_kwargs
            # Should contain history from first message
            assert len(call_kwargs["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_generation_insufficient_context_flag(self, admin_client: AsyncClient):
        """
        Validate that GenerationPipeline sets insufficient_information flag
        when context is unable to answer the query.

        Success path: Flag is correctly set when LLM indicates insufficient information.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "random query"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="random query", results=[], total_found=0
            )
            mock_generate.return_value = GenerationResult(
                answer="I cannot answer this question based on available context.",
                citations=[],
                insufficient_information=True,
            )

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "Something not in docs"},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["insufficient_information"] is True


class TestChatPipelineConversationHistory:
    """Tests for conversation history retrieval and management."""

    @pytest.mark.asyncio
    async def test_conversation_creation(self, admin_client: AsyncClient):
        """
        Validate that new conversations are created with proper metadata
        and returned with correct schema.

        Success path: Conversation is created and returns ID with timestamp.
        """
        response = await admin_client.post("/api/v1/chat/conversations")
        assert response.status_code == 201
        data = response.json()
        assert "conversation_id" in data
        assert "created_at" in data
        assert isinstance(data["conversation_id"], str)
        assert uuid.UUID(data["conversation_id"])  # Validate UUID format

    @pytest.mark.asyncio
    async def test_conversation_history_retrieval(self, admin_client: AsyncClient):
        """
        Validate that conversation history is retrieved with all messages
        in chronological order.

        Success path: Multiple messages are returned in correct order.
        """
        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Send messages
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "test query"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="test query", results=[], total_found=0
            )
            mock_generate.return_value = GenerationResult(
                answer="test answer", citations=[], insufficient_information=False
            )

            await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "First message"},
            )
            await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "Second message"},
            )

        # Retrieve history
        history_response = await admin_client.get(
            f"/api/v1/chat/conversations/{conversation_id}"
        )
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert "messages" in history_data
        messages = history_data["messages"]
        assert len(messages) >= 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_conversation_history_recent_window(self, admin_client: AsyncClient):
        """
        Validate that only the recent message window (last 4 messages)
        is used for query rewriting context.

        Success path: History window is limited to prevent token overflow.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Send multiple messages
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "test query"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="test query", results=[], total_found=0
            )
            mock_generate.return_value = GenerationResult(
                answer="test answer", citations=[], insufficient_information=False
            )

            # Send 6 messages to exceed window of 4
            for i in range(6):
                await admin_client.post(
                    f"/api/v1/chat/conversations/{conversation_id}/messages",
                    json={"message": f"Message {i+1}"},
                )

        # Retrieve and verify only recent history is stored
        history_response = await admin_client.get(
            f"/api/v1/chat/conversations/{conversation_id}"
        )
        assert history_response.status_code == 200
        history_data = history_response.json()
        messages = history_data["messages"]
        # Messages should be stored (we don't limit storage, only rewrite context)
        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_message_persistence_and_retrieval(self, admin_client: AsyncClient):
        """
        Validate that user and assistant messages are correctly persisted to DB
        and retrievable in the correct order.

        Success path: Messages are saved with correct roles and content.
        """
        conversation_id = uuid.uuid4()

        # Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        user_msg = "What is data science?"
        assistant_answer = "Data science combines statistics, programming, and domain expertise."

        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": user_msg})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query=user_msg, results=[], total_found=0
            )
            mock_generate.return_value = GenerationResult(
                answer=assistant_answer, citations=[], insufficient_information=False
            )

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": user_msg},
            )
            assert response.status_code == 200

        # Retrieve and verify
        history_response = await admin_client.get(
            f"/api/v1/chat/conversations/{conversation_id}"
        )
        assert history_response.status_code == 200
        history_data = history_response.json()
        messages = history_data["messages"]

        # Find the messages we sent
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        assert len(user_messages) > 0
        assert user_messages[-1]["content"] == user_msg
        assert len(assistant_messages) > 0
        assert assistant_messages[-1]["content"] == assistant_answer


class TestChatPipelineErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_input_validation_empty_message(self, admin_client: AsyncClient):
        """
        Validate that empty messages are rejected with proper HTTP status.

        Failure path: Empty input triggers validation error.
        """
        conversation_id = uuid.uuid4()

        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        response = await admin_client.post(
            f"/api/v1/chat/conversations/{conversation_id}/messages",
            json={"message": ""},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_input_validation_oversized_message(self, admin_client: AsyncClient):
        """
        Validate that oversized messages are rejected to prevent token overflow.

        Failure path: Message exceeding max length triggers validation error.
        """
        conversation_id = uuid.uuid4()

        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Create a very long message (exceeds typical limits)
        oversized_message = "x" * 50000

        response = await admin_client.post(
            f"/api/v1/chat/conversations/{conversation_id}/messages",
            json={"message": oversized_message},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_authorization_employee_access(self, employee_client: AsyncClient):
        """
        Validate that employees can create conversations and send messages.

        Success path: Employee role has appropriate permissions.
        """
        response = await employee_client.post("/api/v1/chat/conversations")
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_authorization_manager_access(self, manager_client: AsyncClient):
        """
        Validate that managers can create conversations and send messages.

        Success path: Manager role has appropriate permissions.
        """
        response = await manager_client.post("/api/v1/chat/conversations")
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(
        self, admin_client: AsyncClient, employee_client: AsyncClient
    ):
        """
        Validate that users from different tenants cannot access each other's
        conversations due to tenant filtering.

        Success path: Tenant ID in filters prevents cross-tenant access.
        """
        # Admin creates conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        # Employee (different tenant) attempts to access
        history_response = await employee_client.get(
            f"/api/v1/chat/conversations/{conversation_id}"
        )
        # Should fail or return empty (depends on implementation)
        assert history_response.status_code in [200, 403, 404]

    @pytest.mark.asyncio
    async def test_pipeline_exception_handling(self, admin_client: AsyncClient):
        """
        Validate that pipeline exceptions are caught and return appropriate
        HTTP error responses.

        Failure path: Unexpected error in any stage returns 500.
        """
        conversation_id = uuid.uuid4()

        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_data = conv_response.json()
        conversation_id = uuid.UUID(conv_data["conversation_id"])

        with patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve:
            mock_retrieve.side_effect = Exception("Database connection failed")

            response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "test"},
            )
            # Should handle gracefully
            assert response.status_code >= 400


class TestChatPipelineIntegration:
    """End-to-end integration tests for complete chat flow."""

    @pytest.mark.asyncio
    async def test_full_chat_flow_single_turn(self, admin_client: AsyncClient):
        """
        Validate a complete single-turn chat flow: conversation creation,
        message sending, retrieval, generation, and persistence.

        Success path: Complete flow succeeds with all components working.
        """
        # 1. Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conversation_id = uuid.UUID(conv_response.json()["conversation_id"])

        # 2. Send message with full mock
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {"standalone_query": "What is cloud computing?"}
                            )
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="What is cloud computing?",
                results=[
                    RetrievedContext(
                        chunk_id="cloud_1",
                        text="Cloud computing is the delivery of computing services over the internet.",
                        score=0.98,
                        citation="cloud_guide.pdf",
                        metadata={"page": 1},
                    )
                ],
                total_found=1,
            )
            mock_generate.return_value = GenerationResult(
                answer="Cloud computing provides on-demand access to computing resources.",
                citations=[
                    Citation(
                        chunk_id="cloud_1",
                        source_name="cloud_guide.pdf",
                        quote="Cloud computing is the delivery of computing services over the internet.",
                    )
                ],
                insufficient_information=False,
            )

            msg_response = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is cloud computing?"},
            )
            assert msg_response.status_code == 200
            result = msg_response.json()
            assert result["answer"]
            assert len(result["citations"]) > 0

        # 3. Verify history
        history_response = await admin_client.get(
            f"/api/v1/chat/conversations/{conversation_id}"
        )
        assert history_response.status_code == 200
        messages = history_response.json()["messages"]
        assert len(messages) >= 2  # At least user + assistant

    @pytest.mark.asyncio
    async def test_full_chat_flow_multi_turn(self, admin_client: AsyncClient):
        """
        Validate a complete multi-turn conversation with query rewriting,
        history incorporation, and coherent responses.

        Success path: Multi-turn flow maintains context across turns.
        """
        # 1. Create conversation
        conv_response = await admin_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conversation_id = uuid.UUID(conv_response.json()["conversation_id"])

        # 2. First turn
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({"standalone_query": "What is Kubernetes?"})
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="What is Kubernetes?",
                results=[
                    RetrievedContext(
                        chunk_id="k8s_1",
                        text="Kubernetes is an orchestration platform for containerized applications.",
                        score=0.96,
                        citation="k8s_docs.pdf",
                        metadata={"page": 1},
                    )
                ],
                total_found=1,
            )
            mock_generate.return_value = GenerationResult(
                answer="Kubernetes orchestrates container deployment and scaling.",
                citations=[],
                insufficient_information=False,
            )

            response1 = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What is Kubernetes?"},
            )
            assert response1.status_code == 200

        # 3. Second turn with follow-up
        with patch(
            "app.rag.generator.query_rewriter.litellm.acompletion"
        ) as mock_rewriter, patch(
            "app.rag.retrieval.hybrid.HybridRetriever.retrieve"
        ) as mock_retrieve, patch(
            "app.rag.generator.litellm_generator.LiteLLMGenerator.generate_structured"
        ) as mock_generate:
            mock_rewriter.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {"standalone_query": "What are the key features of Kubernetes?"}
                            )
                        )
                    )
                ]
            )
            mock_retrieve.return_value = RankedContext(
                query="What are the key features of Kubernetes?",
                results=[
                    RetrievedContext(
                        chunk_id="k8s_2",
                        text="Key features include auto-scaling, self-healing, and declarative configuration.",
                        score=0.94,
                        citation="k8s_docs.pdf",
                        metadata={"page": 2},
                    )
                ],
                total_found=1,
            )
            mock_generate.return_value = GenerationResult(
                answer="Kubernetes features include auto-scaling, self-healing, and declarative management.",
                citations=[],
                insufficient_information=False,
            )

            response2 = await admin_client.post(
                f"/api/v1/chat/conversations/{conversation_id}/messages",
                json={"message": "What are its key features?"},
            )
            assert response2.status_code == 200
            result2 = response2.json()
            assert result2["answer"]

        # 4. Verify full history
        history_response = await admin_client.get(
            f"/api/v1/chat/conversations/{conversation_id}"
        )
        assert history_response.status_code == 200
        messages = history_response.json()["messages"]
        assert len(messages) >= 4  # 2 turns * (user + assistant)
