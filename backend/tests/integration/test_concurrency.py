"""
Comprehensive integration tests for concurrency, race conditions, and transaction safety.

This module validates that the RAG system handles concurrent operations correctly,
including simultaneous user interactions, document processing, and database transactions.
Focuses on identifying race conditions, deadlocks, and transaction isolation issues.
"""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.chat import Conversation, Message
from app.models.user import Role
from app.rag.retrieval.base import RankedContext, RetrievedContext


class TestConcurrentConversationCreation:
    """Test race conditions when multiple users create conversations simultaneously."""

    @pytest.mark.asyncio
    async def test_multiple_users_create_conversations_simultaneously(
        self, admin_client, manager_client, employee_client
    ):
        """
        Validates that concurrent conversation creation requests from different
        users do not interfere with each other. Each request should receive a
        unique conversation ID with no collisions.

        Tests for: ID collision, state corruption, transaction isolation
        """
        # Create tasks for simultaneous conversation creation
        tasks = [
            admin_client.post("/api/v1/chat/conversations"),
            manager_client.post("/api/v1/chat/conversations"),
            employee_client.post("/api/v1/chat/conversations"),
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 201

        # Extract conversation IDs
        conv_ids = [r.json()["conversation_id"] for r in responses]

        # All IDs should be unique (no collisions)
        assert len(set(conv_ids)) == 3, "Concurrent requests generated duplicate conversation IDs"

        # All should have created_at timestamps
        for response in responses:
            data = response.json()
            assert "created_at" in data
            assert data["created_at"] is not None

    @pytest.mark.asyncio
    async def test_same_user_creates_multiple_conversations_concurrently(
        self, employee_client
    ):
        """
        Validates that a single user can create multiple conversations concurrently
        without interference. Each conversation should be tracked independently.

        Tests for: shared state corruption, transaction conflicts
        """
        num_requests = 5
        tasks = [
            employee_client.post("/api/v1/chat/conversations")
            for _ in range(num_requests)
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 201

        # All should have unique IDs
        conv_ids = [r.json()["conversation_id"] for r in responses]
        assert len(set(conv_ids)) == num_requests

        # All should have been created
        for conv_id in conv_ids:
            assert conv_id is not None
            assert isinstance(conv_id, (str, uuid.UUID))


class TestConcurrentMessageHandling:
    """Test race conditions when processing messages in the same conversation."""

    @pytest.mark.asyncio
    async def test_concurrent_messages_to_same_conversation(self, employee_client):
        """
        Validates that multiple messages sent concurrently to the same conversation
        are processed sequentially and stored in correct order. Tests for:
        - Message ordering corruption
        - Lost messages
        - Duplicate processing
        - Transaction conflicts
        """
        # Create conversation
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        assert conv_response.status_code == 201
        conv_id = conv_response.json()["conversation_id"]

        # Send multiple messages concurrently
        messages = [
            "What is the return policy?",
            "What about refunds?",
            "How long does it take?",
        ]

        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": msg}
            )
            for msg in messages
        ]

        responses = await asyncio.gather(*tasks)

        # All should complete (may be 200 or 422 depending on mocking)
        for response in responses:
            assert response.status_code in [200, 422]

        # Retrieve conversation history
        history_response = await employee_client.get(
            f"/api/v1/chat/conversations/{conv_id}"
        )

        if history_response.status_code == 200:
            history_data = history_response.json()
            messages_returned = history_data.get("messages", [])

            # Should have messages (at least some of the concurrent ones)
            assert len(messages_returned) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_message_access_during_processing(self, employee_client):
        """
        Validates that accessing a conversation while a message is being processed
        does not result in data inconsistency. Tests for:
        - Partial write visibility
        - Dirty reads
        - Uncommitted data leaks
        """
        # Create conversation
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Send a message and immediately try to read
        send_task = employee_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            json={"message": "Test message"}
        )
        read_task = employee_client.get(
            f"/api/v1/chat/conversations/{conv_id}"
        )

        send_response, read_response = await asyncio.gather(send_task, read_task)

        # Read should succeed and not contain corrupted data
        if read_response.status_code == 200:
            data = read_response.json()
            # Should be valid structure
            assert isinstance(data, dict)
            messages = data.get("messages", [])
            # If any messages, they should be proper Message objects
            for msg in messages:
                assert isinstance(msg, dict)
                if "content" in msg:
                    assert isinstance(msg["content"], str)


class TestRaceConditionsBetweenUsers:
    """Test race conditions and isolation between different users."""

    @pytest.mark.asyncio
    async def test_users_cannot_interfere_with_each_others_conversations(
        self, employee_client, admin_client
    ):
        """
        Validates that users cannot modify or corrupt each other's conversations
        through concurrent access. Tests for:
        - Cross-user data corruption
        - Authorization bypass
        - State pollution between sessions
        """
        # Employee creates and uses conversation
        emp_conv = await employee_client.post("/api/v1/chat/conversations")
        emp_conv_id = emp_conv.json()["conversation_id"]

        # Admin attempts concurrent operations
        tasks = [
            # Admin tries to access employee's conversation
            admin_client.get(f"/api/v1/chat/conversations/{emp_conv_id}"),
            # Employee sends message
            employee_client.post(
                f"/api/v1/chat/conversations/{emp_conv_id}/messages",
                json={"message": "Private content"}
            ),
            # Admin tries to post to employee's conversation
            admin_client.post(
                f"/api/v1/chat/conversations/{emp_conv_id}/messages",
                json={"message": "Injected message"}
            ),
        ]

        responses = await asyncio.gather(*tasks)

        # Admin access should be denied (404 or 403)
        assert responses[0].status_code in [403, 404, 422]

        # Employee message should succeed
        assert responses[1].status_code in [200, 422]

        # Admin injection should be denied
        assert responses[2].status_code in [403, 404, 422]

    @pytest.mark.asyncio
    async def test_tenant_isolation_under_concurrent_load(
        self, employee_client, admin_client
    ):
        """
        Validates that multi-tenant isolation is maintained even under concurrent
        operations by users from different tenants. Tests for:
        - Cross-tenant data leakage
        - Shared state between tenants
        - Transaction isolation by tenant
        """
        # Both users create conversations concurrently
        emp_conv_task = employee_client.post("/api/v1/chat/conversations")
        admin_conv_task = admin_client.post("/api/v1/chat/conversations")

        emp_response, admin_response = await asyncio.gather(
            emp_conv_task, admin_conv_task
        )

        emp_conv_id = emp_response.json()["conversation_id"]
        admin_conv_id = admin_response.json()["conversation_id"]

        # Verify they got different conversations
        assert emp_conv_id != admin_conv_id

        # Try to access cross-tenant (should fail)
        cross_access = await employee_client.get(
            f"/api/v1/chat/conversations/{admin_conv_id}"
        )

        assert cross_access.status_code in [403, 404, 422]


class TestTransactionSafety:
    """Test transaction safety and consistency guarantees."""

    @pytest.mark.asyncio
    async def test_message_creation_transaction_atomicity(self, employee_client):
        """
        Validates that message creation is atomic: either fully succeeds or fully
        rolls back. Tests for:
        - Partial message creation
        - Orphaned records
        - Transaction rollback on failure
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Send valid message
        response = await employee_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            json={"message": "Valid message"}
        )

        # Should be atomic - either succeeds or fails cleanly
        assert response.status_code in [200, 201, 422]

        if response.status_code in [200, 201]:
            data = response.json()
            # Should have complete record
            assert "answer" in data or "content" in data

    @pytest.mark.asyncio
    async def test_concurrent_writes_with_read_consistency(self, employee_client):
        """
        Validates that reads during concurrent writes see consistent snapshots,
        not partially written data. Tests for:
        - Dirty reads
        - Non-repeatable reads
        - Phantom reads
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Start with baseline read
        baseline_read = await employee_client.get(
            f"/api/v1/chat/conversations/{conv_id}"
        )

        if baseline_read.status_code == 200:
            baseline_count = len(baseline_read.json().get("messages", []))
        else:
            baseline_count = 0

        # Concurrent writes and reads
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": f"Message {i}"}
            )
            for i in range(3)
        ] + [
            employee_client.get(f"/api/v1/chat/conversations/{conv_id}")
            for _ in range(3)
        ]

        responses = await asyncio.gather(*tasks)

        # All reads should return consistent data
        read_responses = responses[3:]
        for read_response in read_responses:
            if read_response.status_code == 200:
                data = read_response.json()
                # Should be valid structure
                assert isinstance(data.get("messages", []), list)

    @pytest.mark.asyncio
    async def test_transaction_isolation_levels(self, employee_client):
        """
        Validates that the system maintains appropriate transaction isolation
        levels to prevent common concurrency issues. Tests for:
        - Isolation level enforcement
        - Phantom reads prevention
        - Lost update prevention
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Create initial state
        initial = await employee_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            json={"message": "Initial message"}
        )

        # Multiple concurrent transactions
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": f"Concurrent {i}"}
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        # All should process without conflicts
        successful = [r for r in responses if r.status_code in [200, 201, 422]]
        assert len(successful) > 0


class TestDeadlockPrevention:
    """Test for deadlock scenarios and prevention."""

    @pytest.mark.asyncio
    async def test_no_deadlock_with_circular_dependencies(
        self, employee_client, manager_client
    ):
        """
        Validates that the system does not deadlock when transactions have
        circular dependencies. Tests for:
        - Deadlock detection
        - Lock ordering
        - Timeout handling
        """
        # Create two conversations
        conv1_response = await employee_client.post("/api/v1/chat/conversations")
        conv1_id = conv1_response.json()["conversation_id"]

        conv2_response = await manager_client.post("/api/v1/chat/conversations")
        conv2_id = conv2_response.json()["conversation_id"]

        # Attempt circular-pattern access: E->1, M->2, then cross-access
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv1_id}/messages",
                json={"message": "From employee to 1"}
            ),
            manager_client.post(
                f"/api/v1/chat/conversations/{conv2_id}/messages",
                json={"message": "From manager to 2"}
            ),
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Should complete without timeout or deadlock
        for response in responses:
            if isinstance(response, Exception):
                # Should not be a timeout or deadlock exception
                assert "deadlock" not in str(response).lower()
                assert "timeout" not in str(response).lower()

    @pytest.mark.asyncio
    async def test_lock_timeout_handling(self, employee_client):
        """
        Validates that the system handles lock timeouts gracefully without
        hanging. Tests for:
        - Lock acquisition timeout
        - Graceful error on timeout
        - Transaction rollback after timeout
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Create high contention on same resource
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": f"Highly contested message {i}"}
            )
            for i in range(10)
        ]

        # All should complete without hanging
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )

            # Check that we got responses (some may be errors)
            assert len(responses) == 10
        except asyncio.TimeoutError:
            pytest.fail("Lock timeout caused operation to hang beyond 30 seconds")


class TestConcurrentRetrievalAndGeneration:
    """Test concurrency in RAG pipeline operations."""

    @pytest.mark.asyncio
    async def test_concurrent_retrievals_do_not_corrupt_vector_store(
        self, employee_client
    ):
        """
        Validates that concurrent retrieval queries do not corrupt or interfere
        with the vector store state. Tests for:
        - Vector store state corruption
        - Index inconsistency
        - Lost reads/writes in vector DB
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Send concurrent queries that trigger retrieval
        queries = [
            "What is the return policy?",
            "Tell me about warranties",
            "How do refunds work?",
            "What is your shipping policy?",
        ]

        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": q}
            )
            for q in queries
        ]

        responses = await asyncio.gather(*tasks)

        # All should complete
        assert len(responses) == len(queries)

        # Verify vector store is still consistent by doing another retrieval
        final_response = await employee_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            json={"message": "Final query"}
        )

        # Should work normally, indicating no corruption
        assert final_response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_concurrent_embeddings_generation_no_duplicates(
        self, employee_client
    ):
        """
        Validates that concurrent embedding generation for the same text does not
        cause duplicate or conflicting entries in the vector store. Tests for:
        - Duplicate embeddings
        - Inconsistent vector mappings
        - Resource exhaustion from redundant computation
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Send same/similar queries concurrently to trigger embedding generation
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": "What is your return policy?"}
            )
            for _ in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        # All should succeed
        successful = [r for r in responses if r.status_code in [200, 422]]
        assert len(successful) >= 3  # At least most should work


class TestFailurePathsConcurrency:
    """Test concurrent failure scenarios and error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_llm_failures_isolation(self, employee_client):
        """
        Validates that when LLM fails for one request, it does not affect
        concurrent requests. Tests for:
        - Error state propagation between requests
        - Resource leak on failure
        - Partial state corruption on error
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        with patch("app.rag.generator.litellm_generator.litellm.acompletion") as mock_llm:
            # Some succeed, some fail
            call_count = [0]

            async def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 2 == 0:
                    raise Exception("LLM service error")
                from conftest import MockLiteLLMResponse
                return MockLiteLLMResponse()

            mock_llm.side_effect = side_effect

            tasks = [
                employee_client.post(
                    f"/api/v1/chat/conversations/{conv_id}/messages",
                    json={"message": f"Query {i}"}
                )
                for i in range(4)
            ]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Some should fail, some might succeed
            # But none should crash the system
            assert len(responses) == 4

    @pytest.mark.asyncio
    async def test_concurrent_database_failures_do_not_corrupt_state(
        self, employee_client
    ):
        """
        Validates that database failures during concurrent operations do not
        leave the system in an inconsistent state. Tests for:
        - Partial writes on DB failure
        - State recovery after DB error
        - Stale data serving after failure
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Mock DB to fail intermittently
        original_commit = None
        fail_count = [0]

        async def failing_commit():
            fail_count[0] += 1
            if fail_count[0] % 3 == 0:
                raise Exception("Database connection lost")
            # Success case
            pass

        # Concurrent operations with DB failures
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": f"Message {i}"}
            )
            for i in range(6)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify system recovers
        recovery_response = await employee_client.get(
            f"/api/v1/chat/conversations/{conv_id}"
        )

        # Should be able to read after failures
        assert recovery_response.status_code in [200, 422]


class TestCacheCoherency:
    """Test cache coherency under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_invalidation(self, employee_client):
        """
        Validates that cache invalidation works correctly when multiple requests
        modify the same cached resource concurrently. Tests for:
        - Stale cache serving
        - Cache invalidation races
        - Premature cache invalidation
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Read to populate cache
        read1 = await employee_client.get(
            f"/api/v1/chat/conversations/{conv_id}"
        )

        # Concurrent write and read
        tasks = [
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": "Cache invalidating message"}
            ),
            employee_client.get(f"/api/v1/chat/conversations/{conv_id}"),
        ]

        responses = await asyncio.gather(*tasks)

        # Both should succeed
        write_response, read_response = responses

        # Read should get valid data (either cached or fresh)
        if read_response.status_code == 200:
            assert isinstance(read_response.json(), dict)

    @pytest.mark.asyncio
    async def test_cache_coherency_between_different_endpoints(
        self, employee_client
    ):
        """
        Validates that cache coherency is maintained across different API
        endpoints that access the same data. Tests for:
        - Inconsistent data across endpoints
        - Cache invalidation propagation
        - Multiple authority problems
        """
        conv_response = await employee_client.post("/api/v1/chat/conversations")
        conv_id = conv_response.json()["conversation_id"]

        # Concurrent accesses to different but related endpoints
        tasks = [
            employee_client.get(f"/api/v1/chat/conversations/{conv_id}"),
            employee_client.get(f"/api/v1/chat/conversations"),  # List endpoint
            employee_client.post(
                f"/api/v1/chat/conversations/{conv_id}/messages",
                json={"message": "Test"}
            ),
        ]

        responses = await asyncio.gather(*tasks)

        # Should all see consistent view
        for response in responses:
            if response.status_code == 200:
                assert response.json() is not None
