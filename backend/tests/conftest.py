import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

# Mock celery task BEFORE importing app to prevent Redis connection during tests
import sys
from unittest.mock import MagicMock

class MockCeleryTask:
    def __init__(self):
        self.delay = MagicMock(return_value=None)
        self.__name__ = "process_document_task"
        self.__module__ = "app.tasks.document_tasks"
        self.__qualname__ = "process_document_task"

mock_task = MockCeleryTask()

mock_module = MagicMock()
mock_module.__name__ = "app.tasks.document_tasks"
mock_module.__file__ = "app/tasks/document_tasks.py"
mock_module.process_document_task = mock_task

sys.modules['app.tasks.document_tasks'] = mock_module

from app.core.rate_limiter import reset_rate_limiter
from app.core.security import create_access_token
from app.main import app
from app.models.db import ConversationModel, MessageModel, RefreshTokenModel, UserModel
from app.models.user import Role


@pytest.fixture(autouse=True)
async def _reset_rate_limiter():
    """Reset rate limiter state before each test."""
    await reset_rate_limiter()
    yield
    await reset_rate_limiter()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


MOCK_USERS = {}  # user_id -> {"role": Role, "tenant_id": str}
MOCK_CONVERSATIONS = {}
MOCK_MESSAGES = []  # List of MessageModel objects
MOCK_REFRESH_TOKENS = {}  # jti -> RefreshTokenModel


class MockResult:
    def __init__(self, role, user_exists=True, user_id=None, tenant_id=None):
        self.role = role
        self.user_exists = user_exists
        self.user_id = user_id
        self.tenant_id = tenant_id

    def _get_mock_user(self):
        import uuid

        from app.models.db import UserModel

        if not self.user_exists:
            return None

        return UserModel(
            id=uuid.UUID(self.user_id) if self.user_id else uuid.uuid4(),
            email="mock@example.com",
            full_name="Mock User",
            role=self.role,
            tenant_id=uuid.UUID(self.tenant_id) if self.tenant_id else uuid.uuid4(),
            is_active=True,
        )

    def _get_mock_conversation(self, conversation_id):
        """Get a mock conversation if it exists."""
        return MOCK_CONVERSATIONS.get(str(conversation_id))

    def scalars(self):
        parent = self

        class MockScalars:
            def all(self):
                return []

            def first(self):
                return parent._get_mock_user()

        return MockScalars()

    def scalar(self):
        """Return a single scalar result."""
        return None

    def scalar_one_or_none(self):
        return self._get_mock_user()

    def all(self):
        return []


class MockDbSession:
    def __init__(self, raise_error=None):
        self.raise_error = raise_error

    async def execute(self, stmt, *args, **kwargs):
        if self.raise_error:
            raise self.raise_error
        from app.models.user import Role

        role = Role.ADMINISTRATOR
        user_exists = True
        tenant_id = None
        user_id = None
        try:
            # Check if this is a conversation query
            stmt_str = str(stmt)
            stmt_lower = stmt_str.lower()
            if "conversations" in stmt_lower and "where" in stmt_lower:
                # Extract conversation_id, tenant_id, user_id from the query params
                params = stmt.compile().params
                conv_id = None
                req_tenant_id = None
                req_user_id = None
                for k, v in params.items():
                    if "conversation" in k.lower() or ("id" in k.lower() and "tenant" not in k.lower() and "user" not in k.lower()):
                        conv_id = str(v)
                    if "tenant" in k.lower():
                        req_tenant_id = str(v)
                    if "user" in k.lower():
                        req_user_id = str(v)

                if conv_id and conv_id in MOCK_CONVERSATIONS:
                    conversation = MOCK_CONVERSATIONS[conv_id]
                    # Verify tenant_id and user_id match
                    if (req_tenant_id is None or str(conversation.tenant_id) == req_tenant_id) and \
                       (req_user_id is None or str(conversation.user_id) == req_user_id):
                        mock_conv_result = MagicMock()
                        mock_conv_result.scalar_one_or_none.return_value = conversation
                        return mock_conv_result

                # Conversation not found or doesn't match tenant/user
                mock_conv_result = MagicMock()
                mock_conv_result.scalar_one_or_none.return_value = None
                return mock_conv_result

            # Check if this is a messages query (for fetching conversation history)
            if "messages" in stmt_lower and "where" in stmt_lower:
                # Extract conversation_id from params
                params = stmt.compile().params
                conv_id = None
                for k, v in params.items():
                    if "conversation" in k.lower():
                        conv_id = str(v)
                
                # Filter messages for this conversation
                conv_messages = [m for m in MOCK_MESSAGES if str(m.conversation_id) == conv_id]
                # Sort by created_at
                conv_messages.sort(key=lambda m: m.created_at)
                
                mock_msg_result = MagicMock()
                mock_msg_result.scalars.return_value.all.return_value = conv_messages
                return mock_msg_result

            # User query - extract user_id and tenant_id from params
            params = stmt.compile().params
            for k, v in params.items():
                key_lower = k.lower()
                if "sub" in key_lower or "user" in key_lower or ("id" in key_lower and "conversation" not in key_lower and "tenant" not in key_lower):
                    user_id = str(v)
                if "tenant" in key_lower:
                    tenant_id = str(v)

            # Get tenant_id from MOCK_USERS if not in query params
            if user_id and user_id in MOCK_USERS and not tenant_id:
                tenant_id = MOCK_USERS[user_id].get("tenant_id")

            if user_id and user_id in MOCK_USERS:
                role = MOCK_USERS[user_id]["role"]
            else:
                # User doesn't exist in our mock database
                user_exists = False
        except Exception as e:
            import traceback
            print(f"EXCEPTION in MockDbSession.execute: {e}")
            traceback.print_exc()
            pass
        return MockResult(role, user_exists=user_exists, user_id=user_id, tenant_id=tenant_id)

    def add(self, obj):
        if self.raise_error:
            raise self.raise_error
        import uuid
        from datetime import datetime

        if not getattr(obj, "id", None):
            obj.id = uuid.uuid4()
        if hasattr(obj, "created_at") and not getattr(obj, "created_at", None):
            obj.created_at = datetime.now(timezone.utc)
        if hasattr(obj, "updated_at") and not getattr(obj, "updated_at", None):
            obj.updated_at = datetime.now(timezone.utc)

        # Store conversations in mock storage
        if isinstance(obj, ConversationModel):
            MOCK_CONVERSATIONS[str(obj.id)] = obj
        # Store messages in mock storage
        if isinstance(obj, MessageModel):
            MOCK_MESSAGES.append(obj)
        # Store refresh tokens in mock storage
        if isinstance(obj, RefreshTokenModel):
            MOCK_REFRESH_TOKENS[str(obj.jti)] = obj

    def add_all(self, objs):
        for obj in objs:
            self.add(obj)

    async def commit(self):
        if self.raise_error:
            raise self.raise_error

    async def refresh(self, obj):
        pass

    async def delete(self, obj):
        pass

    async def rollback(self):
        pass


import redis.asyncio as redis_async  # noqa: E402

from app.db.session import get_db_session  # noqa: E402


class MockRedis:
    def __init__(self):
        self.data = {}

    async def ping(self):
        return True

    async def aclose(self):
        pass

    async def close(self):
        pass

    async def script_load(self, script):
        return "mock_sha"

    async def evalsha(self, sha, num_keys, key, limit, window_seconds, now):
        if key not in self.data:
            self.data[key] = {"tokens": limit, "last_refill": now}

        bucket = self.data[key]
        time_passed = now - bucket["last_refill"]
        refill_rate = limit / window_seconds
        tokens_to_add = int(time_passed * refill_rate)

        if tokens_to_add > 0:
            bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] += tokens_to_add / refill_rate

        reset_at = bucket["last_refill"] + window_seconds

        if bucket["tokens"] > 0:
            bucket["tokens"] -= 1
            return [bucket["tokens"], reset_at]
        else:
            return [-1, reset_at]

    async def delete(self, key):
        if key in self.data:
            del self.data[key]


async def mock_from_url(*args, **kwargs):
    return MockRedis()


# Patch Redis
redis_async.from_url = mock_from_url


import litellm  # noqa: E402


class MockLiteLLMResponse:
    def __init__(self):
        import json

        self.data = [{"embedding": [0.1] * 1536}]
        mock_json_content = json.dumps({"answer": "Mocked LLM Response", "citations": []})
        self.choices = [
            type("Choice", (), {"message": type("Message", (), {"content": mock_json_content})()})
        ]


def mock_embedding(*args, **kwargs):  # noqa: ARG001
    return MockLiteLLMResponse()


async def mock_acompletion(*args, **kwargs):  # noqa: ARG001
    return MockLiteLLMResponse()


litellm.embedding = mock_embedding
litellm.acompletion = mock_acompletion

app.dependency_overrides[get_db_session] = lambda: MockDbSession()

# Initialize app state for reranker to prevent AttributeError in tests
app.state.reranker = MagicMock()

def _get_auth_headers(role) -> dict[str, str]:
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    MOCK_USERS[user_id] = {"role": role, "tenant_id": tenant_id}
    token = create_access_token(
        {
            "sub": user_id,
            "email": f"{role.lower()}@example.com",
            "role": role,
            "tenant_id": tenant_id,
        }
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_client() -> AsyncGenerator[AsyncClient, None]:
    headers = _get_auth_headers(Role.ADMINISTRATOR)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=headers
    ) as ac:
        yield ac


@pytest.fixture
async def manager_client() -> AsyncGenerator[AsyncClient, None]:
    headers = _get_auth_headers(Role.MANAGER)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=headers
    ) as ac:
        yield ac


@pytest.fixture
async def employee_client() -> AsyncGenerator[AsyncClient, None]:
    headers = _get_auth_headers(Role.EMPLOYEE)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=headers
    ) as ac:
        yield ac