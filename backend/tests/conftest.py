import uuid
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.main import app
from app.models.user import Role


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


MOCK_USERS = {}


class MockResult:
    def __init__(self, role, user_exists=True):
        self.role = role
        self.user_exists = user_exists

    def _get_mock_user(self):
        import uuid

        from app.models.db import UserModel

        if not self.user_exists:
            return None

        return UserModel(
            id=uuid.uuid4(),
            email="mock@example.com",
            full_name="Mock User",
            role=self.role,
            tenant_id=uuid.uuid4(),
            is_active=True,
        )

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
        try:
            params = stmt.compile().params
            for _k, v in params.items():
                user_id = str(v)
                if user_id in MOCK_USERS:
                    role = MOCK_USERS[user_id]
                else:
                    # User doesn't exist in our mock database
                    user_exists = False
        except Exception:
            pass
        return MockResult(role, user_exists=user_exists)

    def add(self, obj):
        if self.raise_error:
            raise self.raise_error
        import uuid
        from datetime import datetime

        if not getattr(obj, "id", None):
            obj.id = uuid.uuid4()
        if hasattr(obj, "created_at") and not getattr(obj, "created_at", None):
            obj.created_at = datetime.utcnow()
        if hasattr(obj, "updated_at") and not getattr(obj, "updated_at", None):
            obj.updated_at = datetime.utcnow()

    def add_all(self, objs):
        pass

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
    async def ping(self):
        return True

    async def aclose(self):
        pass


# Patch Redis
redis_async.from_url = lambda *args, **kwargs: MockRedis()


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

# Mock celery task to prevent Redis connection during tests
import sys
from unittest.mock import MagicMock
mock_task = MagicMock()
mock_task.delay = MagicMock(return_value=None)
sys.modules['app.tasks.document_tasks'] = MagicMock(process_document_task=mock_task)

# Initialize app state for reranker to prevent AttributeError in tests
app.state.reranker = MagicMock()

def _get_auth_headers(role) -> dict[str, str]:
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    MOCK_USERS[user_id] = role
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
