from uuid import uuid4

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from app.core.config import settings
from app.core.security import ALGORITHM, get_password_hash
from app.db.session import get_db_session
from app.main import create_application
from app.models.db import UserModel


@pytest.fixture
async def app_with_db():
    # Force rate limiter to use memory backend for tests
    with patch("app.core.config.settings.RATE_LIMIT_BACKEND", "memory"):
        with patch("app.core.config.settings.RATE_LIMIT_FAIL_MODE", "open"):
            app = create_application()

    # We will override the db session with a mock that returns a user
    class MockResult:
        def __init__(self, user):
            self.user = user

        def scalar_one_or_none(self):
            return self.user

        def scalars(self):
            class MockScalars:
                def __init__(self, user):
                    self.user = user

                def first(self):
                    return self.user

            return MockScalars(self.user)

    class MockDbSession:
        def __init__(self, user):
            self.user = user

        async def execute(self, stmt):
            # For login and me
            return MockResult(self.user)

        def add(self, obj):
            pass

        async def commit(self):
            pass

    # Create seeded user
    seeded_user = UserModel(
        id=uuid4(),
        email="test@example.com",
        hashed_password=get_password_hash("correct_password"),
        full_name="Test User",
        role="Employee",
        tenant_id=uuid4(),
        is_active=True,
    )

    app.dependency_overrides[get_db_session] = lambda: MockDbSession(seeded_user)
    yield app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_auth_verification(app_with_db):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_db), base_url="http://test"
    ) as client:
        # [x] Attempt login via POST /api/v1/auth/login with wrong password
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "wrong_password"},
        )
        assert response.status_code == 401

        # [x] Attempt login with correct seeded password
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "correct_password"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

        # [x] Decode JWT - verify claims include sub, email, role, and tenant_id
        token = data["access_token"]
        secret_key = settings.JWT_SECRET.get_secret_value()
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM], audience=settings.JWT_AUDIENCE, issuer=settings.JWT_ISSUER)

        assert "sub" in payload
        assert "email" in payload
        assert "role" in payload
        assert "tenant_id" in payload
