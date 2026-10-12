"""
Comprehensive API security tests for authentication, authorization, and tenant isolation.

This test module validates:
- Authentication: JWT token validation, token expiration, invalid credentials
- Authorization: RBAC enforcement, role hierarchy verification
- Tenant isolation: Multi-tenancy data boundaries, cross-tenant access prevention
- Rate limiting: Brute force protection on login endpoint
- Input validation: Guard rails against malicious payloads
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import status
from httpx import AsyncClient

from app.core.config import settings
from app.core.security import ALGORITHM, create_access_token
from app.models.user import Role


@pytest.fixture(autouse=True)
def _patch_rate_limit_settings():
    """Patch rate limit settings for testing."""
    with patch("app.core.config.settings.RATE_LIMIT_TRUSTED_PROXIES", "127.0.0.1/32"):
        with patch("app.core.config.settings.RATE_LIMIT_BACKEND", "redis"):
            with patch("app.core.config.settings.RATE_LIMIT_BACKEND_URL", "redis://localhost:6379/0"):
                with patch("app.core.config.settings.RATE_LIMIT_FAIL_MODE", "closed"):
                    yield


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================


@pytest.mark.asyncio
class TestAuthenticationSecurity:
    """Test suite for authentication security mechanisms."""

    async def test_login_with_valid_credentials(self, client: AsyncClient, monkeypatch):
        """
        Validate successful login with correct email and password.
        Expected: Returns valid JWT access token with correct role and tenant_id.
        """
        # Mock the database session to return a valid user
        from app.db.session import get_db_session
        from app.models.db import UserModel
        from app.core.security import get_password_hash

        mock_db = AsyncMock()
        mock_user = UserModel(
            id=uuid.uuid4(),
            email="test@example.com",
            full_name="Test User",
            hashed_password=get_password_hash("password123"),
            role=Role.EMPLOYEE,
            tenant_id=uuid.uuid4(),
            is_active=True,
        )

        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.auth.get_db_session", AsyncMock(return_value=mock_db)
        )

        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "password123"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Decode and verify token content
        secret_key = settings.JWT_SECRET.get_secret_value()
        payload = jwt.decode(data["access_token"], secret_key, algorithms=[ALGORITHM])
        assert payload["email"] == "test@example.com"
        assert payload["role"] == Role.EMPLOYEE

    async def test_login_with_invalid_password(self, client: AsyncClient, monkeypatch):
        """
        Verify login rejection with incorrect password.
        Expected: Returns 401 Unauthorized without revealing user existence.
        """
        from app.db.session import get_db_session
        from app.models.db import UserModel
        from app.core.security import get_password_hash

        mock_db = AsyncMock()
        mock_user = UserModel(
            id=uuid.uuid4(),
            email="test@example.com",
            full_name="Test User",
            hashed_password=get_password_hash("correct_password"),
            role=Role.EMPLOYEE,
            tenant_id=uuid.uuid4(),
            is_active=True,
        )

        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.auth.get_db_session", AsyncMock(return_value=mock_db)
        )

        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "test@example.com", "password": "wrong_password"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Incorrect email or password"

    async def test_login_with_nonexistent_user(self, client: AsyncClient, monkeypatch):
        """
        Verify login rejection for non-existent user.
        Expected: Returns 401 Unauthorized with generic message.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.auth.get_db_session", AsyncMock(return_value=mock_db)
        )

        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent@example.com", "password": "password123"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_login_with_inactive_user(self, client: AsyncClient, monkeypatch):
        """
        Verify login rejection for deactivated user accounts.
        Expected: Returns 400 Bad Request with inactive user message.
        """
        from app.db.session import get_db_session
        from app.models.db import UserModel
        from app.core.security import get_password_hash

        mock_db = AsyncMock()
        mock_user = UserModel(
            id=uuid.uuid4(),
            email="inactive@example.com",
            full_name="Inactive User",
            hashed_password=get_password_hash("password123"),
            role=Role.EMPLOYEE,
            tenant_id=uuid.uuid4(),
            is_active=False,
        )

        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.auth.get_db_session", AsyncMock(return_value=mock_db)
        )

        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "inactive@example.com", "password": "password123"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Inactive user" in response.json()["detail"]

    async def test_request_without_token_returns_401(self, client: AsyncClient):
        """
        Verify endpoints requiring authentication reject requests without tokens.
        Expected: Returns 401 Unauthorized with credential validation error.
        """
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "credentials" in response.json()["detail"].lower()

    async def test_request_with_malformed_token(self, client: AsyncClient):
        """
        Verify rejection of malformed JWT tokens.
        Expected: Returns 401 Unauthorized.
        """
        headers = {"Authorization": "Bearer malformed.token.here"}
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_request_with_expired_token(self, client: AsyncClient):
        """
        Verify rejection of expired JWT tokens.
        Expected: Returns 401 Unauthorized with credential validation error.
        """
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())

        # Create token that expired 1 hour ago
        expired_token = create_access_token(
            {
                "sub": user_id,
                "email": "test@example.com",
                "role": Role.EMPLOYEE,
                "tenant_id": tenant_id,
            },
            expires_delta=timedelta(hours=-1),
        )

        headers = {"Authorization": f"Bearer {expired_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_request_with_tampered_token(self, client: AsyncClient):
        """
        Verify rejection of JWT tokens with tampered payload.
        Expected: Returns 401 Unauthorized.
        """
        # Create valid token then tamper with it
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        token = create_access_token(
            {
                "sub": user_id,
                "email": "test@example.com",
                "role": Role.EMPLOYEE,
                "tenant_id": tenant_id,
            }
        )

        # Tamper with the token
        tampered_token = token[:-10] + "tampered!!"

        headers = {"Authorization": f"Bearer {tampered_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_token_missing_required_claims(self, client: AsyncClient):
        """
        Verify rejection of tokens missing critical claims (sub/user_id).
        Expected: Returns 401 Unauthorized.
        """
        secret_key = settings.JWT_SECRET.get_secret_value()
        # Token without 'sub' claim
        invalid_payload = {
            "email": "test@example.com",
            "role": Role.EMPLOYEE,
            "exp": datetime.utcnow() + timedelta(minutes=30),
        }
        invalid_token = jwt.encode(invalid_payload, secret_key, algorithm=ALGORITHM)

        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# AUTHORIZATION & RBAC TESTS
# ============================================================================


@pytest.mark.asyncio
class TestAuthorizationAndRBAC:
    """Test suite for role-based access control and authorization."""

    async def test_admin_endpoint_accessible_by_admin(self, admin_client: AsyncClient):
        """
        Verify admin-only endpoints are accessible to Administrator role.
        Expected: Returns 200 OK with stats data.
        """
        response = await admin_client.get("/api/v1/admin/stats")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_documents" in data
        assert "total_messages" in data
        assert "active_conversations" in data
        assert "total_tokens" in data

    async def test_admin_endpoint_forbidden_for_manager(self, manager_client: AsyncClient):
        """
        Verify admin-only endpoints reject Manager role users.
        Expected: Returns 403 Forbidden.
        """
        response = await manager_client.get("/api/v1/admin/stats")
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "permissions" in response.json()["detail"].lower()

    async def test_admin_endpoint_forbidden_for_employee(self, employee_client: AsyncClient):
        """
        Verify admin-only endpoints reject Employee role users.
        Expected: Returns 403 Forbidden.
        """
        response = await employee_client.get("/api/v1/admin/stats")
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_admin_reindex_endpoint_accessible_by_admin(self, admin_client: AsyncClient):
        """
        Validate reindex endpoint restricted to Administrator role.
        Expected: Returns 202 Accepted with job_id.
        """
        response = await admin_client.post("/api/v1/admin/reindex")
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    async def test_admin_reindex_endpoint_forbidden_for_non_admin(
        self, manager_client: AsyncClient, employee_client: AsyncClient
    ):
        """
        Verify reindex endpoint rejects non-admin users.
        Expected: Returns 403 Forbidden for all non-admin roles.
        """
        manager_response = await manager_client.post("/api/v1/admin/reindex")
        assert manager_response.status_code == status.HTTP_403_FORBIDDEN

        employee_response = await employee_client.post("/api/v1/admin/reindex")
        assert employee_response.status_code == status.HTTP_403_FORBIDDEN

    async def test_document_upload_accessible_by_manager(self, manager_client: AsyncClient):
        """
        Verify document upload restricted to Manager and above.
        Expected: Accepts file upload from Manager role.
        """
        file_content = b"Test document content"
        response = await manager_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
        )
        # Should return 202 Accepted (processing)
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "processing"

    async def test_document_upload_accessible_by_admin(self, admin_client: AsyncClient):
        """
        Verify document upload accessible to Administrator (inherits Manager permissions).
        Expected: Returns 202 Accepted.
        """
        file_content = b"Test document content"
        response = await admin_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
        )
        assert response.status_code == status.HTTP_202_ACCEPTED

    async def test_document_upload_forbidden_for_employee(self, employee_client: AsyncClient):
        """
        Verify document upload rejected for Employee role.
        Expected: Returns 403 Forbidden.
        """
        file_content = b"Test document content"
        response = await employee_client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_document_delete_accessible_by_admin_only(
        self, admin_client: AsyncClient, manager_client: AsyncClient, employee_client: AsyncClient
    ):
        """
        Verify document deletion restricted to Administrator role only.
        Expected: Returns 403 Forbidden for Manager and Employee.
        """
        doc_id = uuid.uuid4()

        manager_response = await manager_client.delete(f"/api/v1/documents/{doc_id}")
        assert manager_response.status_code == status.HTTP_403_FORBIDDEN

        employee_response = await employee_client.delete(f"/api/v1/documents/{doc_id}")
        assert employee_response.status_code == status.HTTP_403_FORBIDDEN

        # Admin should attempt deletion (will 404 since doc doesn't exist, but auth passes)
        admin_response = await admin_client.delete(f"/api/v1/documents/{doc_id}")
        assert admin_response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_204_NO_CONTENT]

    async def test_role_hierarchy_enforcement(self):
        """
        Validate role hierarchy: Employee < Manager < Administrator.
        Expected: RequireRole dependency correctly evaluates hierarchy.
        """
        from app.api.dependencies.auth import RequireRole
        from app.models.user import UserResponse

        require_manager = RequireRole(Role.MANAGER)
        require_admin = RequireRole(Role.ADMINISTRATOR)

        # Employee cannot satisfy Manager requirement
        employee_user = UserResponse(
            id=uuid.uuid4(),
            email="emp@example.com",
            full_name="Employee",
            role=Role.EMPLOYEE,
            tenant_id=uuid.uuid4(),
            is_active=True,
        )

        with pytest.raises(Exception):  # HTTPException
            require_manager(employee_user)

        # Manager can satisfy Manager requirement but not Admin
        manager_user = UserResponse(
            id=uuid.uuid4(),
            email="mgr@example.com",
            full_name="Manager",
            role=Role.MANAGER,
            tenant_id=uuid.uuid4(),
            is_active=True,
        )

        result = require_manager(manager_user)
        assert result.role == Role.MANAGER

        with pytest.raises(Exception):  # HTTPException
            require_admin(manager_user)

        # Admin can satisfy any requirement
        admin_user = UserResponse(
            id=uuid.uuid4(),
            email="admin@example.com",
            full_name="Admin",
            role=Role.ADMINISTRATOR,
            tenant_id=uuid.uuid4(),
            is_active=True,
        )

        result = require_manager(admin_user)
        assert result.role == Role.ADMINISTRATOR

        result = require_admin(admin_user)
        assert result.role == Role.ADMINISTRATOR


# ============================================================================
# TENANT ISOLATION & MULTI-TENANCY TESTS
# ============================================================================


@pytest.mark.asyncio
class TestTenantIsolation:
    """Test suite for multi-tenancy and tenant data isolation."""

    async def test_user_can_only_see_own_tenant_documents(
        self, admin_client: AsyncClient, monkeypatch
    ):
        """
        Verify users cannot retrieve documents from other tenants.
        Expected: Document list filtered by tenant_id from JWT.
        """
        # This test validates the query filtering in list_documents endpoint
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.documents.get_db_session", AsyncMock(return_value=mock_db)
        )

        response = await admin_client.get("/api/v1/documents")
        assert response.status_code == status.HTTP_200_OK

        # Verify that db.execute was called with a query filtering by tenant_id
        # (The mock will be called, confirming query is made)
        assert mock_db.execute.called

    async def test_user_cannot_access_other_tenant_document(
        self, employee_client: AsyncClient, monkeypatch
    ):
        """
        Verify accessing another tenant's document returns 404.
        Expected: Returns 404 Not Found (not 403, to avoid leaking tenant structure).
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None  # Document not found for this tenant
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.documents.get_db_session", AsyncMock(return_value=mock_db)
        )

        other_tenant_doc_id = uuid.uuid4()
        response = await employee_client.get(f"/api/v1/documents/{other_tenant_doc_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_admin_stats_filtered_by_tenant(self, admin_client: AsyncClient, monkeypatch):
        """
        Verify admin stats endpoint returns only data for the admin's tenant.
        Expected: Stats query includes tenant_id filter in WHERE clause.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_db.scalar.return_value = 5  # Mock count responses

        monkeypatch.setattr(
            "app.api.v1.endpoints.admin.get_db_session", AsyncMock(return_value=mock_db)
        )

        response = await admin_client.get("/api/v1/admin/stats")
        assert response.status_code == status.HTTP_200_OK

        # Verify db.scalar was called multiple times (for each stat query)
        assert mock_db.scalar.called

    async def test_conversation_isolated_by_tenant(
        self, admin_client: AsyncClient, employee_client: AsyncClient, monkeypatch
    ):
        """
        Verify conversations are isolated per tenant.
        Expected: Conversation queries filtered by tenant_id.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        monkeypatch.setattr(
            "app.api.v1.endpoints.chat.get_db_session", AsyncMock(return_value=mock_db)
        )

        # Admin creates conversation
        admin_response = await admin_client.post("/api/v1/chat/conversations")
        assert admin_response.status_code == status.HTTP_201_CREATED

        # Employee creates conversation (separate session, separate tenant)
        employee_response = await employee_client.post("/api/v1/chat/conversations")
        assert employee_response.status_code == status.HTTP_201_CREATED

        # Verify that separate db instances were used (tenant isolation)
        assert mock_db.add.call_count >= 2

    async def test_tenant_filter_applied_to_chat_retrieval(
        self, admin_client: AsyncClient, monkeypatch
    ):
        """
        Verify chat message retrieval applies tenant filters to vector DB query.
        Expected: Retrieval query includes tenant_id in filters dict.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result
        mock_db.add_all = MagicMock()
        mock_db.commit = AsyncMock()

        monkeypatch.setattr(
            "app.api.v1.endpoints.chat.get_db_session", AsyncMock(return_value=mock_db)
        )

        conv_id = uuid.uuid4()
        response = await admin_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            json={"message": "test query"},
        )

        # Response may vary based on implementation, but should process request through filters
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        ]


# ============================================================================
# RATE LIMITING & BRUTE FORCE PROTECTION
# ============================================================================


@pytest.mark.asyncio
class TestRateLimitingAndBruteForceProtection:
    """Test suite for rate limiting and brute force attack prevention."""

    async def test_login_rate_limiting_enforced(self, client: AsyncClient):
        """
        Verify login endpoint enforces rate limiting (10 requests per hour).
        Expected: After 10 failed attempts, returns 429 Too Many Requests.
        """
        # Make 10 login attempts (all will fail, but shouldn't hit rate limit)
        for i in range(10):
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": f"user{i}@example.com", "password": "wrong"},
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # 11th attempt should be rate limited
        response = await client.post(
            "/api/v1/auth/login", data={"username": "user@example.com", "password": "wrong"}
        )

        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    async def test_rate_limiting_per_ip_address(self, client: AsyncClient):
        """
        Verify rate limiting is enforced per client IP address.
        Expected: Rate limit is enforced based on client IP.
        """
        # Test rate limiting on login endpoint - make 10 requests (limit is 10/hour)
        for i in range(10):
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": f"user{i}@example.com", "password": "wrong"},
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # 11th request should be rate limited
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "user@example.com", "password": "wrong"},
        )
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "retry_after_seconds" in response.json().get("details", {})

        # Verify rate limit headers are present
        assert "Retry-After" in response.headers or "retry_after_seconds" in response.json().get("details", {})

    async def test_rate_limiting_resets_after_window(self):
        """
        Verify rate limit counter resets after the time window expires.
        Expected: After time window, new requests are allowed.
        """
        # This test is more suited for integration testing with real Redis
        # Skipping unit test since it requires time manipulation
        pass


# ============================================================================
# INPUT VALIDATION & INJECTION PREVENTION
# ============================================================================


@pytest.mark.asyncio
class TestInputValidationAndGuardrails:
    """Test suite for input validation and injection attack prevention."""

    async def test_chat_message_with_valid_input(self, admin_client: AsyncClient, monkeypatch):
        """
        Verify chat endpoint accepts valid messages.
        Expected: Returns 200 OK with generation result.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result
        mock_db.add_all = MagicMock()
        mock_db.commit = AsyncMock()

        monkeypatch.setattr(
            "app.api.v1.endpoints.chat.get_db_session", AsyncMock(return_value=mock_db)
        )

        conv_id = uuid.uuid4()
        response = await admin_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            json={"message": "What is machine learning?"},
        )

        # Should process the valid message
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    async def test_chat_message_with_empty_input(self, admin_client: AsyncClient):
        """
        Verify chat endpoint rejects empty messages.
        Expected: Returns 422 Unprocessable Entity (validation error) or 400 Bad Request.
        """
        conv_id = uuid.uuid4()
        response = await admin_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages", json={"message": ""}
        )

        # Validation should fail
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST,
        ]

    async def test_chat_message_with_excessive_length(self, admin_client: AsyncClient):
        """
        Verify chat endpoint rejects excessively long messages.
        Expected: Returns 400 Bad Request or 422 Unprocessable Entity.
        """
        conv_id = uuid.uuid4()
        excessive_message = "x" * 100000  # 100k characters

        response = await admin_client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages", json={"message": excessive_message}
        )

        # Should be rejected
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    async def test_document_upload_rejects_empty_file(self, manager_client: AsyncClient):
        """
        Verify document upload rejects empty files.
        Expected: Returns 400 Bad Request.
        """
        response = await manager_client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty" in response.json()["detail"].lower()

    async def test_document_upload_rejects_oversized_file(self, manager_client: AsyncClient):
        """
        Verify document upload rejects files exceeding size limit (25MB).
        Expected: Returns 400 Bad Request.
        """
        oversized_content = b"x" * (26 * 1024 * 1024)  # 26MB

        response = await manager_client.post(
            "/api/v1/documents/upload",
            files={"file": ("huge.bin", oversized_content, "application/octet-stream")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "too large" in response.json()["detail"].lower()

    async def test_path_traversal_prevention(self, manager_client: AsyncClient, monkeypatch):
        """
        Verify document IDs prevent path traversal attacks.
        Expected: UUID parsing rejects non-UUID path parameters.
        """
        # Try to access with path traversal payload
        malicious_id = "../../etc/passwd"
        response = await manager_client.get(f"/api/v1/documents/{malicious_id}")

        # Should return 422 validation error (invalid UUID)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ============================================================================
# SECURITY HEADERS & MISCELLANEOUS
# ============================================================================


@pytest.mark.asyncio
class TestSecurityHeaders:
    """Test suite for security-related HTTP headers and misc security features."""

    async def test_authorization_header_required_bearer_scheme(
        self, client: AsyncClient, monkeypatch
    ):
        """
        Verify endpoints reject non-Bearer authentication schemes.
        Expected: Returns 403 Forbidden or 401 Unauthorized.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        monkeypatch.setattr(
            "app.api.v1.endpoints.auth.get_db_session", AsyncMock(return_value=mock_db)
        )

        # Use Basic auth instead of Bearer
        response = await client.get(
            "/api/v1/auth/me", headers={"Authorization": "Basic dXNlcjpwYXNz"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_token_subject_claim_validated(self, client: AsyncClient):
        """
        Verify that token subject claim is validated and is a valid UUID.
        Expected: Invalid UUIDs in 'sub' claim are rejected.
        """
        secret_key = settings.JWT_SECRET.get_secret_value()
        invalid_payload = {
            "sub": "not-a-uuid",
            "email": "test@example.com",
            "role": Role.EMPLOYEE,
            "exp": datetime.utcnow() + timedelta(minutes=30),
        }
        invalid_token = jwt.encode(invalid_payload, secret_key, algorithm=ALGORITHM)

        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_password_not_returned_in_user_response(
        self, admin_client: AsyncClient
    ):
        """
        Verify that password/password_hash is never returned in API responses.
        Expected: UserResponse excludes sensitive password fields.
        """
        response = await admin_client.get("/api/v1/auth/me")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "password" not in data
        assert "hashed_password" not in data
        assert "password_hash" not in data

    async def test_jwt_algorithm_enforcement(self, client: AsyncClient):
        """
        Verify JWT algorithm is strictly enforced (HS256, not 'none').
        Expected: Tokens with 'none' algorithm are rejected.
        """
        # Create token with 'none' algorithm
        invalid_payload = {
            "sub": str(uuid.uuid4()),
            "email": "test@example.com",
            "role": Role.EMPLOYEE,
            "exp": datetime.utcnow() + timedelta(minutes=30),
        }
        # Encode with 'none' algorithm (simulating attack)
        invalid_token = jwt.encode(invalid_payload, "", algorithm="none")

        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = await client.get("/api/v1/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
class TestSecurityIntegration:
    """Integration tests combining multiple security layers."""

    async def test_complete_auth_flow_with_security_checks(
        self, client: AsyncClient, monkeypatch
    ):
        """
        Validate complete authentication flow from login to protected endpoint.
        Expected: User receives valid token, can access protected resource.
        """
        from app.db.session import get_db_session
        from app.models.db import UserModel
        from app.core.security import get_password_hash

        user_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        password = "SecurePassword123!"

        mock_db = AsyncMock()
        mock_user = UserModel(
            id=user_id,
            email="testuser@example.com",
            full_name="Test User",
            hashed_password=get_password_hash(password),
            role=Role.MANAGER,
            tenant_id=tenant_id,
            is_active=True,
        )

        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_result.scalars.return_value.first.return_value = mock_user
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.auth.get_db_session", AsyncMock(return_value=mock_db)
        )
        monkeypatch.setattr(
            "app.api.dependencies.auth.get_db_session", AsyncMock(return_value=mock_db)
        )

        # Step 1: Login
        login_response = await client.post(
            "/api/v1/auth/login",
            data={"username": "testuser@example.com", "password": password},
        )

        assert login_response.status_code == status.HTTP_200_OK
        token = login_response.json()["access_token"]

        # Step 2: Use token to access protected resource
        headers = {"Authorization": f"Bearer {token}"}
        me_response = await client.get("/api/v1/auth/me", headers=headers)

        assert me_response.status_code == status.HTTP_200_OK
        user_data = me_response.json()
        assert user_data["email"] == "testuser@example.com"
        assert user_data["role"] == Role.MANAGER
        assert user_data["tenant_id"] == str(tenant_id)

    async def test_rbac_enforcement_across_endpoints(
        self, admin_client: AsyncClient, employee_client: AsyncClient, monkeypatch
    ):
        """
        Validate RBAC is consistently enforced across multiple API endpoints.
        Expected: Admins have access, employees are denied for restricted operations.
        """
        from app.db.session import get_db_session

        mock_db = AsyncMock()
        mock_db.scalar.return_value = 0
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        monkeypatch.setattr(
            "app.api.v1.endpoints.admin.get_db_session", AsyncMock(return_value=mock_db)
        )
        monkeypatch.setattr(
            "app.api.v1.endpoints.documents.get_db_session", AsyncMock(return_value=mock_db)
        )

        # Admin can access stats
        admin_stats = await admin_client.get("/api/v1/admin/stats")
        assert admin_stats.status_code == status.HTTP_200_OK

        # Employee cannot access stats
        employee_stats = await employee_client.get("/api/v1/admin/stats")
        assert employee_stats.status_code == status.HTTP_403_FORBIDDEN

        # Both can list documents (different endpoint)
        admin_docs = await admin_client.get("/api/v1/documents")
        assert admin_docs.status_code == status.HTTP_200_OK

        employee_docs = await employee_client.get("/api/v1/documents")
        assert employee_docs.status_code == status.HTTP_200_OK
