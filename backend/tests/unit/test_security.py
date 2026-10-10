import uuid

import pytest
from fastapi import HTTPException

from app.api.dependencies.auth import RequireRole, get_tenant_filters
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.user import Role, UserResponse


def test_password_hashing():
    password = "supersecretpassword"
    hashed = get_password_hash(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False


def test_create_access_token():
    token = create_access_token(data={"sub": "test@example.com", "role": "Employee"})
    # It should be a valid string, JWT format has two periods
    assert isinstance(token, str)
    assert token.count(".") == 2


def test_require_role_hierarchy():
    admin_user = UserResponse(
        id=uuid.uuid4(),
        email="a@a.com",
        full_name="A",
        role=Role.ADMINISTRATOR,
        tenant_id=uuid.uuid4(),
        is_active=True,
    )
    manager_user = UserResponse(
        id=uuid.uuid4(),
        email="m@m.com",
        full_name="M",
        role=Role.MANAGER,
        tenant_id=uuid.uuid4(),
        is_active=True,
    )
    employee_user = UserResponse(
        id=uuid.uuid4(),
        email="e@e.com",
        full_name="E",
        role=Role.EMPLOYEE,
        tenant_id=uuid.uuid4(),
        is_active=True,
    )

    require_admin = RequireRole(Role.ADMINISTRATOR)
    require_manager = RequireRole(Role.MANAGER)
    require_employee = RequireRole(Role.EMPLOYEE)

    # Admin can do anything
    assert require_admin(admin_user) == admin_user
    assert require_manager(admin_user) == admin_user
    assert require_employee(admin_user) == admin_user

    # Manager can do manager+employee
    with pytest.raises(HTTPException) as exc:
        require_admin(manager_user)
    assert exc.value.status_code == 403

    assert require_manager(manager_user) == manager_user
    assert require_employee(manager_user) == manager_user

    # Employee can only do employee
    with pytest.raises(HTTPException):
        require_admin(employee_user)
    with pytest.raises(HTTPException):
        require_manager(employee_user)
    assert require_employee(employee_user) == employee_user


def test_get_tenant_filters():
    tenant_id = uuid.uuid4()
    user = UserResponse(
        id=uuid.uuid4(),
        email="e@e.com",
        full_name="E",
        role=Role.EMPLOYEE,
        tenant_id=tenant_id,
        is_active=True,
    )

    filters = get_tenant_filters(user)
    assert filters["tenant_id"] == str(tenant_id)
