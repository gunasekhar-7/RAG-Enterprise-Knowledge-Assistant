import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError

from app.core.config import settings
from app.core.exceptions import (
    MissingTokenError,
    InvalidTokenError,
    ExpiredTokenError,
    InvalidSignatureError,
    MissingClaimsError,
    UserNotFoundError,
)
from app.core.security import ALGORITHM
from app.models.user import Role, UserResponse

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.future import select  # noqa: E402

from app.db.session import get_db_session  # noqa: E402
from app.models.db import UserModel  # noqa: E402


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    secret_key = settings.JWT_SECRET.get_secret_value()

    if not token:
        raise MissingTokenError()

    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as e:
        if isinstance(e, jwt.ExpiredSignatureError):
            raise ExpiredTokenError() from e
        elif isinstance(e, jwt.InvalidSignatureError):
            raise InvalidSignatureError() from e
        else:
            raise InvalidTokenError() from e
    except ValidationError as e:
        raise InvalidTokenError() from e

    user_id: str = payload.get("sub")  # type: ignore
    if user_id is None:
        raise MissingClaimsError("sub")

    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise InvalidTokenError()

    result = await db.execute(select(UserModel).where(UserModel.id == uid))
    db_user = result.scalars().first()

    if db_user is None:
        raise UserNotFoundError(user_id)

    if not db_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return UserResponse(
        id=db_user.id,  # type: ignore
        email=db_user.email,  # type: ignore
        full_name=db_user.full_name,  # type: ignore
        role=db_user.role,  # type: ignore
        tenant_id=db_user.tenant_id,  # type: ignore
        is_active=db_user.is_active,  # type: ignore,
    )


class RequireRole:
    """Dependency class to enforce RBAC based on minimum required role."""

    def __init__(self, required_role: Role):
        self.required_role = required_role

        # Define hierarchy
        self.hierarchy = {Role.EMPLOYEE: 1, Role.MANAGER: 2, Role.ADMINISTRATOR: 3}

    def __call__(self, current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        user_level = self.hierarchy.get(current_user.role, 0)
        required_level = self.hierarchy.get(self.required_role, 0)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
            )
        return current_user


def get_tenant_filters(current_user: UserResponse = Depends(get_current_user)) -> dict:
    """
    Dependency that generates vector DB filters enforcing document-level authorization.
    Inject this into retrieval routes.
    """
    return {"tenant_id": str(current_user.tenant_id)}
