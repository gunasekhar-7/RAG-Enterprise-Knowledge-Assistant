from datetime import timedelta
from hashlib import sha256
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.rate_limit_checker import check_auth_limit
from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.db.session import get_db_session
from app.models.db import RefreshTokenModel, UserModel
from app.models.user import UserResponse

router = APIRouter()

REFRESH_COOKIE_NAME = "refresh_token"
REFRESH_COOKIE_MAX_AGE = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60  # seconds


def _hash_token(token: str) -> str:
    """Hash a token for storage."""
    return sha256(token.encode()).hexdigest()


@router.post("/login")
async def login_access_token(  # type: ignore
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
    _: None = Depends(check_auth_limit),
):
    """
    OAuth2 compatible token login, get an access token and refresh token for future requests.
    Protected by aggressive rate limiting against brute force.
    """
    # 1. Authenticate User
    stmt = select(UserModel).where(UserModel.email == form_data.username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

    # 2. Prepare token data
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value if hasattr(user.role, "value") else user.role,
        "tenant_id": str(user.tenant_id),
    }

    # 3. Issue Access Token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=token_data, expires_delta=access_token_expires)

    # 4. Issue Refresh Token with rotation support
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(data=token_data, expires_delta=refresh_token_expires)

    # 5. Store refresh token hash for rotation tracking
    refresh_payload = decode_token(refresh_token, expected_type="refresh")
    refresh_jti = refresh_payload["jti"]
    refresh_family_id = str(uuid4())

    refresh_token_record = RefreshTokenModel(
        user_id=user.id,
        jti=refresh_jti,
        family_id=refresh_family_id,
        expires_at=refresh_payload["exp"],
        token_hash=_hash_token(refresh_token),
    )
    db.add(refresh_token_record)
    await db.commit()

    # 6. Set refresh token in secure HttpOnly cookie
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.ENVIRONMENT == "production",
        samesite="lax",
        max_age=REFRESH_COOKIE_MAX_AGE,
        path="/api/v1/auth",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
    }


@router.post("/refresh")
async def refresh_access_token(  # type: ignore
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Refresh access token using refresh token.
    Implements refresh token rotation: old token is revoked, new one issued.
    """
    # 1. Get refresh token from cookie
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found",
        )

    # 2. Decode and validate refresh token
    try:
        payload = decode_token(refresh_token, expected_type="refresh")
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {e}",
        )

    # 3. Verify token exists and is not revoked
    jti = payload["jti"]
    stmt = select(RefreshTokenModel).where(RefreshTokenModel.jti == jti)
    result = await db.execute(stmt)
    stored_token = result.scalar_one_or_none()

    if not stored_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not recognized",
        )

    if stored_token.revoked_at is not None:
        # Token reuse detected! Revoke entire family for security
        await _revoke_token_family(db, stored_token.family_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token reuse detected. Please log in again.",
        )

    if stored_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )

    # 4. Verify token hash matches (prevents use of stolen token hashes)
    if stored_token.token_hash != _hash_token(refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    # 5. Rotate: revoke old token, issue new access + refresh tokens
    stored_token.revoked_at = datetime.now(timezone.utc)

    # Issue new tokens with same family_id for rotation tracking
    token_data = {
        "sub": payload["sub"],
        "email": payload["email"],
        "role": payload["role"],
        "tenant_id": payload["tenant_id"],
    }

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=token_data, expires_delta=access_token_expires)

    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    new_refresh_token = create_refresh_token(data=token_data, expires_delta=refresh_token_expires)

    # Store new refresh token
    new_refresh_payload = decode_token(new_refresh_token, expected_type="refresh")
    new_refresh_jti = new_refresh_payload["jti"]

    new_refresh_token_record = RefreshTokenModel(
        user_id=stored_token.user_id,
        jti=new_refresh_jti,
        family_id=stored_token.family_id,  # Same family for rotation tracking
        expires_at=new_refresh_payload["exp"],
        token_hash=_hash_token(new_refresh_token),
    )
    db.add(new_refresh_token_record)
    await db.commit()

    # 6. Set new refresh token in cookie
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=new_refresh_token,
        httponly=True,
        secure=settings.ENVIRONMENT == "production",
        samesite="lax",
        max_age=REFRESH_COOKIE_MAX_AGE,
        path="/api/v1/auth",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
    }


@router.post("/logout")
async def logout(  # type: ignore
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Logout: revoke the current refresh token.
    """
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME)
    if refresh_token:
        try:
            payload = decode_token(refresh_token, expected_type="refresh")
            jti = payload["jti"]
            stmt = select(RefreshTokenModel).where(RefreshTokenModel.jti == jti)
            result = await db.execute(stmt)
            stored_token = result.scalar_one_or_none()
            if stored_token:
                stored_token.revoked_at = datetime.now(timezone.utc)
                await db.commit()
        except jwt.InvalidTokenError:
            pass  # Ignore invalid tokens on logout

    # Clear cookie
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/api/v1/auth")
    return {"detail": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):  # type: ignore
    """
    Get current user profile.
    """
    return current_user


async def _revoke_token_family(db: AsyncSession, family_id: str | None) -> None:
    """Revoke all tokens in a family (used on reuse detection)."""
    if not family_id:
        return
    from datetime import timezone
    stmt = select(RefreshTokenModel).where(
        RefreshTokenModel.family_id == family_id,
        RefreshTokenModel.revoked_at.is_(None),
    )
    result = await db.execute(stmt)
    tokens = result.scalars().all()
    for token in tokens:
        token.revoked_at = datetime.now(timezone.utc)
    await db.commit()


# Import at bottom to avoid circular imports
from datetime import datetime, timezone
import jwt