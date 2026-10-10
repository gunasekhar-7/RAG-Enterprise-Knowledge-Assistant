from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.rate_limiter import RateLimiter
from app.core.security import create_access_token, verify_password
from app.db.session import get_db_session
from app.models.db import UserModel
from app.models.user import UserResponse

router = APIRouter()
login_rate_limiter = RateLimiter(requests_per_minute=5)


@router.post("/login")
async def login_access_token(  # type: ignore
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    Protected by aggressive rate limiting against brute force.
    """
    # 1. Enforce Rate Limit
    login_rate_limiter(request)

    # 2. Authenticate User
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
        raise HTTPException(status_code=400, detail="Inactive user")

    # 3. Issue Token
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value if hasattr(user.role, "value") else user.role,
            "tenant_id": str(user.tenant_id),
        },
        expires_delta=access_token_expires,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):  # type: ignore
    """
    Get current user profile.
    """
    return current_user
