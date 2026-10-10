from datetime import datetime, timedelta

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

pwd_context = PasswordHash.recommended()

ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Default 30 minutes
        expire = datetime.utcnow() + timedelta(minutes=30)

    to_encode.update({"exp": expire})
    secret_key = settings.JWT_SECRET.get_secret_value()
    if settings.ENVIRONMENT == "production" and secret_key == "insecure-dev-secret-key-change-me":
        raise ValueError("Fatal: Default insecure JWT secret is used in production environment.")
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt
