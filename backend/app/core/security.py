from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

pwd_context = PasswordHash.recommended()

# Supported algorithms - never trust algorithm from incoming token
SUPPORTED_ALGORITHMS = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]

# Default algorithm for backward compatibility
ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def _get_utc_now() -> datetime:
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)


def _validate_algorithm(algorithm: str) -> str:
    """Validate that the algorithm is in the supported list."""
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {SUPPORTED_ALGORITHMS}")
    return algorithm


def _create_token(
    data: dict,
    expires_delta: timedelta,
    token_type: str,
    secret_key: str | None = None,
    algorithm: str | None = None,
) -> str:
    """Internal function to create a JWT token with standard claims."""
    if secret_key is None:
        secret_key = settings.JWT_SECRET.get_secret_value()

    if algorithm is None:
        algorithm = _validate_algorithm(settings.JWT_ALGORITHM)
    else:
        _validate_algorithm(algorithm)

    now = _get_utc_now()
    expire = now + expires_delta
    jti = str(uuid4())

    to_encode = data.copy()
    to_encode.update(
        {
            "exp": expire,
            "iat": now,
            "jti": jti,
            "type": token_type,
            "iss": settings.JWT_ISSUER,
            "aud": settings.JWT_AUDIENCE,
        }
    )

    # Production secret check
    if settings.ENVIRONMENT == "production" and secret_key == "insecure-dev-secret-key-change-me":
        raise ValueError("Fatal: Default insecure JWT secret is used in production environment.")

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
    secret_key: str | None = None,
    algorithm: str | None = None,
) -> str:
    """
    Create an access token.

    Args:
        data: Claims to include in the token (must include 'sub')
        expires_delta: Custom expiration (default: ACCESS_TOKEN_EXPIRE_MINUTES)
        secret_key: Override secret key (for testing)
        algorithm: Override algorithm (for testing)

    Returns:
        Encoded JWT string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return _create_token(data, expires_delta, "access", secret_key, algorithm)


def create_refresh_token(
    data: dict,
    expires_delta: timedelta | None = None,
    secret_key: str | None = None,
    algorithm: str | None = None,
) -> str:
    """
    Create a refresh token.

    Args:
        data: Claims to include in the token (must include 'sub')
        expires_delta: Custom expiration (default: REFRESH_TOKEN_EXPIRE_DAYS)
        secret_key: Override secret key (for testing)
        algorithm: Override algorithm (for testing)

    Returns:
        Encoded JWT string
    """
    if expires_delta is None:
        expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    return _create_token(data, expires_delta, "refresh", secret_key, algorithm)


def decode_token(
    token: str,
    secret_key: str | None = None,
    algorithm: str | None = None,
    expected_type: str | None = None,
) -> dict:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT string to decode
        secret_key: Override secret key (for testing)
        algorithm: Override algorithm (for testing)
        expected_type: If provided, validate the 'type' claim matches

    Returns:
        Decoded payload dict

    Raises:
        jwt.ExpiredSignatureError: Token has expired
        jwt.InvalidSignatureError: Signature verification failed
        jwt.InvalidTokenError: Other validation errors
    """
    if secret_key is None:
        secret_key = settings.JWT_SECRET.get_secret_value()

    if algorithm is None:
        algorithm = _validate_algorithm(settings.JWT_ALGORITHM)
    else:
        _validate_algorithm(algorithm)

    # Always specify algorithms explicitly - never accept 'none' or token-specified algorithm
    payload = jwt.decode(
        token,
        secret_key,
        algorithms=[algorithm],
        audience=settings.JWT_AUDIENCE,
        issuer=settings.JWT_ISSUER,
        options={"require": ["exp", "iat", "jti", "type", "sub"]},
    )

    # Validate token type if expected
    if expected_type is not None:
        token_type = payload.get("type")
        if token_type != expected_type:
            raise jwt.InvalidTokenError(f"Expected token type '{expected_type}', got '{token_type}'")

    return payload