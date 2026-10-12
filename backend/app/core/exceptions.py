"""
Centralized exception handling for the RAG Enterprise Knowledge Assistant.

Provides:
- Custom exception hierarchy with error codes
- Security-safe error responses (no stack traces in production)
- Request ID correlation for distributed tracing
- Consistent error response format across all endpoints
"""

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # Authentication & Authorization
    AUTH_MISSING_TOKEN = "AUTH_MISSING_TOKEN"
    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    AUTH_EXPIRED_TOKEN = "AUTH_EXPIRED_TOKEN"
    AUTH_INVALID_SIGNATURE = "AUTH_INVALID_SIGNATURE"
    AUTH_MISSING_CLAIMS = "AUTH_MISSING_CLAIMS"
    AUTH_USER_NOT_FOUND = "AUTH_USER_NOT_FOUND"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_INSUFFICIENT_PERMISSIONS"

    # Input Validation
    VALIDATION_ERROR = "VALIDATION_ERROR"
    VALIDATION_EMPTY_INPUT = "VALIDATION_EMPTY_INPUT"
    VALIDATION_INVALID_FORMAT = "VALIDATION_INVALID_FORMAT"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"

    # Document Processing
    DOCUMENT_EMPTY = "DOCUMENT_EMPTY"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    DOCUMENT_UNSUPPORTED_TYPE = "DOCUMENT_UNSUPPORTED_TYPE"
    DOCUMENT_CORRUPT = "DOCUMENT_CORRUPT"
    DOCUMENT_DUPLICATE = "DOCUMENT_DUPLICATE"
    DOCUMENT_PARSING_FAILED = "DOCUMENT_PARSING_FAILED"

    # RAG Pipeline
    RAG_EMBEDDING_FAILED = "RAG_EMBEDDING_FAILED"
    RAG_LLM_FAILED = "RAG_LLM_FAILED"
    RAG_LLM_RATE_LIMITED = "RAG_LLM_RATE_LIMITED"
    RAG_LLM_TIMEOUT = "RAG_LLM_TIMEOUT"
    RAG_RETRIEVAL_FAILED = "RAG_RETRIEVAL_FAILED"
    RAG_RERANKER_FAILED = "RAG_RERANKER_FAILED"
    RAG_EMPTY_QUERY = "RAG_EMPTY_QUERY"
    RAG_NO_RESULTS = "RAG_NO_RESULTS"
    RAG_INVALID_FILTERS = "RAG_INVALID_FILTERS"

    # Database
    DB_CONNECTION_FAILED = "DB_CONNECTION_FAILED"
    DB_QUERY_FAILED = "DB_QUERY_FAILED"
    DB_TIMEOUT = "DB_TIMEOUT"
    DB_CONSTRAINT_VIOLATION = "DB_CONSTRAINT_VIOLATION"
    DB_TRANSACTION_FAILED = "DB_TRANSACTION_FAILED"

    # External Services
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_RATE_LIMITED = "EXTERNAL_SERVICE_RATE_LIMITED"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"

    # Configuration
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_MISSING_SECRET = "CONFIG_MISSING_SECRET"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Internal/Unexpected
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class AppError(Exception):
    """
    Base application exception with error code and user-safe message.

    All application-specific exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[dict[str, Any]] = None,
        log_level: str = "error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.log_level = log_level
        self.request_id: Optional[str] = None

    def to_dict(self, include_details: bool = True) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "error": self.error_code.value,
            "message": self.message,
        }
        if include_details and self.details:
            result["details"] = self.details
        if self.request_id:
            result["request_id"] = self.request_id
        return result


# =============================================================================
# Authentication & Authorization Exceptions
# =============================================================================

class AuthenticationError(AppError):
    """Base class for authentication errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: ErrorCode = ErrorCode.AUTH_INVALID_TOKEN,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
            log_level="warning",
        )


class MissingTokenError(AuthenticationError):
    """Raised when Authorization header is missing."""

    def __init__(self, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message="Missing authentication token",
            error_code=ErrorCode.AUTH_MISSING_TOKEN,
            details=details,
        )


class InvalidTokenError(AuthenticationError):
    """Raised when token format is invalid."""

    def __init__(self, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message="Invalid token format",
            error_code=ErrorCode.AUTH_INVALID_TOKEN,
            details=details,
        )


class ExpiredTokenError(AuthenticationError):
    """Raised when token has expired."""

    def __init__(self, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message="Token has expired",
            error_code=ErrorCode.AUTH_EXPIRED_TOKEN,
            details=details,
        )


class InvalidSignatureError(AuthenticationError):
    """Raised when token signature verification fails."""

    def __init__(self, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message="Invalid token signature",
            error_code=ErrorCode.AUTH_INVALID_SIGNATURE,
            details=details,
        )


class MissingClaimsError(AuthenticationError):
    """Raised when required JWT claims are missing."""

    def __init__(self, missing_claim: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Token missing required claim: {missing_claim}",
            error_code=ErrorCode.AUTH_MISSING_CLAIMS,
            details={**(details or {}), "missing_claim": missing_claim},
        )


class UserNotFoundError(AuthenticationError):
    """Raised when user from token doesn't exist in database."""

    def __init__(self, user_id: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message="User not found",
            error_code=ErrorCode.AUTH_USER_NOT_FOUND,
            details={**(details or {}), "user_id": user_id},
        )


class AuthorizationError(AppError):
    """Raised when user lacks required permissions."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_role: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            status_code=status.HTTP_403_FORBIDDEN,
            details={**(details or {}), "required_role": required_role} if required_role else details,
            log_level="warning",
        )


# =============================================================================
# Input Validation & Security Exceptions
# =============================================================================

class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str = "Invalid input",
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_400_BAD_REQUEST,
            details={**(details or {}), "field": field} if field else details,
            log_level="warning",
        )


class SecurityViolationError(AppError):
    """Raised when security guardrails detect malicious input."""

    def __init__(
        self,
        message: str = "Request blocked by security policy",
        violation_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.SECURITY_VIOLATION,
            status_code=status.HTTP_400_BAD_REQUEST,
            details={**(details or {}), "violation_type": violation_type} if violation_type else details,
            log_level="warning",
        )


# =============================================================================
# Document Processing Exceptions
# =============================================================================

class DocumentError(AppError):
    """Base class for document processing errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DOCUMENT_PARSING_FAILED,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        filename: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details={**(details or {}), "filename": filename} if filename else details,
            log_level="warning",
        )


class EmptyDocumentError(DocumentError):
    """Raised when uploaded document is empty."""

    def __init__(self, filename: Optional[str] = None) -> None:
        super().__init__(
            message="Document is empty",
            error_code=ErrorCode.DOCUMENT_EMPTY,
            status_code=status.HTTP_400_BAD_REQUEST,
            filename=filename,
        )


class DocumentTooLargeError(DocumentError):
    """Raised when document exceeds size limit."""

    def __init__(
        self,
        size_bytes: int,
        max_bytes: int,
        filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Document size ({size_bytes} bytes) exceeds maximum ({max_bytes} bytes)",
            error_code=ErrorCode.DOCUMENT_TOO_LARGE,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            filename=filename,
            details={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )


class UnsupportedDocumentTypeError(DocumentError):
    """Raised when document type is not supported."""

    def __init__(
        self,
        detected_type: str,
        filename: Optional[str] = None,
        allowed_types: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            message=f"Unsupported document type: {detected_type}",
            error_code=ErrorCode.DOCUMENT_UNSUPPORTED_TYPE,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            filename=filename,
            details={"detected_type": detected_type, "allowed_types": allowed_types or []},
        )


class CorruptDocumentError(DocumentError):
    """Raised when document appears corrupted."""

    def __init__(
        self,
        filename: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            message="Document appears to be corrupted",
            error_code=ErrorCode.DOCUMENT_CORRUPT,
            status_code=status.HTTP_400_BAD_REQUEST,
            filename=filename,
            details={"reason": reason} if reason else None,
        )


class DuplicateDocumentError(DocumentError):
    """Raised when duplicate document is detected."""

    def __init__(
        self,
        sha256_hash: str,
        filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            message="Document with identical content already exists",
            error_code=ErrorCode.DOCUMENT_DUPLICATE,
            status_code=status.HTTP_409_CONFLICT,
            filename=filename,
            details={"sha256_hash": sha256_hash},
        )


class DocumentParsingError(DocumentError):
    """Raised when document parsing fails."""

    def __init__(
        self,
        filename: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            message="Failed to parse document",
            error_code=ErrorCode.DOCUMENT_PARSING_FAILED,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            filename=filename,
            details={"reason": reason} if reason else None,
        )


# =============================================================================
# RAG Pipeline Exceptions
# =============================================================================

class RAGError(AppError):
    """Base class for RAG pipeline errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.RAG_RETRIEVAL_FAILED,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[dict[str, Any]] = None,
        log_level: str = "error",
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            log_level=log_level,
        )


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str = "Failed to generate embeddings",
        provider: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.RAG_EMBEDDING_FAILED,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={**(details or {}), "provider": provider} if provider else details,
        )


class LLMError(RAGError):
    """Raised when LLM generation fails."""

    def __init__(
        self,
        message: str = "LLM generation failed",
        model: Optional[str] = None,
        retryable: bool = False,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if retryable else status.HTTP_500_INTERNAL_SERVER_ERROR
        super().__init__(
            message=message,
            error_code=ErrorCode.RAG_LLM_FAILED,
            status_code=status_code,
            details={**(details or {}), "model": model, "retryable": retryable} if model else details,
        )


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded, please retry",
        model: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=message,
            model=model,
            retryable=True,
            details={"retry_after_seconds": retry_after} if retry_after else None,
        )
        self.error_code = ErrorCode.RAG_LLM_RATE_LIMITED


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    def __init__(
        self,
        message: str = "LLM request timed out",
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(
            message=message,
            model=model,
            retryable=True,
            details={"timeout_seconds": timeout_seconds} if timeout_seconds else None,
        )
        self.error_code = ErrorCode.RAG_LLM_TIMEOUT


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""

    def __init__(
        self,
        message: str = "Document retrieval failed",
        retriever_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.RAG_RETRIEVAL_FAILED,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={**(details or {}), "retriever_type": retriever_type} if retriever_type else details,
        )


class RerankerError(RAGError):
    """Raised when reranking fails."""

    def __init__(
        self,
        message: str = "Reranking failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.RAG_RERANKER_FAILED,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
        )


class EmptyQueryError(RAGError):
    """Raised when query is empty."""

    def __init__(self) -> None:
        super().__init__(
            message="Query cannot be empty",
            error_code=ErrorCode.RAG_EMPTY_QUERY,
            status_code=status.HTTP_400_BAD_REQUEST,
            log_level="warning",
        )


class NoResultsError(RAGError):
    """Raised when retrieval returns no results (not an error, but handled specially)."""

    def __init__(self, query: str) -> None:
        super().__init__(
            message="No relevant documents found for query",
            error_code=ErrorCode.RAG_NO_RESULTS,
            status_code=status.HTTP_404_NOT_FOUND,
            log_level="info",
            details={"query": query},
        )


class InvalidFiltersError(RAGError):
    """Raised when retrieval filters are invalid."""

    def __init__(
        self,
        message: str = "Invalid filter parameters",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.RAG_INVALID_FILTERS,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
            log_level="warning",
        )


# =============================================================================
# Database Exceptions
# =============================================================================

class DatabaseError(AppError):
    """Base class for database errors."""

    def __init__(
        self,
        message: str = "Database operation failed",
        error_code: ErrorCode = ErrorCode.DB_QUERY_FAILED,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            log_level="error",
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(
        self,
        message: str = "Database connection failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DB_CONNECTION_FAILED,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class DatabaseTimeoutError(DatabaseError):
    """Raised when database query times out."""

    def __init__(
        self,
        message: str = "Database query timed out",
        query_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DB_TIMEOUT,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details={**(details or {}), "query_type": query_type} if query_type else details,
        )


class ConstraintViolationError(DatabaseError):
    """Raised when database constraint is violated."""

    def __init__(
        self,
        message: str = "Database constraint violation",
        constraint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DB_CONSTRAINT_VIOLATION,
            status_code=status.HTTP_409_CONFLICT,
            details={**(details or {}), "constraint": constraint} if constraint else details,
        )


class TransactionError(DatabaseError):
    """Raised when database transaction fails."""

    def __init__(
        self,
        message: str = "Database transaction failed",
        operation: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.DB_TRANSACTION_FAILED,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={**(details or {}), "operation": operation} if operation else details,
        )


# =============================================================================
# External Service Exceptions
# =============================================================================

class ExternalServiceError(AppError):
    """Raised when external service is unavailable."""

    def __init__(
        self,
        message: str = "External service unavailable",
        service: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={**(details or {}), "service": service} if service else details,
            log_level="error",
        )


class ExternalServiceRateLimitError(ExternalServiceError):
    """Raised when external service rate limits are exceeded."""

    def __init__(
        self,
        service: str,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(
            message=f"{service} rate limit exceeded",
            service=service,
            error_code=ErrorCode.EXTERNAL_SERVICE_RATE_LIMITED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"retry_after_seconds": retry_after} if retry_after else None,
        )


class ExternalServiceTimeoutError(ExternalServiceError):
    """Raised when external service times out."""

    def __init__(
        self,
        service: str,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(
            message=f"{service} request timed out",
            service=service,
            error_code=ErrorCode.EXTERNAL_SERVICE_TIMEOUT,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            details={"timeout_seconds": timeout_seconds} if timeout_seconds else None,
        )


# =============================================================================
# Configuration Exceptions
# =============================================================================

class ConfigurationError(AppError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        setting: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIG_INVALID,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={**(details or {}), "setting": setting} if setting else details,
            log_level="critical",
        )


class MissingSecretError(ConfigurationError):
    """Raised when required secret is missing."""

    def __init__(self, secret_name: str) -> None:
        super().__init__(
            message=f"Required secret not configured: {secret_name}",
            setting=secret_name,
            error_code=ErrorCode.CONFIG_MISSING_SECRET,
        )


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standardized error response format."""

    error: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    request_id: str = Field(..., description="Request ID for tracing")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error context")


class ErrorResponseList(BaseModel):
    """Container for multiple errors (validation errors)."""

    errors: list[ErrorResponse] = Field(..., description="List of validation errors")
    request_id: str = Field(..., description="Request ID for tracing")


# =============================================================================
# Exception Handlers
# =============================================================================

def get_request_id(request: Request) -> str:
    """Extract or generate request ID from request."""
    # Check for existing request ID in headers (from load balancer, etc.)
    request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID")
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    return request_id


def create_error_response(
    request: Request,
    exc: AppError,
    include_details: bool = True,
) -> JSONResponse:
    """Create standardized JSON error response."""
    request_id = get_request_id(request)
    exc.request_id = request_id

    # In production, sanitize details to avoid information leakage
    include_details_flag = include_details and settings.ENVIRONMENT != "production"

    log_data = {
        "request_id": request_id,
        "error_code": exc.error_code.value,
        "message": exc.message,
        "status_code": exc.status_code,
        "path": request.url.path,
        "method": request.method,
        **exc.details,
    }

    # Use the appropriate logger method based on log level
    if exc.log_level == "critical":
        logger.critical("Request error", **log_data)
    elif exc.log_level == "error":
        logger.error("Request error", **log_data)
    elif exc.log_level == "warning":
        logger.warning("Request error", **log_data)
    elif exc.log_level == "info":
        logger.info("Request error", **log_data)
    else:
        logger.debug("Request error", **log_data)

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(include_details=include_details_flag),
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle application-specific exceptions."""
    return create_error_response(request, exc)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPExceptions with standardized format."""
    request_id = get_request_id(request)

    # Map common HTTP status codes to error codes
    error_code_map = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.AUTH_INVALID_TOKEN,
        403: ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
        404: ErrorCode.RAG_NO_RESULTS,
        409: ErrorCode.DB_CONSTRAINT_VIOLATION,
        413: ErrorCode.DOCUMENT_TOO_LARGE,
        415: ErrorCode.DOCUMENT_UNSUPPORTED_TYPE,
        422: ErrorCode.VALIDATION_ERROR,
        429: ErrorCode.EXTERNAL_SERVICE_RATE_LIMITED,
        500: ErrorCode.INTERNAL_ERROR,
        502: ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
        503: ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
        504: ErrorCode.EXTERNAL_SERVICE_TIMEOUT,
    }

    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    # Sanitize detail in production
    detail = exc.detail
    if settings.ENVIRONMENT == "production" and isinstance(detail, str):
        # In production, use generic messages for server errors
        if exc.status_code >= 500:
            detail = "Internal server error"

    logger.warning(
        "HTTP exception",
        request_id=request_id,
        error_code=error_code.value,
        status_code=exc.status_code,
        detail=detail,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_code.value,
            "message": detail if isinstance(detail, str) else "Request failed",
            "request_id": request_id,
        },
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation errors."""
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return await unhandled_exception_handler(request, exc)

    request_id = get_request_id(request)

    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append(
            ErrorResponse(
                error=ErrorCode.VALIDATION_ERROR.value,
                message=error["msg"],
                request_id=request_id,
                details={"field": field, "type": error["type"]},
            )
        )

    logger.warning(
        "Validation error",
        request_id=request_id,
        error_count=len(errors),
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponseList(errors=errors, request_id=request_id).model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions - last resort."""
    request_id = get_request_id(request)

    # Safely encode error message to avoid Unicode issues
    try:
        error_message = str(exc)
    except Exception:
        error_message = repr(exc)

    # Log full exception with traceback for debugging
    try:
        logger.exception(
            "Unhandled exception",
            request_id=request_id,
            error_type=type(exc).__name__,
            error_message=error_message,
            path=request.url.path,
            method=request.method,
        )
    except Exception:
        # If logging fails, silently continue
        pass

    # In production, return generic error without details
    if settings.ENVIRONMENT == "production":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": ErrorCode.INTERNAL_ERROR.value,
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
        )

    # In development, include error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": ErrorCode.INTERNAL_ERROR.value,
            "message": error_message,
            "request_id": request_id,
            "details": {
                "type": type(exc).__name__,
                "traceback": None,  # Don't include traceback in response
            },
        },
    )


def register_exception_handlers(app) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Pydantic validation errors
    from pydantic import ValidationError
    app.add_exception_handler(ValidationError, validation_exception_handler)