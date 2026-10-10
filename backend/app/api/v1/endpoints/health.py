import redis.asyncio as redis_async
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db_session

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]


@router.get("", response_model=HealthResponse)
async def check_health(db: AsyncSession = Depends(get_db_session)):  # type: ignore
    """
    Check the health of the API and backing services (DB, Vector Store, Redis).
    """
    components = {}
    is_healthy = True

    # Check DB (which also hosts pgvector)
    try:
        await db.execute(text("SELECT 1"))
        components["database"] = "ok"
        components["vector_store"] = "ok"
    except Exception:
        components["database"] = "down"
        components["vector_store"] = "down"
        is_healthy = False

    # Check Redis
    try:
        r = redis_async.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        components["redis"] = "ok"
    except Exception:
        components["redis"] = "down"
        is_healthy = False

    status_str = "ok" if is_healthy else "error"
    response = HealthResponse(status=status_str, components=components)

    if not is_healthy:
        return JSONResponse(status_code=503, content=response.model_dump())

    return response
