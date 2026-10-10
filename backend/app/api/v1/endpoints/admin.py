import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import RequireRole
from app.db.session import get_db_session
from app.models.db import ConversationModel, DocumentModel, MessageModel
from app.models.user import Role, UserResponse

router = APIRouter()


class StatsResponse(BaseModel):
    total_documents: int
    total_messages: int
    active_conversations: int
    total_tokens: int


class ReindexResponse(BaseModel):
    job_id: uuid.UUID
    status: str


@router.post("/reindex", response_model=ReindexResponse)
async def trigger_reindex(current_user: UserResponse = Depends(RequireRole(Role.ADMINISTRATOR))):  # type: ignore  # noqa: ARG001
    """
    Trigger a full re-embedding of all documents.
    Only accessible by users with the Administrator role.
    """
    # This would kick off a background Celery/Redis-Queue job
    return ReindexResponse(job_id=uuid.uuid4(), status="started")


@router.get("/stats", response_model=StatsResponse)
async def get_dashboard_stats(  # type: ignore
    current_user: UserResponse = Depends(RequireRole(Role.ADMINISTRATOR)),
    db: AsyncSession = Depends(get_db_session),
) -> StatsResponse:
    """
    Get tenant-specific aggregate statistics for the dashboard.
    """
    tenant_id = current_user.tenant_id

    # Total Documents
    doc_count_stmt = select(func.count(DocumentModel.id)).where(
        DocumentModel.tenant_id == tenant_id
    )
    doc_count = await db.scalar(doc_count_stmt) or 0

    # Total Tokens Ingested
    token_sum_stmt = select(func.sum(DocumentModel.token_count)).where(  # type: ignore
        DocumentModel.tenant_id == tenant_id
    )
    token_sum = await db.scalar(token_sum_stmt) or 0

    # Active Conversations
    conv_count_stmt = select(func.count(ConversationModel.id)).where(
        ConversationModel.tenant_id == tenant_id
    )
    conv_count = await db.scalar(conv_count_stmt) or 0

    # Total Messages
    # Join MessageModel and ConversationModel to filter by tenant_id
    msg_count_stmt = (
        select(func.count(MessageModel.id))
        .join(ConversationModel, MessageModel.conversation_id == ConversationModel.id)
        .where(ConversationModel.tenant_id == tenant_id)
    )
    msg_count = await db.scalar(msg_count_stmt) or 0

    return StatsResponse(
        total_documents=doc_count,
        total_messages=msg_count,
        active_conversations=conv_count,
        total_tokens=token_sum,
    )
