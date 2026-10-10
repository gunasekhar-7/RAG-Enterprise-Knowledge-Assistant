import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.dependencies.auth import get_current_user, get_tenant_filters
from app.core.exceptions import (
    SecurityViolationError,
    EmptyQueryError,
    RetrievalError,
    LLMError,
    DatabaseError,
)
from app.models.user import UserResponse
from app.rag.guardrails.input_validator import InputValidator  # noqa: E402

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input query.")


class ConversationResponse(BaseModel):
    conversation_id: uuid.UUID
    created_at: datetime


from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.future import select  # noqa: E402

from app.db.session import get_db_session  # noqa: E402
from app.models.db import ConversationModel, MessageModel  # noqa: E402


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(  # type: ignore
    current_user: UserResponse = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db_session),
):
    """
    Start a new chat session.
    """
    new_conv = ConversationModel(
        tenant_id=current_user.tenant_id, user_id=current_user.id, title="New Chat"
    )
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    return ConversationResponse(conversation_id=new_conv.id, created_at=new_conv.created_at)  # type: ignore


from app.models.chat import Message  # noqa: E402
from app.rag.embedder.factory import EmbedderFactory  # noqa: E402
from app.rag.generator.base import GenerationResult  # noqa: E402
from app.rag.generator.context_manager import ContextFormatter  # noqa: E402
from app.rag.generator.litellm_generator import LiteLLMGenerator  # noqa: E402
from app.rag.generator.pipeline import GenerationPipeline  # noqa: E402
from app.rag.generator.prompt import PromptBuilder  # noqa: E402
from app.rag.generator.query_rewriter import QueryRewriter  # noqa: E402
from app.rag.retrieval.dense import VectorRetriever  # noqa: E402
from app.rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.rag.retrieval.sparse import KeywordRetriever, PostgresFTSStore  # noqa: E402
from app.rag.vectorstore.pgvector_store import PGVectorStore  # noqa: E402


@router.post("/conversations/{conversation_id}/messages", response_model=GenerationResult)
async def send_message(  # type: ignore
    conversation_id: uuid.UUID,
    req: ChatRequest,
    request: Request,
    current_user: UserResponse = Depends(get_current_user),  # noqa: ARG001
    tenant_filters: dict = Depends(get_tenant_filters),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Send a message to an existing conversation.
    Executes the full RAG pipeline (Retrieval + Generation) with multi-turn memory.
    """
    # 1. Validate input - Security guardrails
    if not req.message or not req.message.strip():
        raise EmptyQueryError()

    try:
        InputValidator().validate(req.message)
    except SecurityViolationError:
        raise
    except Exception as e:
        raise SecurityViolationError(message=str(e), violation_type="validation")

    # 2. Fetch History (last 4 messages for context window bounds)
    try:
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.asc())
        )
        result = await db.execute(stmt)
        db_messages = result.scalars().all()
    except Exception as e:
        raise DatabaseError(message="Failed to fetch conversation history", operation="select_messages", details={"error": str(e)})

    history = [
        Message(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
        )
        for m in db_messages
    ]
    recent_history = history[-4:]

    # 3. Rewrite Query (if history exists)
    rewriter = QueryRewriter()
    standalone_query = await rewriter.rewrite(recent_history, req.message)

    # 4. Retrieve Context using Standalone Query
    try:
        embedder = EmbedderFactory.get_embedder()
        vector_store = PGVectorStore(db)
        dense_retriever = VectorRetriever(embedder=embedder, vector_store=vector_store)

        keyword_store = PostgresFTSStore(db)
        sparse_retriever = KeywordRetriever(keyword_store=keyword_store)

        reranker = request.app.state.reranker
        retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, reranker=reranker
        )

        combined_filters = {"tenant_id": current_user.tenant_id, **(tenant_filters or {})}
        context = await retriever.retrieve(standalone_query, filters=combined_filters)
    except Exception as e:
        raise RetrievalError(message="Failed to retrieve relevant documents", retriever_type="hybrid", details={"error": str(e)})

    # 5. Generate Answer using History + Original Query
    try:
        llm = LiteLLMGenerator()
        context_formatter = ContextFormatter()
        prompt_builder = PromptBuilder()
        pipeline = GenerationPipeline(
            llm=llm, context_formatter=context_formatter, prompt_builder=prompt_builder
        )

        history_dicts = [{"role": m.role, "content": m.content} for m in recent_history]
        result = await pipeline.generate_answer(
            req.message, context.results, conversation_history=history_dicts
        )
    except Exception as e:
        raise LLMError(message="Failed to generate answer", model=llm.model_name, retryable=True, details={"error": str(e)})

    # 6. Save interaction to DB
    try:
        user_msg = MessageModel(conversation_id=conversation_id, role="user", content=req.message)
        assistant_msg = MessageModel(
            conversation_id=conversation_id, role="assistant", content=result.answer
        )
        db.add_all([user_msg, assistant_msg])
        await db.commit()
    except Exception as e:
        raise DatabaseError(message="Failed to save conversation messages", operation="commit", details={"error": str(e)})

    return result


@router.get("/conversations/{conversation_id}")
async def get_conversation_history(  # type: ignore
    conversation_id: uuid.UUID,
    current_user: UserResponse = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retrieve past messages in a conversation.
    """
    stmt = (
        select(MessageModel)
        .where(MessageModel.conversation_id == conversation_id)
        .order_by(MessageModel.created_at.asc())
    )
    result = await db.execute(stmt)
    db_messages = result.scalars().all()
    history = [
        Message(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
        )
        for m in db_messages
    ]
    return {"messages": history}
