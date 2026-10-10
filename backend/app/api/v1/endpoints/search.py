from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user, get_tenant_filters
from app.core.exceptions import (
    EmptyQueryError,
    RetrievalError,
    InvalidFiltersError,
    EmbeddingError,
)
from app.db.session import get_db_session
from app.models.user import UserResponse
from app.rag import RankedContext
from app.rag.embedder.factory import EmbedderFactory
from app.rag.retrieval.dense import VectorRetriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.sparse import KeywordRetriever, PostgresFTSStore
from app.rag.vectorstore.pgvector_store import PGVectorStore

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search string")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    filters: dict[str, Any] | None = Field(
        None, description="Metadata filters to apply (e.g., role, department)"
    )


@router.post("", response_model=RankedContext)
async def direct_search(  # type: ignore
    req: SearchRequest,
    request: Request,
    current_user: UserResponse = Depends(get_current_user),
    tenant_filters: dict = Depends(get_tenant_filters),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Perform a direct hybrid semantic search without LLM answer generation.
    Returns the exact document chunks and their confidence scores.
    """
    # Validate query
    if not req.query or not req.query.strip():
        raise EmptyQueryError()

    try:
        embedder = EmbedderFactory.get_embedder()
    except Exception as e:
        raise EmbeddingError(message="Failed to initialize embedder", details={"error": str(e)})

    try:
        vector_store = PGVectorStore(db)
        dense_retriever = VectorRetriever(embedder=embedder, vector_store=vector_store)

        keyword_store = PostgresFTSStore(db)
        sparse_retriever = KeywordRetriever(keyword_store=keyword_store)

        reranker = request.app.state.reranker
        retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, reranker=reranker
        )

        combined_filters = {
            "tenant_id": current_user.tenant_id,
            **(req.filters or {}),
            **tenant_filters,
        }
        return await retriever.retrieve(req.query, top_k=req.top_k, filters=combined_filters)
    except ValueError as e:
        raise InvalidFiltersError(message="Invalid filter parameters", details={"error": str(e)})
    except Exception as e:
        raise RetrievalError(message="Failed to perform search", retriever_type="hybrid", details={"error": str(e)})
