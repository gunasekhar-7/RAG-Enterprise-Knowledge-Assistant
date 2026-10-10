import asyncio
import time
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.rag.embedder.factory import EmbedderFactory
from app.rag.models import ChunkMetadata, DocumentChunk
from app.rag.retrieval.dense import VectorRetriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.reranker import FlashRankReranker
from app.rag.retrieval.sparse import KeywordRetriever, PostgresFTSStore
from app.rag.vectorstore.pgvector_store import PGVectorStore


async def run_benchmark():
    print("--- Hybrid Retrieval Benchmark ---")

    # Setup dummy database & embedder
    embedder = EmbedderFactory.get_embedder()
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_knowledge_db", echo=False
    )
    AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)

    # Create test chunks
    doc_id = uuid4()
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_c0001",
            document_id=doc_id,
            text="The internal network password is 'AlphaBeta99'.",
            embedding=embedder.embed_text("The internal network password is 'AlphaBeta99'."),
            metadata=ChunkMetadata(
                document_id=doc_id,
                chunk_index=1,
                page_number=1,
                filename="IT_Sec.pdf",
                token_count=10,
                char_count=50,
            ),
        ),
        DocumentChunk(
            chunk_id=f"{doc_id}_c0002",
            document_id=doc_id,
            text="All network security policies must be adhered to at all times to prevent data leaks.",
            embedding=embedder.embed_text(
                "All network security policies must be adhered to at all times to prevent data leaks."
            ),
            metadata=ChunkMetadata(
                document_id=doc_id,
                chunk_index=2,
                page_number=1,
                filename="IT_Sec.pdf",
                token_count=10,
                char_count=50,
            ),
        ),
    ]

    async with AsyncSessionLocal() as session:
        # Setup vector store
        vector_store = PGVectorStore(session=session)
        await vector_store.create_collection()
        await vector_store.batch_insert(chunks)

        # Setup FTS store
        bm25_store = PostgresFTSStore(session)

        # Instantiate Retrievers
        dense_retriever = VectorRetriever(embedder, vector_store)
        sparse_retriever = KeywordRetriever(bm25_store)
        reranker = FlashRankReranker()

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, reranker=None
        )

        hybrid_rerank_retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, reranker=reranker
        )

        # Query where dense search might fail but BM25 succeeds
        query = "AlphaBeta99"
        print(f"\n[Query]: {query}")

        strategies = {
            "Vector-Only": dense_retriever,
            "Keyword-Only": sparse_retriever,
            "Hybrid (No Rerank)": hybrid_retriever,
            "Hybrid + Reranking": hybrid_rerank_retriever,
        }

        for name, ret in strategies.items():
            start_t = time.perf_counter()
            res = await ret.retrieve(query)
            elapsed = (time.perf_counter() - start_t) * 1000

            top_text = res.results[0].text if res.total_found > 0 else "None"
            print(f"  [{name}] ({elapsed:.1f}ms) -> Top match: '{top_text}'")

        # Cleanup
        await vector_store.delete(doc_id)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
