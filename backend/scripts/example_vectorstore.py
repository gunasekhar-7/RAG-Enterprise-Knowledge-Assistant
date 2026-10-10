import asyncio
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.rag.models import ChunkMetadata, DocumentChunk
from app.rag.vectorstore import PGVectorStore


async def run_example():
    print("--- Vector Store Example Usage ---")

    # 1. Initialize DB Connection (normally done by dependency injection in FastAPI)
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_knowledge_db"
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)

    # 2. Setup schema and indexes
    async with AsyncSessionLocal() as session:
        store = PGVectorStore(session=session)
        print("1. Creating collection (HNSW index & tables)...")
        try:
            await store.create_collection()
            print("   -> Success")
        except Exception as e:
            print(f"   -> Failed (Ensure postgres is running): {e}")
            return

        # 3. Create dummy data
        print("\n2. Creating dummy document chunks...")
        doc_id = uuid4()
        chunk1 = DocumentChunk(
            chunk_id=f"{doc_id}_c0001",
            document_id=doc_id,
            text="Generative AI policies require that no customer data is used for training.",
            embedding=[0.01] * 1536,  # Dummy 1536-dimensional vector
            metadata=ChunkMetadata(
                document_id=doc_id,
                chunk_index=1,
                page_number=1,
                filename="AI_Policy.pdf",
                token_count=14,
                char_count=74,
                custom_metadata={"department": "Security"},
            ),
        )

        # 4. Insert data
        print("3. Inserting chunk into pgvector...")
        await store.insert(chunk1)
        print("   -> Inserted successfully.")

        # 5. Search
        print("\n4. Performing similarity search with metadata filtering...")
        query_vector = [0.01] * 1536
        filters = {"filename": "AI_Policy.pdf"}

        results = await store.similarity_search(query_vector, top_k=5, filters=filters)
        for rank, result in enumerate(results, 1):
            print(f"   [{rank}] Score: {result.score:.4f} | Text: '{result.chunk.text}'")

        # 6. Cleanup
        print("\n5. Deleting document from vector store...")
        await store.delete(doc_id)
        print("   -> Deleted successfully.")


if __name__ == "__main__":
    asyncio.run(run_example())
