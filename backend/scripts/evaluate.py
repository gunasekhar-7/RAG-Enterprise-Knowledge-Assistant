import asyncio
import json
import time
import uuid
from pathlib import Path

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from app.db.session import AsyncSessionLocal  # noqa: E402
from app.ingestion.pipeline import DocumentIngestionPipeline  # noqa: E402
from app.rag.chunker import DocumentChunker  # noqa: E402
from app.rag.embedder.factory import EmbedderFactory  # noqa: E402
from app.rag.retrieval.dense import VectorRetriever  # noqa: E402
from app.rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.rag.retrieval.reranker import FlashRankReranker  # noqa: E402
from app.rag.retrieval.sparse import KeywordRetriever, PostgresFTSStore  # noqa: E402
from app.rag.vectorstore.pgvector_store import PGVectorStore  # noqa: E402

logger = structlog.get_logger(__name__)


class EvalResult(BaseModel):
    id: str
    architecture: str
    precision: float
    recall: float
    hit_at_k: bool
    mrr: float
    answer_correctness: int = 0
    faithfulness: int = 0
    latency_ms: float
    token_usage: int


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str], ground_truth_ids: list[str], k: int = 5
):
    """Deterministic retrieval metrics calculation."""
    if not ground_truth_ids:
        # If there are no expected chunks (negative query), we should hopefully retrieve nothing
        # or at least the generated answer should be 'Insufficient information'.
        # For retrieval metrics on negative queries, hit_at_k is false.
        return 0.0, 0.0, False, 0.0

    retrieved_k = retrieved_chunk_ids[:k]

    # Precision
    relevant_retrieved = set(retrieved_k).intersection(set(ground_truth_ids))
    precision = len(relevant_retrieved) / len(retrieved_k) if retrieved_k else 0.0

    # Recall
    recall = len(relevant_retrieved) / len(ground_truth_ids)

    # Hit@K
    hit_at_k = len(relevant_retrieved) > 0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, r_id in enumerate(retrieved_k):
        if r_id in ground_truth_ids:
            mrr = 1.0 / (i + 1)
            break

    return precision, recall, hit_at_k, mrr


async def llm_as_a_judge(
    question: str, generated_answer: str, ground_truth: str, context: str
) -> dict[str, int]:
    """Uses Claude 3.5 Sonnet to judge answer correctness and faithfulness on a 1-5 scale."""
    import litellm

    prompt = f"""
You are an expert evaluator for an enterprise search system.
Evaluate the following generated answer against the ground truth and the provided context.

Question: {question}
Context Used: {context}
Generated Answer: {generated_answer}
Ground Truth: {ground_truth}

Output a JSON object with two keys:
"correctness": A score from 1-5 comparing the generated answer to the ground truth.
"faithfulness": A score from 1-5 assessing if the generated answer is entirely derived from the context (no hallucination).
Do not output anything else other than valid JSON.
"""
    try:
        response = await litellm.acompletion(
            model="claude-3-5-sonnet-20240620",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content

        # Simple extraction in case the model adds markdown formatting
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())
    except Exception as e:
        logger.error(f"LLM Judge failed: {e}")
        return {"correctness": 0, "faithfulness": 0}


async def setup_environment():
    """Ingest sample documents to prepare the environment."""
    logger.info("Setting up evaluation environment...")

    tenant_id = uuid.uuid4()
    docs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample_documents"

    pipeline = DocumentIngestionPipeline()
    chunker = DocumentChunker()
    embedder = EmbedderFactory.get_embedder()

    all_chunks = []

    async with AsyncSessionLocal() as db:
        store = PGVectorStore(db)
        await store.create_collection()
        bm25_store = PostgresFTSStore(db)
        # Clear previous data for tenant
        # (Assuming fresh db for evaluation)

        for file_path in docs_dir.glob("*.*"):
            logger.info(f"Processing {file_path.name}")
            content = file_path.read_bytes()
            try:
                parsed_doc = await pipeline.process(
                    filename=file_path.name,
                    content=content,
                    tenant_id=str(tenant_id),
                    custom_metadata={"tenant_id": str(tenant_id)},
                )
                chunks = chunker.chunk_document(parsed_doc)

                # Assign specific chunk IDs matching our dataset format (doc_name_c000X)
                # Since we don't have absolute control over the chunker's output IDs in this pipeline,
                # we'll forcibly overwrite chunk IDs based on their order so they match the dataset.
                doc_prefix = f"doc_{file_path.stem.lower()}"
                for i, c in enumerate(chunks):
                    c.chunk_id = f"{doc_prefix}_c{i:04d}"

                embeddings = embedder.embed_batch([c.text for c in chunks])
                for c, emb in zip(chunks, embeddings, strict=False):
                    c.embedding = emb
                await store.batch_insert(chunks)
                await bm25_store.add_documents(chunks)
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        # Initialize retrievers
        dense_retriever = VectorRetriever(vector_store=store, embedder=embedder)
        sparse_retriever = KeywordRetriever(keyword_store=bm25_store)
        reranker = FlashRankReranker()

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, reranker=None
        )

        hybrid_rerank_retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, reranker=reranker
        )

        return {
            "tenant_id": tenant_id,
            "Dense": dense_retriever,
            "Sparse": sparse_retriever,
            "Hybrid": hybrid_retriever,
            "Hybrid+Reranker": hybrid_rerank_retriever,
        }


async def evaluate_dataset(dataset_path: str, architecture: str, retriever, tenant_id: uuid.UUID):
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []

    for item in dataset:
        logger.info("Evaluating item", id=item["id"], question=item["question"], arch=architecture)

        start_time = time.time()

        # 1. LIVE RETRIEVAL STEP
        filters = {"tenant_id": tenant_id}
        retrieved = await retriever.retrieve(item["question"], top_k=5, filters=filters)
        retrieved_chunk_ids = [c.chunk_id for c in retrieved.results]

        # 2. DETERMINISTIC METRICS
        precision, recall, hit_at_k, mrr = compute_retrieval_metrics(
            retrieved_chunk_ids=retrieved_chunk_ids,
            ground_truth_ids=item["relevant_chunk_ids"],
            k=5,
        )

        # 3. GENERATION STEP
        context = "\n\n".join([c.text for c in retrieved.results])
        from app.rag.generator.context_manager import ContextFormatter
        from app.rag.generator.litellm_generator import LiteLLMGenerator
        from app.rag.generator.pipeline import GenerationPipeline
        from app.rag.generator.prompt import PromptBuilder

        try:
            llm = LiteLLMGenerator()
            pipeline = GenerationPipeline(
                llm=llm, context_formatter=ContextFormatter(), prompt_builder=PromptBuilder()
            )
            gen_res = await pipeline.generate_answer(item["question"], retrieved.results)
            generated_answer = gen_res.answer
            token_usage = 0  # Not exposed in standard schema without deeper changes
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            generated_answer = "Error generating answer"
            token_usage = 0

        # 4. LLM AS A JUDGE
        if item["negative_query"]:
            ground_truth = "Insufficient information"
        else:
            ground_truth = item["expected_answer"]

        judge_scores = await llm_as_a_judge(
            item["question"], generated_answer, ground_truth, context
        )

        end_time = time.time()

        res = EvalResult(
            id=item["id"],
            architecture=architecture,
            precision=precision,
            recall=recall,
            hit_at_k=hit_at_k,
            mrr=mrr,
            answer_correctness=judge_scores.get("correctness", 0),
            faithfulness=judge_scores.get("faithfulness", 0),
            latency_ms=(end_time - start_time) * 1000,
            token_usage=token_usage,
        )
        results.append(res)

    return results


async def main(dataset_file: str):
    logger.info("Starting Evaluation Pipeline")

    env = await setup_environment()
    tenant_id = env["tenant_id"]

    all_results = []

    architectures = ["Dense", "Sparse", "Hybrid", "Hybrid+Reranker"]

    for arch in architectures:
        retriever = env[arch]
        arch_results = await evaluate_dataset(dataset_file, arch, retriever, tenant_id)
        all_results.extend(arch_results)

    # Export results to JSON
    out_file = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump([r.dict() for r in all_results], f, indent=2)

    # Generate Markdown Table
    print("\n# Evaluation Results\n")
    print(
        "| Architecture | Hit@5 | Precision@5 | Recall@5 | MRR | Faithfulness | Correctness | Latency (ms) |"
    )
    print(
        "|--------------|-------|-------------|----------|-----|--------------|-------------|--------------|"
    )

    for arch in architectures:
        arch_res = [r for r in all_results if r.architecture == arch]
        n = len(arch_res)
        if n == 0:
            continue

        avg_hit = sum(1 for r in arch_res if r.hit_at_k) / n
        avg_prec = sum(r.precision for r in arch_res) / n
        avg_recall = sum(r.recall for r in arch_res) / n
        avg_mrr = sum(r.mrr for r in arch_res) / n
        avg_faith = sum(r.faithfulness for r in arch_res) / n
        avg_corr = sum(r.answer_correctness for r in arch_res) / n
        avg_lat = sum(r.latency_ms for r in arch_res) / n

        print(
            f"| {arch} | {avg_hit:.2%} | {avg_prec:.2f} | {avg_recall:.2f} | {avg_mrr:.2f} | {avg_faith:.2f}/5 | {avg_corr:.2f}/5 | {avg_lat:.0f} |"
        )

    print(f"\nEvaluation pipeline successfully executed. Results saved to {out_file.resolve()}")


if __name__ == "__main__":
    import sys

    default_dataset = (
        Path(__file__).resolve().parent.parent.parent / "data" / "golden_evaluation_dataset.json"
    )
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else str(default_dataset)
    asyncio.run(main(dataset_file))
