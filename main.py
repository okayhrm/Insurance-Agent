# main.py
"""
FastAPI entry-point for the HackRx webhook.

Pipeline (all-async):
1.  Bearer-token auth (header -> .env AUTH_TOKEN)
2.  SHA-256 hash of PDF URL (cache key)
3.  If unseen -> async download -> Unstructured parse -> embed -> store
4.  Embed all questions in one batch
5.  For each question (concurrently):
      - HNSW cosine search (doc-scoped)
      - Gemini answer (strict-JSON prompt)
6.  Return {"answers": [...]}

Dependencies:
  - parser.py      (async PDF -> chunks)
  - embedder.py    (async embeddings)
  - retriever.py   (AsyncPGVectorClient, HNSW)
  - llm.py         (ask: prompt -> Gemini -> dict)
  - utils.py       (verify_bearer_token)
"""

from __future__ import annotations
import asyncio
import hashlib
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, NoReturn

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import httpx # Used for async parser errors
import psycopg # Used for async retriever errors
import google.api_core.exceptions as api_exceptions

from models.schema import RunRequest, RunResponse
from utils import verify_bearer_token
from parser import parse_pdf_chunks
from embedder import embed_texts
from retriever import AsyncPGVectorClient
from llm import ask

# ── setup ───────────────────────────────────────────────────────
load_dotenv() # Uvicorn's --env-file flag is used for reliability
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TOP_K      = int(os.getenv("TOP_K", 8))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── global db client (set in lifespan) ──────────────────────────
db_client: AsyncPGVectorClient | None = None


# ── lifespan: open / close async PG pool ────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initializes and cleans up the async DB connection pool."""
    global db_client
    try:
        db_client = AsyncPGVectorClient()
        await db_client.connect()
        logger.info("✅ DB pool ready")
        yield
    except (psycopg.OperationalError, ValueError) as e:
        logger.critical(f"Fatal error during DB startup: {e}")
        raise RuntimeError("Database connection failed on startup.") from e
    finally:
        if db_client:
            await db_client.shutdown()
            logger.info("🛑 DB pool closed")


app = FastAPI(
    title="HackRx RAG Webhook", 
    docs_url="/docs",
    redoc_url=None, # Disable ReDoc
    lifespan=lifespan
)


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="HackRx RAG Webhook Documentation"
    )

def _unauth() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized"
    )

# ── core endpoint --------------------------------------------------------
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def hackrx_run(
    payload: RunRequest,
    authorization: str = Header(..., description="Bearer <token>"),
):
    # 1. auth check -------------------------------------------------------
    if not verify_bearer_token(authorization, AUTH_TOKEN):
        _unauth()

    if db_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DB not ready"
        )

    t0 = time.perf_counter()
    
    # 2. cache key --------------------------------------------------------
    doc_url = payload.document_url.unicode_string()
    doc_hash = hashlib.sha256(doc_url.encode()).hexdigest()
    
    # Placeholder for chunks to be used in step 5
    chunks: List[str] = []

    try:
        # 3. get or ingest document ------------------------------------------
        doc_id = await db_client.get_doc_id(doc_hash)
        if doc_id is None:
            logger.info(f"PDF {doc_url[:50]}... not in cache. Starting ingestion.")
            chunks = await parse_pdf_chunks(doc_url)
            vectors = await embed_texts(chunks)
            doc_id = await db_client.insert_doc(doc_hash, doc_url)
            await db_client.insert_chunks(doc_id, chunks, vectors)
            logger.info(f"PDF ingested and stored with doc_id: {doc_id}.")
        else:
            logger.info(f"PDF {doc_url[:50]}... found in cache. Using doc_id: {doc_id}.")
            chunks = await db_client.fetch_chunks(doc_id)

        # 4. embed all questions at once -------------------------------------
        q_vectors = await embed_texts(payload.questions)

    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.error(f"PDF download or parsing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download or parse PDF: {e}"
        ) from e
    except (psycopg.Error, api_exceptions.GoogleAPIError) as e:
        logger.error(f"Database or embedding service error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream service error during ingestion/embedding."
        ) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred."
        ) from e

    # 5. process each question concurrently ------------------------------
    async def answer_one(question: str, q_vec: List[float]) -> str:
        try:
            ctx_chunks = await db_client.top_k(doc_id, q_vec, k=TOP_K)
            llm_json = await ask(question, ctx_chunks)
            return llm_json.get("answer", "")
        except api_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API error for question '{question}': {e}", exc_info=True)
            return "An error occurred while generating the answer."
        except psycopg.Error as e:
            logger.error(f"Database error for question '{question}': {e}", exc_info=True)
            return "An error occurred while retrieving information."
        except Exception as e:
            logger.error(f"An unexpected error occurred answering '{question}': {e}", exc_info=True)
            return "An unexpected error occurred."

    coros = [answer_one(q, v) for q, v in zip(payload.questions, q_vectors)]
    answers = await asyncio.gather(*coros)

    total_time_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"⏱ total {total_time_ms:.0f} ms for {len(payload.questions)} questions.")
    
    return RunResponse(answers=answers)

# In main.py, add logging:
import logging
logging.basicConfig(level=logging.INFO)

# This will show you:
# - What chunks are being found
# - What the LLM is actually receiving
# - What the raw LLM response looks like/q
