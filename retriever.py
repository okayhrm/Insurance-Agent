# retriever.py
"""
Async wrapper around PostgreSQL + pgvector (Postgres 16 “vector” extension).

Key features
------------
• Async connection pool (psycopg-pool) – plays nicely with FastAPI.
• Document-level caching:  documents table + chunks table.
• HNSW index on the embedding column for fast cosine-similarity search.
• Helper methods:
        - connect() / shutdown()
        - get_doc_id / insert_doc
        - insert_chunks
        - fetch_chunks     (texts only)
        - top_k            (doc-scoped cosine search)
"""

from __future__ import annotations
import os
from typing import List, Tuple

import psycopg_pool
from psycopg.rows import dict_row

# ── env / constants ─────────────────────────────────────────────
PG_URL      = os.getenv("PGVECTOR_URL")
VECTOR_DIM  = 768            # models/embedding-001 vectors


class AsyncPGVectorClient:
    def __init__(self) -> None:
        if not PG_URL:
            raise ValueError("PGVECTOR_URL env var is not set")
        self.pool = psycopg_pool.AsyncConnectionPool(PG_URL, open=False)

    # ── lifecycle ------------------------------------------------
    async def connect(self) -> None:
        await self.pool.open()
        await self._setup_schema()

    async def shutdown(self) -> None:
        await self.pool.close()

    async def _setup_schema(self) -> None:
        """Create extension, tables, and HNSW index (idempotent)."""
        async with self.pool.connection() as aconn:
            async with aconn.cursor(row_factory=dict_row) as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await cur.execute(
                    """CREATE TABLE IF NOT EXISTS documents (
                            id           SERIAL PRIMARY KEY,
                            hash         TEXT UNIQUE,
                            url          TEXT,
                            created_at   TIMESTAMPTZ DEFAULT NOW()
                        );"""
                )
                await cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS chunks (
                            id           SERIAL PRIMARY KEY,
                            doc_id       INT REFERENCES documents(id) ON DELETE CASCADE,
                            chunk_idx    INT,
                            text         TEXT,
                            embedding    VECTOR({VECTOR_DIM})
                        );"""
                )
                await cur.execute(
                    """
                    DO $$
                    BEGIN
                      IF NOT EXISTS (
                          SELECT 1 FROM pg_class WHERE relname = 'chunks_vec_idx'
                      ) THEN
                          CREATE INDEX chunks_vec_idx
                          ON chunks USING hnsw (embedding vector_cosine_ops);
                      END IF;
                    END $$;
                    """
                )

    # ── caching helpers -----------------------------------------
    async def get_doc_id(self, doc_hash: str) -> int | None:
        async with self.pool.connection() as aconn:
            async with aconn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT id FROM documents WHERE hash=%s;", (doc_hash,))
                row = await cur.fetchone()
                return row["id"] if row else None

    async def insert_doc(self, doc_hash: str, url: str) -> int:
        async with self.pool.connection() as aconn:
            async with aconn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "INSERT INTO documents(hash,url) VALUES (%s,%s) RETURNING id;",
                    (doc_hash, url),
                )
                return (await cur.fetchone())["id"]

    async def insert_chunks(
        self,
        doc_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Bulk-insert chunk texts + vectors for a single document."""
        args = [
            (doc_id, idx, text, vec)
            for idx, (text, vec) in enumerate(zip(chunks, embeddings))
        ]
        async with self.pool.connection() as aconn:
            async with aconn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks(doc_id,chunk_idx,text,embedding) VALUES (%s,%s,%s,%s);",
                    args,
                )

    async def fetch_chunks(self, doc_id: int) -> List[str]:
        """Return chunk texts for a cached document (no embeddings)."""
        async with self.pool.connection() as aconn:
            async with aconn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT text FROM chunks WHERE doc_id=%s;", (doc_id,))
                return [row["text"] for row in await cur.fetchall()]

    # ── retrieval ------------------------------------------------
    async def top_k(
        self, doc_id: int, query_vec: List[float], k: int = 4
    ) -> List[str]:
        """
        HNSW cosine-similarity search limited to the given document.
        Returns top-k chunk texts.
        """
        async with self.pool.connection() as aconn:
            async with aconn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SET LOCAL hnsw.ef_search = 100;")
                await cur.execute(
                    "SELECT text "
                    "FROM chunks "
                    "WHERE doc_id = %s "
                    "ORDER BY embedding <=> %s::vector ASC "
                    "LIMIT %s;",
                    (doc_id, query_vec, k),
                )
                return [row["text"] for row in await cur.fetchall()]