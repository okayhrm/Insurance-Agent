# parser.py
"""
Downloads a PDF from a public URL, extracts clean text with
Unstructured, and returns token-aware chunks for RAG.
"""

from __future__ import annotations
import asyncio
import io
import os
from typing import List

import httpx
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace
from utils import chunk_text, timeit

PDF_TIMEOUT = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", 20))


async def download_pdf(url: str) -> bytes:
    """Fetch PDF bytes asynchronously with a timeout."""
    async with httpx.AsyncClient(timeout=PDF_TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _parse_pdf_sync(pdf_bytes: bytes, max_tokens: int) -> List[str]:
    """
    Blocking parsing logic that runs in a separate thread.
    """
    elements = partition_pdf(
        file=io.BytesIO(pdf_bytes),
        strategy="fast",
    )

    full_text = "\n".join(
        clean_extra_whitespace(el.text)
        for el in elements
        if getattr(el, "text", None)
    ).strip()

    if not full_text:
        raise ValueError("Empty text extracted from PDF – check file contents.")

    return chunk_text(full_text, max_tokens=max_tokens, overlap=150)


async def parse_pdf_chunks(url: str, max_tokens: int =1200) -> List[str]:
    """
    1. Download PDF asynchronously
    2. Run blocking text extraction in a thread
    3. Return chunks
    """
    pdf_bytes = await download_pdf(url)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _parse_pdf_sync, pdf_bytes, max_tokens)