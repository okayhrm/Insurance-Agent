# embedder.py
"""
Async wrapper around Google Generative AI embeddings.

• Splits large inputs into token-aware batches (to respect API limits).
• Runs blocking SDK calls in a threadpool so FastAPI stays responsive.
• Exposes two helpers:
      embed_texts(...)    -> list[list[float]]
      embed_query(...)    -> same, convenience for questions
"""

from __future__ import annotations
import asyncio
import os
from typing import List, Iterable

import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from tiktoken import get_encoding
from collections import deque

# ── Configuration ───────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use the correct model name for embedding with the API
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# The model's context window. Adjust this based on documentation.
MAX_TOKENS_PER_BATCH = int(os.getenv("EMBED_MAX_TOKENS_PER_BATCH", 8192))
_TOKENIZER = get_encoding("cl100k_base")

_POOL: ThreadPoolExecutor | None = None


def _get_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=1)
    return _POOL


# ── Core embedding helpers ──────────────────────────────────────
def _embed_sync(batch: List[str]) -> List[List[float]]:
    """Blocking call to Gemini Embedding endpoint (runs in threadpool)."""
    # Use the correct function for embedding content
    result = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=batch,
        task_type="retrieval_document"
    )
    return result['embedding']


async def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """
    Asynchronously embed a list of strings.
    Splits into token-aware batches to respect model limits.
    """
    if not isinstance(texts, list):
        texts = list(texts)
    
    if not texts:
        return []

    loop = asyncio.get_running_loop()
    tasks = []
    
    text_queue = deque(texts)
    
    while text_queue:
        current_batch_texts = []
        current_batch_tokens = 0
        
        while text_queue:
            next_text = text_queue[0]
            num_tokens = len(_TOKENIZER.encode(next_text))
            
            if current_batch_tokens + num_tokens <= MAX_TOKENS_PER_BATCH:
                current_batch_tokens += num_tokens
                current_batch_texts.append(text_queue.popleft())
            else:
                if not current_batch_texts:
                    current_batch_texts.append(text_queue.popleft())
                
                break
        
        tasks.append(loop.run_in_executor(_get_pool(), _embed_sync, current_batch_texts))

    results: List[List[List[float]]] = await asyncio.gather(*tasks)
    return [vec for batch in results for vec in batch]


embed_query = embed_texts