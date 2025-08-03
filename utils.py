# utils.py
import hashlib
import time
from typing import List, Iterable, Tuple
from functools import wraps

# ────────────────────────────────────────────────────────────────
# 1. Hash helper
# ────────────────────────────────────────────────────────────────
def md5_hash(text: str) -> str:
    """Deterministic fingerprint for document URLs or raw bytes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# ────────────────────────────────────────────────────────────────
# 2. Lightweight token counter
# ────────────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  # Fixed: was *enc
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    # Fallback: rough 1-token ≈ 4 characters heuristic
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

# ────────────────────────────────────────────────────────────────
# 3. Token-aware chunker
# ────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    max_tokens: int = 1200,
    overlap: int = 150,
) -> List[str]:
    """
    Split a long string into overlapping chunks of ≈ max_tokens.
    Overlap helps RAG maintain context at boundaries.
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    
    words: List[str] = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    
    for word in words:
        token_len = count_tokens(word + " ")
        
        if current_tokens + token_len > max_tokens:
            # flush chunk
            if current:  # Only add non-empty chunks
                chunks.append(" ".join(current).strip())
            
            # start new chunk with overlap
            overlap_words = current[-overlap:] if overlap and len(current) > overlap else current
            current = overlap_words.copy()
            current_tokens = count_tokens(" ".join(current)) if current else 0
        
        current.append(word)
        current_tokens += token_len
    
    # Add final chunk
    if current:
        chunks.append(" ".join(current).strip())
    
    return [c for c in chunks if c.strip()]  # Filter empty chunks

# ────────────────────────────────────────────────────────────────
# 4. Simple stopwatch decorator (debug / profiling)
# ────────────────────────────────────────────────────────────────
def timeit(fn):
    """Decorator to print function runtime (optional DEBUG use)."""
    @wraps(fn)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        dt_ms = (time.perf_counter() - start) * 1000
        print(f"[⏱  {fn.__name__}] {dt_ms:.1f} ms")
        return result
    return wrapper

# ────────────────────────────────────────────────────────────────
# 5. Bearer token validation helper
# ────────────────────────────────────────────────────────────────
def verify_bearer_token(authorization: str, expected_token: str) -> bool:
    """Verify Bearer token format and value."""
    if not authorization.startswith("Bearer "):
        return False
    token = authorization[7:]  # Remove "Bearer " prefix
    return token == expected_token