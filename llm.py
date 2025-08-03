# llm.py
"""
Lightweight wrapper around Gemini (Generative AI SDK) for answer generation.
Returns a dict with keys: answer, justification, clause_reference, confidence.
"""

from __future__ import annotations
import json
import os
import asyncio
from typing import List
import logging

import google.generativeai as genai
import google.api_core.exceptions as api_exceptions

logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = genai.GenerativeModel("gemini-2.5-flash")  # Updated model

SYSTEM_PROMPT = """You are an expert insurance policy analyst. Your job is to answer questions about insurance policies based on provided document excerpts.

IMPORTANT GUIDELINES:
1. Always try to provide a helpful answer based on the context
2. If you find relevant information, give a direct, specific answer
3. Only say information is "not provided" if you truly cannot find anything relevant
4. Use exact quotes and references from the policy text
5. Be confident when the information is clearly stated
6. Respond in JSON format only

Your response must be valid JSON with these exact keys:
- "answer": Direct answer to the question
- "justification": Brief explanation with evidence from context  
- "clause_reference": Exact text/section that supports your answer
- "decision": One of: "covered", "not covered", "waiting period", "unclear"
- "confidence": Number between 0.0 and 1.0"""

def build_prompt(question: str, chunks: List[str]) -> str:
    # Combine chunks with clear separation
    context_block = "\n\n--- POLICY SECTION ---\n".join(chunks)
    
    return f"""{SYSTEM_PROMPT}

POLICY CONTEXT:
{context_block}

QUESTION: {question}

Provide your answer as valid JSON:"""

def _call_gemini_sync(prompt: str) -> str:
    """Blocking Gemini call (runs in threadpool via ask_async)."""
    try:
        response = MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Lower temperature for more consistent output
                max_output_tokens=1000,
            )
        )
        
        if response.text:
            return response.text.strip()
        return ""
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return ""

async def ask(question: str, context_chunks: List[str]) -> dict:
    """
    Async helper: sends prompt in a threadpool so FastAPI remains non-blocking.
    Returns parsed JSON (dict). Falls back to structured dict on parse error.
    """
    # Debug logging
    logger.info(f"Processing question: {question}")
    logger.info(f"Context chunks: {len(context_chunks)}")
    for i, chunk in enumerate(context_chunks[:2]):  # Log first 2 chunks
        logger.info(f"Chunk {i}: {chunk[:200]}...")
    
    prompt = build_prompt(question, context_chunks)
    loop = asyncio.get_running_loop()
    
    try:
        raw = await loop.run_in_executor(None, _call_gemini_sync, prompt)
        logger.info(f"Raw LLM response: {raw[:500]}...")  # Log first 500 chars
        
        # Clean up the response
        cleaned = raw.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Try to parse JSON
        result = json.loads(cleaned)
        
        # Validate required keys
        required_keys = ["answer", "justification", "clause_reference", "decision", "confidence"]
        for key in required_keys:
            if key not in result:
                result[key] = ""
        
        # Ensure confidence is a float
        if not isinstance(result["confidence"], (int, float)):
            result["confidence"] = 0.5
        
        logger.info(f"Parsed result: {result}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Raw response was: {raw}")
        
        # Fallback: try to extract answer from non-JSON response
        fallback_answer = raw if raw else "Unable to process the question due to formatting error."
        
        return {
            "answer": fallback_answer,
            "justification": "Response parsing failed",
            "clause_reference": "",
            "decision": "unclear",
            "confidence": 0.0,
        }
        
    except Exception as e:
        logger.error(f"LLM processing error: {e}")
        return {
            "answer": f"Error processing question: {str(e)}",
            "justification": "",
            "clause_reference": "",
            "decision": "unclear",
            "confidence": 0.0,
        }

# Simple function to extract just the answer for the API response
def extract_answer(llm_result: dict) -> str:
    """Extract just the answer text for the simple API response format"""
    return llm_result.get("answer", "Unable to process question")