#!/usr/bin/env python3
"""
Dedicated Search API for MongoDB Chatbot.
Provides a standalone search endpoint that handles the same search routine as the main chatbot.

Endpoint:
- POST /search → perform search with message and return results
"""

import os
import sys
import io
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr, nullcontext

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import query_gpt  # to access global console for suppression
from query_gpt import MongoChatbot
from bson import ObjectId
import numpy as np


def _sanitize(value: Any) -> Any:
    """Sanitize values for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, np.ndarray):
        try:
            return value.tolist()
        except Exception:
            return [str(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in list(value)]
    # Fallback to string to avoid serialization errors
    return str(value)


class SearchRequest(BaseModel):
    message: str = Field(..., description="Search message/query")
    user_id: str = Field(..., description="User identifier")


class SearchResponse(BaseModel):
    success: bool
    message: str
    results: List[Dict[str, Any]] = []
    search_info: Optional[Dict[str, Any]] = None
    count: int = 0


app = FastAPI(title="Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory user sessions: user_id → MongoChatbot instance
_search_sessions: Dict[str, MongoChatbot] = {}
_lock = asyncio.Lock()
DEBUG_SEARCH = os.getenv("SEARCH_DEBUG", "0") == "1"


async def get_search_session(user_id: str) -> MongoChatbot:
    """Get or create a search session for a user."""
    async with _lock:
        if user_id not in _search_sessions:
            _search_sessions[user_id] = MongoChatbot()
        return _search_sessions[user_id]


@contextmanager
def suppress_chatbot_output():
    """Suppress rich console prints and stdio from query_gpt during API calls."""
    # Monkey-patch console.print
    original_print = getattr(query_gpt.console, "print", None)
    query_gpt.console.print = lambda *args, **kwargs: None
    # Redirect stdout/stderr
    f_out, f_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            yield
    finally:
        # Restore
        if original_print is not None:
            query_gpt.console.print = original_print


def exec_context():
    return nullcontext() if DEBUG_SEARCH else suppress_chatbot_output()


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Search API for MongoDB Chatbot",
        "version": "1.0.0",
        "time": datetime.now().isoformat(),
        "endpoints": {
            "search": "/search",
            "health": "/health",
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "search_sessions": len(_search_sessions),
        "time": datetime.now().isoformat(),
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Perform search using the same routine as the main chatbot."""
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")

    chatbot = await get_search_session(req.user_id)
    
    # Basic request debug
    print(f"[DEBUG] /search user={req.user_id} message={message}")

    try:
        # Print the context section used by LLM (outside suppression)
        try:
            previous_search_context = chatbot._get_chat_history_context(include_categories=["search"])
            if previous_search_context and previous_search_context.strip():
                print("[CONTEXT] previous_search_context=\n" + previous_search_context)
        except Exception:
            pass

        # Deduce search plan within suppressed context
        with exec_context():
            embedding_decision = await chatbot._determine_search_embeddings(message)
            search_params = await chatbot._get_search_parameters_from_llm(message)

        # Print deduced queries to the API server terminal (outside suppression)
        try:
            print("[DEDUCED] embedding_decision=", embedding_decision)
            print("[DEDUCED] search_params=", search_params)
        except Exception:
            pass

        # Execute the search with suppressed internal output
        with exec_context():
            results = await chatbot._execute_unified_search(message, embedding_decision, search_params)

        # Add search to chat history
        chatbot._add_to_chat_history(message, f"Found {len(results)} results.", "search")

        # Enrich results with trend information if TREND_STATUS is "EXISTS"
        enriched_results = []
        for result in (results or []):
            enriched_result = {
                "song_name": result.get("song_name"),
                "artist_name": result.get("artist_name"),
                "song_id": result.get("song_id"),
                "genres": result.get("genres", []),
                "TREND_STATUS": result.get("TREND_STATUS")
            }
            
            # If TREND_STATUS is "EXISTS", fetch trend_description from TOP_TIKTOK_TRENDS
            if result.get("TREND_STATUS") == "EXISTS":
                try:
                    trend_doc = chatbot.trends_collection.find_one({"song_id": result.get("song_id")})
                    if trend_doc:
                        enriched_result["trend_description"] = trend_doc.get("trend_description", "")
                    else:
                        enriched_result["trend_description"] = ""
                except Exception as e:
                    print(f"[WARNING] Failed to fetch trend description for song_id {result.get('song_id')}: {e}")
                    enriched_result["trend_description"] = ""
            else:
                enriched_result["trend_description"] = ""
            
            enriched_results.append(enriched_result)

        # Sanitize results for JSON response
        sanitized_results = [_sanitize(doc) for doc in enriched_results]

        # Build search info
        search_info = {
            "embedding_decision": embedding_decision,
            "search_params": search_params,
            "search_type": embedding_decision.get("search_type", "unknown"),
            "result_count": len(sanitized_results)
        }

        return SearchResponse(
            success=True,
            message=f"Search completed. Found {len(sanitized_results)} results.",
            results=sanitized_results,
            search_info=search_info,
            count=len(sanitized_results)
        )

    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return SearchResponse(
            success=False,
            message=f"Search failed: {str(e)}",
            results=[],
            count=0
        )


@app.delete("/user/{user_id}")
async def remove_user(user_id: str) -> Dict[str, Any]:
    """Remove a user's search session."""
    async with _lock:
        existed = user_id in _search_sessions
        if existed:
            del _search_sessions[user_id]
    return {"removed": existed, "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    print("Starting Search API server...")
    print("Endpoint: POST /search")
    print("Example: curl -X POST 'http://localhost:8002/search' -H 'Content-Type: application/json' -d '{\"message\": \"find me songs trending in Brazil\", \"user_id\": \"test_user\"}'")
    uvicorn.run("search_api:app", host="0.0.0.0", port=8002, reload=True, log_level="info")
