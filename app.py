#!/usr/bin/env python3
"""
FastAPI server for the MongoDB Chatbot.
Wraps the CLI chatbot to support multiple concurrent users via HTTP.

Endpoints:
- GET    /           → basic info
- GET    /health     → health check
- POST   /chat       → send a message/command for a user
- GET    /context/{user_id} → get user's context songs
- DELETE /user/{user_id}    → clear and remove user session
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


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or command")
    user_id: str = Field(..., description="User identifier")


class ChatResponse(BaseModel):
    response: str
    category: str
    context_count: int = 0
    search_results: Optional[List[Dict[str, Any]]] = None


app = FastAPI(title="MongoDB Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory user sessions: user_id → MongoChatbot instance
_sessions: Dict[str, MongoChatbot] = {}
_lock = asyncio.Lock()
DEBUG_CHAT = os.getenv("CHAT_DEBUG", "0") == "1"


async def get_session(user_id: str) -> MongoChatbot:
    async with _lock:
        if user_id not in _sessions:
            _sessions[user_id] = MongoChatbot()
        return _sessions[user_id]


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
    return nullcontext() if DEBUG_CHAT else suppress_chatbot_output()


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "MongoDB Chatbot API",
        "version": "1.0.0",
        "time": datetime.now().isoformat(),
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "context": "/context/{user_id}",
            "remove_user": "/user/{user_id}",
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "sessions": len(_sessions),
        "time": datetime.now().isoformat(),
    }


def _build_history_text(chatbot: MongoChatbot) -> str:
    if not chatbot.chat_history:
        return "No chat history available."
    lines: List[str] = []
    for i, exchange in enumerate(chatbot.chat_history[-10:], 1):
        category = exchange.get("category", "general")
        user = exchange.get("user", "")
        assistant = exchange.get("assistant", "")
        lines.append(f"{i}. [{category}]\nYou: {user}\nAssistant: {assistant[:200]}...")
    return "\n\n".join(lines)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")

    chatbot = await get_session(req.user_id)
    # Basic request debug
    print(f"[DEBUG] /chat user={req.user_id} message={message}")

    # Commands handled directly
    if message.startswith('/loadtext'):
        args_line = message[len('/loadtext'):]
        with exec_context():
            await chatbot._handle_loadtext_command(args_line)
        response_text = "Processed /loadtext via embeddings API."
        return ChatResponse(
            response=response_text,
            category="load",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    if message.startswith('/load'):
        parts = message.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            return ChatResponse(
                response="Please provide a file path after /load. Example: /load /path/to/file.mp3",
                category="load",
                context_count=len(getattr(chatbot, 'context_songs', []) or []),
            )
        file_path = parts[1].strip()
        with exec_context():
            await chatbot._handle_load_command(file_path)
        response_text = "Audio loaded via embeddings API. You can now run /search."
        return ChatResponse(
            response=response_text,
            category="load",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    if message.startswith('/search'):
        file_path, prompt = chatbot._parse_search_command(message)
        with exec_context():
            await chatbot._handle_search_command(file_path, prompt)
        results = getattr(chatbot, 'proposed_results', []) or []
        sanitized_results = [_sanitize(doc) for doc in results]
        return ChatResponse(
            response=f"Search completed. Found {len(results)} results.",
            category="search",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
            search_results=sanitized_results,
        )

    if message.lower() == '/history':
        text = _build_history_text(chatbot)
        return ChatResponse(
            response=text,
            category="help",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    if message.lower() == '/clear':
        chatbot._clear_chat_history()
        return ChatResponse(
            response="Chat history cleared.",
            category="help",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    if message.lower().startswith('/add'):
        await chatbot._handle_add_command(message)
        return ChatResponse(
            response="Processed /add command.",
            category="add",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    # Otherwise, let the orchestrator decide
    with exec_context():
        category = await chatbot._categorize_input(message)
    print(f"[DEBUG] /chat category={category}")
    if category == "help":
        help_response = (
            "Here's how I can help you explore the music database:\n\n"
            "- Search for songs by artist, genre, country charts, etc.\n"
            "- Find similar songs using audio and lyrics embeddings\n"
            "- Get information about music trends and chart rankings\n"
            "- Use commands like /load, /search, /history for advanced features\n\n"
            "Just ask me anything about music or use the special commands!"
        )
        chatbot._add_to_chat_history(message, help_response, "help")
        return ChatResponse(
            response=help_response,
            category="help",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    if category == "talk":
        with exec_context():
            chat_response = await chatbot._generate_chat_response(message)
        chatbot._add_to_chat_history(message, chat_response, "talk")
        return ChatResponse(
            response=chat_response,
            category="talk",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    if category == "search":
        # Print the context section used by LLM (outside suppression)
        try:
            previous_search_context = chatbot._get_chat_history_context(include_categories=["search"])  # type: ignore[attr-defined]
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
        chatbot._add_to_chat_history(message, f"Found {len(results)} results.", "search")
        sanitized_results = [_sanitize(doc) for doc in (results or [])]
        return ChatResponse(
            response=f"Search completed. Found {len(results)} results.",
            category="search",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
            search_results=sanitized_results,
        )

    if category == "analysis":
        with exec_context():
            analysis_response = await chatbot._handle_analysis_query(message)
        chatbot._add_to_chat_history(message, analysis_response, "analysis")
        return ChatResponse(
            response=analysis_response,
            category="analysis",
            context_count=len(getattr(chatbot, 'context_songs', []) or []),
        )

    # Fallback
    return ChatResponse(
        response="I'm here to help with music searches. Try /help for options.",
        category="talk",
        context_count=len(getattr(chatbot, 'context_songs', []) or []),
    )


@app.get("/context/{user_id}")
async def get_context(user_id: str) -> Dict[str, Any]:
    chatbot = await get_session(user_id)
    context_items = getattr(chatbot, 'context_songs', []) or []
    return {"context_songs": context_items}


@app.delete("/user/{user_id}")
async def remove_user(user_id: str) -> Dict[str, Any]:
    async with _lock:
        existed = user_id in _sessions
        if existed:
            del _sessions[user_id]
    return {"removed": existed, "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


