#!/usr/bin/env python3
"""
MCP server for the MongoDB Chatbot.

This wraps the same functionality as app.py but exposes it via Model Context Protocol (MCP)
so compatible clients (IDEs/agents) can invoke tools.

Tools exposed:
- chat(message, user_id)
- load_file(file_path, user_id)
- load_text(text, user_id)
- search(user_id, file_path=None, prompt=None)
- get_context(user_id)
- clear_user(user_id)

Notes:
- Requires an MCP Python SDK. This file targets the FastMCP interface.
- Suppresses Rich console output from query_gpt to keep clean tool outputs.
"""

import io
import asyncio
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Optional
from datetime import datetime

from bson import ObjectId  # type: ignore
import numpy as np  # type: ignore

import query_gpt
from query_gpt import MongoChatbot

try:
    # FastMCP is a simple decorator-based interface
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception as e:  # pragma: no cover
    FastMCP = None  # type: ignore


# Per-user chatbot sessions
_sessions: Dict[str, MongoChatbot] = {}
_lock = asyncio.Lock()


async def get_session(user_id: str) -> MongoChatbot:
    async with _lock:
        if user_id not in _sessions:
            _sessions[user_id] = MongoChatbot()
        return _sessions[user_id]


@contextmanager
def suppress_chatbot_output():
    """Suppress rich console prints and stdio during tool execution."""
    original_print = getattr(query_gpt.console, "print", None)
    query_gpt.console.print = lambda *args, **kwargs: None
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            yield
    finally:
        if original_print is not None:
            query_gpt.console.print = original_print


def sanitize(value: Any) -> Any:
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
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in list(value)]
    return str(value)


if FastMCP is None:
    raise ImportError(
        "MCP server not available. Please install an MCP Python SDK, e.g.: pip install mcp"
    )


app = FastMCP("music-chatbot")


@app.tool()
async def chat(message: str, user_id: str) -> dict:
    """Chat with the assistant (routes to talk/search/analysis/help)."""
    bot = await get_session(user_id)
    with suppress_chatbot_output():
        category = await bot._categorize_input(message)

    if category == "talk":
        with suppress_chatbot_output():
            response = await bot._generate_chat_response(message)
        bot._add_to_chat_history(message, response, "talk")
        return {"category": "talk", "response": response}

    if category == "search":
        with suppress_chatbot_output():
            decision = await bot._determine_search_embeddings(message)
            params = await bot._get_search_parameters_from_llm(message)
            results = await bot._execute_unified_search(message, decision, params)
        bot._add_to_chat_history(message, f"Found {len(results)} results.", "search")
        return {"category": "search", "results": [sanitize(r) for r in (results or [])]}

    if category == "analysis":
        with suppress_chatbot_output():
            response = await bot._handle_analysis_query(message)
        bot._add_to_chat_history(message, response, "analysis")
        return {"category": "analysis", "response": response}

    help_text = (
        "Use /load, /loadtext, /search, /add, /history, /clear or ask general questions."
    )
    bot._add_to_chat_history(message, help_text, "help")
    return {"category": "help", "response": help_text}


@app.tool()
async def load_file(file_path: str, user_id: str) -> dict:
    """Load an audio file via embeddings API and cache embeddings in session."""
    bot = await get_session(user_id)
    with suppress_chatbot_output():
        await bot._handle_load_command(file_path)
    return {"ok": True}


@app.tool()
async def load_text(text: str, user_id: str) -> dict:
    """Load text for embeddings via embeddings API and cache in session."""
    bot = await get_session(user_id)
    with suppress_chatbot_output():
        # Reuse same parser path
        await bot._handle_loadtext_command(f"--text {text}")
    return {"ok": True}


@app.tool()
async def search(user_id: str, file_path: Optional[str] = None, prompt: Optional[str] = None) -> dict:
    """Perform a search; optional file or prompt accepted."""
    bot = await get_session(user_id)
    with suppress_chatbot_output():
        await bot._handle_search_command(file_path, prompt)
    results = getattr(bot, "proposed_results", []) or []
    return {"results": [sanitize(r) for r in results]}


@app.tool()
async def get_context(user_id: str) -> dict:
    """Return context songs for the user session."""
    bot = await get_session(user_id)
    items = getattr(bot, "context_songs", []) or []
    return {"context_songs": sanitize(items)}


@app.tool()
async def clear_user(user_id: str) -> dict:
    """Clear chat history and remove user session."""
    async with _lock:
        existed = user_id in _sessions
        if existed:
            try:
                _sessions[user_id]._clear_chat_history()
            except Exception:
                pass
            del _sessions[user_id]
    return {"removed": existed}


if __name__ == "__main__":
    # Run MCP server over stdio (common for editor integrations)
    app.run()


