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


class TrendInfoRequest(BaseModel):
    song_id: str = Field(..., description="Song ID to get trend information for")


class TrendInfoResponse(BaseModel):
    success: bool
    message: str
    song_id: str
    trend_description: Optional[str] = None
    detailed_description: Optional[str] = None
    trend_explanation: Optional[str] = None
    video_gcs_uris: List[str] = []
    tiktok_uris: List[str] = []
    video_count: int = 0


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
            "trend_info": "/trend_info",
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


@app.post("/trend_info", response_model=TrendInfoResponse)
async def get_trend_info(req: TrendInfoRequest):
    """Get comprehensive trend information for a specific song."""
    song_id = req.song_id.strip()
    if not song_id:
        raise HTTPException(status_code=400, detail="Song ID must not be empty")

    print(f"[DEBUG] /trend_info song_id={song_id}")

    try:
        # Get a search session to access the database
        chatbot = await get_search_session("trend_info_user")
        
        # First check if the song exists and has TREND_STATUS = "EXISTS"
        song_doc = chatbot.collection.find_one({"song_id": song_id})
        if not song_doc:
            return TrendInfoResponse(
                success=False,
                message=f"Song with ID '{song_id}' not found",
                song_id=song_id
            )
        
        trend_status = song_doc.get("TREND_STATUS")
        if trend_status != "EXISTS":
            return TrendInfoResponse(
                success=False,
                message=f"Song '{song_id}' does not have trend information (TREND_STATUS: {trend_status})",
                song_id=song_id
            )

        # Get trend information from TOP_TIKTOK_TRENDS
        trend_doc = chatbot.trends_collection.find_one({"song_id": song_id})
        if not trend_doc:
            return TrendInfoResponse(
                success=False,
                message=f"Trend information not found for song '{song_id}'",
                song_id=song_id
            )

        trend_description = trend_doc.get("trend_description", "")
        detailed_description = trend_doc.get("detailedDescription", "")
        trend_explanation = trend_doc.get("trendExplanation", "")

        # Get video IDs from Hugo_final2.clusters
        video_ids = []
        try:
            clusters_collection = chatbot.db.Hugo_final2.clusters
            print(f"[DEBUG] Querying Hugo_final2.clusters for song_id: {song_id}")
            cluster_doc = clusters_collection.find_one({"song_id": song_id})
            if cluster_doc:
                video_ids = cluster_doc.get("video_ids", [])
                print(f"[DEBUG] Found cluster document: {cluster_doc}")
                print(f"[DEBUG] Found {len(video_ids)} video IDs in clusters: {video_ids}")
            else:
                print(f"[DEBUG] No cluster document found for song_id: {song_id}")
                # Try to see what's in the clusters collection
                sample_clusters = list(clusters_collection.find().limit(3))
                print(f"[DEBUG] Sample clusters: {sample_clusters}")
        except Exception as e:
            print(f"[WARNING] Failed to fetch video IDs from clusters: {e}")

        # Get GCS URIs from Hugo_final2.videos
        video_gcs_uris = []
        tiktok_uris = []
        if video_ids:
            try:
                videos_collection = chatbot.db.Hugo_final2.videos
                print(f"[DEBUG] Querying Hugo_final2.videos for {len(video_ids)} video IDs")
                for video_id in video_ids:
                    print(f"[DEBUG] Looking for video with _id: {video_id}")
                    video_doc = videos_collection.find_one({"_id": video_id})
                    if video_doc:
                        print(f"[DEBUG] Found video document: {video_doc}")
                        gcs_uri = video_doc.get("gcs_uri")
                        if gcs_uri:
                            video_gcs_uris.append(gcs_uri)
                            print(f"[DEBUG] Added gcs_uri: {gcs_uri}")
                        else:
                            print(f"[WARNING] No gcs_uri found for video_id: {video_id}")
                        tiktok_url = video_doc.get("tiktok_url")
                        if tiktok_url:
                            tiktok_uris.append(tiktok_url)
                            print(f"[DEBUG] Added tiktok_url: {tiktok_url}")
                        else:
                            print(f"[WARNING] No tiktok_url found for video_id: {video_id}")
                    else:
                        print(f"[WARNING] Video document not found for video_id: {video_id}")
                
                print(f"[DEBUG] Retrieved {len(video_gcs_uris)} GCS URIs and {len(tiktok_uris)} TikTok URIs")
            except Exception as e:
                print(f"[WARNING] Failed to fetch URIs from videos: {e}")
        else:
            print(f"[DEBUG] No video_ids to process")

        return TrendInfoResponse(
            success=True,
            message=f"Trend information retrieved successfully for song '{song_id}'",
            song_id=song_id,
            trend_description=trend_description,
            detailed_description=detailed_description,
            trend_explanation=trend_explanation,
            video_gcs_uris=video_gcs_uris,
            tiktok_uris=tiktok_uris,
            video_count=len(video_gcs_uris)
        )

    except Exception as e:
        print(f"[ERROR] Trend info failed: {e}")
        return TrendInfoResponse(
            success=False,
            message=f"Failed to retrieve trend information: {str(e)}",
            song_id=song_id
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
    print("Endpoints:")
    print("  POST /search")
    print("  POST /trend_info")
    print("\nExamples:")
    print("  Search: curl -X POST 'http://localhost:8002/search' -H 'Content-Type: application/json' -d '{\"message\": \"find me songs trending in Brazil\", \"user_id\": \"test_user\"}'")
    print("  Trend Info: curl -X POST 'http://localhost:8002/trend_info' -H 'Content-Type: application/json' -d '{\"song_id\": \"your_song_id_here\"}'")
    uvicorn.run("search_api:app", host="0.0.0.0", port=8002, reload=True, log_level="info")
