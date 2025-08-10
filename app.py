#!/usr/bin/env python3
"""
FastAPI Chatbot API
A REST API wrapper for the MongoDB-Connected Chatbot that allows multiple users
to interact with the chatbot simultaneously through a single chat endpoint.

Dependencies:
- fastapi
- uvicorn
- python-multipart
- All dependencies from query_gpt.py
"""

import os
import asyncio
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import the chatbot classes
from query_gpt import MongoChatbot, MongoDBQueryGenerator

# Load environment variables
load_dotenv()

# Global chatbot instances storage
chatbot_instances: Dict[str, MongoChatbot] = {}

# Embeddings API configuration
EMBEDDINGS_API_URL = os.getenv("EMBEDDINGS_API_URL", "http://localhost:8001")

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to process")
    user_id: str = Field(..., description="Unique identifier for the user")
    session_id: Optional[str] = Field(None, description="Session identifier for chat history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot response")
    category: str = Field(..., description="Category of the response (help, talk, search, analysis)")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Timestamp of the response")
    search_results: Optional[List[Dict[str, Any]]] = Field(None, description="Search results if applicable")
    context_count: Optional[int] = Field(None, description="Number of songs in context after operation")

class ContextResponse(BaseModel):
    context_songs: List[Dict[str, Any]] = Field(..., description="Songs in user's context")
    user_id: str = Field(..., description="User identifier")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    user_id: str = Field(..., description="User identifier")

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Chatbot API...")
    print("📚 Loading models and initializing connections...")
    print(f"🔗 Embeddings API URL: {EMBEDDINGS_API_URL}")
    
    # Shutdown
    yield
    print("🛑 Shutting down Chatbot API...")
    # Clean up chatbot instances
    for user_id in list(chatbot_instances.keys()):
        del chatbot_instances[user_id]

# Create FastAPI app
app = FastAPI(
    title="MongoDB Chatbot API",
    description="A REST API for the MongoDB-Connected Chatbot with unified chat endpoint",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get or create chatbot instance
async def get_chatbot(user_id: str) -> MongoChatbot:
    """Get or create a chatbot instance for a user"""
    if user_id not in chatbot_instances:
        try:
            chatbot_instances[user_id] = MongoChatbot()
            print(f"✅ Created new chatbot instance for user: {user_id}")
        except Exception as e:
            print(f"❌ Error creating chatbot for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {str(e)}")
    
    return chatbot_instances[user_id]

# Function to call embeddings API
async def call_embeddings_api(file_path: Optional[str] = None, text: Optional[str] = None, user_id: str = None) -> Dict[str, Any]:
    """Call the embeddings API service"""
    try:
        # Check if embeddings API is available
        health_response = requests.get(f"{EMBEDDINGS_API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            raise Exception(f"Embeddings API not available: {health_response.status_code}")
        
        # Prepare request data
        request_data = {
            "user_id": user_id
        }
        
        if file_path:
            request_data["file_path"] = file_path
        if text:
            request_data["text"] = text
        
        # Call embeddings API
        response = requests.post(
            f"{EMBEDDINGS_API_URL}/generate-embeddings",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Embeddings API error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to embeddings API: {str(e)}")
    except Exception as e:
        raise Exception(f"Embeddings API error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check embeddings API health
    embeddings_api_health = "unknown"
    try:
        health_response = requests.get(f"{EMBEDDINGS_API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            embeddings_api_health = "healthy"
        else:
            embeddings_api_health = "unhealthy"
    except:
        embeddings_api_health = "unreachable"
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now(), 
        "active_users": len(chatbot_instances),
        "embeddings_api": embeddings_api_health
    }

# Unified chat endpoint - handles all functionality
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Unified chat endpoint that handles all chatbot functionality"""
    try:
        chatbot = await get_chatbot(request.user_id)
        user_input = request.message.strip()
        
        # Initialize response variables
        response = ""
        category = ""
        search_results = None
        context_count = len(chatbot.context_songs)
        
        # Handle special commands first
        if user_input.startswith("/"):
            # Handle /load command with new syntax
            if user_input.startswith("/load"):
                try:
                    # Parse the /load command
                    parts = user_input.split()
                    file_path = None
                    text = None
                    
                    # Parse arguments
                    i = 1
                    while i < len(parts):
                        if parts[i] == "--file" and i + 1 < len(parts):
                            file_path = parts[i + 1]
                            i += 2
                        elif parts[i] == "--text" and i + 1 < len(parts):
                            text = parts[i + 1]
                            i += 2
                        else:
                            i += 1
                    
                    if not file_path and not text:
                        response = """Usage: /load --file <file_path> --text <text>
Examples:
• /load --file /path/to/song.mp3
• /load --text "song lyrics here"
• /load --file /path/to/song.mp3 --text "additional lyrics" """
                        category = "help"
                    else:
                        # Call embeddings API
                        print(f"🔗 Calling embeddings API: file={file_path}, text={text}")
                        embeddings_result = await call_embeddings_api(
                            file_path=file_path,
                            text=text,
                            user_id=request.user_id
                        )
                        
                        if embeddings_result.get("success"):
                            # Add embeddings to chatbot context
                            if file_path:
                                # Store file path and embeddings info
                                context_item = {
                                    "type": "audio_file",
                                    "file_path": file_path,
                                    "embeddings": embeddings_result["embeddings"],
                                    "timestamp": datetime.now().isoformat()
                                }
                                chatbot.context_songs.append(context_item)
                                
                                response = f"✅ Successfully processed audio file: {file_path}\n"
                                if "audio_embedding" in embeddings_result["embeddings"]:
                                    response += f"🎵 Audio embeddings generated\n"
                                if "lyrics_embedding" in embeddings_result["embeddings"]:
                                    response += f"📝 Lyrics embeddings generated\n"
                                response += f"Added to context. You now have {len(chatbot.context_songs)} items in context."
                            
                            if text:
                                # Store text and embeddings info
                                context_item = {
                                    "type": "text_content",
                                    "text": text,
                                    "embeddings": embeddings_result["embeddings"],
                                    "timestamp": datetime.now().isoformat()
                                }
                                chatbot.context_songs.append(context_item)
                                
                                response += f"✅ Successfully processed text content\n"
                                if "text_embedding" in embeddings_result["embeddings"]:
                                    response += f"📝 Text embeddings generated\n"
                                response += f"Added to context. You now have {len(chatbot.context_songs)} items in context."
                            
                            category = "load"
                            context_count = len(chatbot.context_songs)
                        else:
                            response = f"❌ Error processing content: {embeddings_result.get('message', 'Unknown error')}"
                            category = "error"
                
                except Exception as e:
                    response = f"❌ Error processing /load command: {str(e)}"
                    category = "error"
            
            # Handle /add command
            elif user_input.startswith("/add "):
                try:
                    await chatbot._handle_add_command(user_input)
                    response = f"Successfully added item to context. You now have {len(chatbot.context_songs)} songs in context."
                    category = "context"
                    context_count = len(chatbot.context_songs)
                except Exception as e:
                    response = f"Error adding to context: {str(e)}"
                    category = "error"
            
            # Handle /clear command
            elif user_input == "/clear":
                chatbot.context_songs.clear()
                response = "Context cleared successfully"
                category = "clear"
                context_count = 0
            
            # Handle /history command
            elif user_input == "/history":
                history = chatbot._get_chat_history_context()
                response = f"Chat History:\n{history}"
                category = "history"
            
            # Handle /help command
            elif user_input == "/help":
                response = """Available commands:
• /load --file <file_path> --text <text> - Load audio file and/or text content
  Examples:
  - /load --file /path/to/song.mp3
  - /load --text "song lyrics here"
  - /load --file /path/to/song.mp3 --text "additional lyrics"
• /add <index> - Add search result to context
• /clear - Clear context
• /history - Show chat history
• /help - Show this help message

You can also just chat normally or ask me to search for music!"""
                category = "help"
            
            # Unknown command
            else:
                response = f"Unknown command: {user_input}. Type /help for available commands."
                category = "error"
        
        else:
            # Regular chat - categorize and handle
            category = await chatbot._categorize_input(user_input)
            
            if category == "search":
                # Handle search queries
                try:
                    # Get search parameters from LLM
                    search_params = await chatbot._get_search_parameters_from_llm(user_input)
                    
                    # Execute search
                    results = await chatbot._handle_search_command(
                        prompt=user_input,
                        limit=search_params.get('limit', 10)
                    )
                    
                    # Format response
                    if results:
                        response = f"Found {len(results)} results for your search. Here are the top results:\n\n"
                        for i, result in enumerate(results[:5]):  # Show top 5
                            title = result.get('title', 'Unknown Title')
                            artist = result.get('artist', 'Unknown Artist')
                            trend_desc = result.get('trend_description', '')
                            trend_info = f" (Trend: {trend_desc})" if trend_desc else ""
                            response += f"{i+1}. {title} by {artist}{trend_info}\n"
                        
                        response += f"\nUse /add <index> to add any result to your context for analysis."
                        search_results = results
                    else:
                        response = "No results found for your search. Try different keywords or filters."
                    
                except Exception as e:
                    response = f"Error during search: {str(e)}"
                    category = "error"
            
            elif category == "analysis":
                # Handle analysis queries
                try:
                    if not chatbot.context_songs:
                        response = """To use analysis, you need to add songs to your context first:

1. Search for songs using natural language (e.g., "find me popular songs")
2. Use /add <index> to add interesting results to your context
3. Then ask me to analyze them (e.g., "explain why this trend became popular")

Try searching for some songs first!"""
                    else:
                        response = await chatbot._handle_analysis_query(user_input)
                        response += f"\n\nYou have {len(chatbot.context_songs)} items in context for analysis."
                    
                except Exception as e:
                    response = f"Error during analysis: {str(e)}"
                    category = "error"
            
            elif category == "help":
                # Handle help queries
                response = """I'm your music analysis assistant! Here's what I can do:

🔍 **Search**: Ask me to find songs, artists, or trends
   Example: "find me popular songs from Sweden"

🧠 **Analysis**: Get deep insights about music trends
   Example: "explain why this trend became popular"

📚 **Context Management**: Build a collection of songs and content to analyze
   • Search for songs first
   • Use /add <index> to save interesting results
   • Use /load to add audio files or text content
   • Ask me to analyze your collection

🎵 **Audio Processing**: Load and analyze audio files
   Use: /load --file <file_path>

💬 **General Chat**: Just chat with me about music!

Type /help for command reference."""
            
            else:
                # Handle general chat
                response = await chatbot._generate_chat_response(user_input)
        
        # Add to chat history
        chatbot._add_to_chat_history(user_input, response, category)
        
        return ChatResponse(
            response=response,
            category=category,
            user_id=request.user_id,
            timestamp=datetime.now(),
            search_results=search_results,
            context_count=context_count
        )
        
    except Exception as e:
        print(f"❌ Error in chat endpoint for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get context endpoint (for checking current context)
@app.get("/context/{user_id}", response_model=ContextResponse)
async def get_context_endpoint(user_id: str):
    """Get the user's current context"""
    try:
        chatbot = await get_chatbot(user_id)
        
        return ContextResponse(
            context_songs=chatbot.context_songs,
            user_id=user_id
        )
        
    except Exception as e:
        print(f"❌ Error getting context for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Remove user endpoint (cleanup)
@app.delete("/user/{user_id}")
async def remove_user_endpoint(user_id: str):
    """Remove a user and clean up their resources"""
    try:
        if user_id in chatbot_instances:
            del chatbot_instances[user_id]
            print(f"🗑️ Removed chatbot instance for user: {user_id}")
        
        return {
            "message": f"User {user_id} removed successfully"
        }
        
    except Exception as e:
        print(f"❌ Error removing user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MongoDB Chatbot API - Unified Chat Endpoint",
        "version": "1.0.0",
        "description": "All functionality is available through the /chat endpoint",
        "endpoints": {
            "chat": "/chat - Send any message or command (handles everything)",
            "context": "/context/{user_id} - Get user's context",
            "remove_user": "/user/{user_id} (DELETE) - Remove user and cleanup",
            "health": "/health - Health check"
        },
        "active_users": len(chatbot_instances),
        "embeddings_api": EMBEDDINGS_API_URL,
        "usage": {
            "chat": "Send any message to /chat - it will automatically classify and handle it",
            "commands": "Use /help in chat to see available commands",
            "search": "Just ask naturally: 'find me popular songs'",
            "analysis": "Build context with /add, then ask for analysis",
            "load": "Use /load --file <path> --text <content> to add audio/text to context"
        }
    }

if __name__ == "__main__":
    # Run the API with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
