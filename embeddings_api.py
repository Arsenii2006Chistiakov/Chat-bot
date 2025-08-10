#!/usr/bin/env python3
"""
Embeddings API Service
Handles MERT audio embeddings and multilingual text embeddings
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import tempfile
import shutil

# Import the embedding models
from query_gpt import MongoChatbot

# Load environment variables
load_dotenv()

# Global variables for models
mert_embedder = None
multilingual_model = None

# Pydantic models for API requests/responses
class EmbeddingsRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Path to audio file")
    text: Optional[str] = Field(None, description="Text content to embed")
    user_id: str = Field(..., description="User identifier")

class EmbeddingsResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    embeddings: Dict[str, Any] = Field(..., description="Generated embeddings")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Timestamp of the operation")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    user_id: str = Field(..., description="User identifier")

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Embeddings API Service...")
    print("📚 Loading MERT and Multilingual models...")
    
    global mert_embedder, multilingual_model
    
    try:
        # Initialize a temporary chatbot instance to get the models
        temp_chatbot = MongoChatbot()
        mert_embedder = temp_chatbot.mert_embedder
        multilingual_model = temp_chatbot.multilingual_model
        print("✅ Models loaded successfully")
        
        # Clean up temporary instance
        del temp_chatbot
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e
    
    # Shutdown
    yield
    print("🛑 Shutting down Embeddings API Service...")

# Create FastAPI app
app = FastAPI(
    title="Embeddings API Service",
    description="API service for MERT audio embeddings and multilingual text embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "models_loaded": {
            "mert": mert_embedder is not None,
            "multilingual": multilingual_model is not None
        }
    }

# Main embeddings endpoint
@app.post("/generate-embeddings", response_model=EmbeddingsResponse)
async def generate_embeddings(request: EmbeddingsRequest):
    """Generate embeddings for audio file and/or text"""
    try:
        if not request.file_path and not request.text:
            raise HTTPException(status_code=400, detail="Either file_path or text must be provided")
        
        embeddings = {}
        
        # Process audio file if provided
        if request.file_path:
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail=f"Audio file not found: {request.file_path}")
            
            print(f"🎵 Processing audio file: {request.file_path}")
            
            # Create temporary chatbot instance for audio processing
            temp_chatbot = MongoChatbot()
            
            try:
                # Load and process audio
                await temp_chatbot._handle_load_command(request.file_path)
                
                # Get the embeddings
                if temp_chatbot.current_embedding is not None:
                    embeddings["audio_embedding"] = {
                        "shape": temp_chatbot.current_embedding.shape,
                        "dtype": str(temp_chatbot.current_embedding.dtype),
                        "embedding_id": id(temp_chatbot.current_embedding)
                    }
                    print(f"✅ Audio embeddings generated: {embeddings['audio_embedding']['shape']}")
                else:
                    print("⚠️ No audio embeddings generated")
                
                # Get lyrics if available
                if temp_chatbot.current_text_embedding is not None:
                    embeddings["lyrics_embedding"] = {
                        "shape": temp_chatbot.current_text_embedding.shape,
                        "dtype": str(temp_chatbot.current_text_embedding.dtype),
                        "embedding_id": id(temp_chatbot.current_text_embedding)
                    }
                    print(f"✅ Lyrics embeddings generated: {embeddings['lyrics_embedding']['shape']}")
                else:
                    print("⚠️ No lyrics embeddings generated")
                
            finally:
                # Clean up temporary instance
                del temp_chatbot
        
        # Process text if provided
        if request.text:
            print(f"📝 Processing text: {request.text[:100]}...")
            
            # Create temporary chatbot instance for text processing
            temp_chatbot = MongoChatbot()
            
            try:
                # Process text
                await temp_chatbot._handle_loadtext_command(request.text)
                
                # Get the text embeddings
                if temp_chatbot.current_text_embedding is not None:
                    embeddings["text_embedding"] = {
                        "shape": temp_chatbot.current_text_embedding.shape,
                        "dtype": str(temp_chatbot.current_text_embedding.dtype),
                        "embedding_id": id(temp_chatbot.current_text_embedding)
                    }
                    print(f"✅ Text embeddings generated: {embeddings['text_embedding']['shape']}")
                else:
                    print("⚠️ No text embeddings generated")
                
            finally:
                # Clean up temporary instance
                del temp_chatbot
        
        # Prepare response
        if embeddings:
            message = f"Successfully generated embeddings for: {', '.join(embeddings.keys())}"
            success = True
        else:
            message = "No embeddings were generated"
            success = False
        
        return EmbeddingsResponse(
            success=success,
            message=message,
            embeddings=embeddings,
            user_id=request.user_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ Error generating embeddings for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint for audio files
@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """Upload and process audio file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process the uploaded file
            request = EmbeddingsRequest(
                file_path=temp_file_path,
                user_id=user_id
            )
            
            result = await generate_embeddings(request)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return result
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        print(f"❌ Error uploading audio for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Embeddings API Service",
        "version": "1.0.0",
        "description": "API service for MERT audio embeddings and multilingual text embeddings",
        "endpoints": {
            "generate_embeddings": "/generate-embeddings - Generate embeddings for audio/text",
            "upload_audio": "/upload-audio - Upload and process audio file",
            "health": "/health - Health check"
        },
        "models": {
            "mert": mert_embedder is not None,
            "multilingual": multilingual_model is not None
        }
    }

if __name__ == "__main__":
    # Run the API with uvicorn
    uvicorn.run(
        "embeddings_api:app",
        host="0.0.0.0",
        port=8001,  # Different port from main chatbot API
        reload=True,
        log_level="info"
    )
