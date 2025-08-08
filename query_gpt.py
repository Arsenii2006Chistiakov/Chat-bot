#!/usr/bin/env python3
"""
MongoDB-Connected Chatbot
A CLI chatbot that uses an LLM to translate natural language into
MongoDB queries and displays the results.

Dependencies:
- pymongo
- python-dotenv
- rich
- openai (used for the LLM API call)
- torch
- torchaudio
- transformers
- pydub

You will need to have a MongoDB instance running and
set the OPENAI_API_KEY and MONGO_URI environment variables.
"""

import os
import json
import sys
import warnings
import asyncio
import requests
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.syntax import Syntax
import json
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
import tempfile
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from contextlib import contextmanager
from elevenlabs.client import ElevenLabs

# Suppress specific warnings
warnings.filterwarnings("ignore", message="feature_extractor_cqt requires the libray 'nnAudio'")
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")
warnings.filterwarnings("ignore", message="torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")

# Initialize Rich console
console = Console()

@contextmanager
def temp_file(suffix=None):
    """Context manager for temporary files that ensures cleanup"""
    temp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        yield temp.name
    finally:
        try:
            os.unlink(temp.name)
        except OSError:
            pass

class ElevenLabsClient:
    def __init__(self, api_key: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        if not self.api_key:
            self.enabled = False
            self.client = None
        else:
            self.enabled = True
            self.client = ElevenLabs(api_key=self.api_key)

    def transcribe_sync(
        self,
        audio_path: str,
        model_id: str = "scribe_v1",
        language_code: str = "eng",
        diarize: bool = True,
        tag_audio_events: bool = True,
    ) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        if not os.path.exists(audio_path):
            return None
        try:
            with open(audio_path, "rb") as audio_file:
                result = self.client.speech_to_text.convert(
                    file=audio_file,
                    model_id=model_id,
                    tag_audio_events=tag_audio_events,
                    language_code=language_code,
                    diarize=diarize,
                )
            # Normalize to dict-like
            if isinstance(result, dict):
                return result
            # Fallback: try to extract attributes
            text_val = getattr(result, "text", None)
            return {"text": text_val} if text_val is not None else {"raw": str(result)}
        except Exception as e:
            console.print(f"[red]Transcription request failed: {e}[/red]")
            return None

class MERTEmbedder:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_sr: int = 24000,  # Target sample rate for MERT
        snippet_duration: int = 15,  # Duration in seconds to extract from start
    ):
        """
        Initialize the MERT model for generating music embeddings.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            target_sr: Target sample rate for audio processing
            snippet_duration: Duration in seconds to extract from start of file
        """
        self.device = device
        self.target_sr = target_sr
        self.snippet_duration = snippet_duration
        
        #        console.print(f"[green]Initializing MERT model on {device}[/green]")
        
        # Load MERT model and processor with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mert_model = AutoModel.from_pretrained(
                "m-a-p/MERT-v1-330M", 
                trust_remote_code=True
            ).to(device)
            
            self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-330M",
                trust_remote_code=True
            )
        
        self.mert_model.eval()

    def extract_snippet_and_waveform(self, audio_path: str) -> tuple[torch.Tensor, str]:
        """
        Decode audio once with pydub, extract ~20s snippet, export to a temp wav, and load waveform.
        Returns (waveform_tensor [1, T], wav_path). Caller is responsible for deleting wav_path.
        """
        try:
            audio = AudioSegment.from_file(str(audio_path))
        except Exception as e:
            console.print(f"[red]Error loading audio file {audio_path}: {str(e)}[/red]")
            raise
        # Extract snippet (first 20s to match user's change)
        snippet = audio[:20000]
        # Export snippet to a persistent temp wav file
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        snippet.export(wav_path, format="wav")
        # Load with torchaudio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sr = torchaudio.load(wav_path)
        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr, dtype=waveform.dtype)
            waveform = resampler(waveform)
        # Force mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, wav_path

    def embedding_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        inputs = self.mert_processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)  # [layers, batch, time, dim]
        embedding = hidden_states.mean(dim=[0, 1, 2])  # [1024]
        return embedding.cpu()

    def _extract_middle_snippet_local(self, audio_path: str) -> torch.Tensor:
        """
        Extract a snippet from the middle of a local audio file.
        
        Args:
            audio_path: Path to the local audio file
            
        Returns:
            torch.Tensor: Audio waveform of the middle snippet
        """
        # Load audio file using pydub (more reliable for various formats)
        try:
            audio = AudioSegment.from_file(str(audio_path))
        except Exception as e:
            console.print(f"[red]Error loading audio file {audio_path}: {str(e)}[/red]")
            raise
        
        # Extract snippet
        snippet = audio[:20000]
        
        # Save to temporary file and load with torchaudio using modern approach
        with temp_file(suffix='.wav') as wav_path:
            snippet.export(wav_path, format="wav")
            
            # Use torchaudio.load with suppress_warnings context
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sr = torchaudio.load(wav_path)
        
        # Resample if necessary using modern resampler
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr, dtype=waveform.dtype)
            waveform = resampler(waveform)
        
        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform

    def process_local_file(self, audio_path: str) -> torch.Tensor:
        """
        Process a single local audio file and return its embedding.
        
        Args:
            audio_path: Path to the local audio file
            
        Returns:
            torch.Tensor: MERT embedding of shape [1024]
        """
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Extract first 20 seconds snippet
            waveform = self._extract_first_20s_snippet_local(audio_path)
            # Process audio
            inputs = self.mert_processor(
                waveform.squeeze().numpy(),
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.mert_model(**inputs, output_hidden_states=True)
                
            # Get all hidden states and stack them
            hidden_states = torch.stack(outputs.hidden_states)  # [25, 1, T, 1024]
            
            # Mean pool over time dimension and layers
            embedding = hidden_states.mean(dim=[0, 1, 2])  # [1024]
            
            return embedding.cpu()
            
        except Exception as e:
            console.print(f"[red]Error processing {audio_path}: {str(e)}[/red]")
            raise

class MongoDBQueryGenerator:
    """
    A class to generate MongoDB query JSON from natural language using an LLM.
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            console.print(Panel(
                "[red]Error: OPENAI_API_KEY not found in environment variables![/red]",
                title="Configuration Error", border_style="red"
            ))
            sys.exit(1)
        
        # Load example document
        try:
            with open('example_doc.json', 'r') as f:
                self.example_doc = json.load(f)
        except FileNotFoundError:
            console.print(Panel(
                "[red]Error: example_doc.json not found![/red]",
                title="Configuration Error", border_style="red"
            ))
            sys.exit(1)
            
    async def get_query_from_llm(self, natural_language_query: str, collection_schema: Dict[str, Any]) -> str:
        """
        Sends the user's query and the schema to an LLM to get a structured MongoDB query.
        
        Args:
            natural_language_query (str): The user's question.
            collection_schema (Dict[str, Any]): A JSON-like representation of the collection schema.
            
        Returns:
            str: A JSON string of the MongoDB query.
        """
        prompt = f"""
You are an expert at converting natural language questions into structured MongoDB queries.
You will be provided with an example document from the collection and a user's question.
Your task is to generate a JSON object that can be used directly in a PyMongo find() operation.
The JSON object should have the following keys:
- "filter": A MongoDB query filter (e.g., {{ "artist": "Capone" }}).
- "projection": A MongoDB projection to select specific fields (e.g., {{ "title": 1, "artist": 1, "views": 1, "_id": 0 }}).
- "sort": A MongoDB sort object (e.g., {{ "views": -1 }}).
- "limit": An integer limit for the number of results.

        Here is an example document from the collection:
        {json.dumps(self.example_doc, indent=2)}

---
Here are some examples of natural language queries and the correct JSON output:

User: "Find all songs that were ever ranked #1 in Brazil"
{{
  "filter": {{ "charts.Brazil.rank": 1 }},
  "projection": {{ "song_name": 1, "artist_name": 1, "charts.Brazil": 1, "_id": 0 }},
  "sort": {{}},
  "limit": 10
}}

User: "List the top 5 songs by Ponte Perro that charted in Argentina"
{{
  "filter": {{
    "artist_name": "Ponte Perro",
    "charts.Argentina.rank": {{ "$lte": 5 }}
  }},
  "projection": {{ "song_name": 1, "artist_name": 1, "charts.Argentina": 1, "_id": 0 }},
  "sort": {{ "charts.Argentina.rank": 1 }},
  "limit": 5
}}

User: "Show me all Latin songs with Spanish lyrics"
{{
  "filter": {{ "genres": "Latin", "language_code": "spa" }},
  "projection": {{ "song_name": 1, "artist_name": 1, "genres": 1, "language_code": 1, "_id": 0 }},
  "sort": {{ "first_seen": -1 }},
  "limit": 10
}}

User: "Find songs that are longer than 3 minutes and have high language probability"
{{
  "filter": {{ 
    "audio_metadata.duration": {{ "$gt": 180 }},
    "language_probability": {{ "$gt": 0.8 }}
  }},
  "projection": {{ "song_name": 1, "artist_name": 1, "audio_metadata.duration": 1, "language_probability": 1, "_id": 0 }},
  "sort": {{ "language_probability": -1 }},
  "limit": 10
}}

User: "Get all songs with RKT genre that are still being processed"
{{
  "filter": {{ "genres": "RKT", "TREND_STATUS": "UNPROCESSED" }},
  "projection": {{ "song_name": 1, "artist_name": 1, "genres": 1, "TREND_STATUS": 1, "_id": 0 }},
  "sort": {{ "first_seen": -1 }},
  "limit": 10
}}

User: "{natural_language_query}"

Provide only the JSON output. Do not include any other text or explanation.
"""
        
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=self.api_key)
        
        try:
            # Make API call to OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert at converting natural language questions into structured MongoDB queries. Provide only JSON output."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            # Extract the response text
            text = response.choices[0].message.content.strip()
            #print(text)
            return text
            
        except Exception as e:
            console.print(f"[red]Debug - OpenAI API call failed: {e}[/red]")
            return "{}"

class MongoChatbot:
    """
    Main chatbot class to handle user interaction and database operations.
    """
    def __init__(self):
        load_dotenv()
        self.mongo_uri = os.getenv('MONGO_URI')
        if not self.mongo_uri:
            console.print(Panel(
                "[red]Error: MONGO_URI not found in environment variables![/red]",
                title="Configuration Error", border_style="red"
            ))
        
            sys.exit(1)

        try:
            self.client = pymongo.MongoClient(self.mongo_uri)
            self.db = self.client.HUGO_MODEL_DB
            self.collection = self.db.TOP_TIKTOK_SOUNDS
            self.query_generator = MongoDBQueryGenerator()
            
            # Initialize MERT embedder for audio processing
            self.mert_embedder = None  # Will be initialized on first use
            self.current_embedding = None  # Store the current audio embedding
            self.current_transcription_text = None  # Store last transcription text
        except pymongo.errors.ConnectionFailure as e:
            console.print(Panel(
                f"[red]Error connecting to MongoDB: {e}[/red]",
                title="Connection Error", border_style="red"
            ))
            sys.exit(1)
        
        # Updated schema to reflect the new document structure with charts
        self.collection_schema = {
            "_id": "ObjectId",
            "song_id": "string",
            "artist_name": "string",
            "charts": {
                "country_code_example": [{
                    "timestamp": "string (date)",
                    "rank": "number"
                }]
            },
            "first_seen": "string (date)",
            "song_name": "string",
            "sound_link": "string (URL)",
            "genres": ["string"],
            "gcs_path": "string",
            "status": "string",
            "embedding_path": "string",
            "transcription_status": "string",
            "audio_metadata": {
                "format_name": "string",
                "duration": "number",
                "bit_rate": "number",
                "size": "number",
                "codec_name": "string",
                "sample_rate": "number",
                "channels": "number"
            },
            "language_code": "string",
            "language_probability": "number",
            "lyrics": "string",
            "music_embedding": ["number"],
            "TREND_STATUS": "string"
        }

    def _format_results(self, documents: List[Dict[str, Any]]):
        """Formats the list of documents for a nice display."""
        if not documents:
            console.print("[yellow]No documents found matching your query.[/yellow]")
            return

        for i, doc in enumerate(documents, 1):
            song_name = doc.get("song_name", "N/A")
            artist_name = doc.get("artist_name", "N/A")
            genres = ', '.join(doc.get("genres", []))
            
            charts_info = ""
            charts = doc.get("charts", {})
            for country, chart_entries in charts.items():
                if chart_entries:
                    latest_entry = chart_entries[0]  # Assuming sorted by timestamp
                    rank_data = latest_entry.get("rank", {})
                    # Handle both formats: {"$numberInt": "1"} and direct integer
                    if isinstance(rank_data, dict) and "$numberInt" in rank_data:
                        rank = rank_data["$numberInt"]
                    elif isinstance(rank_data, (int, str)):
                        rank = str(rank_data)
                    else:
                        rank = "N/A"
                    timestamp = latest_entry.get("timestamp", "N/A")
                    charts_info += f"  - [bold]{country}:[/bold] Rank {rank} (as of {timestamp})\n"

            panel_content = (
                f"[bold blue]Result {i}[/bold blue]\n\n"
                f"[bold]Song Name:[/bold] {song_name}\n"
                f"[bold]Artist:[/bold] {artist_name}\n"
                f"[bold]Genres:[/bold] {genres}\n"
                f"[bold]First Seen:[/bold] {doc.get('first_seen', 'N/A')}\n"
            )
            
            if charts_info:
                panel_content += f"\n[bold]Latest Chart Ranks:[/bold]\n{charts_info}"

            console.print(Panel(
                panel_content,
                border_style="green"
            ))

    async def _handle_load_command(self, file_path: str):
        """Handle the /load command: preprocess, transcribe, and store in memory."""
        try:
            # Initialize MERT embedder if not already done
            if self.mert_embedder is None:
                self.mert_embedder = MERTEmbedder()
            
            console.print(Panel(
                f"[bold blue]🎵 Loading Audio[/bold blue]\n\n"
                f"File: {file_path}\n"
                f"Extracting snippet, embedding, and transcribing...",
                title="Audio Load", border_style="blue"
            ))

            # Extract snippet and waveform, and create temp wav
            waveform, wav_path = self.mert_embedder.extract_snippet_and_waveform(file_path)

            # Kick off transcription concurrently
            transcription_task = asyncio.create_task(self._transcribe_audio(wav_path))

            # Compute embedding from waveform
            embedding = self.mert_embedder.embedding_from_waveform(waveform)
            self.current_embedding = embedding

            # Await transcription
            transcription_result = await transcription_task
            self.current_transcription_text = transcription_result.get("text", "")

            # Cleanup temp wav
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

            console.print(Panel(
                f"[bold green]✅ Loaded & Stored[/bold green]\n\n"
                f"Embedding: {list(embedding.shape)}\n"
                f"Transcript chars: {len(self.current_transcription_text or '')}\n\n"
                f"You can now run /search (uses the loaded embedding).",
                title="Success", border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error loading audio: {str(e)}[/red]",
                title="Error", border_style="red"
            ))

    async def _categorize_input(self, user_input: str) -> str:
        """
        Categorize user input into one of four categories using GPT-3.5.
        
        Args:
            user_input (str): The user's input message
            
        Returns:
            str: Category - "help", "talk", "search", or "match"
        """
        prompt = f"""
You are an input categorizer for a music database chatbot. Categorize the user's input into exactly one of these four categories:

1. "help" - User is asking for help, instructions, or wants to know what the system can do
   Examples: "help", "what can you do", "how does this work", "show me the commands"

2. "talk" - User is making casual conversation, greetings, or general chat
   Examples: "hello", "how are you", "thanks", "goodbye", "nice to meet you"

3. "search" - User wants to search the database with natural language queries
   Examples: "find songs by artist X", "show me Latin songs", "songs from Brazil", "popular songs"

4. "match" - User wants to match/find similar songs (usually involves audio or specific song matching)
   Examples: "find similar songs", "match this song", "what songs are like this", "/search", "/load"

User input: "{user_input}"

Respond with ONLY the category name: help, talk, search, or match
"""
        
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=self.query_generator.api_key)
        
        try:
            # Make API call to OpenAI with cheaper model
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a simple categorizer. Respond with only one word: help, talk, search, or match."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract the response text and clean it
            category = response.choices[0].message.content.strip().lower()
            
            # Ensure it's one of the valid categories
            if category not in ["help", "talk", "search", "match"]:
                category = "talk"  # Default fallback
                
            return category
            
        except Exception as e:
            console.print(f"[red]Error categorizing input: {e}[/red]")
            return "talk"  # Default fallback

    async def _get_search_parameters_from_llm(self, search_comment: str) -> Dict[str, Any]:
        """
        Use ChatGPT to deduce additional search parameters from user comment.
        Args:
            search_comment (str): User's search comment (e.g., "in Germany", "Latin songs", etc.)
            
        Returns:
            Dict[str, Any]: Search parameters including filters and limit
        """
        prompt = f"""
You are an expert at converting natural language search requests into MongoDB query parameters.
Given a user's search comment, determine what additional filters should be applied to a music vector search.

Available fields in the database:
- genres: Array of strings (e.g., ["Latin", "Pop", "RKT"])
- charts: Object with country codes as keys (e.g., "Germany", "Brazil", "Argentina")
- language_code: String (e.g., "spa", "eng", "por")
- language_probability: Number (0-1)
- audio_metadata.duration: Number (in seconds)
- TREND_STATUS: String (e.g., "PROCESSED", "UNPROCESSED")

User comment: "{search_comment}"

Return a JSON object with the following structure:
{{
    "filters": {{}},  // MongoDB filters to apply
    "limit": 10,     // Number of results to return
    "description": "Brief description of what this search is looking for"
}}

Examples:
- "in Germany" → {{"filters": {{"charts.Germany": {{"$exists": true}}}}, "limit": 10, "description": "Songs that charted in Germany"}}
- "Latin songs" → {{"filters": {{"genres": "Latin"}}, "limit": 10, "description": "Latin genre songs"}}
- "Spanish lyrics" → {{"filters": {{"language_code": "spa"}}, "limit": 10, "description": "Songs with Spanish lyrics"}}
- "popular in Brazil" → {{"filters": {{"charts.Brazil": {{"$exists": true}}}}, "limit": 15, "description": "Songs popular in Brazil"}}
- "long songs" → {{"filters": {{"audio_metadata.duration": {{"$gt": 180}}}}, "limit": 10, "description": "Songs longer than 3 minutes"}}

Provide only the JSON output. Do not include any other text or explanation.
"""
        
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=self.query_generator.api_key)
        
        try:
            # Make API call to OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert at converting natural language search requests into MongoDB query parameters. Provide only JSON output."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            # Extract the response text
            text = response.choices[0].message.content.strip()
            return json.loads(text)
            
        except Exception as e:
            console.print(f"[red]Error parsing search parameters: {e}[/red]")
            return {"filters": {}, "limit": 10, "description": "Simple music search"}

    def _parse_search_command(self, user_input: str) -> tuple:
        """
        Parse /search command with optional --file and --prompt parameters.
        
        Args:
            user_input (str): Raw user input (e.g., "/search --file audio.mp3 --prompt in Germany")
            
        Returns:
            tuple: (file_path, prompt) where either can be None
        """
        file_path = None
        prompt = None
        
        # Remove /search prefix
        command = user_input[8:].strip()
        
        if not command:
            return None, None
        
        # Parse arguments
        parts = command.split()
        i = 0
        while i < len(parts):
            if parts[i] == "--file" and i + 1 < len(parts):
                file_path = parts[i + 1]
                i += 2
            elif parts[i] == "--prompt" and i + 1 < len(parts):
                # Collect all remaining parts as the prompt
                prompt = " ".join(parts[i + 1:])
                break
            else:
                i += 1
        
        return file_path, prompt

    async def _handle_search_command(self, file_path: str = None, prompt: str = None, limit: int = 10):
        """Handle the /search command with optional --file and --prompt parameters."""
        try:
            query_vector = None
            search_comment = ""
            
            # Process file if provided
            if file_path:
                console.print(Panel(
                    f"[bold blue]🎵 Processing Audio File[/bold blue]\n\n"
                    f"File: {file_path}\n"
                    f"Extracting audio embedding...",
                    title="Audio Processing", border_style="blue"
                ))
                
                # Initialize MERT embedder if not already done
                if self.mert_embedder is None:
                    self.mert_embedder = MERTEmbedder()
                
                # Extract snippet and waveform
                waveform, wav_path = self.mert_embedder.extract_snippet_and_waveform(file_path)
                
                # Transcribe audio concurrently
                transcription_task = asyncio.create_task(self._transcribe_audio(wav_path))
                
                # Process the waveform and get embedding
                embedding = self.mert_embedder.embedding_from_waveform(waveform)
                query_vector = embedding.numpy().tolist()
                
                # Wait for transcription to complete
                transcription_result = await transcription_task
                search_comment = transcription_result.get("text", "")
                print(search_comment)
                
                # If user did not pass a prompt, use transcription text as the prompt for parameter extraction
                if (prompt is None or str(prompt).strip() == "") and search_comment:
                    prompt = search_comment
                
                # Clean up the temporary wav file
                try:
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                except Exception:
                    pass
                
                # console.print(Panel(
                #     f"[bold green]✅ Audio Processing Complete![/bold green]\n\n"
                #     f"Embedding shape: {embedding.shape}\n"
                #     f"Embedding values range: {embedding.min().item():.4f} to {embedding.max().item():.4f}",
                #     title="Success", border_style="green"
                # ))
            else:
                # Use stored embedding if available, otherwise set to None for text-only search
                if hasattr(self, 'current_embedding') and self.current_embedding is not None:
                    query_vector = self.current_embedding.numpy().tolist()
                    console.print(Panel(
                        "[blue]Using previously loaded audio embedding[/blue]",
                        title="Info", border_style="blue"
                    ))
                else:
                    query_vector = None
                    console.print(Panel(
                        "[blue]Performing text-only search (no audio embedding)[/blue]",
                        title="Info", border_style="blue"
                    ))
            
            # Process prompt if provided
            if prompt:
                # console.print(Panel(
                #     f"[bold blue]🔍 Analyzing Search Request[/bold blue]\n\n"
                #     f"Prompt: '{prompt}'\n"
                #     f"Determining additional search parameters...",
                #     title="Processing", border_style="blue"
                # ))
                
                # Get search parameters from LLM
                search_params = await self._get_search_parameters_from_llm(prompt)
                filters = search_params.get("filters", {})
                limit = search_params.get("limit", 10)
                description = search_params.get("description", "Custom search")
                
                console.print(Panel(
                    f"[bold green]✅ Search Parameters Determined[/bold green]\n\n"
                    f"Description: {description}\n"
                    f"Filters: {json.dumps(filters, indent=2)}\n"
                    f"Limit: {limit} results",
                    title="Parameters", border_style="green"
                ))
            else:
                filters = {}
                description = "Simple music similarity search"
                # console.print(Panel(
                #     f"[bold blue]🔍 Performing Simple Music Vector Search[/bold blue]\n\n"
                #     f"Searching for similar songs...\n"
                #     f"Limit: {limit} results",
                #     title="Vector Search", border_style="blue"
                # ))
            
            # Build vector search pipeline with optional filters
            if query_vector:
                # Build vector search stage - only include filter if it's not empty
                vector_search_stage = {
                    "index": "lyrics_n_music_search",
                    "path": "music_embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": limit
                }
                
                # Only add filter if it's not empty
                if filters and filters != {}:
                    vector_search_stage["filter"] = filters
                
                pipeline = [
                    {
                        "$vectorSearch": vector_search_stage
                    },
                    {
                        "$addFields": {
                            "musicScore": {"$meta": "vectorSearchScore"}
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "song_id": 1,
                            "song_name": 1,
                            "artist_name": 1,
                            "lyrics": 1,
                            "genres": 1,
                            "musicScore": 1,
                            "first_seen": 1,
                            "charts": 1,
                            "language_code": 1,
                            "audio_metadata": 1
                        }
                    },
                    {
                        "$sort": {"musicScore": -1}
                    }
                ]
            else:
                # Perform a simple MongoDB find query using only filters (no vector search)
                pipeline = [
                    {
                        "$match": filters if filters and filters != {} else {}
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "song_id": 1,
                            "song_name": 1,
                            "artist_name": 1,
                            "lyrics": 1,
                            "genres": 1,
                            "first_seen": 1,
                            "charts": 1,
                            "language_code": 1,
                            "audio_metadata": 1
                        }
                    },
                    {
                        "$limit": limit
                    }
                ]
                
            results = list(self.collection.aggregate(pipeline))
            
            if not results:
                console.print(Panel(
                    "[yellow]No similar songs found matching your criteria.[/yellow]",
                    title="No Results", border_style="yellow"
                ))
                return
            
            # Display results
            console.print(Panel(
                f"[bold green]🎵 Vector Search Results - {description}[/bold green]\n"
                f"Found {len(results)} results",
                title="Search Results", border_style="green"
            ))
            
            for i, result in enumerate(results, 1):
                song_name = result.get("song_name", "N/A")
                artist_name = result.get("artist_name", "N/A")
                genres = ', '.join(result.get("genres", []))
                music_score = result.get("musicScore", 0)
                lyrics = result.get("lyrics", "N/A")[:100] + "..." if result.get("lyrics") else "N/A"
                language = result.get("language_code", "N/A")
                
                charts_info = ""
                charts = result.get("charts", {})
                for country, chart_entries in charts.items():
                    if chart_entries:
                        latest_entry = chart_entries[0]
                        rank_data = latest_entry.get("rank", {})
                        if isinstance(rank_data, dict) and "$numberInt" in rank_data:
                            rank = rank_data["$numberInt"]
                        elif isinstance(rank_data, (int, str)):
                            rank = str(rank_data)
                        else:
                            rank = "N/A"
                        timestamp = latest_entry.get("timestamp", "N/A")
                        charts_info += f"  - [bold]{country}:[/bold] Rank {rank} (as of {timestamp})\n"
                
                panel_content = (
                    f"[bold blue]Result {i}[/bold blue]\n\n"
                    f"[bold]Song Name:[/bold] {song_name}\n"
                    f"[bold]Artist:[/bold] {artist_name}\n"
                    f"[bold]Genres:[/bold] {genres}\n"
                    f"[bold]Language:[/bold] {language}\n"
                    f"[bold]Similarity Score:[/bold] {music_score:.4f}\n"
                    f"[bold]First Seen:[/bold] {result.get('first_seen', 'N/A')}\n"
                    f"[bold]Lyrics Preview:[/bold] {lyrics}\n"
                )
                
                if charts_info:
                    panel_content += f"\n[bold]Latest Chart Ranks:[/bold]\n{charts_info}"
                
                console.print(Panel(
                    panel_content,
                    border_style="green"
                ))
            
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error performing vector search: {str(e)}[/red]",
                title="Error", border_style="red"
            ))

    async def _transcribe_audio(self, wav_path: str) -> Dict[str, Any]:
        """
        Transcribes the audio file using ElevenLabs API (SDK).
        Returns a dictionary containing the transcription text.
        """
        elevenlabs_client = ElevenLabsClient()
        if not elevenlabs_client.enabled:
            console.print(Panel(
                "[red]ElevenLabs API key not found. Cannot perform transcription.[/red]",
                title="Configuration Error", border_style="red"
            ))
            return {"text": "Transcription failed due to missing API key."}

        console.print(f"[blue]Transcribing audio file: {wav_path}[/blue]")
        # Offload blocking SDK call to a worker thread so it doesn't block the event loop
        transcription_result = await asyncio.to_thread(elevenlabs_client.transcribe_sync, wav_path)
        if transcription_result and transcription_result.get("text"):
            console.print(f"[green]Transcription successful. Text: {transcription_result['text']}[/green]")
            return transcription_result
        else:
            console.print(f"[red]Transcription failed for {wav_path}.[/red]")
            return {"text": "Transcription failed."}

    async def run(self):
        """Main application loop."""
        console.print(Panel(
            "[bold blue]🤖 MongoDB Song Database Chatbot CLI[/bold blue]\n\n"
            "Ask me questions about the song database!\n\n"
            "Example questions:\n"
            "• Find all songs that were ranked #1 in Brazil.\n"
            "• Show me songs by Ponte Perro that charted in Argentina.\n"
            "• List all Latin songs with Spanish lyrics.\n"
            "• Find songs longer than 3 minutes with high language probability.\n\n"
            "[bold green]Special Commands:[/bold green]\n"
            "• /load local/path/to/file - Preprocess an audio file with MERT\n"
            "• /search - Find similar songs using vector search (requires loaded audio)\n"
            "• /search --file audio.mp3 --prompt 'in Germany' - Search with file and filters\n"
            "• /search --file audio.mp3 - Search with new file only\n"
            "• /search --prompt 'Latin songs' - Text-only search with filters\n\n"
            "[yellow]Type 'quit' or 'exit' to end the session.[/yellow]",
            title="Welcome", border_style="blue"
        ))

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if not user_input:
                    continue

                # Check for /load command
                if user_input.startswith('/load '):
                    file_path = user_input[6:].strip()  # Remove '/load ' prefix
                    if file_path:
                        await self._handle_load_command(file_path)
                    else:
                        console.print(Panel(
                            "[red]Please provide a file path after /load[/red]\n"
                            "Example: /load /path/to/your/audio/file.mp3",
                            title="Usage Error", border_style="red"
                        ))
                    continue

                # Check for /search command
                if user_input.startswith('/search'):
                    # Parse the search command
                    file_path, prompt = self._parse_search_command(user_input)
                    await self._handle_search_command(file_path, prompt)
                    continue

                # Orchestrator: Categorize the input
                with console.status("[bold blue]Analyzing input...[/bold blue]", spinner="dots"):
                    category = await self._categorize_input(user_input)
                
                # Print the category to command line
                console.print(f"[bold magenta]🎯 Input Category: {category.upper()}[/bold magenta]")
                
                # Handle based on category
                if category == "help":
                    console.print(Panel(
                        "[bold blue]🤖 MongoDB Song Database Chatbot CLI[/bold blue]\n\n"
                        "Ask me questions about the song database!\n\n"
                        "Example questions:\n"
                        "• Find all songs that were ranked #1 in Brazil.\n"
                        "• Show me songs by Ponte Perro that charted in Argentina.\n"
                        "• List all Latin songs with Spanish lyrics.\n"
                        "• Find songs longer than 3 minutes with high language probability.\n\n"
                        "[bold green]Special Commands:[/bold green]\n"
                        "• /load local/path/to/file - Preprocess an audio file with MERT\n"
                        "• /search - Find similar songs using vector search (requires loaded audio)\n"
                        "• /search --file audio.mp3 --prompt 'in Germany' - Search with file and filters\n"
                        "• /search --file audio.mp3 - Search with new file only\n"
                        "• /search --prompt 'Latin songs' - Text-only search with filters\n\n"
                        "[yellow]Type 'quit' or 'exit' to end the session.[/yellow]",
                        title="Help", border_style="blue"
                    ))
                
                elif category == "talk":
                    console.print(Panel(
                        f"[bold green]Hello! 👋[/bold green]\n\n"
                        f"I'm your music database assistant. I can help you:\n"
                        f"• Search for songs and artists\n"
                        f"• Find similar songs using audio\n"
                        f"• Answer questions about the music database\n\n"
                        f"Just ask me anything about music!",
                        title="Chat", border_style="green"
                    ))
                
                elif category == "search":
                    # Use a typing indicator while the LLM and DB are working
                    with console.status("[bold green]Searching database...[/bold green]", spinner="dots"):
                        # 1. Get the structured query from the LLM
                        query_json_str = await self.query_generator.get_query_from_llm(user_input, self.collection_schema)
                        
                        try:
                            query_dict = json.loads(query_json_str)
                        except json.JSONDecodeError:
                            console.print(Panel(
                                f"[red]Error parsing LLM response. Got: {query_json_str}[/red]",
                                title="LLM Error", border_style="red"
                            ))
                            continue

                        # 2. Execute the query
                        filter_criteria = query_dict.get('filter', {})
                        projection = query_dict.get('projection', {})
                        sort = query_dict.get('sort', None)
                        limit = query_dict.get('limit', 10) # Default limit

                        # Check for empty filter to prevent full collection scans
                        if not filter_criteria or filter_criteria == {}:
                             console.print(Panel(
                                "[yellow]Your query was too broad. Please be more specific![/yellow]",
                                title="Query Error", border_style="yellow"
                            ))
                             continue

                        # Execute the find operation
                        cursor = self.collection.find(filter_criteria, projection)
                        if sort:
                            cursor = cursor.sort(sort)
                        if limit:
                            cursor = cursor.limit(limit)

                        results = list(cursor)

                    # 3. Display the formatted results
                    console.print()
                    self._format_results(results)
                    console.print()
                
                elif category == "match":
                    console.print(Panel(
                        f"[bold yellow]🎵 Song Matching[/bold yellow]\n\n"
                        f"It looks like you want to find similar songs!\n\n"
                        f"Try these commands:\n"
                        f"• /load /path/to/audio/file.mp3 - Load an audio file\n"
                        f"• /search --file /path/to/audio/file.mp3 - Search with a new file\n"
                        f"• /search --prompt 'similar to this style' - Text-based matching\n\n"
                        f"Or ask me to find songs similar to a specific artist or genre!",
                        title="Song Matching", border_style="yellow"
                    ))
            
            except Exception as e:
                console.print(f"[red]An unexpected error occurred: {e}[/red]")

async def main():
    """Entry point of the application."""
    try:
        chatbot = MongoChatbot()
        await chatbot.run()
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())