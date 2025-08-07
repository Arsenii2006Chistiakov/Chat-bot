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
        
        console.print(f"[green]Initializing MERT model on {device}[/green]")
        
        # Load MERT model and processor
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", 
            trust_remote_code=True
        ).to(device)
        
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M",
            trust_remote_code=True
        )
        
        self.mert_model.eval()
        console.print("[green]MERT model initialized successfully[/green]")

    def _extract_middle_snippet_local(self, audio_path: str) -> torch.Tensor:
        """
        Extract a snippet from the middle of a local audio file.
        
        Args:
            audio_path: Path to the local audio file
            
        Returns:
            torch.Tensor: Audio waveform of the middle snippet
        """
        # Load audio file as M4A regardless of extension
        try:
            audio = AudioSegment.from_file(str(audio_path), format='m4a')
        except Exception as e:
            console.print(f"[red]Error loading audio file {audio_path}: {str(e)}[/red]")
            raise
        
        # Calculate middle position
        total_duration = len(audio)  # in milliseconds
        snippet_duration_ms = self.snippet_duration * 1000
        
        start_ms = (total_duration - snippet_duration_ms) // 2
        end_ms = start_ms + snippet_duration_ms
        
        # Extract snippet
        snippet = audio[start_ms:end_ms]
        
        # Save to temporary file and load with torchaudio
        with temp_file(suffix='.wav') as wav_path:
            snippet.export(wav_path, format="wav")
            waveform, sr = torchaudio.load(wav_path)
        
        # Resample if necessary
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
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
            
            console.print(f"[blue]Processing local file: {audio_path}[/blue]")
            
            # Extract middle snippet
            waveform = self._extract_middle_snippet_local(audio_path)
            
            # Print waveform shape for debugging
            console.print(f"[yellow]Waveform shape: {waveform.shape}[/yellow]")
            console.print(f"[yellow]Sample rate: {self.target_sr} Hz[/yellow]")
            console.print(f"[yellow]Duration: {waveform.shape[1] / self.target_sr:.2f} seconds[/yellow]")
            
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
            
            console.print(f"[green]Embedding shape: {embedding.shape}[/green]")
            console.print(f"[green]Embedding values range: {embedding.min().item():.4f} to {embedding.max().item():.4f}[/green]")
            
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

    def _handle_load_command(self, file_path: str):
        """Handle the /load command to preprocess an audio file."""
        try:
            # Initialize MERT embedder if not already done
            if self.mert_embedder is None:
                #console.print("[blue]Initializing MERT model for audio processing...[/blue]")
                self.mert_embedder = MERTEmbedder()
            
            # Process the audio file
            #console.print(Panel(
                # f"[bold blue]🎵 Audio File Preprocessing[/bold blue]\n\n"
                # f"Processing: {file_path}\n\n"
                # "This will extract a 15-second snippet from the middle of the file,\n"
                # "resample to 24kHz, and generate a MERT embedding.",
                # title="Audio Processing", border_style="blue"
            #))
            
            # Process the file and store the embedding
            self.current_embedding = self.mert_embedder.process_local_file(file_path)
            
            console.print(Panel(
                f"[bold green]✅ Audio Processing Complete![/bold green]\n\n"
                #f"File: {file_path}\n"
                #f"Embedding shape: {self.current_embedding.shape}\n"
                #f"Embedding values range: {self.current_embedding.min().item():.4f} to {self.current_embedding.max().item():.4f}\n\n"
                #f"[yellow]Type /search to find similar songs in the database![/yellow]",
                #title="Success", border_style="green"
            ))
            
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error processing audio file: {str(e)}[/red]",
                title="Error", border_style="red"
            ))

    def _handle_search_command(self, limit: int = 10):
        """Handle the /search command to perform vector search on music embeddings."""
        try:
            if not hasattr(self, 'current_embedding') or self.current_embedding is None:
                console.print(Panel(
                    "[red]❌ No audio file loaded! Please use /load first.[/red]\n"
                    "Example: /load /path/to/your/audio/file.mp3",
                    title="Error", border_style="red"
                ))
                return
            
            console.print(Panel(
                f"[bold blue]🔍 Performing Music Vector Search[/bold blue]\n\n"
                f"Searching for songs similar to your loaded audio...\n"
                f"Limit: {limit} results",
                title="Vector Search", border_style="blue"
            ))
            
            # Convert embedding to list for MongoDB
            query_vector = self.current_embedding.numpy().tolist()
            
            # Perform vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "lyrics_n_music_search",
                        "path": "music_embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": limit
                    }
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
                        "charts": 1
                    }
                },
                {
                    "$sort": {"musicScore": -1}
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            if not results:
                console.print(Panel(
                    "[yellow]No similar songs found in the database.[/yellow]",
                    title="No Results", border_style="yellow"
                ))
                return
            
            # Display results
            console.print(Panel(
                f"[bold green]🎵 Vector Search Results (Top {len(results)})[/bold green]",
                title="Search Results", border_style="green"
            ))
            
            for i, result in enumerate(results, 1):
                song_name = result.get("song_name", "N/A")
                artist_name = result.get("artist_name", "N/A")
                genres = ', '.join(result.get("genres", []))
                music_score = result.get("musicScore", 0)
                lyrics = result.get("lyrics", "N/A")[:100] + "..." if result.get("lyrics") else "N/A"
                
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
            "• /search - Find similar songs using vector search (requires loaded audio)\n\n"
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
                        self._handle_load_command(file_path)
                    else:
                        console.print(Panel(
                            "[red]Please provide a file path after /load[/red]\n"
                            "Example: /load /path/to/your/audio/file.mp3",
                            title="Usage Error", border_style="red"
                        ))
                    continue

                # Check for /search command
                if user_input == '/search':
                    self._handle_search_command()
                    continue

                # Use a typing indicator while the LLM and DB are working
                with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
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