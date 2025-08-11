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
from typing import List, Dict, Any, Optional
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
from sentence_embeddings import MultilingualSentenceEmbeddings
from split_lyrics import get_sentences
from torch import nn

# Suppress specific warnings
warnings.filterwarnings("ignore", message="feature_extractor_cqt requires the libray 'nnAudio'")
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")
warnings.filterwarnings("ignore", message="torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")

# Initialize Rich console
console = Console()

class SimpleFCNN(nn.Module):
    """
    A small fully-connected neural network for multi-label genre prediction.
    Uses BCEWithLogitsLoss downstream, so outputs raw logits.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(features)

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
        language_code: str | None = None,
        diarize: bool | None = False,
        tag_audio_events: bool | None = False,
    ) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        if not os.path.exists(audio_path):
            return None
        try:
            with open(audio_path, "rb") as audio_file:
                kwargs: Dict[str, Any] = {
                    "model_id": model_id,
                }
                # Only include optional params if explicitly provided
                if tag_audio_events is not None:
                    kwargs["tag_audio_events"] = tag_audio_events
                if language_code is not None:
                    kwargs["language_code"] = language_code
                if diarize is not None:
                    kwargs["diarize"] = diarize

                result = self.client.speech_to_text.convert(
                    file=audio_file,
                    **kwargs,
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

    def _extract_first_20s_snippet_local(self, audio_path: str) -> torch.Tensor:
        """
        Extract a snippet from the first 20 seconds of a local audio file.
        
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

    def _process_lyrics_text(self, lyrics_text: str, language_code: Optional[str] = None) -> Dict[str, Any] | None:
        """
        Split lyrics into sentences and compute sentence embeddings + mean pooled embedding.
        Returns an embeddings doc similar to the external app implementation.
        """
        console.print(f"[blue]Processing lyrics text (length: {len(lyrics_text)})[/blue]")
        
        if not lyrics_text or str(lyrics_text).strip() == "":
            console.print("[yellow]No lyrics text provided[/yellow]")
            return None
            
        if self.text_embeddings_model is None:
            console.print(Panel(
                "[red]Text embeddings model not available.[/red]",
                title="Embedding Error", border_style="red"
            ))
            return None
            
        try:
            console.print(f"[blue]Splitting sentences with language_code: {language_code}[/blue]")
            sentences = get_sentences(lyrics_text, language_code)
            console.print(f"[blue]Got {len(sentences)} sentences[/blue]")
            
            if not sentences:
                console.print("[yellow]No sentences extracted[/yellow]")
                return None
                
            console.print("[blue]Generating embeddings for sentences...[/blue]")
            embeddings = self.text_embeddings_model.get_embeddings(sentences)
            console.print(f"[blue]Generated embeddings shape: {embeddings.shape}[/blue]")
            
            mean_embedding = np.mean(embeddings, axis=0)
            console.print(f"[blue]Mean embedding shape: {mean_embedding.shape}[/blue]")
            
            embeddings_doc: Dict[str, Any] = {
                "language_code": language_code,
                "sentences": sentences,
                "embeddings": embeddings.tolist(),
                "mean_embedding": mean_embedding.tolist(),
                "embedding_dim": int(embeddings.shape[1]),
                "num_sentences": int(len(sentences)),
                "processed_at": datetime.now().isoformat(),
            }
            console.print("[green]✅ Text embeddings processing completed successfully[/green]")
            return embeddings_doc
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error embedding lyrics text: {str(e)}[/red]",
                title="Embedding Error", border_style="red"
            ))
            return None

    def _save_text_embedding_to_mongodb(self, song_id: str, embeddings_doc: Dict[str, Any]) -> bool:
        """
        Save mean pooled text embedding to the current collection under `text_embedding`.
        """
        try:
            result = self.collection.update_one(
                {"song_id": song_id},
                {
                    "$set": {
                        "text_embedding": embeddings_doc["mean_embedding"],
                        "embedding_dim": embeddings_doc["embedding_dim"],
                        "num_sentences": embeddings_doc["num_sentences"],
                        "embedding_processed_at": datetime.now().isoformat(),
                    }
                }
            )
            return result.matched_count > 0
        except Exception as e:
            console.print(Panel(
                f"[red]Error saving text embedding to MongoDB: {str(e)}[/red]",
                title="MongoDB Error", border_style="red"
            ))
            return False

    # async def _handle_loadtext_command(self, args_line: str):
    #     """
    #     Handle the /loadtext command by delegating text embedding to embeddings_api.
    #     Usage examples:
    #       /loadtext --text "some lyrics here" [--song_id XYZ]
    #       /loadtext --file /path/to/lyrics.txt [--song_id XYZ]
    #     """
    #     # Default values
    #     lyrics_text: str | None = None
    #     song_id: str | None = None
    #     file_path: str | None = None

    #     # Simple arg parsing
    #     parts = args_line.strip().split()
    #     i = 0
    #     while i < len(parts):
    #         token = parts[i]
    #         if token == "--text" and i + 1 < len(parts):
    #             lyrics_text = " ".join(parts[i + 1:])
    #             break
    #         elif token == "--file" and i + 1 < len(parts):
    #             file_path = parts[i + 1]
    #             i += 2
    #             continue
    #         elif token == "--song_id" and i + 1 < len(parts):
    #             song_id = parts[i + 1]
    #             i += 2
    #             continue
    #         else:
    #             i += 1

    #     # If not provided via flags, treat the raw remainder as text
    #     if lyrics_text is None and file_path is None:
    #         raw = args_line.strip()
    #         lyrics_text = raw if raw else None

    #     if file_path and not lyrics_text:
    #         try:
    #             with open(file_path, "r", encoding="utf-8") as f:
    #                 lyrics_text = f.read()
    #         except Exception as e:
    #             console.print(Panel(
    #                 f"[red]Failed to read file: {e}[/red]",
    #                 title="File Error", border_style="red"
    #             ))
    #             return

    #     if not lyrics_text:
    #         console.print(Panel(
    #             "[red]Please provide lyrics via --text or --file[/red]",
    #             title="Usage Error", border_style="red"
    #         ))
    #         return

    #     try:
    #         api_base = os.getenv("EMBEDDINGS_API_URL", "http://localhost:8001")
    #         url = f"{api_base}/generate-embeddings"
    #         payload = {"text": lyrics_text, "user_id": os.getenv("USER_ID", "cli_user")}

    #         resp = requests.post(url, json=payload, timeout=120)
    #         if resp.status_code != 200:
    #             raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")
    #         data = resp.json()
    #         if not data.get("success"):
    #             raise RuntimeError(data.get("message", "Failed to generate text embedding"))

    #         embeddings = data.get("embeddings", {})
    #         vec = embeddings.get("text_embedding") or embeddings.get("lyrics_embedding")
    #         if not vec:
    #             raise RuntimeError("Embeddings API did not return text embedding")

    #         # Keep in memory for search
    #         self.current_text_embedding = np.array(vec, dtype=np.float32)

    #         # Optionally save to MongoDB if song_id provided
    #         saved_msg = ""
    #         if song_id:
    #             try:
    #                 result = self.collection.update_one(
    #                     {"song_id": song_id},
    #                     {"$set": {
    #                         "text_embedding": vec,
    #                         "embedding_dim": int(len(vec)),
    #                         "embedding_processed_at": datetime.now().isoformat(),
    #                     }}
    #                 )
    #                 if result.matched_count > 0:
    #                     saved_msg = f"\nSaved to MongoDB for song_id: {song_id}"
    #                 else:
    #                     saved_msg = f"\n[Warning] No document updated for song_id: {song_id}"
    #             except Exception as e:
    #                 saved_msg = f"\n[Warning] Failed to save to MongoDB: {e}"

    #         console.print(Panel(
    #             f"[bold green]✅ Text Loaded via API & Embedded[/bold green]\n\n"
    #             f"Embedding dim: {len(vec)}\n"
    #             f"You can now run /search (uses the loaded text embedding)."
    #             f"{saved_msg}",
    #             title="Success", border_style="green"
    #         ))
    #     except Exception as e:
    #         console.print(Panel(
    #             f"[red]Could not generate embeddings via API: {e}[/red]",
    #             title="Embedding Error", border_style="red"
    #         ))
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
            
    async def get_query_from_llm(self, natural_language_query: str, collection_schema: Dict[str, Any], chat_history_context: str = "") -> str:
        """
        Sends the user's query and the schema to an LLM to get a structured MongoDB query.
        
        Args:
            natural_language_query (str): The user's question.
            collection_schema (Dict[str, Any]): A JSON-like representation of the collection schema.
            chat_history_context (str): Previous conversation context for better understanding.
            
        Returns:
            str: A JSON string of the MongoDB query.
        """
        # Build the context section
        context_section = ""
        if chat_history_context.strip():
            context_section = f"""
CONVERSATION CONTEXT:
{chat_history_context}

---
"""

        prompt = f"""
You are an expert at converting natural language questions into structured MongoDB queries.
You will be provided with an example document from the collection, conversation context, and a user's question.
Your task is to generate a JSON object that can be used directly in a PyMongo find() operation.
The JSON object should have the following keys:
- "filter": A MongoDB query filter (e.g., {{ "artist": "Capone" }}).
- "projection": A MongoDB projection to select specific fields (e.g., {{ "title": 1, "artist": 1, "views": 1, "_id": 0 }}).
- "sort": A MongoDB sort object (e.g., {{ "views": -1 }}).
- "limit": An integer limit for the number of results.

{context_section}
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
  "filter": {{ "genres": "latin", "language_code": "spa" }},
  "projection": {{ "song_name": 1, "artist_name": 1, "genres": 1, "language_code": 1, "_id": 0 }},
  "sort": {{ "first_seen": -1 }},
  "limit": 10
}}

User: "Get all songs with RKT"
{{
  "filter": {{ "genres": "rkt"}},
  "projection": {{ "song_name": 1, "artist_name": 1, "genres": 1, "TREND_STATUS": 1, "_id": 0 }},
  "sort": {{ "first_seen": -1 }},
  "limit": 10
}}

User: "give me all jazz songs which trended in july" <- important point, since dates are in charts.Country we need to temporarily convert charts to and array. 
if user asks for dates - we first apply any of the other filters he was asking for, then transform charts and filter by date: 
{{
  "pipeline": [
    {{
      "$match": {{
        "genres": "jazz"
      }}
    }},
    {{
      "$addFields": {{
        "charts_array": {{ "$objectToArray": "$charts" }}
      }}
    }},
    {{
      "$match": {{
        "charts_array.v": {{
          "$elemMatch": {{
            "timestamp": {{
              "$gte": "2025-08-01",
              "$lt": "2025-08-08"
            }}
          }}
        }}
      }}
    }}
  ],
  "projection": {{ 
    "song_name": 1, 
    "artist_name": 1, 
    "charts": 1,
    "_id": 0 
  }},
  "limit": 10
}}

User: "Find me songs which were popular in the last week"
{{
  "pipeline": [
    {{
      "$addFields": {{
        "charts_array": {{ "$objectToArray": "$charts" }}
      }}
    }},
    {{
      "$match": {{
        "charts_array.v": {{
          "$elemMatch": {{
            "timestamp": {{
              "$gte": "2025-08-04"
            }}
          }}
        }}
      }}
    }},
    {{
      "$sort": {{
        "first_seen": -1
      }}
    }}
  ],
  "projection": {{
    "song_name": 1,
    "artist_name": 1,
    "first_seen": 1,
    "charts": 1,
    "_id": 0
  }},
  "limit": 10
}}

Notes: 

-genres always are lowercase, usually with a space between words: hip hop, r&b, electronic dance music, etc.
-only set TREND_STATUS to "PROCESSED" if user asks for a song with a distinct video trend (e.g "find me a song with a dance trend related to it", "find me song wiht a meme trend")
-country names always start with a capital letter: Brazil, Argentina, etc. no short forms. United States, United Kingdom, etc.
-when user asks for "trending" songs, it doesn't mean that the TREND_STATUS is PROCESSED - but that the song is in charts. 

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
            try:
                print(f"[DEBUG] LLM(get_query_from_llm) raw=\n{text}")
            except Exception:
                pass
            #print(text)
            return text
            
        except Exception as e:
            console.print(f"[red]OpenAI API call failed: {e}[/red]")
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
            self.trends_collection=self.db.TOP_TIKTOK_TRENDS
            self.query_generator = MongoDBQueryGenerator()
            

            
            # Do not initialize heavy models in chat runtime; embeddings handled by external API
            self.mert_embedder = None
            self.current_embedding = None  # Store the current audio embedding
            self.current_transcription_text = None  # Store last transcription text
            
            # Initialize chat history
            self.chat_history = []  # Store conversation history as list of dicts
            self.max_history_length = 10  # Maximum number of exchanges to keep
            
            # Initialize search results storage
            self.proposed_results = []  # Store last search results for /add command
            
            # Initialize context storage
            self.context_songs = []  # Store songs and content in user's context
            
            # Do not load genre/text models here; rely on embeddings_api instead
            self.genre_model = None
            self.genre_labels = None
            self.text_embeddings_model = None
            self.current_text_embedding = None  # Mean-pooled lyrics embedding kept in memory
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
                    "timestamp": "string (date) - YYYY-MM-DD",
                    "rank": "number"
                }]
            },
            "first_seen": "string (date) - YYYY-MM-DD",
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

    def _process_lyrics_text(self, lyrics_text: str, language_code: Optional[str] = None) -> Dict[str, Any] | None:
        """
        Split lyrics into sentences and compute sentence embeddings + mean pooled embedding.
        Returns an embeddings doc similar to the external app implementation.
        """
        console.print(f"[blue]Processing lyrics text (length: {len(lyrics_text)})[/blue]")
        
        if not lyrics_text or str(lyrics_text).strip() == "":
            console.print("[yellow]No lyrics text provided[/yellow]")
            return None
            
        if self.text_embeddings_model is None:
            console.print(Panel(
                "[red]Text embeddings model not available.[/red]",
                title="Embedding Error", border_style="red"
            ))
            return None
            
        try:
            console.print(f"[blue]Splitting sentences with language_code: {language_code}[/blue]")
            sentences = get_sentences(lyrics_text, language_code)
            console.print(f"[blue]Got {len(sentences)} sentences[/blue]")
            
            if not sentences:
                console.print("[yellow]No sentences extracted[/yellow]")
                return None
                
            console.print("[blue]Generating embeddings for sentences...[/blue]")
            embeddings = self.text_embeddings_model.get_embeddings(sentences)
            console.print(f"[blue]Generated embeddings shape: {embeddings.shape}[/blue]")
            
            mean_embedding = np.mean(embeddings, axis=0)
            console.print(f"[blue]Mean embedding shape: {mean_embedding.shape}[/blue]")
            
            embeddings_doc: Dict[str, Any] = {
                "language_code": language_code,
                "sentences": sentences,
                "embeddings": embeddings.tolist(),
                "mean_embedding": mean_embedding.tolist(),
                "embedding_dim": int(embeddings.shape[1]),
                "num_sentences": int(len(sentences)),
                "processed_at": datetime.now().isoformat(),
            }
            console.print("[green]✅ Text embeddings processing completed successfully[/green]")
            return embeddings_doc
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error embedding lyrics text: {str(e)}[/red]",
                title="Embedding Error", border_style="red"
            ))
            return None

    def _save_text_embedding_to_mongodb(self, song_id: str, embeddings_doc: Dict[str, Any]) -> bool:
        """
        Save mean pooled text embedding to the current collection under `text_embedding`.
        """
        try:
            result = self.collection.update_one(
                {"song_id": song_id},
                {
                    "$set": {
                        "text_embedding": embeddings_doc["mean_embedding"],
                        "embedding_dim": embeddings_doc["embedding_dim"],
                        "num_sentences": embeddings_doc["num_sentences"],
                        "embedding_processed_at": datetime.now().isoformat(),
                    }
                }
            )
            return result.matched_count > 0
        except Exception as e:
            console.print(Panel(
                f"[red]Error saving text embedding to MongoDB: {str(e)}[/red]",
                title="MongoDB Error", border_style="red"
            ))
            return False
    async def _handle_loadtext_command(self, args_line: str):
        """
        Handle the /loadtext command by delegating text embedding to embeddings_api.
        Usage examples:
          /loadtext --text "some lyrics here" [--song_id XYZ]
          /loadtext --file /path/to/lyrics.txt [--song_id XYZ]
        """
        # Default values
        lyrics_text: str | None = None
        song_id: str | None = None
        file_path: str | None = None

        # Simple arg parsing
        parts = args_line.strip().split()
        i = 0
        while i < len(parts):
            token = parts[i]
            if token == "--text" and i + 1 < len(parts):
                lyrics_text = " ".join(parts[i + 1:])
                break
            elif token == "--file" and i + 1 < len(parts):
                file_path = parts[i + 1]
                i += 2
                continue
            elif token == "--song_id" and i + 1 < len(parts):
                song_id = parts[i + 1]
                i += 2
                continue
            else:
                i += 1

        # If not provided via flags, treat the raw remainder as text
        if lyrics_text is None and file_path is None:
            raw = args_line.strip()
            lyrics_text = raw if raw else None

        if file_path and not lyrics_text:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lyrics_text = f.read()
            except Exception as e:
                console.print(Panel(
                    f"[red]Failed to read file: {e}[/red]",
                    title="File Error", border_style="red"
                ))
                return

        if not lyrics_text:
            console.print(Panel(
                "[red]Please provide lyrics via --text or --file[/red]",
                title="Usage Error", border_style="red"
            ))
            return

        try:
            api_base = os.getenv("EMBEDDINGS_API_URL", "http://localhost:8001")
            url = f"{api_base}/generate-embeddings"
            payload = {"text": lyrics_text, "user_id": os.getenv("USER_ID", "cli_user")}

            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")
            data = resp.json()
            if not data.get("success"):
                raise RuntimeError(data.get("message", "Failed to generate text embedding"))

            embeddings = data.get("embeddings", {})
            vec = embeddings.get("text_embedding") or embeddings.get("lyrics_embedding")
            if not vec:
                raise RuntimeError("Embeddings API did not return text embedding")

            # Keep in memory for search
            self.current_text_embedding = np.array(vec, dtype=np.float32)

            # Optionally save to MongoDB if song_id provided
            saved_msg = ""
            if song_id:
                try:
                    result = self.collection.update_one(
                        {"song_id": song_id},
                        {"$set": {
                            "text_embedding": vec,
                            "embedding_dim": int(len(vec)),
                            "embedding_processed_at": datetime.now().isoformat(),
                        }}
                    )
                    if result.matched_count > 0:
                        saved_msg = f"\nSaved to MongoDB for song_id: {song_id}"
                    else:
                        saved_msg = f"\n[Warning] No document updated for song_id: {song_id}"
                except Exception as e:
                    saved_msg = f"\n[Warning] Failed to save to MongoDB: {e}"

            console.print(Panel(
                f"[bold green]✅ Text Loaded via API & Embedded[/bold green]\n\n"
                f"Embedding dim: {len(vec)}\n"
                f"You can now run /search (uses the loaded text embedding)."
                f"{saved_msg}",
                title="Success", border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                f"[red]Could not generate embeddings via API: {e}[/red]",
                title="Embedding Error", border_style="red"
            ))

    def _add_to_chat_history(self, user_message: str, assistant_response: str, category: str = "general"):
        """
        Add an exchange to the chat history.
        
        Args:
            user_message (str): The user's message
            assistant_response (str): The assistant's response
            category (str): The category of interaction (search, talk, help, etc.)
        """
        self.chat_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "category": category,
            
        })
        
        # Keep only the last max_history_length exchanges
        if len(self.chat_history) > self.max_history_length:
            self.chat_history = self.chat_history[-self.max_history_length:]
    
    def _get_chat_history_context(self, include_categories: Optional[List[str]] = None) -> str:
        """
        Get formatted chat history for context in LLM calls.
        
        Args:
            include_categories (Optional[List[str]]): Only include exchanges from these categories
            
        Returns:
            str: Formatted chat history
        """
        if not self.chat_history:
            return ""
        
        filtered_history = self.chat_history
        if include_categories:
            filtered_history = [
                exchange for exchange in self.chat_history 
                if exchange.get("category", "general") in include_categories
            ]
        
        if not filtered_history:
            return ""
        
        context_parts = ["Previous conversation context:"]
        for exchange in filtered_history[-5:]:  # Last 5 exchanges for context
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant'][:200]}...")  # Truncate long responses
        
        return "\n".join(context_parts)
    
    def _clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history.clear()
        console.print(Panel(
            "[green]Chat history cleared successfully![/green]",
            title="History Cleared", border_style="green"
        ))
    
    def _show_chat_history(self):
        """Display the current chat history."""
        if not self.chat_history:
            console.print(Panel(
                "[yellow]No chat history available.[/yellow]",
                title="Chat History", border_style="yellow"
            ))
            return
        
        history_text = []
        for i, exchange in enumerate(self.chat_history, 1):
            timestamp = exchange.get("timestamp", "Unknown")
            category = exchange.get("category", "general")
            history_text.append(f"[bold blue]{i}. [{category.upper()}] - {timestamp}[/bold blue]")
            history_text.append(f"[cyan]You:[/cyan] {exchange['user']}")
            history_text.append(f"[green]Assistant:[/green] {exchange['assistant'][:150]}...")
            history_text.append("")
        
        console.print(Panel(
            "\n".join(history_text),
            title=f"Chat History ({len(self.chat_history)} exchanges)", 
            border_style="blue"
        ))

    async def _handle_add_command(self, user_input: str):
        """Handle /add i command to add i'th search result to context."""
        try:
            # Parse the index from the command
            parts = user_input.strip().split()
            if len(parts) != 2:
                console.print("[red]Usage: /add <index> (e.g., /add 1)[/red]")
                return
            
            try:
                index = int(parts[1]) - 1  # Convert to 0-based index
            except ValueError:
                console.print("[red]Please provide a valid number (e.g., /add 1)[/red]")
                return
            
            # Check if we have search results
            if not self.proposed_results:
                console.print("[yellow]No search results available. Please perform a search first.[/yellow]")
                return
            
            # Check if index is valid
            if index < 0 or index >= len(self.proposed_results):
                console.print(f"[red]Invalid index. Please choose between 1 and {len(self.proposed_results)}.[/red]")
                return
            
            # Get the selected result
            selected_result = self.proposed_results[index]
            trend_status = selected_result.get("TREND_STATUS")
            
            # Check if the song has trend status
            if trend_status != "PROCESSED":
                console.print("[red]Can't add: Selected song doesn't have trend information.[/red]")
                return
            
            # Get detailed trend information from TOP_TIKTOK_TRENDS collection
            trend_details = self.trends_collection.find_one({})  # Get the first (and likely only) trend
            
            if not trend_details:
                console.print("[red]Can't add: No trend details found in database.[/red]")
                return
            
            # Extract detailed information
            detailed_description = trend_details.get("detailedDescription", "")
            trend_explanation = trend_details.get("trendExplanation", "")
            
            # Create context entry
            context_entry = {
                "song_name": selected_result.get("song_name", "N/A"),
                "artist_name": selected_result.get("artist_name", "N/A"),
                "genres": selected_result.get("genres", []),
                "lyrics": selected_result.get("lyrics", ""),
                "trend_description": selected_result.get("trend_description", ""),
                "detailed_description": detailed_description,
                "trend_explanation": trend_explanation,
                "added_at": str(datetime.now())
            }
            
            # Store in context (initialize if needed)
            if not hasattr(self, 'context_songs'):
                self.context_songs = []
            
            self.context_songs.append(context_entry)
            
            # Simple success message
            console.print(f"[green]✅ Added '{context_entry['song_name']}' by {context_entry['artist_name']} to context[/green]")
            
        except Exception as e:
            console.print(f"[red]Error adding song to context: {e}[/red]")

    async def _handle_analysis_query(self, user_input: str) -> str:
        """Handle deep analysis queries using o3-mini reasoning model with context data."""
        try:
            console.print(f"[blue]Debug Analysis Handler: Called with input '{user_input}'[/blue]")
            
            # Check if we have context songs to analyze
            if not hasattr(self, 'context_songs') or not self.context_songs:
                console.print("[blue]Debug Analysis Handler: No context songs found[/blue]")
                return """I don't have any songs in context to analyze yet. To perform deep analysis, please:

1. Search for songs related to your question (e.g., "find songs with trends in Sweden")
2. Use /add <index> to add interesting results to context
3. Then ask your analysis questions

For example:
• Search: "find songs that charted in Sweden"
• Add: "/add 1" (adds first result to context)  
• Analyze: "explain why this trend became popular in Sweden"

Would you like me to help you search for relevant songs first?"""
            
            console.print(f"[blue]Debug Analysis Handler: Found {len(self.context_songs)} context songs[/blue]")
            
            # Build context from stored songs
            context_parts = []
            context_parts.append("CONTEXT INFORMATION:")
            
            for i, song in enumerate(self.context_songs, 1):
                context_parts.append(f"\nSONG {i}:")
                context_parts.append(f"Title: {song.get('song_name', 'N/A')}")
                context_parts.append(f"Artist: {song.get('artist_name', 'N/A')}")
                
                if song.get('genres'):
                    context_parts.append(f"Genres: {', '.join(song['genres'])}")
                
                if song.get('trend_description'):
                    context_parts.append(f"Trend: {song['trend_description']}")
                
                if song.get('detailed_description'):
                    context_parts.append(f"Detailed Description: {song['detailed_description']}")
                
                if song.get('trend_explanation'):
                    context_parts.append(f"Trend Explanation: {song['trend_explanation']}")
                
                if song.get('lyrics'):
                    # Truncate lyrics for context
                    lyrics_preview = song['lyrics'][:500] + "..." if len(song['lyrics']) > 500 else song['lyrics']
                    context_parts.append(f"Lyrics Preview: {lyrics_preview}")
            
            context_data = "\n".join(context_parts)
            
            # Build the analysis prompt
            analysis_prompt = f"""
You are an expert music analyst and cultural researcher. Use the provided context information to give a deep, insightful analysis.

{context_data}

USER QUESTION: {user_input}

Please provide a comprehensive analysis that:
1. Uses the specific context information provided above
2. Explains cultural, social, and musical factors
3. Provides reasoning backed by the data
4. Connects trends to broader cultural phenomena
5. Offers insights into why certain patterns emerged

Be specific, detailed, and use the context data to support your analysis.
"""
            
            import openai
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=self.query_generator.api_key)
            
            # Use o3-mini for deep reasoning
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an expert music analyst and cultural researcher. Provide deep, insightful analysis using the context data provided."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_completion_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            console.print(f"[red]Error in analysis: {e}[/red]")
            return "I encountered an error while performing the analysis. Please try again."

    def _predict_genres(self, music_embedding: torch.Tensor, text_embedding: Optional[torch.Tensor] = None) -> List[str]:
        """
        Predict genres using concatenated music and text embeddings.
        
        Args:
            music_embedding: MERT embedding tensor of shape [1024]
            text_embedding: Optional text embedding tensor of shape [768] or similar
            
        Returns:
            List[str]: List of predicted genre labels
        """
        if self.genre_model is None or self.genre_labels is None:
            return []
        
        try:
            with torch.no_grad():
                # Prepare text embedding (use zeros if not available)
                if text_embedding is None:
                    # Use zeros for text embedding if not available
                    # Assuming text embedding dimension is 768 (common for sentence transformers)
                    text_embedding = torch.zeros(768, dtype=music_embedding.dtype, device=music_embedding.device)
                elif text_embedding.dim() == 1:
                    text_embedding = text_embedding.unsqueeze(0)  # Add batch dimension
                
                # Ensure music embedding has batch dimension
                if music_embedding.dim() == 1:
                    music_embedding = music_embedding.unsqueeze(0)  # Add batch dimension
                
                # Concatenate music + text embeddings (same order as training)
                combined_embedding = torch.cat([music_embedding, text_embedding], dim=1)
                
                # Get logits from model
                logits = self.genre_model(combined_embedding)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(logits)
                
                # Get the top-1 predicted genre (highest probability)
                top_index = torch.argmax(probabilities, dim=1)[0]
                predicted_genres = [self.genre_labels[top_index]]
                return predicted_genres
        except Exception as e:
            console.print(f"[red]Error predicting genres: {e}[/red]")
            return []

    async def _handle_loadtext_command_legacy(self, args_line: str):
        """
        Handle the /loadtext command.
        Usage examples:
          /loadtext --text "some lyrics here" --lang eng [--song_id XYZ]
          /loadtext --file /path/to/lyrics.txt --lang spa [--song_id XYZ]
        If --song_id is provided, will also save the mean embedding to MongoDB under `text_embedding`.
        Always keeps the current mean text embedding in memory for vector search.
        """
        # Default values
        lyrics_text: str | None = None
        language_code: Optional[str] = None
        song_id: str | None = None
        file_path: str | None = None

        # Simple arg parsing
        parts = args_line.strip().split()
        i = 0
        while i < len(parts):
            token = parts[i]
            if token == "--text" and i + 1 < len(parts):
                # Collect the rest as text
                lyrics_text = " ".join(parts[i + 1:])
                break
            elif token == "--file" and i + 1 < len(parts):
                file_path = parts[i + 1]
                i += 2
                continue
            elif token == "--lang" and i + 1 < len(parts):
                language_code = parts[i + 1]
                i += 2
                continue
            elif token == "--song_id" and i + 1 < len(parts):
                song_id = parts[i + 1]
                i += 2
                continue
            else:
                i += 1

        # If not provided via flags, treat the raw remainder as text
        if lyrics_text is None and file_path is None:
            raw = args_line.strip()
            lyrics_text = raw if raw else None

        if file_path and not lyrics_text:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lyrics_text = f.read()
            except Exception as e:
                console.print(Panel(
                    f"[red]Failed to read file: {e}[/red]",
                    title="File Error", border_style="red"
                ))
                return

        if not lyrics_text:
            console.print(Panel(
                "[red]Please provide lyrics via --text or --file[/red]",
                title="Usage Error", border_style="red"
            ))
            return

        embeddings_doc = self._process_lyrics_text(lyrics_text, language_code)
        if not embeddings_doc:
            console.print(Panel(
                "[red]Could not generate embeddings for the provided lyrics.[/red]",
                title="Embedding Error", border_style="red"
            ))
            return

        # Keep in memory for search
        self.current_text_embedding = np.array(embeddings_doc["mean_embedding"], dtype=np.float32)

        saved_msg = ""
        if song_id:
            if self._save_text_embedding_to_mongodb(song_id, embeddings_doc):
                saved_msg = f"\nSaved to MongoDB for song_id: {song_id}"
            else:
                saved_msg = f"\n[Warning] No document updated for song_id: {song_id}"

        console.print(Panel(
            f"[bold green]✅ Text Loaded & Embedded[/bold green]\n\n"
            f"Sentences: {embeddings_doc['num_sentences']}\n"
            f"Embedding dim: {embeddings_doc['embedding_dim']}\n"
            f"Language: {embeddings_doc['language_code']}\n"
            f"You can now run /search (uses the loaded text embedding)."
            f"{saved_msg}",
            title="Success", border_style="green"
        ))

    def _format_results(self, documents: List[Dict[str, Any]]):
        """Formats the list of documents for a nice display."""
        if not documents:
            console.print("[yellow]No documents found matching your query.[/yellow]")
            return

        for i, doc in enumerate(documents, 1):
            song_name = doc.get("song_name", "N/A")
            artist_name = doc.get("artist_name", "N/A")
            genres = ', '.join(doc.get("genres", []))
            
            # charts_info = ""
            # charts = doc.get("charts", {})
            # for country, chart_entries in charts.items():
            #     if chart_entries:
            #         latest_entry = chart_entries[0]  # Assuming sorted by timestamp
            #         rank_data = latest_entry.get("rank", {})
            #         # Handle both formats: {"$numberInt": "1"} and direct integer
            #         if isinstance(rank_data, dict) and "$numberInt" in rank_data:
            #             rank = rank_data["$numberInt"]
            #         elif isinstance(rank_data, (int, str)):
            #             rank = str(rank_data)
            #         else:
            #             rank = "N/A"
            #         timestamp = latest_entry.get("timestamp", "N/A")
            #         charts_info += f"  - [bold]{country}:[/bold] Rank {rank} (as of {timestamp})\n"

            # Add trend description if available
            trend_description = doc.get("trend_description")
            trend_status = doc.get("TREND_STATUS")
            
            trend_info = ""
            if trend_description and trend_status == "PROCESSED":
                trend_info = f"[bold]Trend:[/bold] {trend_description}\n"
            
            panel_content = (
                f"[bold blue]Result {i}[/bold blue]\n\n"
                f"[bold]Song Name:[/bold] {song_name}\n"
                f"[bold]Artist:[/bold] {artist_name}\n"
                f"[bold]Genres:[/bold] {genres}\n"
                f"{trend_info}"
                #f"[bold]First Seen:[/bold] {doc.get('first_seen', 'N/A')}\n"
            )
            
            # if charts_info:
            #     panel_content += f"\n[bold]Latest Chart Ranks:[/bold]\n{charts_info}"

            console.print(Panel(
                panel_content,
                border_style="green"
            ))

    async def _handle_load_command(self, file_path: str):
        """Handle the /load command by delegating embedding to embeddings_api service."""
        try:
            console.print(Panel(
                f"[bold blue]🎵 Loading Audio via Embeddings API[/bold blue]\n\n"
                f"File: {file_path}",
                title="Audio Load", border_style="blue"
            ))

            api_base = os.getenv("EMBEDDINGS_API_URL", "http://localhost:8001")
            url = f"{api_base}/generate-embeddings"
            payload = {"file_path": file_path, "user_id": os.getenv("USER_ID", "cli_user")}

            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")

            data = resp.json()
            if not data.get("success"):
                raise RuntimeError(data.get("message", "Failed to generate embeddings"))

            embeddings = data.get("embeddings", {})
            audio_vec = embeddings.get("audio_embedding")
            lyrics_vec = embeddings.get("lyrics_embedding") or embeddings.get("text_embedding")
            transcript_text = embeddings.get("transcript")

            if audio_vec:
                self.current_embedding = torch.tensor(audio_vec, dtype=torch.float32)
            else:
                self.current_embedding = None

            if lyrics_vec:
                self.current_text_embedding = np.array(lyrics_vec, dtype=np.float32)
            else:
                self.current_text_embedding = None

            self.current_transcription_text = transcript_text or ""

            emb_info = f"Embedding: {[len(audio_vec)] if audio_vec else 'N/A'}"
            text_info = "\nTranscript chars: " + (str(len(self.current_transcription_text)) if self.current_transcription_text else "0")

            console.print(Panel(
                f"[bold green]✅ Loaded via API & Stored[/bold green]\n\n"
                f"{emb_info}{text_info}\n\n"
                f"You can now run /search (uses the loaded embedding).",
                title="Success", border_style="green"
            ))
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error loading audio via API: {str(e)}[/red]",
                title="Error", border_style="red"
            ))

    async def _categorize_input(self, user_input: str) -> str:
        """
        Categorize user input into one of four categories using GPT-3.5.
        
        Args:
            user_input (str): The user's input message
            
        Returns:
            str: Category - "help", "talk", "search", or "analysis"
        """
        prompt = f"""
You are an input categorizer for a music database chatbot. Categorize the user's input into exactly one of these four categories:

1. "help" - User is asking for help, instructions, or wants to know what the system can do
   Examples: "help", "what can you do", "how does this work", "show me the commands"

2. "talk" - User is making casual conversation, greetings, or general chat
   Examples: "hello", "how are you", "thanks", "goodbye", "nice to meet you"

3. "search" - User wants to search the database with natural language queries or find similar songs/lyrics
   Examples: "find songs by artist X", "show me Latin songs", "songs from Brazil", "popular songs", "find similar songs", "match this song", "what songs are like this", "find similar lyrics",
   "now make sure the song is in Japan" <- extending the query, "now filter the songs whcih have a genger rap" <- extending the query, 

4. "analysis" - User wants deep analysis, explanations, or reasoning about trends, songs, genres, or cultural phenomena
   Examples: "explain why this trend took off in Germany", "tell me more about this song and genre", "why did this become popular", "analyze the cultural impact", "what makes this genre unique", "explain the trend phenomenon", "analyze why this trend became popular", "explain me why this trend became so popular", "how could I use this trend", "why this trend mecame so popular"
   
   KEY INDICATORS for analysis:
   - Contains words like: "explain", "analyze", "why", "how", "what makes", "tell me more about"
   - Asks about trends, popularity, cultural phenomena, genre characteristics
   - Seeks reasoning or deep understanding rather than just data

User input: "{user_input}"

Respond with ONLY the category name: help, talk, search, or analysis
"""
        
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=self.query_generator.api_key)
        
        try:
            # Make API call to OpenAI with cheaper model
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a simple categorizer. Respond with only one word: help, talk, search, or analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract the response text and clean it
            raw_response = response.choices[0].message.content.strip()
            try:
                print(f"[DEBUG] LLM(_categorize_input) raw=\n{raw_response}")
            except Exception:
                pass
            category = raw_response.lower()
            
            # Debug: Print raw orchestrator response
            console.print(f"[blue]Debug Orchestrator Raw Response: '{raw_response}'[/blue]")
            
            # Ensure it's one of the valid categories
            if category not in ["help", "talk", "search", "analysis"]:
                console.print(f"[yellow]Warning: Invalid category '{category}' returned, defaulting to 'talk'[/yellow]")
                category = "talk"  # Default fallback
                
            return category
            
        except Exception as e:
            console.print(f"[red]Error categorizing input: {e}[/red]")
            return "talk"  # Default fallback

    async def _generate_chat_response(self, user_input: str) -> str:
        """
        Generate a contextual chat response using chat history.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            str: Generated response
        """
        # Get chat history context for talk category
        chat_context = self._get_chat_history_context(include_categories=["talk", "help"])
        
        # Build the context section
        context_section = ""
        if chat_context.strip():
            context_section = f"""
CONVERSATION HISTORY:
{chat_context}

---
"""

        prompt = f"""
You are a friendly and knowledgeable music database assistant. You have access to a MongoDB database of TikTok songs with chart information, genres, lyrics, and audio embeddings.

{context_section}
Your capabilities include:
- Searching for songs by various criteria (artist, genre, country charts, etc.)
- Finding similar songs using audio and lyrics embeddings
- Providing information about music trends and charts
- Helping users understand how to use the system

Current user message: "{user_input}"

Please provide a helpful, conversational response that:
1. Acknowledges the conversation history if relevant
2. Offers specific assistance related to music database queries
3. Suggests concrete next steps or examples
4. Maintains a friendly, professional tone
5. Keeps the response concise (2-3 sentences)

Respond naturally as if you're having a conversation with the user.
"""
        
        import openai
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=self.query_generator.api_key)
        
        try:
            # Make API call to OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a friendly music database assistant. Keep responses concise and helpful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            # Extract the response text
            text = response.choices[0].message.content.strip()
            try:
                print(f"[DEBUG] LLM(_generate_chat_response) raw=\n{text}")
            except Exception:
                pass
            return text
            
        except Exception as e:
            console.print(f"[red]Error generating chat response: {e}[/red]")
            return "I'm here to help you explore the music database! You can ask me to find songs, search by artist or genre, or discover similar music. What would you like to explore?"

    async def _get_search_parameters_from_llm(self, search_comment: str, include_chat_history: bool = True) -> Dict[str, Any]:
        """
        Use ChatGPT to deduce additional search parameters from user comment.
        Args:
            search_comment (str): User's search comment (e.g., "in Germany", "Latin songs", etc.)
            include_chat_history (bool): Whether to include chat history for context
            
        Returns:
            Dict[str, Any]: Search parameters including filters and limit
        """
        # Get chat history context for search queries
        context_section = ""
        if include_chat_history:
            chat_context = self._get_chat_history_context(include_categories=["search"])
            if chat_context.strip():
                context_section = f"""
PREVIOUS SEARCH CONTEXT:
{chat_context}

---
"""

        prompt = f"""
You are an expert at converting natural language search requests into MongoDB query parameters.
Given a user's search comment and previous conversation context, determine what additional filters should be applied to a music vector search.

{context_section}

Available fields in the database:
- genres: lowercase, Array of strings (e.g., ["latin", "pop", "rkt"])
- charts: Object with country codes as keys (e.g., "Germany", "Brazil", "Argentina")
  Each country chart contains an array of objects: [{{"timestamp": "2025-07-08", "rank": 2}}, ...]
- first_seen: Date (e.g., "2025-07-08")
- last_seen: Date (e.g., "2025-08-08")
- language_code: String (e.g., "spa", "eng", "por")
- language_probability: Number (0-1)
- audio_metadata.duration: Number (in seconds)
- TREND_STATUS: String (e.g., "PROCESSED", "UNPROCESSED")<- songs which have a consistent trend with videos, don't use if user asks for "trending"
- country names are always full names like "United States" or "United Kingdom" or "United Arab Emirates"
- genres are always lowercase, usually with a space between words: hip hop, r&b, electronic dance music, etc.
- only set TREND_STATUS to "PROCESSED" if user asks for a song with a distinct video trend (e.g "find me a song with a dance trend related to it", "find me song wiht a meme trend")
User comment: "{search_comment}"

Return a JSON object with the following structure:
{{
    "filters": {{}},  // MongoDB filters to apply
    "limit": 10,     // Number of results to return
    "description": "Brief description of what this search is looking for"
}}

Examples:
- "in Germany" → {{"filters": {{"charts.Germany": {{"$exists": true}}}}, "limit": 10, "description": "Songs that charted in Germany"}}
- "trending in Germany in July" → {{"filters": {{"country": "Germany", "first_seen": {{"$lte": "2025-08-01"}}, "last_seen": {{"$gte": "2025-07-01"}}}}, "limit": 10, "description": "Songs trending in Germany during July 2025"}}
- "popular in Brazil in March" → {{"filters": {{"charts.Brazil": {{"$exists": true}}, "first_seen": {{"$lte": "2025-04-01"}}, "last_seen": {{"$gte": "2025-03-01"}}}}, "limit": 15, "description": "Songs popular in Brazil during March 2025"}}
- "Latin songs" → {{"filters": {{"genres": "latin"}}, "limit": 10, "description": "Latin genre songs"}}
- "Spanish lyrics" → {{"filters": {{"language_code": "spa"}}, "limit": 10, "description": "Songs with Spanish lyrics"}}



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
            try:
                print(f"[DEBUG] LLM(_get_search_parameters_from_llm) raw=\n{text}")
            except Exception:
                pass
            return json.loads(text)
            
        except Exception as e:
            console.print(f"[red]Error parsing search parameters: {e}[/red]")
            return {"filters": {}, "limit": 10, "description": "Simple music search"}

    async def _determine_search_embeddings(self, search_query: str) -> Dict[str, Any]:
        """
        Use LLM to determine which embeddings should be used for the search.
        
        Args:
            search_query (str): The user's search query
            
        Returns:
            Dict[str, Any]: Search parameters including which embeddings to use
        """
        # Check what embeddings are available
        has_music_embedding = hasattr(self, 'current_embedding') and self.current_embedding is not None
        has_text_embedding = hasattr(self, 'current_text_embedding') and self.current_text_embedding is not None
        
        # If no embeddings are available, skip LLM decision and return filter-only search
        if not has_music_embedding and not has_text_embedding:
            return {
                "use_music_embedding": False,
                "use_text_embedding": False,
                "search_type": "filter_only",
                "reasoning": "No embeddings available, using filter-only search"
            }
        
        embedding_status = {
            "music_embedding": has_music_embedding,
            "text_embedding": has_text_embedding
        }
        
        prompt = f"""
You are an expert at determining which embeddings should be used for music database searches.
Given a user's search query and available embeddings, determine the best search approach.

Available embeddings: {embedding_status}
User query: "{search_query}"

Return a JSON object with the following structure:
{{
    "use_music_embedding": true/false,  // Whether to use music (audio) embedding
    "use_text_embedding": true/false,   // Whether to use text (lyrics) embedding
    "search_type": "vector_only" | "filter_only" | "hybrid",  // Type of search to perform
    "reasoning": "Brief explanation of the decision"
}}

Guidelines:
- If query mentions "similar songs", "like this", "match", "audio", "sound" → prefer music embedding
- If query mentions "lyrics", "text", "words", "meaning" → prefer text embedding  
- If query mentions both audio and lyrics → use both embeddings
- If query is purely text-based filtering (artist, genre, country) → use filter_only
- If no embeddings available → use filter_only

Examples:
- "find similar songs" → {{"use_music_embedding": true, "use_text_embedding": false, "search_type": "vector_only", "reasoning": "Looking for similar audio"}}
- "songs with similar lyrics" → {{"use_music_embedding": false, "use_text_embedding": true, "search_type": "vector_only", "reasoning": "Looking for similar text content"}}
- "Latin songs in Brazil" → {{"use_music_embedding": false, "use_text_embedding": false, "search_type": "filter_only", "reasoning": "Pure text filtering"}}

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
                    {"role": "system", "content": "You are an expert at determining which embeddings should be used for music database searches. Provide only JSON output."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            # Extract the response text
            text = response.choices[0].message.content.strip()
            try:
                print(f"[DEBUG] LLM(_determine_search_embeddings) raw=\n{text}")
            except Exception:
                pass
            return json.loads(text)
            
        except Exception as e:
            console.print(f"[red]Error determining search embeddings: {e}[/red]")
            # Fallback: use available embeddings if any
            return {
                "use_music_embedding": has_music_embedding,
                "use_text_embedding": has_text_embedding,
                "search_type": "vector_only" if (has_music_embedding or has_text_embedding) else "filter_only",
                "reasoning": "Fallback decision"
            }

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
            vector_path = "music_embedding"
            search_comment = ""
            
            # Process file if provided
            if file_path:
                console.print(Panel(
                    f"[bold blue]🎵 Processing Audio File via Embeddings API[/bold blue]\n\n"
                    f"File: {file_path}",
                    title="Audio Processing", border_style="blue"
                ))

                api_base = os.getenv("EMBEDDINGS_API_URL", "http://localhost:8001")
                url = f"{api_base}/generate-embeddings"
                payload = {"file_path": file_path, "user_id": os.getenv("USER_ID", "cli_user")}
                resp = requests.post(url, json=payload, timeout=120)
                if resp.status_code != 200:
                    raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")
                data = resp.json()
                if not data.get("success"):
                    raise RuntimeError(data.get("message", "Failed to generate embeddings"))

                embeddings = data.get("embeddings", {})
                audio_vec = embeddings.get("audio_embedding")
                search_comment = embeddings.get("transcript", "")
                if audio_vec:
                    query_vector = audio_vec
                    vector_path = "music_embedding"
                else:
                    query_vector = None
                # If user did not pass a prompt, use transcription text as prompt
                if (prompt is None or str(prompt).strip() == "") and search_comment:
                    prompt = search_comment
                
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
                    vector_path = "music_embedding"
                    console.print(Panel(
                        "[blue]Using previously loaded audio embedding[/blue]",
                        title="Info", border_style="blue"
                    ))
                elif hasattr(self, 'current_text_embedding') and self.current_text_embedding is not None:
                    query_vector = self.current_text_embedding.tolist()
                    vector_path = "text_embedding"
                    console.print(Panel(
                        "[blue]Using previously loaded text embedding[/blue]",
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
                    "path": vector_path,
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
                        "$lookup": {
                            "from": "TOP_TIKTOK_TRENDS",
                            "pipeline": [{"$limit": 1}],  # Get the first (and likely only) trend
                            "as": "trend_info"
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
                            "audio_metadata": 1,
                            "TREND_STATUS": 1,
                            "trend_description": {
                                "$cond": {
                                    "if": {"$eq": ["$TREND_STATUS", "PROCESSED"]},
                                    "then": {"$arrayElemAt": ["$trend_info.trend_description", 0]},
                                    "else": None
                                }
                            }
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
                        "$lookup": {
                            "from": "TOP_TIKTOK_TRENDS",
                            "pipeline": [{"$limit": 1}],  # Get the first (and likely only) trend
                            "as": "trend_info"
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
                            "first_seen": 1,
                            "charts": 1,
                            "language_code": 1,
                            "audio_metadata": 1,
                            "TREND_STATUS": 1,
                            "trend_description": {
                                "$cond": {
                                    "if": {"$eq": ["$TREND_STATUS", "PROCESSED"]},
                                    "then": {"$arrayElemAt": ["$trend_info.trend_description", 0]},
                                    "else": None
                                }
                            }
                        }
                    },
                    {
                        "$limit": limit
                    }
                ]
                
            results = list(self.collection.aggregate(pipeline))
            
            # Store results for /add command
            self.proposed_results = results
            
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
                
                # charts_info = ""
                # charts = result.get("charts", {})
                # for country, chart_entries in charts.items():
                #     if chart_entries:
                #         latest_entry = chart_entries[0]
                #         rank_data = latest_entry.get("rank", {})
                #         if isinstance(rank_data, dict) and "$numberInt" in rank_data:
                #             rank = rank_data["$numberInt"]
                #         elif isinstance(rank_data, (int, str)):
                #             rank = str(rank_data)
                #         else:
                #             rank = "N/A"
                #         timestamp = latest_entry.get("timestamp", "N/A")
                #         charts_info += f"  - [bold]{country}:[/bold] Rank {rank} (as of {timestamp})\n"
                
                # Add trend description if available
                trend_description = result.get("trend_description")
                trend_status = result.get("TREND_STATUS")
                
                trend_info = ""
                if trend_description and trend_status == "PROCESSED":
                    trend_info = f"[bold]Trend:[/bold] {trend_description}\n"
                
                panel_content = (
                    f"[bold blue]Result {i}[/bold blue]\n\n"
                    f"[bold]Song Name:[/bold] {song_name}\n"
                    f"[bold]Artist:[/bold] {artist_name}\n"
                    f"[bold]Genres:[/bold] {genres}\n"
                    f"[bold]Language:[/bold] {language}\n"
                    f"{trend_info}"
                    #f"[bold]Similarity Score:[/bold] {music_score:.4f}\n"
                    #f"[bold]First Seen:[/bold] {result.get('first_seen', 'N/A')}\n"
                    f"[bold]Lyrics Preview:[/bold] {lyrics}\n"
                )
                
                # if charts_info:
                #     panel_content += f"\n[bold]Latest Chart Ranks:[/bold]\n{charts_info}"
                
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

    async def _execute_unified_search(self, user_input: str, embedding_decision: Dict[str, Any], search_params: Dict[str, Any]) -> List[Dict]:
        """
        Execute search based on embedding decision and search parameters.
        
        Args:
            user_input (str): Original user query
            embedding_decision (Dict[str, Any]): Decision about which embeddings to use
            search_params (Dict[str, Any]): Search parameters (filters, limit, etc.)
        """
        try:
            search_type = embedding_decision.get('search_type', 'filter_only')
            filters = search_params.get('filters', {})
            limit = search_params.get('limit', 10)
            description = search_params.get('description', 'Custom search')

            console.print(f"[blue]Search type: {search_type}[/blue]")
            console.print(f"[blue]MongoDB search filters: {filters}[/blue]")
            
            console.print(Panel(
                f"[bold green]🎵 Executing {search_type.upper()} Search[/bold green]\n"
                f"Description: {description}\n"
                f"Limit: {limit} results",
                title="Search Execution", border_style="green"
            ))
            
            if search_type == "vector_only":
                # Vector search with embeddings
                return await self._execute_vector_search(embedding_decision, filters, limit, description)
                
            elif search_type == "filter_only":
                # Traditional MongoDB find with filters
                return await self._execute_filter_search(filters, limit, description)
                
            elif search_type == "hybrid":
                # Combine vector and filter search
                return await self._execute_hybrid_search(embedding_decision, filters, limit, description)
                
            else:
                console.print(Panel(
                    f"[red]Unknown search type: {search_type}[/red]",
                    title="Search Error", border_style="red"
                ))
                return []
        
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Error executing search: {str(e)}[/red]",
                title="Search Error", border_style="red"
            ))
            return []

    async def _execute_vector_search(self, embedding_decision: Dict[str, Any], filters: Dict, limit: int, description: str) -> List[Dict]:
        """Execute vector search using available embeddings."""
        query_vector = None
        vector_path = None
        
        # Determine which embedding to use
        if embedding_decision.get('use_music_embedding') and hasattr(self, 'current_embedding') and self.current_embedding is not None:
            query_vector = self.current_embedding.numpy().tolist()
            vector_path = "music_embedding"
            console.print("[blue]Using music embedding for vector search[/blue]")
        elif embedding_decision.get('use_text_embedding') and hasattr(self, 'current_text_embedding') and self.current_text_embedding is not None:
            query_vector = self.current_text_embedding.tolist()
            vector_path = "text_embedding"
            console.print("[blue]Using text embedding for vector search[/blue]")
        else:
            console.print(Panel(
                "[yellow]No suitable embeddings available for vector search[/yellow]",
                title="Vector Search Warning", border_style="yellow"
            ))
            return []
        
        # Separate basic filters (for vector search) from complex filters (for post-match)
        basic_filters = {}
        complex_filters = {}
        
        if filters and filters != {}:
            for key, value in filters.items():
                # Check if this is a complex filter that $vectorSearch doesn't support
                if isinstance(value, dict):
                    # Check for unsupported operators in vector search
                    unsupported_ops = ['$exists', '$regex', '$text', '$geoWithin', '$geoIntersects']
                    has_unsupported = any(op in value for op in unsupported_ops)
                    
                    if has_unsupported:
                        complex_filters[key] = value
                        console.print(f"[yellow]Moving complex filter '{key}: {value}' to post-vector match[/yellow]")
                    else:
                        basic_filters[key] = value
                else:
                    # Simple equality filters are supported
                    basic_filters[key] = value
        
        # Build vector search stage - always query top 100 results first
        vector_search_stage = {
            "index": "lyrics_n_music_search",
            "path": vector_path,
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": 100  # Always get top 100 for similarity search
        }
        
        # Only add basic filters to vector search
        if basic_filters:
            vector_search_stage["filter"] = basic_filters
            console.print(f"[blue]Vector search filters: {basic_filters}[/blue]")
        
        # Build the pipeline
        pipeline = [
            {"$vectorSearch": vector_search_stage},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}
        ]
        
        # Add complex filters as a separate $match stage after vector search
        if complex_filters:
            pipeline.append({"$match": complex_filters})
            console.print(f"[blue]Post-vector match filters: {complex_filters}[/blue]")
        
        # Simple lookup to get trend description for all PROCESSED songs
        pipeline.extend([
            {
                "$lookup": {
                    "from": "TOP_TIKTOK_TRENDS",
                    "pipeline": [{"$limit": 1}],  # Get the first (and likely only) trend
                    "as": "trend_info"
                }
            },
            {"$project": {
                "_id": 1, "song_id": 1, "song_name": 1, "artist_name": 1,
                "lyrics": 1, "genres": 1, "score": 1, "first_seen": 1,
                "charts": 1, "language_code": 1, "audio_metadata": 1,
                "TREND_STATUS": 1, 
                "trend_description": {
                    "$cond": {
                        "if": {"$eq": ["$TREND_STATUS", "PROCESSED"]},
                        "then": {"$arrayElemAt": ["$trend_info.trend_description", 0]},
                        "else": None
                    }
                }
            }},
            {"$sort": {"score": -1}},
            {"$limit": limit}
        ])
        
        results = list(self.collection.aggregate(pipeline))
        
        # Check if we got any results after filtering
        if not results:
            if complex_filters:
                console.print(Panel(
                    f"[yellow]No results found for your query in the top 100 most similar matches.[/yellow]\n\n"
                    f"The search found the top 100 most similar songs, but none matched your additional filters:\n"
                    f"{json.dumps(complex_filters, indent=2)}\n\n"
                    f"Try relaxing your search criteria or searching without the additional filters.",
                    title="No Results in Top 100", border_style="yellow"
                ))
            else:
                console.print(Panel(
                    "[yellow]No similar songs found in the top 100 matches.[/yellow]",
                    title="No Results", border_style="yellow"
                ))
            return []
        
        # Store results for /add command
        self.proposed_results = results
        
        self._display_search_results(results, description, "vector")
        return results

    async def _execute_filter_search(self, filters: Dict, limit: int, description: str) -> List[Dict]:
        """Execute traditional MongoDB find with filters."""
        pipeline = [
            {"$match": filters if filters and filters != {} else {}},
            {
                "$lookup": {
                    "from": "TOP_TIKTOK_TRENDS",
                    "pipeline": [{"$limit": 1}],  # Get the first (and likely only) trend
                    "as": "trend_info"
                }
            },
            {"$project": {
                "_id": 1, "song_id": 1, "song_name": 1, "artist_name": 1,
                "lyrics": 1, "genres": 1, "first_seen": 1, "charts": 1,
                "language_code": 1, "audio_metadata": 1, "TREND_STATUS": 1,
                "trend_description": {
                    "$cond": {
                        "if": {"$eq": ["$TREND_STATUS", "PROCESSED"]},
                        "then": {"$arrayElemAt": ["$trend_info.trend_description", 0]},
                        "else": None
                    }
                }
            }},
            {"$limit": limit}
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        # Store results for /add command
        self.proposed_results = results
        
        self._display_search_results(results, description, "filter")
        return results

    async def _execute_hybrid_search(self, embedding_decision: Dict[str, Any], filters: Dict, limit: int, description: str) -> List[Dict]:
        """Execute hybrid search combining vector and filter approaches."""
        # For now, prioritize vector search if embeddings are available
        if (embedding_decision.get('use_music_embedding') and hasattr(self, 'current_embedding') and self.current_embedding is not None) or \
           (embedding_decision.get('use_text_embedding') and hasattr(self, 'current_text_embedding') and self.current_text_embedding is not None):
            return await self._execute_vector_search(embedding_decision, filters, limit, description)
        else:
            return await self._execute_filter_search(filters, limit, description)

    def _display_search_results(self, results: List[Dict], description: str, search_type: str):
        """Display search results with appropriate formatting."""
        if not results:
            console.print(Panel(
                "[yellow]No results found matching your criteria.[/yellow]",
                title="No Results", border_style="yellow"
            ))
            return
        
        console.print(Panel(
            f"[bold green]🎵 {search_type.upper()} Search Results - {description}[/bold green]\n"
            f"Found {len(results)} results",
            title="Search Results", border_style="green"
        ))
        
        for i, result in enumerate(results, 1):
            song_name = result.get("song_name", "N/A")
            artist_name = result.get("artist_name", "N/A")
            genres = ', '.join(result.get("genres", []))
            score = result.get("score", 0) if search_type == "vector" else None
            lyrics = result.get("lyrics", "N/A")[:100] + "..." if result.get("lyrics") else "N/A"
            language = result.get("language_code", "N/A")
            
            # charts_info = ""
            # charts = result.get("charts", {})
            # for country, chart_entries in charts.items():
            #     if chart_entries:
            #         latest_entry = chart_entries[0]
            #         rank_data = latest_entry.get("rank", {})
            #         if isinstance(rank_data, dict) and "$numberInt" in rank_data:
            #             rank = rank_data["$numberInt"]
            #         elif isinstance(rank_data, (int, str)):
            #             rank = str(rank_data)
            #         else:
            #             rank = "N/A"
            #         timestamp = latest_entry.get("timestamp", "N/A")
            #         charts_info += f"  - [bold]{country}:[/bold] Rank {rank} (as of {timestamp})\n"
            
            panel_content = (
                f"[bold blue]Result {i}[/bold blue]\n\n"
                f"[bold]Song Name:[/bold] {song_name}\n"
                f"[bold]Artist:[/bold] {artist_name}\n"
                f"[bold]Genres:[/bold] {genres}\n"
                #f"[bold]Language:[/bold] {language}\n"
            )
            
            # if score is not None:
            #     panel_content += f"[bold]Similarity Score:[/bold] {score:.4f}\n"
            
            # Add trend description if available
            trend_description = result.get("trend_description")
            trend_status = result.get("TREND_STATUS")
            
            if trend_description and trend_status == "PROCESSED":
                panel_content += f"[bold]Trend:[/bold] {trend_description}\n"
            
            panel_content += (
                #f"[bold]First Seen:[/bold] {result.get('first_seen', 'N/A')}\n"
                f"[bold]Lyrics Preview:[/bold] {lyrics}\n"
            )
            
            # if charts_info:
            #     panel_content += f"\n[bold]Latest Chart Ranks:[/bold]\n{charts_info}"
            
            console.print(Panel(
                panel_content,
                border_style="green"
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
            "• Find songs longer than 3 minutes with high language probability.\n"
            "• Find similar songs (uses loaded audio/lyrics embeddings)\n"
            "• Explain why this trend took off in Germany (analysis)\n"
            "• Tell me more about this song and genre (analysis)\n\n"
            "[bold green]Special Commands:[/bold green]\n"
            "• /load local/path/to/file - Preprocess an audio file with MERT\n"
            "• /loadtext --text 'lyrics here' [--lang eng] [--song_id XYZ] - Embed lyrics text and optionally save to DB\n"
            "• /loadtext --file /path/to/lyrics.txt [--lang spa] [--song_id XYZ] - Embed lyrics from a file\n"
            "• /search - Find similar songs using vector search (requires loaded audio)\n"
            "• /search --file audio.mp3 --prompt 'in Germany' - Search with file and filters\n"
            "• /search --file audio.mp3 - Search with new file only\n"
            "• /search --prompt 'Latin songs' - Text-only search with filters\n\n"
            "[bold cyan]Chat History Commands:[/bold cyan]\n"
            "• /history - Show conversation history\n"
            "• /clear - Clear conversation history\n\n"
            "[bold magenta]Context Commands:[/bold magenta]\n"
            "• /add <index> - Add search result to context (e.g., /add 1)\n\n"
            "[bold purple]Analysis Workflow:[/bold purple]\n"
            "1. Search for songs → 2. /add results to context → 3. Ask analysis questions\n\n"
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

                # Check for /loadtext command
                if user_input.startswith('/loadtext'):
                    args_line = user_input[len('/loadtext'):]
                    await self._handle_loadtext_command(args_line)
                    continue

                # Check for history commands
                if user_input.lower() == '/history':
                    self._show_chat_history()
                    continue

                if user_input.lower() == '/clear':
                    self._clear_chat_history()
                    continue

                # Check for add command
                if user_input.lower().startswith('/add'):
                    await self._handle_add_command(user_input)
                    continue

                # Orchestrator: Categorize the input
                with console.status("[bold blue]Analyzing input...[/bold blue]", spinner="dots"):
                    category = await self._categorize_input(user_input)
                
                # Print the category to command line
                console.print(f"[bold magenta]🎯 Input Category: {category.upper()}[/bold magenta]")
                
                # Handle based on category
                if category == "help":
                    help_response = (
                        "Here's how I can help you explore the music database:\n\n"
                        "🔍 Search for songs by artist, genre, country charts, etc.\n"
                        "🎵 Find similar songs using audio and lyrics embeddings\n"
                        "📊 Get information about music trends and chart rankings\n"
                        "💬 Use commands like /load, /search, /history for advanced features\n\n"
                        "Just ask me anything about music or use the special commands!"
                    )
                    
                    console.print(Panel(
                        f"[bold blue]🤖 MongoDB Song Database Chatbot CLI[/bold blue]\n\n{help_response}",
                        title="Help", border_style="blue"
                    ))
                    
                    # Add to chat history
                    self._add_to_chat_history(user_input, help_response, "help")
                
                elif category == "talk":
                    # Generate contextual chat response using history
                    chat_response = await self._generate_chat_response(user_input)
                    
                    console.print(Panel(
                        f"[bold green]💬 Assistant[/bold green]\n\n{chat_response}",
                        title="Chat", border_style="green"
                    ))
                    
                    # Add to chat history
                    self._add_to_chat_history(user_input, chat_response, "talk")
                
                elif category == "search":
                    # Enhanced search: determine embeddings and search parameters
                    with console.status("[bold green]Analyzing search request...[/bold green]", spinner="dots"):
                        # 1. Determine which embeddings to use
                        embedding_decision = await self._determine_search_embeddings(user_input)
                        
                        # 2. Get search parameters (filters, limit, etc.)
                        search_params = await self._get_search_parameters_from_llm(user_input)
                        
                        console.print(Panel(
                            f"[bold blue]🔍 Search Analysis[/bold blue]\n\n"
                            f"Query: {user_input}\n"
                            f"Embedding Decision: {embedding_decision['reasoning']}\n"
                            f"Search Type: {embedding_decision['search_type']}\n"
                            f"Music Embedding: {'✅' if embedding_decision['use_music_embedding'] else '❌'}\n"
                            f"Text Embedding: {'✅' if embedding_decision['use_text_embedding'] else '❌'}\n"
                            f"Filters: {json.dumps(search_params.get('filters', {}), indent=2)}",
                            title="Search Configuration", border_style="blue"
                        ))
                    
                    # 3. Execute the search based on the determined approach
                    search_results = await self._execute_unified_search(
                        user_input=user_input,
                        embedding_decision=embedding_decision,
                        search_params=search_params
                    )
                    
                    # Add search to chat history
                    search_summary = f"Found {len(search_results) if search_results else 0} results for: {search_params.get('description', 'music search')}"
                    self._add_to_chat_history(user_input, search_summary, "search")
                
                elif category == "analysis":
                    console.print("[blue]Debug: Analysis category detected, calling analysis handler[/blue]")
                    # Deep analysis using o3-mini reasoning model
                    with console.status("[bold purple]Performing deep analysis...[/bold purple]", spinner="dots"):
                        analysis_response = await self._handle_analysis_query(user_input)
                    
                    console.print(f"[blue]Debug: Analysis response received: {len(analysis_response)} characters[/blue]")
                    
                    console.print(Panel(
                        f"[bold purple]🧠 Deep Analysis[/bold purple]\n\n{analysis_response}",
                        title="Analysis Results", border_style="purple"
                    ))
                    
                    # Add to chat history
                    self._add_to_chat_history(user_input, analysis_response, "analysis")
            
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