#!/usr/bin/env python3
"""
Subgenre Prediction Script

This script accepts a local audio file path and a genre, loads the corresponding
FCNN model, extracts both audio and lyrics embeddings, and predicts the most
likely subgenre using the same pipeline as query_gpt.py.

Usage:
    python predict_subgenre.py --local_path /path/to/audio.mp3 --genre hiphop
"""

import argparse
import os
import sys
import warnings
import asyncio
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel

# Import the same classes and functions used in query_gpt.py
from query_gpt import MERTEmbedder, ElevenLabsClient, SimpleFCNN
from sentence_embeddings import MultilingualSentenceEmbeddings
from split_lyrics import get_sentences

# Suppress warnings like in query_gpt.py
warnings.filterwarnings("ignore", message="feature_extractor_cqt requires the libray 'nnAudio'")
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")
warnings.filterwarnings("ignore", message="torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")

console = Console()


class SubgenrePredictor:
    """
    Predicts subgenres for audio files using the same pipeline as query_gpt.py
    """
    
    def __init__(self, genre: str):
        """
        Initialize the predictor with the specified genre.
        
        Args:
            genre (str): The parent genre (e.g., "hiphop", "latin")
        """
        self.genre = genre.lower()
        self.model_path = f"{self.genre}_fcnn.pt"
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            console.print(Panel(
                f"[red]Model file '{self.model_path}' not found![/red]\n\n"
                f"Please ensure you have trained a model for the '{self.genre}' genre\n"
                f"using the pipeline.py script with --genre {self.genre}",
                title="Model Not Found", border_style="red"
            ))
            sys.exit(1)
        
        # Initialize components
        self.mert_embedder = None
        self.text_embeddings_model = None
        self.genre_model = None
        self.genre_labels = None
        self.elevenlabs_client = None
        
        self._load_model()
        self._initialize_embedders()
    
    def _load_model(self):
        """Load the FCNN model from the checkpoint file"""
        try:
            console.print(f"[blue]Loading genre classification model: {self.model_path}[/blue]")
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            input_dim = checkpoint["input_dim"]
            self.genre_labels = checkpoint["output_labels"]
            
            self.genre_model = SimpleFCNN(input_dim=input_dim, output_dim=len(self.genre_labels))
            self.genre_model.load_state_dict(checkpoint["model_state_dict"])
            self.genre_model.eval()
            
            console.print(f"[green]✅ Model loaded successfully[/green]")
            console.print(f"[blue]Available subgenres: {', '.join(self.genre_labels)}[/blue]")
            console.print(f"[blue]Input dimension: {input_dim}[/blue]")
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error loading model: {e}[/red]",
                title="Model Loading Error", border_style="red"
            ))
            sys.exit(1)
    
    def _initialize_embedders(self):
        """Initialize the MERT embedder and text embeddings model"""
        try:
            # Initialize MERT embedder for audio processing
            console.print("[blue]Initializing MERT embedder...[/blue]")
            self.mert_embedder = MERTEmbedder()
            console.print("[green]✅ MERT embedder initialized[/green]")
            
            # Initialize text embeddings model
            console.print("[blue]Initializing text embeddings model...[/blue]")
            self.text_embeddings_model = MultilingualSentenceEmbeddings()
            console.print("[green]✅ Text embeddings model initialized[/green]")
            
            # Initialize ElevenLabs client for transcription
            self.elevenlabs_client = ElevenLabsClient()
            if self.elevenlabs_client.enabled:
                console.print("[green]✅ ElevenLabs transcription available[/green]")
            else:
                console.print("[yellow]⚠️ ElevenLabs API key not found - transcription disabled[/yellow]")
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error initializing embedders: {e}[/red]",
                title="Initialization Error", border_style="red"
            ))
            sys.exit(1)
    
    def _extract_audio_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extract MERT audio embedding from the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: MERT embedding of shape [1024]
        """
        try:
            console.print(f"[blue]Extracting audio embedding from: {audio_path}[/blue]")
            embedding = self.mert_embedder.process_local_file(audio_path)
            console.print(f"[green]✅ Audio embedding extracted - shape: {embedding.shape}[/green]")
            return embedding
        except Exception as e:
            console.print(Panel(
                f"[red]Error extracting audio embedding: {e}[/red]",
                title="Audio Processing Error", border_style="red"
            ))
            raise
    
    async def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using ElevenLabs API.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        if not self.elevenlabs_client.enabled:
            console.print("[yellow]Transcription disabled - using empty text[/yellow]")
            return ""
        
        try:
            console.print(f"[blue]Transcribing audio: {audio_path}[/blue]")
            
            # Use the same approach as query_gpt.py - extract snippet first
            waveform, wav_path = self.mert_embedder.extract_snippet_and_waveform(audio_path)
            
            # Transcribe using the temporary wav file
            transcription_result = await asyncio.to_thread(
                self.elevenlabs_client.transcribe_sync, wav_path
            )
            
            # Clean up temp file
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass
            
            if transcription_result and transcription_result.get("text"):
                transcript_text = transcription_result["text"]
                console.print(f"[green]✅ Transcription successful: {transcript_text[:100]}...[/green]")
                return transcript_text
            else:
                console.print("[yellow]⚠️ Transcription failed or empty[/yellow]")
                return ""
                
        except Exception as e:
            console.print(f"[red]Transcription error: {e}[/red]")
            return ""
    
    def _extract_text_embedding(self, transcript_text: str) -> Optional[torch.Tensor]:
        """
        Extract text embedding from transcribed lyrics.
        
        Args:
            transcript_text (str): Transcribed text
            
        Returns:
            Optional[torch.Tensor]: Text embedding of shape [768] or None if failed
        """
        if not transcript_text or not transcript_text.strip():
            console.print("[yellow]No transcript text available for text embedding[/yellow]")
            return None
        
        try:
            console.print(f"[blue]Processing text embedding (length: {len(transcript_text)})[/blue]")
            
            # Split into sentences using the same approach as query_gpt.py
            sentences = get_sentences(transcript_text, None)  # Auto-detect language
            console.print(f"[blue]Split into {len(sentences)} sentences[/blue]")
            
            if not sentences:
                console.print("[yellow]No sentences extracted from transcript[/yellow]")
                return None
            
            # Generate embeddings for all sentences
            embeddings = self.text_embeddings_model.get_embeddings(sentences)
            console.print(f"[blue]Generated embeddings shape: {embeddings.shape}[/blue]")
            
            # Compute mean embedding (same as query_gpt.py)
            mean_embedding = np.mean(embeddings, axis=0)
            console.print(f"[green]✅ Text embedding extracted - shape: {mean_embedding.shape}[/green]")
            
            return torch.tensor(mean_embedding, dtype=torch.float32)
            
        except Exception as e:
            console.print(f"[red]Error extracting text embedding: {e}[/red]")
            return None
    
    def _predict_subgenre(self, music_embedding: torch.Tensor, text_embedding: Optional[torch.Tensor] = None) -> str:
        """
        Predict the most likely subgenre using the FCNN model.
        
        Args:
            music_embedding (torch.Tensor): MERT embedding of shape [1024]
            text_embedding (Optional[torch.Tensor]): Text embedding of shape [768] or None
            
        Returns:
            str: Predicted subgenre label
        """
        try:
            with torch.no_grad():
                # Prepare text embedding (use zeros if not available, same as query_gpt.py)
                if text_embedding is None:
                    # Use zeros for text embedding if not available (768 is common for sentence transformers)
                    text_embedding = torch.zeros(768, dtype=music_embedding.dtype, device=music_embedding.device)
                elif text_embedding.dim() == 1:
                    text_embedding = text_embedding.unsqueeze(0)  # Add batch dimension
                
                # Ensure music embedding has batch dimension
                if music_embedding.dim() == 1:
                    music_embedding = music_embedding.unsqueeze(0)  # Add batch dimension
                
                # Concatenate music + text embeddings (same order as training)
                combined_embedding = torch.cat([music_embedding, text_embedding], dim=1)
                console.print(f"[blue]Combined embedding shape: {combined_embedding.shape}[/blue]")
                
                # Get logits from model
                logits = self.genre_model(combined_embedding)
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(logits)
                
                # Get the top-1 predicted subgenre (highest probability)
                top_index = torch.argmax(probabilities, dim=1)[0]
                predicted_subgenre = self.genre_labels[top_index]
                confidence = probabilities[0][top_index].item()
                
                console.print(f"[green]✅ Prediction: {predicted_subgenre} (confidence: {confidence:.4f})[/green]")
                
                return predicted_subgenre
                
        except Exception as e:
            console.print(Panel(
                f"[red]Error predicting subgenre: {e}[/red]",
                title="Prediction Error", border_style="red"
            ))
            raise
    
    async def predict(self, audio_path: str) -> str:
        """
        Main prediction method that orchestrates the entire pipeline.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Predicted subgenre
        """
        # Validate audio file
        audio_path = Path(audio_path)
        if not audio_path.exists():
            console.print(Panel(
                f"[red]Audio file not found: {audio_path}[/red]",
                title="File Not Found", border_style="red"
            ))
            sys.exit(1)
        
        console.print(Panel(
            f"[bold blue]🎵 Subgenre Prediction Pipeline[/bold blue]\n\n"
            f"Audio File: {audio_path}\n"
            f"Parent Genre: {self.genre}\n"
            f"Model: {self.model_path}",
            title="Starting Prediction", border_style="blue"
        ))
        
        # Step 1: Extract audio embedding
        music_embedding = self._extract_audio_embedding(str(audio_path))
        
        # Step 2: Transcribe audio and extract text embedding
        transcript_text = await self._transcribe_audio(str(audio_path))
        text_embedding = self._extract_text_embedding(transcript_text)
        
        # Step 3: Predict subgenre
        predicted_subgenre = self._predict_subgenre(music_embedding, text_embedding)
        
        return predicted_subgenre


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict subgenre for an audio file using trained FCNN model"
    )
    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="Path to the local audio file"
    )
    parser.add_argument(
        "--genre",
        type=str,
        required=True,
        help="Parent genre (e.g., 'hiphop', 'latin') - determines which model to load"
    )
    
    return parser.parse_args()


async def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Initialize predictor
        predictor = SubgenrePredictor(args.genre)
        
        # Run prediction
        predicted_subgenre = await predictor.predict(args.local_path)
        
        # Display final result
        console.print(Panel(
            f"[bold green]🎯 PREDICTION RESULT[/bold green]\n\n"
            f"Audio File: {args.local_path}\n"
            f"Parent Genre: {args.genre}\n"
            f"[bold yellow]Most Likely Subgenre: {predicted_subgenre}[/bold yellow]",
            title="Final Result", border_style="green"
        ))
        
        return predicted_subgenre
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Prediction interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Fatal error: {e}[/red]",
            title="Error", border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
