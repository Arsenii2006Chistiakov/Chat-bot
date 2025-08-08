import os
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time
from typing import List, Dict, Any

# Determine local cache directory for the model, with fallbacks
def _resolve_local_model_path() -> str | None:
    candidates = []
    # Highest priority: environment variable
    env_path = os.getenv("MINILM_LOCAL_PATH")
    if env_path:
        candidates.append(os.path.expanduser(env_path))
    # Common local directories relative to the project
    candidates.extend([
        os.path.abspath("my_minilm_cache"),
        os.path.abspath("models/my_minilm_cache"),
        os.path.expanduser("~/lyrics_embedding_api/my_minilm_cache"),
        "/home/arsenii/lyrics_embedding_api/my_minilm_cache",
    ])

    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path):
            print(f"Local model cache found at: {path}")
            return path
    print("Warning: No local MiniLM cache directory found. Will load by model id (may download).")
    return None

local_model_path = _resolve_local_model_path()


class LocalSentenceEmbeddings:
    """
    A class for generating multilingual sentence embeddings using
    paraphrase-multilingual-MiniLM-L12-v2 model locally.
    """

    def __init__(self):
        """
        Initialize the embeddings model. Prefer local cache if available; otherwise load by model id.
        """
        start_time = time.time()
        if local_model_path is not None:
            print(f"\nLoading SentenceTransformer model from local cache: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            model_id = "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"\nLoading SentenceTransformer model by id: {model_id} (may download if not cached)")
            self.model = SentenceTransformer(model_id)

        self.model.to('cpu')  # Explicitly move to CPU unless user config changes it

        load_time = time.time() - start_time

        print(f"Model loaded successfully in {load_time:.2f} seconds")
        print(f"Model device: {self.model.device}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single sentence.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple sentences.
        """
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided for embedding")
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two sentences.
        """
        embeddings = self.get_embeddings([text1, text2])
        emb1, emb2 = embeddings[0], embeddings[1]

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar sentences to a query.
        """
        all_texts = [query] + candidates
        embeddings = self.get_embeddings(all_texts)

        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = np.dot(query_embedding, candidate_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb))
            similarities.append({
                'text': candidates[i],
                'similarity': float(similarity),
                'index': i
            })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

def main():
    """
    Main function to demonstrate the local sentence embeddings.
    """
    print("\n" + "=" * 60)
    print("LOCAL SENTENCE EMBEDDINGS DEMONSTRATION")
    print("=" * 60)

    # Initialize the embeddings model.
    # This will now use the locally downloaded files from 'local_model_path'.
    embeddings_model = LocalSentenceEmbeddings()

    # Trial sentences in different languages
    trial_sentences = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Ciao, come stai oggi?",
        "Hallo, wie geht es dir heute?",
        "The weather is beautiful today.",
        "El clima está hermoso hoy.",
        "Le temps est magnifique aujourd'hui.",
        "Il tempo è bellissimo oggi.",
        "Das Wetter ist heute wunderschön.",
        "I love listening to music.",
        "Me encanta escuchar música.",
        "J'aime écouter de la musique.",
        "Mi piace ascoltare la musica.",
        "Ich liebe es, Musik zu hören."
    ]

    print("\n" + "=" * 40)
    print("1. SINGLE SENTENCE EMBEDDING")
    print("=" * 40)

    text = "Hello, how are you today?"
    embedding = embeddings_model.get_embedding(text)
    print(f"Input text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")

    print("\n" + "=" * 40)
    print("2. MULTIPLE SENTENCES EMBEDDING")
    print("=" * 40)

    sample_texts = trial_sentences[:5]
    embeddings = embeddings_model.get_embeddings(sample_texts)
    print(f"Number of sentences: {len(sample_texts)}")
    print(f"Embeddings shape: {embeddings.shape}")

    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")

    print("\n" + "=" * 40)
    print("3. SIMILARITY COMPUTATION")
    print("=" * 40)

    pairs = [
        ("Hello, how are you today?", "Hola, ¿cómo estás hoy?"),
        ("The weather is beautiful today.", "El clima está hermoso hoy."),
        ("I love listening to music.", "Me encanta escuchar música."),
        ("Hello, how are you today?", "The weather is beautiful today."),  # Different topics
    ]

    for text1, text2 in pairs:
        similarity = embeddings_model.compute_similarity(text1, text2)
        print(f"Similarity between:")
        print(f"  '{text1}'")
        print(f"  '{text2}'")
        print(f"  Score: {similarity:.4f}")
        print()

    print("\n" + "=" * 40)
    print("4. FINDING MOST SIMILAR SENTENCES")
    print("=" * 40)

    query = "How are you doing?"
    candidates = trial_sentences
    top_similar = embeddings_model.find_most_similar(query, candidates, top_k=5)

    print(f"Query: '{query}'")
    print("\nTop 5 most similar sentences:")
    for i, result in enumerate(top_similar, 1):
        print(f"{i}. '{result['text']}' (similarity: {result['similarity']:.4f})")

    print("\n" + "=" * 40)
    print("5. MULTILINGUAL COMPARISON")
    print("=" * 40)

    greetings = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Ciao, come stai oggi?",
        "Hallo, wie geht es dir heute?"
    ]

    base_greeting = "Hello, how are you today?"
    print(f"Base greeting: '{base_greeting}'")
    print("\nSimilarity with other language greetings:")

    for greeting in greetings[1:]:  # Skip the base greeting
        similarity = embeddings_model.compute_similarity(base_greeting, greeting)
        print(f"  '{greeting}' -> {similarity:.4f}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()