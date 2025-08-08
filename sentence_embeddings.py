import numpy as np
from typing import List, Dict, Any
from minilm import LocalSentenceEmbeddings

class MultilingualSentenceEmbeddings:
    """
    Wrapper class for LocalSentenceEmbeddings to provide consistent interface
    """
    
    def __init__(self):
        """Initialize the embeddings model"""
        print(f"\n{'='*50}")
        print("MULTILINGUAL SENTENCE EMBEDDINGS INITIALIZATION")
        print(f"{'='*50}")
        
        try:
            print("Creating LocalSentenceEmbeddings instance...")
            self.model = LocalSentenceEmbeddings()
            print("✅ LocalSentenceEmbeddings created successfully")
        except Exception as e:
            print(f"❌ Error creating LocalSentenceEmbeddings: {e}")
            raise
            
        print(f"{'='*50}")
        print("MULTILINGUAL SENTENCE EMBEDDINGS INITIALIZATION COMPLETE")
        print(f"{'='*50}\n")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single sentence.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        return self.model.get_embedding(text)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple sentences.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        return self.model.get_embeddings(texts)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two sentences.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score
        """
        return self.model.compute_similarity(text1, text2)
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar sentences to a query.
        
        Args:
            query (str): Query text
            candidates (List[str]): Candidate texts
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar sentences with scores
        """
        return self.model.find_most_similar(query, candidates, top_k) 