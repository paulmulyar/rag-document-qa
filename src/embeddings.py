"""
Embeddings Generator
Converts text into vector embeddings for semantic search
"""

import os
from typing import List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class EmbeddingsGenerator:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embeddings generator
        
        Args:
            model_name: Name of sentence-transformers model to use
                       'all-MiniLM-L6-v2' is fast and lightweight
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded (dimension: {self.embedding_dimension})")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Convert to list of floats
        return embedding.tolist()
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Convert to list of lists
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        print(f"✓ Generated {len(embeddings_list)} embeddings")
        return embeddings_list
    
    def get_embedding_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'max_sequence_length': self.model.max_seq_length
        }


def main():
    """Test embeddings generator"""
    print("Testing Embeddings Generator\n" + "="*50)
    
    # Initialize
    embedder = EmbeddingsGenerator()
    
    # Show model info
    info = embedder.get_embedding_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test single embedding
    print(f"\n{'='*50}")
    print("Testing Single Text Embedding:")
    print(f"{'='*50}")
    
    test_text = "What is the cost of electrical outlet installation?"
    embedding = embedder.generate_embedding(test_text)
    
    print(f"Text: {test_text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embeddings
    print(f"\n{'='*50}")
    print("Testing Batch Embeddings:")
    print(f"{'='*50}")
    
    test_texts = [
        "Electrical safety procedures and PPE requirements",
        "Commercial office electrical installation pricing",
        "Emergency lighting and exit sign installation",
        "Service panel upgrade from 100A to 200A",
        "Network cabling and data drop installation"
    ]
    
    embeddings = embedder.generate_embeddings_batch(test_texts, show_progress=False)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    
    # Test similarity (dot product)
    print(f"\n{'='*50}")
    print("Testing Semantic Similarity:")
    print(f"{'='*50}")
    
    import numpy as np
    
    query = "How much does it cost to install outlets?"
    query_embedding = np.array(embedder.generate_embedding(query))
    
    print(f"Query: '{query}'")
    print("\nSimilarity scores with test texts:")
    
    for i, text in enumerate(test_texts):
        text_embedding = np.array(embeddings[i])
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = np.dot(query_embedding, text_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
        )
        
        print(f"  {similarity:.3f} - {text[:60]}...")
    
    print("\n✓ Embeddings working correctly!")


if __name__ == "__main__":
    main()