"""
Vector Store
Stores document embeddings and enables semantic search using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os
from pathlib import Path


class VectorStore:
    """Vector database for storing and searching document embeddings"""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize vector store with ChromaDB
        
        Args:
            collection_name: Name of the collection to store embeddings
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing ChromaDB in {persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"✓ Collection '{collection_name}' ready")
    
    def add_documents(
        self,
        chunks: List[Dict[str, str]],
        embeddings: List[List[float]]
    ):
        """
        Add document chunks and their embeddings to the vector store
        
        Args:
            chunks: List of chunk dictionaries with content, source, chunk_id
            embeddings: List of embedding vectors corresponding to chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        ids = [f"{chunk['source']}_chunk_{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_id': str(chunk['chunk_id']),
                'char_count': str(chunk['char_count'])
            }
            for chunk in chunks
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunks")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Dict:
        """
        Search for most similar documents
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            
        Returns:
            Dictionary with ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': self.persist_directory
        }
    
    def clear(self):
        """Clear all documents from the collection"""
        print(f"Clearing collection '{self.collection_name}'...")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Collection cleared")


def main():
    """Test vector store"""
    from document_loader import DocumentLoader
    from text_chunker import TextChunker
    from embeddings import EmbeddingsGenerator
    
    print("Testing Vector Store\n" + "="*50)
    
    # Load and chunk documents
    print("\n1. Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    print("\n2. Chunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("\n3. Generating embeddings...")
    embedder = EmbeddingsGenerator()
    chunk_texts = [chunk['content'] for chunk in chunks]
    embeddings = embedder.generate_embeddings_batch(chunk_texts, show_progress=False)
    
    # Initialize vector store
    print(f"\n{'='*50}")
    print("4. Initializing vector store...")
    print(f"{'='*50}")
    vector_store = VectorStore()
    
    # Clear any existing data
    vector_store.clear()
    
    # Add documents
    print("\n5. Adding documents to vector store...")
    vector_store.add_documents(chunks, embeddings)
    
    # Show stats
    stats = vector_store.get_stats()
    print(f"\n{'='*50}")
    print("Vector Store Statistics:")
    print(f"{'='*50}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    print(f"\n{'='*50}")
    print("6. Testing semantic search...")
    print(f"{'='*50}")
    
    queries = [
        "What safety equipment is required?",
        "How much does outlet installation cost?",
        "What was the TechCorp project about?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Generate query embedding
        query_embedding = embedder.generate_embedding(query)
        
        # Search
        results = vector_store.search(query_embedding, n_results=3)
        
        # Show results
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"\n  Result {i+1} (similarity: {similarity:.3f}):")
            print(f"  Source: {metadata['source']}")
            print(f"  Content: {doc[:150]}...")
    
    print(f"\n{'='*50}")
    print("✓ Vector store working correctly!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()