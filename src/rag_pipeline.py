"""
RAG Pipeline
Complete Retrieval-Augmented Generation pipeline
Combines vector search with LLM to answer questions based on documents
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from anthropic import Anthropic

from document_loader import DocumentLoader
from text_chunker import TextChunker
from embeddings import EmbeddingsGenerator
from vector_store import VectorStore

load_dotenv()


class RAGPipeline:
    """Complete RAG system for document Q&A"""
    
    def __init__(
        self,
        documents_path: str = "data/documents",
        vector_store_path: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = "claude-sonnet-4-5-20250929"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            documents_path: Path to documents directory
            vector_store_path: Path to vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            model_name: Claude model to use
        """
        print("Initializing RAG Pipeline...")
        
        # Initialize components
        self.loader = DocumentLoader(documents_path)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingsGenerator()
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        
        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        
        print("✓ RAG Pipeline initialized")
    
    def index_documents(self):
        """Load, chunk, embed, and store all documents"""
        print("\nIndexing documents...")
        
        # Load documents
        documents = self.loader.load_documents()
        if not documents:
            raise ValueError("No documents found to index")
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.generate_embeddings_batch(
            chunk_texts,
            show_progress=False
        )
        
        # Clear existing data and add new
        self.vector_store.clear()
        self.vector_store.add_documents(chunks, embeddings)
        
        stats = self.vector_store.get_stats()
        print(f"✓ Indexed {stats['total_chunks']} chunks")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User question
            n_results: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, n_results)
        
        # Format results
        retrieved_chunks = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            retrieved_chunks.append({
                'content': doc,
                'source': metadata['source'],
                'similarity': 1 - distance
            })
        
        return retrieved_chunks
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 1000
    ) -> Dict:
        """
        Generate answer using Claude with retrieved context
        
        Args:
            query: User question
            context_chunks: Retrieved relevant chunks
            max_tokens: Maximum response length
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build context from chunks
        context_parts = []
        sources = set()
        
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Document {i} - {chunk['source']}]:\n{chunk['content']}")
            sources.add(chunk['source'])
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant answering questions based on the provided documents about electrical contracting services.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY information from the provided context
- If the context doesn't contain enough information to answer fully, say so
- Cite which document(s) you're referencing in your answer
- Be specific and include relevant details (prices, procedures, requirements, etc.)
- If there are no relevant documents, say "I don't have information about that in the available documents"

Answer:"""
        
        # Call Claude
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = message.content[0].text
        
        return {
            'answer': answer,
            'sources': list(sources),
            'num_chunks_used': len(context_chunks),
            'model': self.model_name
        }
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        max_tokens: int = 1000
    ) -> Dict:
        """
        Complete RAG query: retrieve + generate
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            max_tokens: Maximum answer length
            
        Returns:
            Dictionary with answer, sources, and chunks
        """
        print(f"\nQuery: {question}")
        
        # Retrieve relevant chunks
        chunks = self.retrieve(question, n_results)
        print(f"✓ Retrieved {len(chunks)} relevant chunks")
        
        # Generate answer
        result = self.generate_answer(question, chunks, max_tokens)
        
        # Add retrieved chunks to result
        result['retrieved_chunks'] = chunks
        
        return result


def main():
    """Test the complete RAG pipeline"""
    print("Testing Complete RAG Pipeline\n" + "="*60)
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Index documents
    rag.index_documents()
    
    # Test queries
    print(f"\n{'='*60}")
    print("Testing RAG Queries")
    print(f"{'='*60}")
    
    test_queries = [
        "What personal protective equipment is required for electrical work?",
        "How much does it cost to install a GFCI outlet?",
        "Tell me about the TechCorp project",
        "What are the lockout/tagout procedures?",
        "What is the hourly rate for a master electrician?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {query}")
        print(f"{'='*60}")
        
        result = rag.query(query, n_results=3)
        
        print(f"\nAnswer:")
        print(result['answer'])
        
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"Chunks used: {result['num_chunks_used']}")
        
        print(f"\nRetrieved Chunks (top 3):")
        for j, chunk in enumerate(result['retrieved_chunks'][:3], 1):
            print(f"  {j}. {chunk['source']} (similarity: {chunk['similarity']:.3f})")
            print(f"     {chunk['content'][:100]}...")
    
    print(f"\n{'='*60}")
    print("✓ RAG Pipeline working successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()