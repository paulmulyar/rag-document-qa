"""
Text Chunker
Splits documents into smaller chunks for efficient retrieval
"""

from typing import List, Dict


class TextChunker:
    """Split text into overlapping chunks"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Preferred split point (paragraph breaks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_text(self, text: str, source: str = "unknown") -> List[Dict[str, str]]:
        """
        Split text into chunks
        
        Args:
            text: Text to chunk
            source: Source document name for tracking
            
        Returns:
            List of chunk dictionaries with content, source, and chunk_id
        """
        if not text.strip():
            return []
        
        # Try to split on paragraphs first
        paragraphs = text.split(self.separator)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk_size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'source': source,
                        'chunk_id': chunk_id,
                        'char_count': len(current_chunk.strip())
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap from previous chunk
                    if self.chunk_overlap > 0:
                        # Take last chunk_overlap characters
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += self.separator + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'source': source,
                'chunk_id': chunk_id,
                'char_count': len(current_chunk.strip())
            })
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries with 'content' and 'source'
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(
                text=doc['content'],
                source=doc.get('source', 'unknown')
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, str]]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [chunk['char_count'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }


def main():
    """Test the text chunker"""
    from document_loader import DocumentLoader
    
    print("Testing Text Chunker\n" + "="*50)
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    if not documents:
        print("No documents to chunk!")
        return
    
    # Initialize chunker
    chunker = TextChunker(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Chunk documents
    print("\nChunking documents...")
    chunks = chunker.chunk_documents(documents)
    
    # Show stats
    stats = chunker.get_chunk_stats(chunks)
    print(f"\n{'='*50}")
    print("Chunk Statistics:")
    print(f"{'='*50}")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Show first few chunks
    print(f"\n{'='*50}")
    print("Sample Chunks:")
    print(f"{'='*50}")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Source: {chunk['source']}")
        print(f"  Chunk ID: {chunk['chunk_id']}")
        print(f"  Length: {chunk['char_count']} chars")
        print(f"  Content preview: {chunk['content'][:150]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()