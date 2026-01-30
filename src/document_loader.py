"""
Document Loader
Loads text documents from a directory for RAG pipeline
"""

import os
from typing import List, Dict
from pathlib import Path


class DocumentLoader:
    """Load and manage text documents"""
    
    def __init__(self, documents_path: str = "data/documents"):
        """
        Initialize document loader
        
        Args:
            documents_path: Path to directory containing documents
        """
        self.documents_path = Path(documents_path)
        
        if not self.documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")
    
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all text documents from the documents directory
        
        Returns:
            List of dictionaries with 'content' and 'source' keys
        """
        documents = []
        
        # Find all .txt files
        txt_files = list(self.documents_path.glob("*.txt"))
        
        if not txt_files:
            print(f"Warning: No .txt files found in {self.documents_path}")
            return documents
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Only add non-empty documents
                    if content.strip():
                        documents.append({
                            'content': content,
                            'source': file_path.name,
                            'path': str(file_path)
                        })
                        print(f"✓ Loaded: {file_path.name} ({len(content)} chars)")
                    else:
                        print(f"⚠ Skipped empty file: {file_path.name}")
                        
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        print(f"\n✓ Loaded {len(documents)} documents")
        return documents
    
    def get_document_stats(self, documents: List[Dict[str, str]]) -> Dict:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'total_words': 0,
                'avg_doc_length': 0
            }
        
        total_chars = sum(len(doc['content']) for doc in documents)
        total_words = sum(len(doc['content'].split()) for doc in documents)
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_doc_length': total_chars // len(documents) if documents else 0
        }


def main():
    """Test the document loader"""
    print("Testing Document Loader\n" + "="*50)
    
    # Initialize loader
    loader = DocumentLoader()
    
    # Load documents
    documents = loader.load_documents()
    
    # Show stats
    if documents:
        print("\nDocument Statistics:")
        stats = loader.get_document_stats(documents)
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
        
        # Show first document preview
        print(f"\n{'='*50}")
        print("First Document Preview:")
        print(f"{'='*50}")
        print(f"Source: {documents[0]['source']}")
        print(f"Length: {len(documents[0]['content'])} characters")
        print(f"\nContent preview (first 200 chars):")
        print(documents[0]['content'][:200] + "...")


if __name__ == "__main__":
    main()