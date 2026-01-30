"""Quick RAG demo"""
from src.rag_pipeline import RAGPipeline

# Initialize
print("Loading RAG system...")
rag = RAGPipeline()

# Index documents (only needed first time or when docs change)
print("Indexing documents...")
rag.index_documents()

print("\n" + "="*60)
print("RAG System Ready! Ask questions about electrical contracting.")
print("="*60)

# Interactive loop
while True:
    question = input("\nYour question (or 'quit' to exit): ")
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not question.strip():
        continue
    
    # Query
    result = rag.query(question, n_results=3)
    
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")