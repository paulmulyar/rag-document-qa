# RAG-Powered Document Q&A System

Production-ready Retrieval-Augmented Generation (RAG) system for intelligent document search and question answering. Built with semantic search, vector embeddings, and Claude AI.

## Overview

This system demonstrates a complete RAG pipeline that:
- Loads and processes documents from any domain
- Splits text into optimized chunks with overlap
- Generates semantic embeddings using sentence transformers
- Stores vectors in ChromaDB for fast similarity search
- Retrieves relevant context for user questions
- Generates accurate answers using Claude AI with source citations

**Current Implementation:** Electrical contracting knowledge base (safety procedures, pricing guides, project documentation)

## Features

- **Semantic Search**: Vector-based document retrieval using cosine similarity
- **Optimized Chunking**: Configurable chunk size with overlap for context preservation
- **Source Citations**: All answers include document sources for verification
- **Performance Tracking**: Component-level latency monitoring
- **Evaluation Framework**: Automated testing of retrieval accuracy and answer quality
- **Modular Architecture**: Clean separation of concerns for easy extension

## Performance Metrics

Based on automated evaluation with 10 test questions:

| Metric | Score |
|--------|-------|
| **Retrieval Accuracy** | 100% |
| **Keyword Match Rate** | 95% |
| **Average Query Time** | 3.8s |
| **Retrieval Latency** | 0.098s |
| **Generation Latency** | 3.65s |

### Component Breakdown
- **Document Indexing**: 0.3s (one-time setup)
- **Retrieval**: 0.098s per query (2.6% of query time)
- **LLM Generation**: 3.65s per query (96% of query time)
- **Vector Search**: Sub-100ms for 13-chunk corpus (0.053s - 0.138s range)

### Performance Notes
- Total corpus: 3 documents, 13 chunks, ~4,100 characters
- Embedding model load time: 3.0s (one-time initialization)
- Query latency range: 2.6s - 5.1s depending on answer complexity
- Bottleneck: Claude API latency (unavoidable for quality responses)

## Architecture
```
User Question
    ↓
Document Loader → Text Chunker → Embeddings Generator
                                        ↓
                                  Vector Store (ChromaDB)
                                        ↓
Query → Embedding → Similarity Search → Top-K Chunks
                                             ↓
                                 Context + Query → Claude AI
                                             ↓
                                    Answer + Citations
```

### Tech Stack

- **Vector Database**: ChromaDB with cosine similarity
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM**: Claude Sonnet 4.5 (Anthropic)
- **Language**: Python 3.13
- **Key Libraries**: chromadb, sentence-transformers, anthropic

## Quick Start

### Prerequisites

- Python 3.9+
- LLM API key

### Installation
```bash
# Clone repository
git clone https://github.com/paulmulyar/rag-document-qa.git
cd rag-document-qa

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your API_KEY
```

### Usage

**Interactive Demo:**
```bash
python3 demo.py
```

Ask questions like:
- "What safety equipment is required?"
- "How much does outlet installation cost?"
- "What was the TechCorp project scope?"

**Programmatic Usage:**
```python
from src.rag_pipeline import RAGPipeline

# Initialize and index
rag = RAGPipeline()
rag.index_documents()

# Query
result = rag.query("Your question here", n_results=5)
print(result['answer'])
print(f"Sources: {result['sources']}")
```

## Project Structure
```
rag-document-qa/
├── src/
│   ├── document_loader.py      # Load documents from directory
│   ├── text_chunker.py         # Split text into chunks
│   ├── embeddings.py           # Generate vector embeddings
│   ├── vector_store.py         # ChromaDB vector storage
│   ├── rag_pipeline.py         # Complete RAG orchestration
│   └── performance_tracker.py  # Latency monitoring
├── tests/
│   ├── test_questions.json     # Evaluation test cases
│   ├── evaluate_rag.py         # Accuracy testing
│   └── performance_report.json # Benchmark results
├── data/
│   └── documents/              # Source documents (*.txt)
├── docs/
│   └── ARCHITECTURE.md         # Detailed technical documentation
├── demo.py                     # Interactive CLI demo
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Evaluation

Run automated evaluation:
```bash
# Test retrieval accuracy and answer quality
python3 tests/evaluate_rag.py

# Benchmark performance
python3 src/performance_tracker.py
```

### Evaluation Methodology

**Retrieval Accuracy:**
- 10 test questions with known correct source documents
- Measures if the expected document appears in top-5 retrieved chunks
- Current: 100% accuracy (10/10 correct)

**Answer Quality:**
- Tests if generated answers contain expected keywords
- Validates factual accuracy against source material
- Current: 95% keyword match rate (38/40 keywords present)

**Performance Benchmarking:**
- Measures component-level latency across 5 test queries
- Tracks retrieval speed, generation time, and total query time
- Identifies bottlenecks for optimization opportunities

## Configuration

Key parameters in `src/rag_pipeline.py`:
```python
RAGPipeline(
    chunk_size=500,           # Characters per chunk
    chunk_overlap=50,         # Overlap between chunks
    documents_path="data/documents",
    vector_store_path="./chroma_db",
    model_name="claude-sonnet-4-5-20250929"
)
```

### Optimization Options

**For Faster Queries:**
- Reduce n_results from 5 to 3 (slight quality trade-off)
- Use Claude Haiku instead of Sonnet (faster, lower quality)
- Implement query caching for repeated questions

**For Better Accuracy:**
- Increase chunk_overlap from 50 to 100 characters
- Reduce chunk_size from 500 to 300 characters
- Retrieve more chunks (increase n_results to 7-10)

## Performance Analysis

### Latency Breakdown

Based on benchmark results across 5 queries:

| Component | Min | Avg | Max | Percent of Total |
|-----------|-----|-----|-----|------------------|
| Retrieval | 0.053s | 0.098s | 0.138s | 2.6% |
| Generation | 2.618s | 3.649s | 5.183s | 96% |
| Complete Query | 2.602s | 3.798s | 5.112s | 100% |

### Key Insights

1. **Retrieval is Fast**: Sub-100ms average, with minimal variance (0.053s - 0.138s)
2. **LLM is Bottleneck**: 96% of query time, varies based on answer complexity
3. **Consistent Performance**: 2.6s - 5.1s range provides predictable user experience
4. **Scalability**: Vector search remains fast even as corpus grows

### Future Optimizations

**Implemented:**
- Batch embedding generation (32 texts at a time)
- Cosine similarity for fast vector search
- Configurable chunk overlap for context retention
- Sentence-level splitting to preserve semantic meaning

**Planned:**
- Query caching for repeated questions (60% latency reduction expected)
- Hybrid search (keyword + semantic) for better precision
- Re-ranking layer for improved relevance
- Streaming responses for better perceived performance

## Use Cases

This RAG system can be adapted for:
- **Technical Documentation Q&A**: Product manuals, API docs, knowledge bases
- **Customer Support**: FAQ automation, ticket deflection
- **Research Assistance**: Paper search, citation extraction
- **Compliance and Policy**: Regulatory document search
- **Enterprise Knowledge Management**: Internal wiki, procedures
- **Contract Analysis**: Legal document search and summarization

## Design Decisions

### Why These Technologies?

**Sentence-Transformers (all-MiniLM-L6-v2):**
- Fast inference with batch processing
- Good semantic understanding
- Runs locally (no API costs)
- 384-dimensional embeddings (good balance of speed vs. quality)

**ChromaDB:**
- Easy setup (no server required)
- Persistent storage
- Fast similarity search with HNSW index
- Built-in metadata filtering

**Claude Sonnet 4.5:**
- Superior reasoning capabilities
- Strong instruction following
- Excellent at citing sources
- 200K token context window

### Trade-offs Considered

| Decision | Alternative | Why Not? |
|----------|------------|----------|
| Local embeddings | OpenAI embeddings | Cost ($0.13/1M tokens), API dependency |
| ChromaDB | Pinecone/Weaviate | Overkill for small corpus, added complexity |
| Claude Sonnet | GPT-4 | Claude better at instruction following |
| Text chunking | Recursive splitting | Paragraph-aware splitting preserves context |

## Contributing

This is a portfolio project demonstrating RAG implementation. Feel free to fork and adapt for your use case.

---

**Note**: This project uses sample electrical contracting documents for demonstration. Replace with your own domain-specific documents by adding .txt files to `data/documents/` and re-running `rag.index_documents()`.

## Technical Deep Dive

For detailed architecture explanation, component breakdown, and data flow examples, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).