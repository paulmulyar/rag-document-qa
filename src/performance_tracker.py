"""
Performance Tracker
Measures latency of RAG pipeline components
"""

import time
from typing import Dict, List
from pathlib import Path
import json


class PerformanceTracker:
    """Track latency of RAG operations"""
    
    def __init__(self):
        """Initialize tracker"""
        self.measurements = []
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return TimedOperation(operation_name, self)
    
    def add_measurement(self, operation: str, duration: float, metadata: dict = None):
        """Add a timing measurement"""
        self.measurements.append({
            'operation': operation,
            'duration_seconds': duration,
            'metadata': metadata or {}
        })
    
    def get_stats(self) -> Dict:
        """Calculate statistics across all measurements"""
        if not self.measurements:
            return {}
        
        # Group by operation
        by_operation = {}
        for m in self.measurements:
            op = m['operation']
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(m['duration_seconds'])
        
        # Calculate stats for each operation
        stats = {}
        for op, durations in by_operation.items():
            stats[op] = {
                'count': len(durations),
                'total_seconds': sum(durations),
                'avg_seconds': sum(durations) / len(durations),
                'min_seconds': min(durations),
                'max_seconds': max(durations)
            }
        
        return stats
    
    def print_report(self):
        """Print performance report"""
        stats = self.get_stats()
        
        if not stats:
            print("No measurements recorded")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        total_time = sum(s['total_seconds'] for s in stats.values())
        
        for operation, data in sorted(stats.items()):
            pct = (data['total_seconds'] / total_time * 100) if total_time > 0 else 0
            print(f"\n{operation}:")
            print(f"  Count: {data['count']}")
            print(f"  Average: {data['avg_seconds']:.3f}s")
            print(f"  Min: {data['min_seconds']:.3f}s")
            print(f"  Max: {data['max_seconds']:.3f}s")
            print(f"  Total: {data['total_seconds']:.3f}s ({pct:.1f}% of total)")
        
        print(f"\n{'='*60}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"{'='*60}")
    
    def save_report(self, filepath: str):
        """Save report to JSON file"""
        stats = self.get_stats()
        
        with open(filepath, 'w') as f:
            json.dump({
                'measurements': self.measurements,
                'statistics': stats
            }, f, indent=2)
        
        print(f"\n✓ Performance report saved to: {filepath}")


class TimedOperation:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, tracker: PerformanceTracker):
        self.operation_name = operation_name
        self.tracker = tracker
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.tracker.add_measurement(self.operation_name, duration)


def benchmark_rag_pipeline():
    """Benchmark the complete RAG pipeline"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.rag_pipeline import RAGPipeline
    
    print("Benchmarking RAG Pipeline...")
    print("="*60)
    
    tracker = PerformanceTracker()
    
    # Initialize (time this)
    with tracker.time_operation("pipeline_initialization"):
        rag = RAGPipeline()
    
    # Index documents (time this)
    with tracker.time_operation("document_indexing"):
        rag.index_documents()
    
    # Test queries
    test_queries = [
        "What safety equipment is required?",
        "How much does outlet installation cost?",
        "What was the TechCorp project about?",
        "What are the lockout tagout procedures?",
        "What is the hourly rate for electricians?"
    ]
    
    print(f"\nRunning {len(test_queries)} test queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}/{len(test_queries)}: {query[:50]}...")
        
        # Time retrieval
        with tracker.time_operation("retrieval"):
            chunks = rag.retrieve(query, n_results=3)
        
        # Time generation
        with tracker.time_operation("generation"):
            result = rag.generate_answer(query, chunks, max_tokens=500)
        
        # Time complete query (retrieval + generation)
        with tracker.time_operation("complete_query"):
            _ = rag.query(query, n_results=3, max_tokens=500)
    
    # Print and save report
    tracker.print_report()
    
    output_path = Path('tests') / 'performance_report.json'
    output_path.parent.mkdir(exist_ok=True)
    tracker.save_report(str(output_path))
    
    # Calculate average query time
    stats = tracker.get_stats()
    if 'complete_query' in stats:
        avg_query_time = stats['complete_query']['avg_seconds']
        print(f"\n{'='*60}")
        print(f"Average Query Time: {avg_query_time:.2f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    benchmark_rag_pipeline()