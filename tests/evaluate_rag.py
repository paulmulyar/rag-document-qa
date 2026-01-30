"""
RAG Evaluation Framework
Tests retrieval accuracy and answer quality
"""

import json
import sys
from pathlib import Path
import time

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_pipeline: RAGPipeline, test_cases_path: str):
        """
        Initialize evaluator
        
        Args:
            rag_pipeline: Initialized RAG pipeline
            test_cases_path: Path to test questions JSON file
        """
        self.rag = rag_pipeline
        
        # Load test cases
        with open(test_cases_path, 'r') as f:
            data = json.load(f)
            self.test_cases = data['test_cases']
        
        print(f"✓ Loaded {len(self.test_cases)} test cases")
    
    def evaluate_retrieval_accuracy(self, n_results: int = 5) -> dict:
        """
        Test if correct documents are retrieved
        
        Args:
            n_results: Number of chunks to retrieve per query
            
        Returns:
            Dictionary with accuracy metrics
        """
        print(f"\nEvaluating Retrieval Accuracy (top {n_results} chunks)...")
        print("="*60)
        
        correct_retrievals = 0
        total_tests = len(self.test_cases)
        results = []
        
        for test_case in self.test_cases:
            question = test_case['question']
            expected_source = test_case['expected_source']
            
            # Retrieve chunks
            chunks = self.rag.retrieve(question, n_results=n_results)
            
            # Check if expected source is in retrieved chunks
            retrieved_sources = [chunk['source'] for chunk in chunks]
            is_correct = expected_source in retrieved_sources
            
            if is_correct:
                correct_retrievals += 1
            
            results.append({
                'question': question,
                'expected_source': expected_source,
                'retrieved_sources': retrieved_sources,
                'correct': is_correct,
                'top_similarity': chunks[0]['similarity'] if chunks else 0
            })
            
            # Print result
            status = "✓" if is_correct else "✗"
            print(f"{status} Q{test_case['id']}: {question[:50]}...")
            print(f"   Expected: {expected_source}")
            print(f"   Retrieved: {retrieved_sources[0] if retrieved_sources else 'None'}")
        
        accuracy = (correct_retrievals / total_tests) * 100
        
        print(f"\n{'='*60}")
        print(f"Retrieval Accuracy: {correct_retrievals}/{total_tests} = {accuracy:.1f}%")
        print(f"{'='*60}")
        
        return {
            'accuracy_percent': accuracy,
            'correct_retrievals': correct_retrievals,
            'total_tests': total_tests,
            'detailed_results': results
        }
    
    def evaluate_answer_quality(self, n_results: int = 3) -> dict:
        """
        Test if answers contain expected keywords
        
        Args:
            n_results: Number of chunks to use for generation
            
        Returns:
            Dictionary with quality metrics
        """
        print(f"\nEvaluating Answer Quality...")
        print("="*60)
        
        keyword_matches = 0
        total_keywords = 0
        results = []
        
        for test_case in self.test_cases:
            question = test_case['question']
            expected_keywords = test_case['expected_keywords']
            
            # Generate answer
            result = self.rag.query(question, n_results=n_results, max_tokens=500)
            answer = result['answer'].lower()
            
            # Check keyword presence
            matched_keywords = []
            for keyword in expected_keywords:
                if keyword.lower() in answer:
                    matched_keywords.append(keyword)
                    keyword_matches += 1
                total_keywords += 1
            
            match_rate = len(matched_keywords) / len(expected_keywords) * 100
            
            results.append({
                'question': question,
                'expected_keywords': expected_keywords,
                'matched_keywords': matched_keywords,
                'match_rate': match_rate,
                'answer_preview': answer[:200]
            })
            
            # Print result
            status = "✓" if match_rate >= 50 else "✗"
            print(f"{status} Q{test_case['id']}: {match_rate:.0f}% keywords matched")
            print(f"   Matched: {matched_keywords}")
        
        overall_rate = (keyword_matches / total_keywords) * 100
        
        print(f"\n{'='*60}")
        print(f"Keyword Match Rate: {keyword_matches}/{total_keywords} = {overall_rate:.1f}%")
        print(f"{'='*60}")
        
        return {
            'keyword_match_percent': overall_rate,
            'keywords_matched': keyword_matches,
            'total_keywords': total_keywords,
            'detailed_results': results
        }
    
    def run_full_evaluation(self) -> dict:
        """Run complete evaluation suite"""
        print("\n" + "="*60)
        print("RUNNING FULL RAG EVALUATION")
        print("="*60)
        
        # Test retrieval
        retrieval_results = self.evaluate_retrieval_accuracy(n_results=5)
        
        # Test answer quality
        quality_results = self.evaluate_answer_quality(n_results=3)
        
        # Combined report
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Retrieval Accuracy: {retrieval_results['accuracy_percent']:.1f}%")
        print(f"Keyword Match Rate: {quality_results['keyword_match_percent']:.1f}%")
        print(f"{'='*60}")
        
        return {
            'retrieval': retrieval_results,
            'quality': quality_results,
            'summary': {
                'retrieval_accuracy': retrieval_results['accuracy_percent'],
                'keyword_match_rate': quality_results['keyword_match_percent']
            }
        }


def main():
    """Run evaluation"""
    print("Initializing RAG Evaluation...\n")
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Index documents
    print("Indexing documents...")
    rag.index_documents()
    
    # Initialize evaluator
    test_cases_path = Path(__file__).parent / 'test_questions.json'
    evaluator = RAGEvaluator(rag, str(test_cases_path))
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Save results
    output_path = Path(__file__).parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()