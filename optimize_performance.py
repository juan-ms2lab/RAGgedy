#!/usr/bin/env python3
"""
Performance Optimization Tool
System Architect Agent Implementation

Analyzes and optimizes RAG system performance including chunk sizes,
retrieval parameters, and system resource utilization.
"""

import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from build_index import VectorIndexBuilder
from extract_and_chunk import DocumentProcessor

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class RAGPerformanceOptimizer:
    """Optimizes RAG system performance through parameter tuning and analysis."""
    
    def __init__(self, docs_dir: str = "docs"):
        """
        Initialize performance optimizer.
        
        Args:
            docs_dir: Directory containing documents
        """
        self.docs_dir = docs_dir
        self.test_queries = [
            "What is machine learning?",
            "How does supervised learning work?",
            "What are the applications of AI in healthcare?",
            "What are the challenges in machine learning?",
            "What is reinforcement learning?",
            "How is ML used in finance?",
            "What are the future trends in AI?",
            "What is unsupervised learning?",
            "What ethical considerations exist for AI?",
            "How does natural language processing work?"
        ]
    
    def benchmark_chunk_sizes(self, chunk_sizes: List[int]) -> Dict:
        """
        Test different chunk sizes and measure performance.
        
        Args:
            chunk_sizes: List of chunk sizes to test
            
        Returns:
            Performance metrics for each chunk size
        """
        results = []
        
        for chunk_size in chunk_sizes:
            print(f"\nTesting chunk size: {chunk_size}")
            
            # Process documents with this chunk size
            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_size//5)
            chunks = processor.process_directory(self.docs_dir)
            
            if not chunks:
                continue
            
            # Build temporary index
            temp_db = f"temp_db_{chunk_size}"
            Path(temp_db).mkdir(exist_ok=True)
            
            try:
                builder = VectorIndexBuilder(
                    db_dir=temp_db,
                    collection_name=f"test_{chunk_size}"
                )
                
                # Measure index building time
                start_time = time.time()
                builder.add_chunks_to_index(chunks)
                build_time = time.time() - start_time
                
                # Measure query performance
                query_times = []
                relevance_scores = []
                
                for query in self.test_queries:
                    start_time = time.time()
                    results_data = builder.query_index(query, n_results=5)
                    query_time = time.time() - start_time
                    
                    query_times.append(query_time)
                    
                    # Calculate average relevance score
                    if results_data.get('distances') and results_data['distances'][0]:
                        avg_distance = statistics.mean(results_data['distances'][0])
                        avg_similarity = 1 - avg_distance
                        relevance_scores.append(avg_similarity)
                
                # Compile metrics
                chunk_result = {
                    'chunk_size': chunk_size,
                    'total_chunks': len(chunks),
                    'avg_chunk_length': statistics.mean(len(chunk['text']) for chunk in chunks),
                    'build_time': build_time,
                    'avg_query_time': statistics.mean(query_times),
                    'avg_relevance': statistics.mean(relevance_scores) if relevance_scores else 0,
                    'query_time_std': statistics.stdev(query_times) if len(query_times) > 1 else 0
                }
                
                results.append(chunk_result)
                print(f"  Chunks: {chunk_result['total_chunks']}")
                print(f"  Build time: {chunk_result['build_time']:.2f}s")
                print(f"  Avg query time: {chunk_result['avg_query_time']:.3f}s")
                print(f"  Avg relevance: {chunk_result['avg_relevance']:.3f}")
                
            except Exception as e:
                print(f"Error testing chunk size {chunk_size}: {e}")
            
            finally:
                # Cleanup temporary database
                import shutil
                if Path(temp_db).exists():
                    shutil.rmtree(temp_db)
        
        return results
    
    def benchmark_retrieval_counts(self, top_k_values: List[int]) -> Dict:
        """
        Test different retrieval counts (top_k) and measure performance.
        
        Args:
            top_k_values: List of top_k values to test
            
        Returns:
            Performance metrics for each top_k value
        """
        builder = VectorIndexBuilder()  # Use existing index
        results = []
        
        for top_k in top_k_values:
            print(f"\nTesting top_k: {top_k}")
            
            query_times = []
            relevance_scores = []
            
            for query in self.test_queries:
                start_time = time.time()
                results_data = builder.query_index(query, n_results=top_k)
                query_time = time.time() - start_time
                
                query_times.append(query_time)
                
                # Calculate average relevance score
                if results_data.get('distances') and results_data['distances'][0]:
                    avg_distance = statistics.mean(results_data['distances'][0])
                    avg_similarity = 1 - avg_distance
                    relevance_scores.append(avg_similarity)
            
            result = {
                'top_k': top_k,
                'avg_query_time': statistics.mean(query_times),
                'avg_relevance': statistics.mean(relevance_scores) if relevance_scores else 0,
                'query_time_std': statistics.stdev(query_times) if len(query_times) > 1 else 0
            }
            
            results.append(result)
            print(f"  Avg query time: {result['avg_query_time']:.3f}s")
            print(f"  Avg relevance: {result['avg_relevance']:.3f}")
        
        return results
    
    def analyze_document_characteristics(self) -> Dict:
        """Analyze characteristics of documents in the collection."""
        processor = DocumentProcessor()
        chunks = processor.process_directory(self.docs_dir)
        
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['text']) for chunk in chunks]
        
        analysis = {
            'total_documents': len(set(chunk['source_file'] for chunk in chunks)),
            'total_chunks': len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_chunk_length': statistics.mean(chunk_lengths),
            'median_chunk_length': statistics.median(chunk_lengths),
            'chunk_length_std': statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
        }
        
        return analysis
    
    def generate_recommendations(
        self, 
        chunk_results: List[Dict], 
        retrieval_results: List[Dict]
    ) -> Dict:
        """
        Generate optimization recommendations based on benchmark results.
        
        Args:
            chunk_results: Results from chunk size benchmarking
            retrieval_results: Results from retrieval count benchmarking
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'optimal_chunk_size': None,
            'optimal_top_k': None,
            'performance_summary': {},
            'suggestions': []
        }
        
        # Find optimal chunk size (balance between relevance and speed)
        if chunk_results:
            # Score each chunk size (higher is better)
            scored_chunks = []
            for result in chunk_results:
                # Normalize metrics (0-1 scale)
                relevance_score = result['avg_relevance']
                speed_score = 1 / (1 + result['avg_query_time'])  # Invert for speed
                
                # Weighted combination (favor relevance slightly)
                combined_score = 0.6 * relevance_score + 0.4 * speed_score
                scored_chunks.append((result['chunk_size'], combined_score, result))
            
            # Find best chunk size
            best_chunk = max(scored_chunks, key=lambda x: x[1])
            recommendations['optimal_chunk_size'] = best_chunk[0]
            recommendations['performance_summary']['chunk_analysis'] = best_chunk[2]
        
        # Find optimal top_k value
        if retrieval_results:
            scored_retrievals = []
            for result in retrieval_results:
                relevance_score = result['avg_relevance']
                speed_score = 1 / (1 + result['avg_query_time'])
                
                # For retrieval, balance relevance and speed equally
                combined_score = 0.5 * relevance_score + 0.5 * speed_score
                scored_retrievals.append((result['top_k'], combined_score, result))
            
            best_retrieval = max(scored_retrievals, key=lambda x: x[1])
            recommendations['optimal_top_k'] = best_retrieval[0]
            recommendations['performance_summary']['retrieval_analysis'] = best_retrieval[2]
        
        # Generate suggestions
        if recommendations['optimal_chunk_size']:
            chunk_size = recommendations['optimal_chunk_size']
            recommendations['suggestions'].append(
                f"Use chunk size of {chunk_size} characters for optimal balance of relevance and performance"
            )
        
        if recommendations['optimal_top_k']:
            top_k = recommendations['optimal_top_k']
            recommendations['suggestions'].append(
                f"Retrieve top {top_k} chunks for optimal query performance"
            )
        
        # Additional suggestions based on patterns
        if chunk_results:
            build_times = [r['build_time'] for r in chunk_results]
            if max(build_times) > 10:
                recommendations['suggestions'].append(
                    "Consider processing documents in smaller batches to reduce index build time"
                )
            
            query_times = [r['avg_query_time'] for r in chunk_results]
            if max(query_times) > 1.0:
                recommendations['suggestions'].append(
                    "Query times are high - consider using a smaller embedding model or reducing chunk count"
                )
        
        return recommendations
    
    def run_full_optimization(self) -> Dict:
        """
        Run complete performance optimization analysis.
        
        Returns:
            Complete optimization report
        """
        print("RAG Performance Optimization Analysis")
        print("=" * 50)
        
        # Document analysis
        print("\n1. Analyzing document characteristics...")
        doc_analysis = self.analyze_document_characteristics()
        
        if doc_analysis:
            print(f"Total documents: {doc_analysis['total_documents']}")
            print(f"Total chunks: {doc_analysis['total_chunks']}")
            print(f"Average chunk length: {doc_analysis['avg_chunk_length']:.0f} characters")
        
        # Chunk size optimization
        print("\n2. Benchmarking chunk sizes...")
        chunk_sizes = [400, 600, 800, 1000, 1200, 1500]
        chunk_results = self.benchmark_chunk_sizes(chunk_sizes)
        
        # Retrieval count optimization
        print("\n3. Benchmarking retrieval counts...")
        top_k_values = [3, 5, 7, 10]
        retrieval_results = self.benchmark_retrieval_counts(top_k_values)
        
        # Generate recommendations
        print("\n4. Generating recommendations...")
        recommendations = self.generate_recommendations(chunk_results, retrieval_results)
        
        # Compile full report
        report = {
            'document_analysis': doc_analysis,
            'chunk_benchmarks': chunk_results,
            'retrieval_benchmarks': retrieval_results,
            'recommendations': recommendations,
            'timestamp': time.time()
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted optimization report."""
        print("\n" + "="*60)
        print("RAG SYSTEM PERFORMANCE OPTIMIZATION REPORT")
        print("="*60)
        
        # Document analysis
        if report.get('document_analysis'):
            doc = report['document_analysis']
            print(f"\nDocument Analysis:")
            print(f"  Total documents: {doc['total_documents']}")
            print(f"  Total chunks: {doc['total_chunks']}")
            print(f"  Avg chunk length: {doc['avg_chunk_length']:.0f} chars")
            print(f"  Chunk length range: {doc['min_chunk_length']}-{doc['max_chunk_length']} chars")
        
        # Recommendations
        if report.get('recommendations'):
            rec = report['recommendations']
            print(f"\nOptimal Configuration:")
            if rec['optimal_chunk_size']:
                print(f"  Chunk size: {rec['optimal_chunk_size']} characters")
            if rec['optimal_top_k']:
                print(f"  Retrieval count: {rec['optimal_top_k']} chunks")
            
            print(f"\nRecommendations:")
            for i, suggestion in enumerate(rec['suggestions'], 1):
                print(f"  {i}. {suggestion}")
        
        # Performance summary
        if report.get('recommendations', {}).get('performance_summary'):
            perf = report['recommendations']['performance_summary']
            print(f"\nPerformance Metrics:")
            if 'chunk_analysis' in perf:
                chunk_perf = perf['chunk_analysis']
                print(f"  Query time: {chunk_perf['avg_query_time']:.3f}s")
                print(f"  Relevance score: {chunk_perf['avg_relevance']:.3f}")
            
        print("\n" + "="*60)


def main():
    """Command line interface for performance optimization."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick optimization with fewer test cases
        optimizer = RAGPerformanceOptimizer()
        
        print("Running quick performance analysis...")
        
        # Test fewer chunk sizes
        chunk_results = optimizer.benchmark_chunk_sizes([600, 800, 1000])
        retrieval_results = optimizer.benchmark_retrieval_counts([3, 5, 7])
        
        recommendations = optimizer.generate_recommendations(chunk_results, retrieval_results)
        
        print("\nQuick Optimization Results:")
        if recommendations['optimal_chunk_size']:
            print(f"Recommended chunk size: {recommendations['optimal_chunk_size']}")
        if recommendations['optimal_top_k']:
            print(f"Recommended retrieval count: {recommendations['optimal_top_k']}")
    
    else:
        # Full optimization analysis
        optimizer = RAGPerformanceOptimizer()
        report = optimizer.run_full_optimization()
        optimizer.print_report(report)
        
        # Save report
        import json
        with open('optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: optimization_report.json")


if __name__ == "__main__":
    main()