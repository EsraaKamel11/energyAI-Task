#!/usr/bin/env python3
"""
Memory Management Test Script
Demonstrates chunked processing, memory monitoring, and optimization strategies
"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile
import shutil

# Import memory management utilities
from src.utils.memory_manager import memory_manager, memory_safe, chunked_processing
from src.utils.config_manager import get_config

# Import pipeline components
from src.data_collection.web_scraper import WebScraper
from src.data_collection.pdf_extractor import PDFExtractor
from src.data_processing.deduplication import Deduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_memory_monitoring():
    """Test memory monitoring capabilities"""
    print("\n" + "="*60)
    print("TESTING MEMORY MONITORING")
    print("="*60)
    
    # Get initial memory stats
    initial_stats = memory_manager.get_memory_stats()
    print(f"Initial memory usage: {initial_stats.used_memory_mb:.2f}MB / {initial_stats.total_memory_mb:.2f}MB ({initial_stats.memory_percentage:.1f}%)")
    
    # Check memory availability
    required_memory = 100  # 100MB
    is_available = memory_manager.is_memory_available(required_memory)
    print(f"Memory available for {required_memory}MB: {is_available}")
    
    # Check if memory is critical
    is_critical = memory_manager.is_memory_critical(threshold_percentage=80.0)
    print(f"Memory critical (>80%): {is_critical}")
    
    # Get comprehensive memory report
    memory_report = memory_manager.get_memory_report()
    print(f"Memory report: {json.dumps(memory_report, indent=2)}")

def test_chunked_processing():
    """Test chunked processing functionality"""
    print("\n" + "="*60)
    print("TESTING CHUNKED PROCESSING")
    print("="*60)
    
    # Create test data
    test_items = list(range(1000))
    print(f"Created {len(test_items)} test items")
    
    # Test chunked iteration
    chunk_size = 100
    chunks = list(memory_manager.chunked(test_items, chunk_size))
    print(f"Split into {len(chunks)} chunks of size {chunk_size}")
    print(f"First chunk: {chunks[0][:5]}...")
    print(f"Last chunk: {chunks[-1][-5:]}...")
    
    # Test chunked processing with a simple function
    def process_chunk(chunk):
        return [x * 2 for x in chunk]
    
    results = memory_manager.process_in_chunks(test_items, process_chunk, chunk_size=50)
    print(f"Processed {len(results)} chunks")
    print(f"Total processed items: {sum(len(r) for r in results)}")

def test_memory_safe_embedding():
    """Test memory-safe embedding generation"""
    print("\n" + "="*60)
    print("TESTING MEMORY-SAFE EMBEDDING")
    print("="*60)
    
    # Create test texts
    test_texts = [f"This is test text number {i} for embedding generation." for i in range(100)]
    print(f"Created {len(test_texts)} test texts")
    
    # Mock embedding function
    def mock_embedder(texts):
        # Simulate embedding generation
        time.sleep(0.1)  # Simulate processing time
        return np.random.rand(len(texts), 384)  # 384-dimensional embeddings
    
    # Test memory-safe embedding
    start_time = time.time()
    embeddings = memory_manager.memory_safe_embedding(test_texts, mock_embedder, chunk_size=20)
    end_time = time.time()
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding generation time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage after embedding: {memory_manager.get_memory_stats().used_memory_mb:.2f}MB")

def test_memory_optimization():
    """Test memory optimization strategies"""
    print("\n" + "="*60)
    print("TESTING MEMORY OPTIMIZATION")
    print("="*60)
    
    # Create some objects to consume memory
    large_objects = []
    for i in range(100):
        large_objects.append(np.random.rand(1000, 1000))  # 8MB each
    
    print(f"Created {len(large_objects)} large objects")
    print(f"Memory usage before optimization: {memory_manager.get_memory_stats().used_memory_mb:.2f}MB")
    
    # Force garbage collection
    memory_manager.force_garbage_collection()
    print(f"Memory usage after garbage collection: {memory_manager.get_memory_stats().used_memory_mb:.2f}MB")
    
    # Test memory optimization
    optimization_results = memory_manager.optimize_memory()
    print(f"Memory optimization results: {json.dumps(optimization_results, indent=2)}")
    
    # Clean up
    del large_objects
    memory_manager.force_garbage_collection()

def test_disk_operations():
    """Test disk save/load operations"""
    print("\n" + "="*60)
    print("TESTING DISK OPERATIONS")
    print("="*60)
    
    # Create test data (JSON serializable only)
    test_data = {
        'numbers': list(range(1000)),
        'strings': [f"string_{i}" for i in range(1000)],
        'nested_dict': {'key1': 'value1', 'key2': [1, 2, 3]}
    }
    
    # Test saving to disk with JSON
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        memory_manager.save_to_disk(test_data, temp_path, "json")
        print(f"Saved data to {temp_path}")
        
        # Test loading from disk
        loaded_data = memory_manager.load_from_disk(temp_path, "json")
        print(f"Loaded data from {temp_path}")
        
        # Check if data was loaded successfully
        if loaded_data is not None:
            print(f"Data integrity check: {test_data['numbers'][:5] == loaded_data['numbers'][:5]}")
        else:
            print("Failed to load data from JSON file")
        
        # Test with pickle for numpy arrays
        test_data_with_numpy = {
            'numbers': list(range(1000)),
            'strings': [f"string_{i}" for i in range(1000)],
            'numpy_array': np.random.rand(100, 100)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle_path = f.name
        
        try:
            memory_manager.save_to_disk(test_data_with_numpy, pickle_path, "pickle")
            print(f"Saved numpy data to {pickle_path}")
            
            loaded_numpy_data = memory_manager.load_from_disk(pickle_path, "pickle")
            print(f"Loaded numpy data from {pickle_path}")
            
            if loaded_numpy_data is not None:
                print(f"Numpy data integrity check: {np.array_equal(test_data_with_numpy['numpy_array'], loaded_numpy_data['numpy_array'])}")
            else:
                print("Failed to load numpy data from pickle file")
        finally:
            if Path(pickle_path).exists():
                Path(pickle_path).unlink()
        
    finally:
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()

def test_streaming_processor():
    """Test streaming processor for large datasets"""
    print("\n" + "="*60)
    print("TESTING STREAMING PROCESSOR")
    print("="*60)
    
    # Create test data iterator
    def data_iterator():
        for i in range(1000):
            yield f"data_item_{i}"
    
    # Define processor function
    def process_item(item):
        return f"processed_{item}"
    
    # Test streaming processor
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        memory_manager.streaming_processor(
            data_iterator(),
            process_item,
            output_path,
            batch_size=50
        )
        
        # Check results
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        print(f"Streaming processor completed: {len(results)} items processed")
        print(f"Sample results: {results[:3]}")
        
    finally:
        # Clean up
        if Path(output_path).exists():
            Path(output_path).unlink()

def test_parallel_processor():
    """Test parallel processing with memory management"""
    print("\n" + "="*60)
    print("TESTING PARALLEL PROCESSOR")
    print("="*60)
    
    # Create test items
    test_items = list(range(100))
    
    # Define processor function
    def process_item(item):
        time.sleep(0.01)  # Simulate work
        return item * 2
    
    # Test parallel processing
    start_time = time.time()
    results = memory_manager.parallel_processor(test_items, process_item, max_workers=4)
    end_time = time.time()
    
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    print(f"Results: {results[:5]}...")
    print(f"Memory usage after parallel processing: {memory_manager.get_memory_stats().used_memory_mb:.2f}MB")

def test_memory_safe_decorator():
    """Test memory-safe decorator"""
    print("\n" + "="*60)
    print("TESTING MEMORY-SAFE DECORATOR")
    print("="*60)
    
    @memory_safe
    def memory_intensive_function(size):
        """Function that consumes a lot of memory"""
        data = []
        for i in range(size):
            data.append(np.random.rand(100, 100))
        return len(data)
    
    # Test with different sizes
    for size in [10, 50, 100]:
        try:
            print(f"Testing with size {size}...")
            result = memory_intensive_function(size)
            print(f"Result: {result}")
            print(f"Memory usage: {memory_manager.get_memory_stats().used_memory_mb:.2f}MB")
        except Exception as e:
            print(f"Error with size {size}: {e}")

def test_chunked_processing_decorator():
    """Test chunked processing decorator"""
    print("\n" + "="*60)
    print("TESTING CHUNKED PROCESSING DECORATOR")
    print("="*60)
    
    # Create a class to test the decorator (since it expects a method)
    class TestProcessor:
        @chunked_processing(chunk_size=50)
        def process_large_list(self, items):
            """Process a large list of items"""
            return [item * 2 for item in items]
    
    # Create processor instance
    processor = TestProcessor()
    
    # Create large list
    large_list = list(range(1000))
    print(f"Created list with {len(large_list)} items")
    
    # Test chunked processing
    start_time = time.time()
    results = processor.process_large_list(large_list)
    end_time = time.time()
    
    print(f"Chunked processing completed in {end_time - start_time:.2f} seconds")
    print(f"Results length: {len(results)}")
    print(f"Sample results: {results[:5]}...")
    
    # Verify results are flattened (since process_in_chunks returns list of lists)
    if results and isinstance(results[0], list):
        flattened_results = []
        for chunk_result in results:
            flattened_results.extend(chunk_result)
        print(f"Flattened results length: {len(flattened_results)}")
        print(f"Sample flattened results: {flattened_results[:5]}...")
    else:
        print("Results are already flattened")

def test_pipeline_integration():
    """Test memory management integration with pipeline components"""
    print("\n" + "="*60)
    print("TESTING PIPELINE INTEGRATION")
    print("="*60)
    
    # Test deduplicator with memory management
    print("Testing deduplicator...")
    deduplicator = Deduplicator(method="levenshtein", similarity_threshold=0.95)
    
    # Create test documents
    test_docs = [
        {"id": f"doc_{i}", "text": f"This is test document number {i} with some content."}
        for i in range(100)
    ]
    
    # Add some duplicates
    for i in range(10):
        test_docs.append({
            "id": f"duplicate_{i}",
            "text": f"This is test document number {i} with some content."
        })
    
    print(f"Created {len(test_docs)} test documents (including duplicates)")
    
    # Test deduplication
    start_time = time.time()
    unique_docs = deduplicator.deduplicate(test_docs, text_column="text")
    end_time = time.time()
    
    print(f"Deduplication completed in {end_time - start_time:.2f} seconds")
    print(f"Original documents: {len(test_docs)}")
    print(f"Unique documents: {len(unique_docs)}")
    print(f"Removed {len(test_docs) - len(unique_docs)} duplicates")
    
    # Get deduplication stats
    stats = deduplicator.get_deduplication_stats(len(test_docs), len(unique_docs))
    print(f"Deduplication stats: {json.dumps(stats, indent=2)}")

def test_memory_monitoring_background():
    """Test background memory monitoring"""
    print("\n" + "="*60)
    print("TESTING BACKGROUND MEMORY MONITORING")
    print("="*60)
    
    # Start memory monitoring
    memory_manager.memory_monitor(interval=2.0)  # Monitor every 2 seconds
    print("Started background memory monitoring")
    
    # Simulate some work
    print("Simulating memory-intensive work...")
    for i in range(5):
        # Create some objects
        temp_objects = [np.random.rand(100, 100) for _ in range(10)]
        print(f"Created batch {i+1}, memory usage: {memory_manager.get_memory_stats().used_memory_mb:.2f}MB")
        time.sleep(1)
        del temp_objects
    
    # Stop monitoring
    memory_manager.monitoring_enabled = False
    print("Stopped background memory monitoring")

def test_configuration_integration():
    """Test memory management with configuration"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION INTEGRATION")
    print("="*60)
    
    # Load configuration
    config = get_config()
    
    # Get memory management configuration
    memory_config = config.get("memory_management", {})
    print(f"Memory management configuration: {json.dumps(memory_config, indent=2)}")
    
    # Get deduplication configuration
    dedup_config = config.get("deduplication", {})
    print(f"Deduplication configuration: {json.dumps(dedup_config, indent=2)}")
    
    # Test configuration-based memory limits
    max_memory_mb = memory_config.get("max_memory_mb", 4096)
    chunk_size = memory_config.get("chunk_size", 1000)
    
    print(f"Configured max memory: {max_memory_mb}MB")
    print(f"Configured chunk size: {chunk_size}")

def main():
    """Run all memory management tests"""
    print("MEMORY MANAGEMENT TEST SUITE")
    print("="*60)
    
    try:
        # Test basic memory monitoring
        test_memory_monitoring()
        
        # Test chunked processing
        test_chunked_processing()
        
        # Test memory-safe embedding
        test_memory_safe_embedding()
        
        # Test memory optimization
        test_memory_optimization()
        
        # Test disk operations
        test_disk_operations()
        
        # Test streaming processor
        test_streaming_processor()
        
        # Test parallel processor
        test_parallel_processor()
        
        # Test decorators
        test_memory_safe_decorator()
        test_chunked_processing_decorator()
        
        # Test pipeline integration
        test_pipeline_integration()
        
        # Test configuration integration
        test_configuration_integration()
        
        # Test background monitoring
        test_memory_monitoring_background()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final memory report
        final_report = memory_manager.get_memory_report()
        print(f"Final memory report: {json.dumps(final_report, indent=2)}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        memory_manager.cleanup()

if __name__ == "__main__":
    main() 
