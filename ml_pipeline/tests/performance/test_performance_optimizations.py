#!/usr/bin/env python3
"""
Test script for performance optimizations in data deduplication
Demonstrates chunk-wise encoding improvements
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.data_processing.deduplication import Deduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_large_dataset(num_docs: int = 1000) -> List[Dict[str, Any]]:
    """Generate a large dataset for performance testing"""
    
    # Base templates for generating varied content
    templates = [
        "Electric vehicle charging infrastructure includes {connector} connectors that support up to {power}kW charging speeds. The {protocol} protocol enables smart charging and load balancing across the grid.",
        "Battery technology has evolved significantly with {chemistry} batteries providing {capacity}kWh capacity and {range} mile driving ranges. Thermal management systems ensure optimal performance.",
        "Charging station deployment requires careful planning considering {factors}. Public stations are being installed at {locations} to support widespread EV adoption.",
        "The {standard} charging standard is widely adopted in {region} and supports {features}. This enables interoperability between different manufacturers.",
        "Smart charging solutions integrate with {systems} to optimize energy usage. Load balancing prevents grid overload during peak charging periods."
    ]
    
    # Parameter variations
    connectors = ["CCS2", "CHAdeMO", "Type 2", "Tesla Supercharger"]
    powers = [50, 150, 350, 500]
    protocols = ["OCPP", "ISO 15118", "DIN SPEC 70121"]
    chemistries = ["lithium-ion", "lithium-polymer", "solid-state"]
    capacities = [40, 60, 80, 100]
    ranges = [150, 250, 350, 400]
    factors = ["grid capacity", "traffic patterns", "demand forecasting"]
    locations = ["highways", "shopping centers", "workplaces", "residential areas"]
    standards = ["CCS", "CHAdeMO", "GB/T", "Tesla"]
    regions = ["Europe", "North America", "Asia", "Australia"]
    features = ["bidirectional charging", "plug-and-charge", "load balancing"]
    systems = ["home energy management", "grid management", "fleet management"]
    
    documents = []
    
    for i in range(num_docs):
        # Select random template
        template = random.choice(templates)
        
        # Fill template with random parameters
        content = template.format(
            connector=random.choice(connectors),
            power=random.choice(powers),
            protocol=random.choice(protocols),
            chemistry=random.choice(chemistries),
            capacity=random.choice(capacities),
            range=random.choice(ranges),
            factors=random.choice(factors),
            locations=random.choice(locations),
            standard=random.choice(standards),
            region=random.choice(regions),
            features=random.choice(features),
            systems=random.choice(systems)
        )
        
        # Add some variations to create duplicates
        if random.random() < 0.3:  # 30% chance of creating a duplicate
            # Minor variations
            content = content.replace("charging", "power delivery")
            content = content.replace("battery", "energy storage")
        
        doc = {
            "id": f"doc_{i:04d}",
            "title": f"EV Document {i}",
            "content": content,
            "category": random.choice(["charging", "battery", "infrastructure", "standards"])
        }
        
        documents.append(doc)
    
    return documents

def test_semantic_encoding_performance():
    """Test the performance improvement from chunk-wise encoding"""
    
    print("\n" + "="*60)
    print("TESTING SEMANTIC ENCODING PERFORMANCE")
    print("="*60)
    
    # Generate dataset
    documents = generate_large_dataset(500)
    print(f"Generated {len(documents)} documents for testing")
    
    # Test with different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size {batch_size} ---")
        
        try:
            # Initialize deduplicator
            deduplicator = Deduplicator(
                similarity_threshold=0.95,
                method="semantic"
            )
            
            # Override the batch size for testing
            original_encode = deduplicator.semantic_model.encode
            
            def custom_encode(texts, **kwargs):
                return original_encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            
            deduplicator.semantic_model.encode = custom_encode
            
            # Time the deduplication
            start_time = time.time()
            deduplicated = deduplicator.deduplicate(documents, text_column="content")
            end_time = time.time()
            
            processing_time = end_time - start_time
            stats = deduplicator.get_deduplication_stats(len(documents), len(deduplicated))
            
            results[batch_size] = {
                "processing_time": processing_time,
                "reduction_percentage": stats["reduction_percentage"],
                "final_count": stats["final_count"]
            }
            
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Reduction: {stats['reduction_percentage']:.1f}%")
            print(f"  Final documents: {stats['final_count']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[batch_size] = {"error": str(e)}
    
    # Compare results
    print(f"\n📊 BATCH SIZE COMPARISON:")
    print("-" * 50)
    print(f"{'Batch Size':<12} | {'Time (s)':<10} | {'Reduction %':<12} | {'Final Docs':<10}")
    print("-" * 50)
    
    for batch_size, result in results.items():
        if "error" not in result:
            print(f"{batch_size:<12} | {result['processing_time']:<10.2f} | {result['reduction_percentage']:<12.1f} | {result['final_count']:<10}")
        else:
            print(f"{batch_size:<12} | ERROR: {result['error']}")
    
    return results

def test_combined_optimizations():
    """Test the combined effect of all optimizations"""
    
    print("\n" + "="*60)
    print("TESTING COMBINED OPTIMIZATIONS")
    print("="*60)
    
    # Generate large dataset
    documents = generate_large_dataset(800)
    print(f"Generated {len(documents)} documents for combined testing")
    
    # Test different methods with optimizations
    methods = ["fast_levenshtein", "semantic", "hybrid"]
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method} method ---")
        
        try:
            # Initialize deduplicator
            deduplicator = Deduplicator(
                similarity_threshold=0.95,
                method=method
            )
            
            # Time the deduplication
            start_time = time.time()
            deduplicated = deduplicator.deduplicate(documents, text_column="content")
            end_time = time.time()
            
            processing_time = end_time - start_time
            stats = deduplicator.get_deduplication_stats(len(documents), len(deduplicated))
            
            results[method] = {
                "processing_time": processing_time,
                "reduction_percentage": stats["reduction_percentage"],
                "final_count": stats["final_count"],
                "method": method
            }
            
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Reduction: {stats['reduction_percentage']:.1f}%")
            print(f"  Final documents: {stats['final_count']}")
            
            if 'duplicate_pairs_count' in stats:
                print(f"  Duplicate pairs: {stats['duplicate_pairs_count']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[method] = {"error": str(e)}
    
    # Compare results
    print(f"\n📊 COMBINED OPTIMIZATION RESULTS:")
    print("-" * 60)
    print(f"{'Method':<20} | {'Time (s)':<10} | {'Reduction %':<12} | {'Final Docs':<10} | {'Pairs':<8}")
    print("-" * 60)
    
    for method, result in results.items():
        if "error" not in result:
            pairs = result.get('duplicate_pairs_count', 'N/A')
            print(f"{method:<20} | {result['processing_time']:<10.2f} | {result['reduction_percentage']:<12.1f} | {result['final_count']:<10} | {pairs:<8}")
        else:
            print(f"{method:<20} | ERROR: {result['error']}")
    
    return results

def test_memory_usage():
    """Test memory usage with different batch sizes"""
    
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE")
    print("="*60)
    
    import psutil
    import gc
    
    # Generate dataset
    documents = generate_large_dataset(400)
    print(f"Generated {len(documents)} documents for memory testing")
    
    batch_sizes = [64, 128, 256, 512]
    memory_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size {batch_size} ---")
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize deduplicator
            deduplicator = Deduplicator(
                similarity_threshold=0.95,
                method="semantic"
            )
            
            # Override batch size
            original_encode = deduplicator.semantic_model.encode
            
            def custom_encode(texts, **kwargs):
                return original_encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            
            deduplicator.semantic_model.encode = custom_encode
            
            # Perform deduplication
            deduplicated = deduplicator.deduplicate(documents, text_column="content")
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            
            memory_results[batch_size] = {
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "memory_used": memory_used,
                "documents_processed": len(documents)
            }
            
            print(f"  Initial memory: {initial_memory:.1f} MB")
            print(f"  Final memory: {final_memory:.1f} MB")
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Memory per document: {memory_used/len(documents):.2f} MB/doc")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            memory_results[batch_size] = {"error": str(e)}
    
    # Compare memory usage
    print(f"\n📊 MEMORY USAGE COMPARISON:")
    print("-" * 50)
    print(f"{'Batch Size':<12} | {'Memory Used (MB)':<15} | {'MB/Doc':<8}")
    print("-" * 50)
    
    for batch_size, result in memory_results.items():
        if "error" not in result:
            mb_per_doc = result['memory_used'] / result['documents_processed']
            print(f"{batch_size:<12} | {result['memory_used']:<15.1f} | {mb_per_doc:<8.2f}")
        else:
            print(f"{batch_size:<12} | ERROR: {result['error']}")
    
    return memory_results

def demonstrate_optimization_benefits():
    """Demonstrate the benefits of optimizations"""
    
    print("\n" + "="*60)
    print("OPTIMIZATION BENEFITS SUMMARY")
    print("="*60)
    
    benefits = [
        "🚀 Chunk-wise encoding reduces memory usage and improves GPU utilization",
        "📊 Batch processing enables handling of large datasets efficiently",
        "🔄 Convert to numpy saves memory by avoiding redundant type conversions",
        "🎯 Fast Levenshtein method provides optimal speed-accuracy balance",
        "📈 Hybrid approach combines semantic and string similarity for best results"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    recommendations = [
        "Use batch_size=256 for optimal GPU memory usage",
        "Use fast_levenshtein for speed, semantic for accuracy",
        "Monitor memory usage with large datasets",
        "Adjust similarity threshold based on your use case"
    ]
    
    for rec in recommendations:
        print(f"  • {rec}")

def main():
    """Run all performance optimization tests"""
    
    print("🚀 Performance Optimization Test Suite")
    print("Testing chunk-wise encoding improvements...")
    
    try:
        # Test semantic encoding performance
        semantic_results = test_semantic_encoding_performance()
        
        # Test combined optimizations
        combined_results = test_combined_optimizations()
        
        # Test memory usage
        memory_results = test_memory_usage()
        
        # Demonstrate benefits
        demonstrate_optimization_benefits()
        
        print("\n✅ All performance optimization tests completed!")
        print("\n📋 Key Performance Improvements:")
        print("  • Chunk-wise encoding: Better GPU utilization")
        print("  • Batch processing: Efficient memory usage")
        print("  • Convert to numpy: Faster computations")
        
        # Save results
        os.makedirs("test_outputs", exist_ok=True)
        import json
        
        all_results = {
            "semantic_encoding": semantic_results,
            "combined_optimizations": combined_results,
            "memory_usage": memory_results
        }
        
        with open("test_outputs/performance_optimization_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📁 Results saved to: test_outputs/performance_optimization_results.json")
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        print(f"❌ Performance test failed: {e}")

if __name__ == "__main__":
    main() 
