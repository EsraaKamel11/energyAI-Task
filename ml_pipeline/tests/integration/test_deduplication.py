#!/usr/bin/env python3
"""
Test script for enhanced data deduplication using Levenshtein ratio
Demonstrates the fast_levenshtein method and comprehensive duplicate analysis
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
import random
import string

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.data_processing.deduplication import Deduplicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_documents(num_docs: int = 100) -> List[Dict[str, Any]]:
    """Generate sample documents with known duplicates for testing"""
    
    # Base documents
    base_documents = [
        {
            "id": "ev_charging_guide",
            "title": "Electric Vehicle Charging Guide",
            "content": "This comprehensive guide covers all aspects of electric vehicle charging, including CCS2, CHAdeMO, and Type 2 connectors. CCS2 charging supports maximum speeds up to 350kW, while CHAdeMO is limited to 150kW. The OCPP protocol enables smart charging and load balancing."
        },
        {
            "id": "battery_technology",
            "title": "Battery Technology Overview",
            "content": "Modern electric vehicle batteries use lithium-ion technology with advanced thermal management systems. Battery capacity ranges from 40kWh to 100kWh, providing driving ranges of 150 to 400 miles. Fast charging can replenish 80% of capacity in 30-45 minutes."
        },
        {
            "id": "charging_infrastructure",
            "title": "Charging Infrastructure Development",
            "content": "The development of charging infrastructure is crucial for widespread EV adoption. Public charging stations are being deployed across highways, shopping centers, and workplaces. Home charging solutions include Level 1 (120V) and Level 2 (240V) options."
        }
    ]
    
    documents = []
    
    # Add base documents
    for doc in base_documents:
        documents.append(doc.copy())
    
    # Generate duplicates with variations
    for i in range(num_docs - len(base_documents)):
        base_doc = random.choice(base_documents)
        
        # Create variations
        variation_type = random.choice(['minor', 'moderate', 'major'])
        
        if variation_type == 'minor':
            # Minor changes (high similarity)
            content = base_doc['content']
            # Replace a few words
            words = content.split()
            if len(words) > 10:
                replace_idx = random.randint(0, len(words) - 1)
                words[replace_idx] = random.choice(['modern', 'advanced', 'contemporary', 'current'])
                content = ' '.join(words)
        elif variation_type == 'moderate':
            # Moderate changes (medium similarity)
            content = base_doc['content']
            # Add or remove sentences
            sentences = content.split('. ')
            if len(sentences) > 2:
                if random.choice([True, False]):
                    # Remove a sentence
                    sentences.pop(random.randint(0, len(sentences) - 1))
                else:
                    # Add a sentence
                    new_sentences = [
                        "This technology continues to evolve rapidly.",
                        "Environmental considerations are paramount.",
                        "Cost-effectiveness is a key factor.",
                        "Safety standards must be maintained."
                    ]
                    sentences.append(random.choice(new_sentences))
                content = '. '.join(sentences)
        else:
            # Major changes (lower similarity)
            content = base_doc['content']
            # Significant rewording
            content = content.replace("electric vehicle", "EV")
            content = content.replace("charging", "power delivery")
            content = content.replace("battery", "energy storage")
        
        # Create new document
        new_doc = {
            "id": f"{base_doc['id']}_variant_{i}",
            "title": f"{base_doc['title']} (Variant {i})",
            "content": content
        }
        
        documents.append(new_doc)
    
    return documents

def test_fast_levenshtein_deduplication():
    """Test the fast Levenshtein deduplication method"""
    
    print("\n" + "="*60)
    print("TESTING FAST LEVENSHTEIN DEDUPLICATION")
    print("="*60)
    
    # Generate sample documents
    print("Generating sample documents...")
    documents = generate_sample_documents(50)
    print(f"Generated {len(documents)} sample documents")
    
    # Initialize deduplicator with fast_levenshtein method
    deduplicator = Deduplicator(
        similarity_threshold=0.95,
        method="fast_levenshtein"
    )
    
    # Perform deduplication
    print("\nPerforming deduplication...")
    deduplicated_docs = deduplicator.deduplicate(documents, text_column="content")
    
    # Get statistics
    stats = deduplicator.get_deduplication_stats(len(documents), len(deduplicated_docs))
    
    # Display results
    print("\n📊 DEDUPLICATION RESULTS:")
    print("-" * 40)
    print(f"Original documents: {stats['original_count']}")
    print(f"After deduplication: {stats['final_count']}")
    print(f"Removed duplicates: {stats['removed_count']}")
    print(f"Reduction: {stats['reduction_percentage']:.1f}%")
    print(f"Method: {stats['method']}")
    print(f"Threshold: {stats['threshold']}")
    
    if 'duplicate_pairs_count' in stats:
        print(f"Duplicate pairs found: {stats['duplicate_pairs_count']}")
        print(f"Average similarity: {stats['avg_similarity']:.3f}")
        print(f"Similarity range: {stats['min_similarity']:.3f} - {stats['max_similarity']:.3f}")
    
    return deduplicated_docs, stats

def test_different_methods():
    """Test different deduplication methods"""
    
    print("\n" + "="*60)
    print("COMPARING DEDUPLICATION METHODS")
    print("="*60)
    
    # Generate sample documents
    documents = generate_sample_documents(30)
    
    methods = ["fast_levenshtein", "levenshtein", "semantic"]
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} method...")
        
        try:
            deduplicator = Deduplicator(
                similarity_threshold=0.95,
                method=method
            )
            
            deduplicated = deduplicator.deduplicate(documents, text_column="content")
            stats = deduplicator.get_deduplication_stats(len(documents), len(deduplicated))
            
            results[method] = stats
            print(f"  ✅ {method}: {stats['reduction_percentage']:.1f}% reduction")
            
        except Exception as e:
            print(f"  ❌ {method}: Failed - {e}")
            results[method] = {"error": str(e)}
    
    # Compare results
    print("\n📈 METHOD COMPARISON:")
    print("-" * 40)
    for method, stats in results.items():
        if "error" not in stats:
            print(f"{method:20} | {stats['reduction_percentage']:6.1f}% | {stats['final_count']:3d} docs")
        else:
            print(f"{method:20} | ERROR: {stats['error']}")
    
    return results

def test_duplicate_analysis():
    """Test duplicate analysis without removal"""
    
    print("\n" + "="*60)
    print("TESTING DUPLICATE ANALYSIS")
    print("="*60)
    
    # Generate sample documents
    documents = generate_sample_documents(40)
    
    # Initialize deduplicator
    deduplicator = Deduplicator(similarity_threshold=0.95)
    
    # Analyze duplicates
    print("Analyzing potential duplicates...")
    analysis = deduplicator.analyze_duplicates(documents, text_column="content")
    
    # Display analysis results
    print("\n📋 DUPLICATE ANALYSIS RESULTS:")
    print("-" * 40)
    print(f"Total documents: {analysis['total_documents']}")
    print(f"Potential duplicates: {analysis['potential_duplicates']}")
    
    # Similarity distribution
    dist = analysis['similarity_distribution']
    print(f"\nSimilarity Distribution:")
    print(f"  Very High (>95%): {dist.get('very_high', 0)}")
    print(f"  High (90-95%): {dist.get('high', 0)}")
    print(f"  Medium (80-90%): {dist.get('medium', 0)}")
    
    # Show some high similarity pairs
    high_similarity = analysis['high_similarity_pairs']
    if high_similarity:
        print(f"\n🔍 HIGH SIMILARITY PAIRS (Sample):")
        for i, pair in enumerate(high_similarity[:3]):
            print(f"  Pair {i+1}:")
            print(f"    Doc1: {pair['doc1_id']}")
            print(f"    Doc2: {pair['doc2_id']}")
            print(f"    Similarity: {pair['similarity']:.3f}")
            print(f"    Preview1: {pair['doc1_preview']}")
            print(f"    Preview2: {pair['doc2_preview']}")
            print()
    
    return analysis

def test_duplicate_clusters():
    """Test duplicate clustering functionality"""
    
    print("\n" + "="*60)
    print("TESTING DUPLICATE CLUSTERS")
    print("="*60)
    
    # Generate sample documents
    documents = generate_sample_documents(25)
    
    # Initialize deduplicator
    deduplicator = Deduplicator(similarity_threshold=0.95)
    
    # Get duplicate clusters
    print("Finding duplicate clusters...")
    clusters = deduplicator.get_duplicate_clusters(documents, text_column="content")
    
    # Display cluster information
    print(f"\n📊 CLUSTER ANALYSIS:")
    print("-" * 40)
    print(f"Total clusters found: {len(clusters)}")
    
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1} (Size: {len(cluster)}):")
        for j, doc in enumerate(cluster):
            print(f"  {j+1}. {doc['id']} - {doc['title'][:50]}...")
    
    return clusters

def test_comprehensive_report():
    """Test comprehensive duplicate report generation"""
    
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE DUPLICATE REPORT")
    print("="*60)
    
    # Generate sample documents
    documents = generate_sample_documents(35)
    
    # Initialize deduplicator
    deduplicator = Deduplicator(similarity_threshold=0.95)
    
    # Generate comprehensive report
    print("Generating comprehensive duplicate report...")
    output_path = "test_outputs/duplicate_report.json"
    os.makedirs("test_outputs", exist_ok=True)
    
    report = deduplicator.export_duplicate_report(
        documents, 
        text_column="content", 
        output_path=output_path
    )
    
    # Display report summary
    print(f"\n📄 REPORT SUMMARY:")
    print("-" * 40)
    summary = report['summary']
    print(f"Original documents: {summary['original_count']}")
    print(f"Final documents: {summary['final_count']}")
    print(f"Removed: {summary['removed_count']} ({summary['reduction_percentage']:.1f}%)")
    
    analysis = report['analysis']
    print(f"Potential duplicates: {len(analysis['potential_duplicates'])}")
    print(f"Duplicate clusters: {len(report['clusters'])}")
    
    # Show recommendations
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print(f"\n✅ Comprehensive report saved to: {output_path}")
    
    return report

def test_batch_deduplication():
    """Test batch deduplication for large datasets"""
    
    print("\n" + "="*60)
    print("TESTING BATCH DEDUPLICATION")
    print("="*60)
    
    # Generate larger dataset
    documents = generate_sample_documents(200)
    
    # Initialize deduplicator
    deduplicator = Deduplicator(similarity_threshold=0.95)
    
    # Test batch deduplication
    print(f"Testing batch deduplication with {len(documents)} documents...")
    batch_size = 50
    
    deduplicated = deduplicator.batch_deduplicate(
        documents, 
        text_column="content", 
        batch_size=batch_size
    )
    
    # Get statistics
    stats = deduplicator.get_deduplication_stats(len(documents), len(deduplicated))
    
    print(f"\n📊 BATCH DEDUPLICATION RESULTS:")
    print("-" * 40)
    print(f"Original documents: {stats['original_count']}")
    print(f"After deduplication: {stats['final_count']}")
    print(f"Removed duplicates: {stats['removed_count']}")
    print(f"Reduction: {stats['reduction_percentage']:.1f}%")
    print(f"Batch size used: {batch_size}")
    
    return deduplicated, stats

def demonstrate_levenshtein_ratio():
    """Demonstrate the specific Levenshtein ratio approach"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING LEVENSHTEIN RATIO APPROACH")
    print("="*60)
    
    # Sample documents with known similarities
    sample_docs = [
        {
            "id": "doc1",
            "content": "CCS2 charging supports maximum speeds up to 350kW with OCPP protocol."
        },
        {
            "id": "doc2", 
            "content": "CCS2 charging supports maximum speeds up to 350kW with OCPP protocol and Plug&Charge functionality."
        },
        {
            "id": "doc3",
            "content": "CCS2 charging supports maximum speeds up to 350kW with OCPP protocol and Plug&Charge functionality for enhanced user experience."
        },
        {
            "id": "doc4",
            "content": "Electric vehicle charging infrastructure includes CCS2 connectors that support up to 350kW charging speeds."
        }
    ]
    
    print("Sample documents:")
    for doc in sample_docs:
        print(f"  {doc['id']}: {doc['content']}")
    
    # Test with different thresholds
    thresholds = [0.8, 0.9, 0.95, 0.98]
    
    for threshold in thresholds:
        print(f"\n--- Testing with threshold {threshold} ---")
        
        deduplicator = Deduplicator(
            similarity_threshold=threshold,
            method="fast_levenshtein"
        )
        
        deduplicated = deduplicator.deduplicate(sample_docs, text_column="content")
        stats = deduplicator.get_deduplication_stats(len(sample_docs), len(deduplicated))
        
        print(f"Threshold {threshold}: {stats['final_count']} documents remain")
        if 'duplicate_pairs_count' in stats:
            print(f"  Duplicate pairs: {stats['duplicate_pairs_count']}")
            print(f"  Avg similarity: {stats['avg_similarity']:.3f}")

def main():
    """Run all deduplication tests"""
    
    print("🚀 Enhanced Data Deduplication Test Suite")
    print("Testing Levenshtein ratio approach and advanced features...")
    
    try:
        # Test the fast Levenshtein method (requested approach)
        test_fast_levenshtein_deduplication()
        
        # Test different methods
        test_different_methods()
        
        # Test duplicate analysis
        test_duplicate_analysis()
        
        # Test duplicate clusters
        test_duplicate_clusters()
        
        # Test comprehensive reporting
        test_comprehensive_report()
        
        # Test batch processing
        test_batch_deduplication()
        
        # Demonstrate Levenshtein ratio
        demonstrate_levenshtein_ratio()
        
        print("\n✅ All deduplication tests completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("  • Fast Levenshtein ratio deduplication")
        print("  • Multiple deduplication methods")
        print("  • Duplicate analysis without removal")
        print("  • Duplicate clustering")
        print("  • Comprehensive reporting")
        print("  • Batch processing for large datasets")
        print("  • Performance optimization")
        
        print("\n📁 Check test_outputs/ directory for generated reports")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 
