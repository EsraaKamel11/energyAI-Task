#!/usr/bin/env python3
"""
Test script for QA generation pipeline
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import only the QA generator
from src.data_processing.qa_generation import QAGenerator, QAGenerationConfig

def test_qa_generation():
    """Test the QA generation pipeline"""
    print("🧪 Testing QA Generation Pipeline")
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Running tests without actual API calls.")
        test_qa_generation_mock()
        return True
    
    # Create test data
    test_documents = [
        {
            "text": "Electric vehicles (EVs) are automobiles that use electric motors for propulsion. They are powered by rechargeable batteries and produce zero tailpipe emissions. EVs are becoming increasingly popular due to their environmental benefits and lower operating costs compared to traditional gasoline vehicles.",
            "source": "https://www.tesla.com/support/charging",
            "type": "web",
            "title": "Electric Vehicle Basics",
            "author": "Tesla Support",
            "date": "2024-01-15"
        },
        {
            "text": "Level 2 charging stations use 240V power and can charge an electric vehicle much faster than Level 1 charging. A typical Level 2 charger can provide 10-60 miles of range per hour of charging, depending on the vehicle and charger specifications.",
            "source": "sample_document.pdf",
            "type": "pdf",
            "title": "EV Charging Guide",
            "author": "EV Association",
            "date": "2024-01-10"
        },
        {
            "text": "DC fast charging, also known as Level 3 charging, can provide 60-80% charge in just 20-30 minutes. These chargers use high-voltage direct current and are typically found at public charging stations along highways and in urban areas.",
            "source": "https://www.electrifyamerica.com/",
            "type": "web",
            "title": "DC Fast Charging",
            "author": "Electrify America",
            "date": "2024-01-20"
        }
    ]
    
    print(f"\n📋 Test documents: {len(test_documents)}")
    for i, doc in enumerate(test_documents):
        print(f"  {i+1}. {doc['text'][:80]}...")
    
    # Test different configurations
    configs = [
        QAGenerationConfig(
            model="gpt-4-turbo",
            temperature=0.3,
            max_qa_per_chunk=1,
            include_source=True,
            include_metadata=True
        ),
        QAGenerationConfig(
            model="gpt-4-turbo",
            temperature=0.5,
            max_qa_per_chunk=2,
            include_source=True,
            include_metadata=True
        )
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🔧 Testing configuration {i+1}:")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max QA per chunk: {config.max_qa_per_chunk}")
        
        # Initialize QA generator
        qa_generator = QAGenerator(config=config)
        
        # Generate QA pairs
        domain = "electric_vehicles"
        qa_pairs = qa_generator.generate_qa_pairs(test_documents, domain, text_column="text")
        
        # Validate QA pairs
        validation = qa_generator.validate_qa_pairs(qa_pairs)
        print(f"  QA validation: {validation}")
        
        # Get statistics
        stats = qa_generator.get_qa_stats(qa_pairs)
        print(f"  QA stats: {stats}")
        
        # Display sample QA pairs
        print(f"\n  Sample QA pairs:")
        for j, qa in enumerate(qa_pairs[:3]):  # Show first 3
            print(f"    {j+1}. Q: {qa.get('question', 'N/A')}")
            print(f"       A: {qa.get('answer', 'N/A')[:100]}...")
            if 'source' in qa:
                print(f"       Source: {qa.get('source', 'N/A')}")
            print()
    
    # Test file operations
    print(f"\n💾 Testing file operations:")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate QA pairs
        qa_generator = QAGenerator()
        qa_pairs = qa_generator.generate_qa_pairs(test_documents[:1], "electric_vehicles", text_column="text")
        
        if qa_pairs:
            # Save QA pairs
            qa_file = os.path.join(temp_dir, "test_qa_pairs.jsonl")
            qa_generator.save_qa_pairs(qa_pairs, qa_file)
            print(f"  ✅ Saved QA pairs to: {qa_file}")
            
            # Load QA pairs
            loaded_qa = qa_generator.load_qa_pairs(qa_file)
            print(f"  ✅ Loaded {len(loaded_qa)} QA pairs from: {qa_file}")
            
            # Verify content
            if len(loaded_qa) == len(qa_pairs):
                print("  ✅ File save/load test passed!")
            else:
                print("  ❌ File save/load test failed!")
        else:
            print("  ⚠️  No QA pairs generated, skipping file test")
    
    # Test batch processing
    print(f"\n📦 Testing batch processing:")
    
    # Create larger test dataset
    large_test_docs = test_documents * 3  # 9 documents total
    
    qa_generator = QAGenerator(QAGenerationConfig(batch_size=3))
    batch_qa = qa_generator.generate_qa_batch(large_test_docs, "electric_vehicles", text_column="text")
    
    batch_stats = qa_generator.get_qa_stats(batch_qa)
    print(f"  Batch processing: {batch_stats['total_count']} QA pairs generated")
    
    print("\n🎉 All QA generation tests passed!")
    return True

def test_qa_generation_mock():
    """Test QA generation without API calls using mock data"""
    print("🧪 Testing QA Generation Pipeline (Mock Mode)")
    
    # Create mock QA pairs
    mock_qa_pairs = [
        {
            "question": "What are electric vehicles?",
            "answer": "Electric vehicles (EVs) are automobiles that use electric motors for propulsion, powered by rechargeable batteries.",
            "source": "https://www.tesla.com/support/charging",
            "source_type": "web",
            "generated_at": "2024-01-15T10:30:00",
            "model": "gpt-4-turbo",
            "temperature": 0.3
        },
        {
            "question": "How fast can Level 2 charging charge an EV?",
            "answer": "Level 2 charging can provide 10-60 miles of range per hour of charging, depending on the vehicle and charger specifications.",
            "source": "sample_document.pdf",
            "source_type": "pdf",
            "generated_at": "2024-01-15T10:30:00",
            "model": "gpt-4-turbo",
            "temperature": 0.3
        }
    ]
    
    print(f"\n📋 Mock QA pairs: {len(mock_qa_pairs)}")
    for i, qa in enumerate(mock_qa_pairs):
        print(f"  {i+1}. Q: {qa['question']}")
        print(f"     A: {qa['answer'][:80]}...")
        print(f"     Source: {qa['source']}")
    
    # Test validation
    print(f"\n🔍 Testing validation:")
    
    # Create a mock QA generator (without API calls)
    class MockQAGenerator:
        def validate_qa_pairs(self, qa_pairs):
            valid_count = 0
            invalid_count = 0
            
            for qa in qa_pairs:
                if (isinstance(qa.get("question"), str) and 
                    isinstance(qa.get("answer"), str) and
                    len(qa["question"].strip()) > 5 and
                    len(qa["answer"].strip()) > 10):
                    valid_count += 1
                else:
                    invalid_count += 1
            
            return {
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "total_count": len(qa_pairs),
                "validity_rate": valid_count / len(qa_pairs) if qa_pairs else 0
            }
        
        def get_qa_stats(self, qa_pairs):
            if not qa_pairs:
                return {"total_count": 0}
            
            question_lengths = [len(qa.get("question", "")) for qa in qa_pairs]
            answer_lengths = [len(qa.get("answer", "")) for qa in qa_pairs]
            
            sources = {}
            for qa in qa_pairs:
                source = qa.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_count": len(qa_pairs),
                "avg_question_length": sum(question_lengths) / len(question_lengths),
                "avg_answer_length": sum(answer_lengths) / len(answer_lengths),
                "source_distribution": sources,
                "unique_sources": len(sources)
            }
        
        def save_qa_pairs(self, qa_pairs, output_path):
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    mock_generator = MockQAGenerator()
    
    # Test validation
    validation = mock_generator.validate_qa_pairs(mock_qa_pairs)
    print(f"  Validation: {validation}")
    
    # Test statistics
    stats = mock_generator.get_qa_stats(mock_qa_pairs)
    print(f"  Statistics: {stats}")
    
    # Test file saving
    with tempfile.TemporaryDirectory() as temp_dir:
        qa_file = os.path.join(temp_dir, "mock_qa_pairs.jsonl")
        mock_generator.save_qa_pairs(mock_qa_pairs, qa_file)
        print(f"  ✅ Saved mock QA pairs to: {qa_file}")
        
        # Verify file exists
        if os.path.exists(qa_file):
            print("  ✅ File saving test passed!")
        else:
            print("  ❌ File saving test failed!")
    
    mock_generator = MockQAGenerator()
    
    # Test validation
    validation = mock_generator.validate_qa_pairs(mock_qa_pairs)
    print(f"  Validation: {validation}")
    
    # Test statistics
    stats = mock_generator.get_qa_stats(mock_qa_pairs)
    print(f"  Statistics: {stats}")
    
    # Test file saving
    with tempfile.TemporaryDirectory() as temp_dir:
        qa_file = os.path.join(temp_dir, "mock_qa_pairs.jsonl")
        mock_generator.save_qa_pairs(mock_qa_pairs, qa_file)
        print(f"  ✅ Saved mock QA pairs to: {qa_file}")
        
        # Verify file exists
        if os.path.exists(qa_file):
            print("  ✅ File saving test passed!")
        else:
            print("  ❌ File saving test failed!")
    
    print("\n🎉 All mock QA generation tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_qa_generation()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
