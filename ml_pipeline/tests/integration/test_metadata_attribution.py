#!/usr/bin/env python3
"""
Test script for metadata attribution pipeline
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.metadata_handler import MetadataHandler
import pandas as pd

def test_metadata_attribution():
    """Test the metadata attribution pipeline"""
    print("🧪 Testing Metadata Attribution Pipeline")
    
    # Create test data
    test_documents = [
        {
            "text": "EV charging stations are essential for electric vehicles.",
            "source": "https://www.tesla.com/support/charging",
            "type": "web",
            "metadata": {
                "status_code": 200,
                "timestamp": "2024-01-01T12:00:00"
            }
        },
        {
            "text": "Level 2 charging uses 240V power and can charge faster than Level 1.",
            "source": "sample_document.pdf",
            "type": "pdf",
            "metadata": {
                "page_number": 1,
                "extractor": "pdfplumber"
            }
        },
        {
            "text": "DC fast charging can provide 60-80% charge in 20-30 minutes.",
            "source": "https://www.electrifyamerica.com/",
            "type": "web",
            "metadata": {
                "status_code": 200,
                "timestamp": "2024-01-01T13:00:00"
            }
        }
    ]
    
    # Test QA pairs
    test_qa_pairs = [
        {
            "context": "EV charging stations are essential for electric vehicles.",
            "question": "What are EV charging stations?",
            "answer": "EV charging stations are essential for electric vehicles."
        },
        {
            "context": "Level 2 charging uses 240V power and can charge faster than Level 1.",
            "question": "What is Level 2 charging?",
            "answer": "Level 2 charging uses 240V power and can charge faster than Level 1."
        },
        {
            "context": "DC fast charging can provide 60-80% charge in 20-30 minutes.",
            "question": "How fast is DC charging?",
            "answer": "DC fast charging can provide 60-80% charge in 20-30 minutes."
        }
    ]
    
    # Initialize metadata handler
    metadata_handler = MetadataHandler()
    
    print("\n📋 Step 1: Adding metadata to documents")
    documents_with_metadata = metadata_handler.add_metadata(test_documents)
    
    print("Documents with metadata:")
    for i, doc in enumerate(documents_with_metadata):
        print(f"  {i+1}. Source: {doc['metadata']['source_type']} - {doc['metadata'].get('url', doc['metadata'].get('file_name', 'Unknown'))}")
    
    print("\n📊 Step 2: Validating metadata")
    validation_stats = metadata_handler.validate_metadata(documents_with_metadata)
    print(f"Validation stats: {validation_stats}")
    
    print("\n🔗 Step 3: Propagating metadata to QA pairs")
    qa_pairs_with_metadata = metadata_handler.propagate_metadata_to_qa(test_qa_pairs, documents_with_metadata)
    
    print("\n📝 Step 4: Adding attribution")
    qa_pairs_with_attribution = metadata_handler.add_attribution_to_qa(qa_pairs_with_metadata)
    
    print("\n✅ Final QA pairs with attribution:")
    for i, qa in enumerate(qa_pairs_with_attribution):
        print(f"  {i+1}. Q: {qa['question']}")
        print(f"     A: {qa['answer']}")
        print(f"     Attribution: {qa['attribution']}")
        print(f"     Source tracking: {qa['source_tracking']}")
        print()
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n💾 Step 5: Testing file saving in {temp_dir}")
        
        # Save test data
        test_df = pd.DataFrame(documents_with_metadata)
        test_df.to_parquet(os.path.join(temp_dir, "test_documents.parquet"))
        
        # Save QA pairs
        import json
        qa_file = os.path.join(temp_dir, "qa_pairs_with_metadata.jsonl")
        with open(qa_file, 'w', encoding='utf-8') as f:
            for qa in qa_pairs_with_attribution:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved test documents to: {temp_dir}/test_documents.parquet")
        print(f"✅ Saved QA pairs to: {temp_dir}/qa_pairs_with_metadata.jsonl")
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_dir, "test_documents.parquet"))
        assert os.path.exists(qa_file)
        print("✅ File saving test passed!")
    
    print("\n🎉 All metadata attribution tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_metadata_attribution()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
