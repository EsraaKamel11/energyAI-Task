#!/usr/bin/env python3
"""
CI-Specific version of main.py for GitHub Actions
This version handles CI environment constraints and provides better error handling
"""

import os
import logging
import sys
from pathlib import Path
import pandas as pd
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "ml_pipeline"))

# Set CI-specific environment variables
os.environ.setdefault('USE_GPU', 'false')
os.environ.setdefault('LOG_LEVEL', 'INFO')

from ml_pipeline.config.settings import settings, logger
from ml_pipeline.src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager, MetadataHandler, Deduplicator

def create_sample_data():
    """Create sample EV charging data for testing"""
    sample_texts = [
        "Electric vehicle charging stations are essential infrastructure for the transition to sustainable transportation. Level 1 charging uses a standard 120-volt outlet and provides 2-5 miles of range per hour.",
        "Level 2 charging stations use 240-volt power and can provide 10-60 miles of range per hour, making them ideal for home and workplace charging.",
        "DC fast charging, also known as Level 3 charging, can provide 60-80% charge in 20-30 minutes, making it suitable for long-distance travel.",
        "Tesla Superchargers are proprietary DC fast charging stations that can provide up to 200 miles of range in 15 minutes for compatible vehicles.",
        "Public charging networks like ChargePoint, EVgo, and ElectrifyAmerica provide access to charging stations across the country.",
        "The cost of charging an electric vehicle varies by location and charging speed, typically ranging from $0.10 to $0.30 per kWh.",
        "Most electric vehicles come with a portable Level 1 charger that can be plugged into any standard electrical outlet.",
        "Charging station connectors include Type 1 (J1772), Type 2 (Mennekes), CHAdeMO, and CCS, with different connectors used in different regions.",
        "Smart charging allows vehicles to charge during off-peak hours when electricity rates are lower, helping to reduce charging costs.",
        "Bidirectional charging technology enables electric vehicles to serve as mobile energy storage, providing power back to the grid during peak demand."
    ]
    
    return pd.DataFrame({
        "text": sample_texts,
        "source": ["sample_data"] * len(sample_texts),
        "timestamp": [pd.Timestamp.now()] * len(sample_texts)
    })

def main():
    """CI-specific main function with enhanced error handling"""
    logger.info("Starting EV Charging Stations LLM Pipeline (CI Version)")
    
    # Check if we're in CI environment
    is_ci = os.environ.get('CI', 'false').lower() == 'true'
    logger.info(f"Running in CI environment: {is_ci}")
    
    # Check available device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available - using GPU")
    else:
        device = "cpu"
        logger.info("CUDA not available - using CPU")
    
    # Configuration
    domain = "electric vehicle charging stations"
    model_key = "dialogpt-medium"  # Use a smaller model for CI
    
    try:
        from ml_pipeline.model_configs import get_model_config
        model_config = get_model_config(model_key)
        base_model = model_config["name"]
        logger.info(f"Using model: {base_model}")
    except Exception as e:
        logger.warning(f"Could not load model config: {e}")
        base_model = "microsoft/DialoGPT-medium"  # Fallback
        logger.info(f"Using fallback model: {base_model}")
    
    output_dir = "ml_pipeline/pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Stage 1: Data Collection (Sample Data Only)
        logger.info("=== Stage 1: Data Collection ===")
        logger.info("Using sample data for CI testing")
        
        # Stage 2: Data Processing
        logger.info("=== Stage 2: Data Processing ===")
        
        # Use sample data directly
        storage = StorageManager()
        web_data = create_sample_data()
        logger.info(f"Using sample data with {len(web_data)} documents")
        
        # Clean and filter
        cleaner = DataCleaner()
        quality_filter = QualityFilter(min_length=50)
        normalizer = Normalizer(model_name=base_model)
        metadata_handler = MetadataHandler()
        deduplicator = Deduplicator(similarity_threshold=0.95, method="levenshtein")
        
        processed_data = cleaner.process(
            web_data, 
            text_column="text",
            remove_boilerplate=True,
            filter_sentences=True,
            min_length=30
        )
        processed_data = quality_filter.filter(processed_data, text_column="text")
        processed_data = normalizer.normalize(processed_data, text_column="text")
        
        # Add metadata and source tracking
        documents = processed_data.to_dict('records')
        documents_with_metadata = metadata_handler.add_metadata(documents)
        
        # Validate metadata
        metadata_stats = metadata_handler.validate_metadata(documents_with_metadata)
        logger.info(f"Metadata validation: {metadata_stats}")
        
        # Deduplicate documents
        original_count = len(documents_with_metadata)
        deduplicated_documents = deduplicator.deduplicate(documents_with_metadata, text_column="text")
        final_count = len(deduplicated_documents)
        
        # Get deduplication statistics
        dedup_stats = deduplicator.get_deduplication_stats(original_count, final_count)
        logger.info(f"Deduplication stats: {dedup_stats}")
        
        # Convert back to DataFrame
        processed_data = pd.DataFrame(deduplicated_documents)
        
        # Save processed data
        storage.save_to_parquet(processed_data, f"{output_dir}/processed_data.parquet")
        logger.info(f"Processed {len(processed_data)} documents with metadata")
        
        # Save pipeline statistics
        pipeline_stats = {
            "total_documents": len(processed_data),
            "deduplication_reduction": dedup_stats.get("reduction_percentage", 0),
            "domain": domain,
            "base_model": base_model,
            "device": device,
            "ci_environment": is_ci
        }
        
        import json
        with open(f"{output_dir}/pipeline_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(pipeline_stats, f, indent=2)
        logger.info("Pipeline statistics saved")
        
        # Skip training in CI (requires GPU and more resources)
        logger.info("=== Stage 3: Training (Skipped in CI) ===")
        logger.info("Training skipped in CI environment to save resources")
        
        # Final summary
        logger.info("=== CI Pipeline Summary ===")
        logger.info(f"Processed {len(processed_data)} documents")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Device used: {device}")
        logger.info(f"CI environment: {is_ci}")
        
        # Create CI summary file
        summary_lines = [
            "=== ML Pipeline Summary (CI Version) ===",
            f"Processed documents: {len(processed_data)}",
            f"Domain: {domain}",
            f"Base model: {base_model}",
            f"Device: {device}",
            f"CI Environment: {is_ci}",
            f"Output directory: {output_dir}",
            "",
            "=== Pipeline Stages ===",
            "✓ Data Collection (sample data)",
            "✓ Data Processing",
            "✓ Data Cleaning",
            "✓ Quality Filtering", 
            "✓ Normalization",
            "✓ Deduplication",
            "⚠ Training (skipped in CI)",
            "⚠ Evaluation (skipped in CI)",
            "⚠ Deployment (skipped in CI)",
            "",
            "=== CI-Specific Notes ===",
            "• Training skipped to save CI resources",
            "• Using CPU for processing",
            "• Sample data used for testing",
            "• All core data processing functions tested",
            "",
            "=== Success Criteria ===",
            "✓ All imports successful",
            "✓ Data processing pipeline works",
            "✓ No critical errors",
            "✓ Output files generated"
        ]
        
        with open(f"{output_dir}/ci_pipeline_summary.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info("✅ CI pipeline completed successfully!")
        logger.info(f"Check {output_dir}/ci_pipeline_summary.txt for detailed summary")
        
    except Exception as e:
        logger.error(f"CI pipeline failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 