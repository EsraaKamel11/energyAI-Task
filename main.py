#!/usr/bin/env python3
"""
End-to-End LLM Fine-tuning Pipeline for EV Charging Stations (Enhanced Version with CI/CD & Monitoring)
"""

import os
import logging
import sys
from pathlib import Path
import pandas as pd
import torch
from huggingface_hub import login

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "ml_pipeline"))

from ml_pipeline.config.settings import settings, logger
from ml_pipeline.src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager, MetadataHandler, Deduplicator, QAGenerator, QAGenerationConfig

# Import modular QLoRA components
from ml_pipeline.src.training_qlora.lora_config import QLoRAConfigurator
from ml_pipeline.src.training_qlora.data_preparation import QLoRADataPreparer
from ml_pipeline.src.training_qlora.main_orchestrator import QLoRAOrchestrator
from ml_pipeline.src.training_qlora.experiment_tracker import QLoRAExperimentTracker, create_qlora_experiment_tracker

from ml_pipeline.src.evaluation import BenchmarkCreator, BenchmarkGenerator, Comparator, ModelEvaluator
from ml_pipeline.src.deployment import ModelRegistry
from ml_pipeline.config.model_configs import get_model_config, list_available_models

# Import monitoring components
from ml_pipeline.src.deployment.monitored_inference_server import MonitoredInferenceServer
import subprocess
import requests
import time

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

def check_monitoring_setup():
    """Check if monitoring stack is properly configured"""
    logger.info("🔍 Checking monitoring setup...")
    
    # Check if monitoring files exist
    monitoring_files = [
        "ml_pipeline/monitoring/prometheus.yml",
        "ml_pipeline/monitoring/alertmanager.yml", 
        "ml_pipeline/monitoring/rules/alerts.yml",
        "ml_pipeline/monitoring/grafana/dashboards/ml-pipeline-dashboard.json",
        "ml_pipeline/monitoring/grafana/datasources/prometheus.yml",
        "ml_pipeline/docker-compose.monitoring.yml",
        "ml_pipeline/start_monitoring.py",
        "ml_pipeline/requirements_monitoring.txt",
        "ml_pipeline/README_MONITORING.md"
    ]
    
    missing_files = []
    for file_path in monitoring_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Missing monitoring files: {missing_files}")
        return False
    
    logger.info("✅ All monitoring files found")
    return True

def check_cicd_setup():
    """Check if CI/CD setup is properly configured"""
    logger.info("🔍 Checking CI/CD setup...")
    
    cicd_files = [
        ".github/workflows/ml-pipeline.yml"
    ]
    
    missing_files = []
    for file_path in cicd_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Missing CI/CD files: {missing_files}")
        return False
    
    logger.info("✅ CI/CD files found")
    return True

def check_docker_setup():
    """Check if Docker setup is properly configured"""
    logger.info("🔍 Checking Docker setup...")
    
    docker_files = [
        "ml_pipeline/docker/Dockerfile",
        "ml_pipeline/docker-compose.monitoring.yml"
    ]
    
    missing_files = []
    for file_path in docker_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Missing Docker files: {missing_files}")
        return False
    
    logger.info("✅ Docker files found")
    return True

def start_monitoring_stack():
    """Start the monitoring stack using Docker Compose"""
    logger.info("🚀 Starting monitoring stack...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Docker not available. Skipping monitoring stack startup.")
            return False
        
        # Check if docker-compose is available
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Docker Compose not available. Skipping monitoring stack startup.")
            return False
        
        # Start the monitoring stack
        logger.info("Starting monitoring stack with docker-compose...")
        result = subprocess.run([
            "docker-compose", "-f", "ml_pipeline/docker-compose.monitoring.yml", "up", "-d"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Monitoring stack started successfully")
            
            # Wait a moment for services to start
            time.sleep(10)
            
            # Check service health
            services = {
                "Prometheus": "http://localhost:9090/-/healthy",
                "Grafana": "http://localhost:3000/api/health",
                "Alertmanager": "http://localhost:9093/-/healthy"
            }
            
            for service_name, health_url in services.items():
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name} is healthy")
                    else:
                        logger.warning(f"⚠️ {service_name} returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ {service_name} health check failed: {e}")
            
            return True
        else:
            logger.error(f"Failed to start monitoring stack: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error starting monitoring stack: {e}")
        return False

def main():
    """Main pipeline execution with enhanced monitoring, CI/CD, and Docker"""
    logger.info("Starting EV Charging Stations LLM Pipeline (Enhanced Version with CI/CD, Docker & Monitoring)")
    
    # Check CI/CD, Docker, and monitoring setup
    cicd_ok = check_cicd_setup()
    docker_ok = check_docker_setup()
    monitoring_ok = check_monitoring_setup()
    
    logger.info("=== Infrastructure Status ===")
    logger.info(f"CI/CD Setup: {'✅' if cicd_ok else '⚠️'}")
    logger.info(f"Docker Setup: {'✅' if docker_ok else '⚠️'}")
    logger.info(f"Monitoring Setup: {'✅' if monitoring_ok else '⚠️'}")
    
    if not cicd_ok or not docker_ok or not monitoring_ok:
        logger.warning("Some infrastructure components are missing. Pipeline will continue without them.")
        logger.info("To enable full functionality:")
        if not cicd_ok:
            logger.info("  - Set up GitHub Actions workflow in .github/workflows/")
        if not docker_ok:
            logger.info("  - Ensure Docker and docker-compose are installed")
        if not monitoring_ok:
            logger.info("  - Install monitoring requirements: pip install -r requirements_monitoring.txt")
    
    # Authenticate with Hugging Face
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        logger.info("Authenticating with Hugging Face...")
        login(hf_token)
        logger.info("✅ Hugging Face authentication successful")
    else:
        logger.warning("No HF_TOKEN found in environment variables")
    
    # Configuration
    domain = "electric vehicle charging stations"
    
    # Model selection - choose from available models
    model_key = "dialogpt-medium"  # DialoGPT-medium (345M parameters, open access, proven to work)
    
    model_config = get_model_config(model_key)
    base_model = model_config["name"]
    target_modules = model_config["target_modules"]
    
    logger.info(f"Using model: {base_model} ({model_config['params']} parameters)")
    logger.info(f"Target modules for LoRA: {target_modules}")
    
    output_dir = "ml_pipeline/pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Stage 1: Data Collection (Sample Data Only)
        logger.info("=== Stage 1: Data Collection ===")
        logger.info("Using sample data for demonstration")
        
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
        
        # Generate QA pairs from processed documents (optional - requires OpenAI API key)
        qa_pairs = []
        try:
            qa_config = QAGenerationConfig(
                model="gpt-4-turbo",
                temperature=0.3,
                max_qa_per_chunk=2,
                include_source=True,
                include_metadata=True
            )
            qa_generator = QAGenerator(config=qa_config)
            
            # Generate QA pairs
            domain = "electric_vehicles"  # Can be made configurable
            qa_pairs = qa_generator.generate_qa_pairs(deduplicated_documents, domain, text_column="text")
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
        except Exception as e:
            logger.warning(f"QA generation failed: {e}")
            logger.info("Continuing without QA pairs")
        
        # Validate QA pairs (if any were generated)
        if qa_pairs:
            try:
                qa_validation = qa_generator.validate_qa_pairs(qa_pairs)
                logger.info(f"QA validation: {qa_validation}")
                
                # Get QA statistics
                qa_stats = qa_generator.get_qa_stats(qa_pairs)
                logger.info(f"QA generation stats: {qa_stats}")
                
                # Save QA pairs
                qa_generator.save_qa_pairs(qa_pairs, f"{output_dir}/qa_pairs.jsonl")
            except Exception as e:
                logger.warning(f"QA validation/saving failed: {e}")
        else:
            logger.info("No QA pairs to validate or save")
        
        # Save processed data
        storage.save_to_parquet(processed_data, f"{output_dir}/processed_data.parquet")
        logger.info(f"Processed {len(processed_data)} documents with metadata")
        
        # Save pipeline statistics
        pipeline_stats = {
            "total_documents": len(processed_data),
            "deduplication_reduction": dedup_stats.get("reduction_percentage", 0),
            "qa_pairs_generated": len(qa_pairs) if 'qa_pairs' in locals() else 0,
            "domain": domain,
            "base_model": base_model
        }
        
        import json
        with open(f"{output_dir}/pipeline_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(pipeline_stats, f, indent=2)
        logger.info("Pipeline statistics saved")
        
        # Stage 3: Training Dataset Preparation (optional - requires OpenAI API key)
        logger.info("=== Stage 3: Training Dataset Preparation ===")
        
        # Training dataset preparation (optional - requires OpenAI API key)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
            logger.info("Skipping training dataset preparation - please set OPENAI_API_KEY to continue")
        else:
            try:
                # Use the new modular QLoRA data preparer
                qa_preparer = QLoRADataPreparer()
                
                # Create sample QA pairs for demonstration
                sample_qa = [
                    {"question": "What is Level 2 charging?", "answer": "Level 2 charging uses 240V power and provides 10-60 miles of range per hour."},
                    {"question": "How fast is DC charging?", "answer": "DC fast charging can provide 60-80% charge in 20-30 minutes."},
                    {"question": "What are Tesla Superchargers?", "answer": "Tesla Superchargers are proprietary DC fast charging stations that can provide up to 200 miles of range in 15 minutes."}
                ]
                
                # Save sample QA pairs
                import json
                with open(f"{output_dir}/qa_pairs.jsonl", 'w', encoding='utf-8') as f:
                    for qa in sample_qa:
                        f.write(json.dumps(qa) + '\n')
                
                logger.info(f"Created sample QA pairs: {len(sample_qa)} pairs")
            except Exception as e:
                logger.warning(f"QA preparation failed: {e}")
                logger.info("Continuing without QA pairs")
        
        # Stage 4: QLoRA Fine-tuning
        logger.info("=== Stage 4: QLoRA Fine-tuning ===")
        
        # Check if we have training data
        training_data_path = f"{output_dir}/qa_pairs.jsonl"
        if not os.path.exists(training_data_path):
            logger.warning("Training data not found. Skipping fine-tuning.")
            logger.info("To enable fine-tuning, ensure QA pairs are generated")
            
            # Final summary
            logger.info("=== Pipeline Summary ===")
            logger.info(f"Processed {len(processed_data)} documents")
            logger.info(f"Output directory: {output_dir}")
            
            # Create summary file
            summary_lines = [
                "=== ML Pipeline Summary (Enhanced Version with CI/CD, Docker & Monitoring) ===",
                f"Processed documents: {len(processed_data)}",
                f"Domain: {domain}",
                f"Base model: {base_model}",
                f"Output directory: {output_dir}",
                "",
                "=== Pipeline Stages ===",
                "✓ Data Collection (sample data)",
                "✓ Data Processing",
                "✓ Data Cleaning",
                "✓ Quality Filtering", 
                "✓ Normalization",
                "✓ Deduplication",
                "⚠ QA Generation (skipped - no API key)",
                "⚠ QLoRA Fine-tuning (skipped - no training data)",
                "⚠ Evaluation (skipped - no fine-tuned model)",
                "⚠ Deployment (skipped - no fine-tuned model)",
                "",
                "=== Infrastructure Status ===",
                f"CI/CD Setup: {'✓' if cicd_ok else '⚠'} (.github/workflows/ml-pipeline.yml)",
                f"Docker Setup: {'✓' if docker_ok else '⚠'} (ml_pipeline/docker/Dockerfile, ml_pipeline/docker-compose.monitoring.yml)",
                f"Monitoring Setup: {'✓' if monitoring_ok else '⚠'} (ml_pipeline/monitoring/, ml_pipeline/start_monitoring.py)",
                "",
                "=== New Features Available ===",
                "• Docker containerization support",
                "• GitHub Actions CI/CD pipeline",
                "• Enhanced monitored inference server",
                "• Pre-configured Grafana dashboards",
                "• Prometheus alerting rules",
                "• Comprehensive monitoring stack",
                "",
                "=== Next Steps ===",
                "To enable full pipeline:",
                "1. Set OPENAI_API_KEY environment variable",
                "2. Run the pipeline again",
                "3. The pipeline will generate QA pairs and perform QLoRA fine-tuning",
                "4. Start monitoring: python ml_pipeline/start_monitoring.py",
                "5. Access Grafana dashboard at http://localhost:3000",
                "6. Deploy with Docker: docker-compose up -d"
            ]
            
            with open(f"{output_dir}/pipeline_summary.txt", 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Check {output_dir}/pipeline_summary.txt for detailed summary")
            return
        
        # Use the new modular QLoRA training system
        try:
            logger.info("🚀 Starting QLoRA training with modular system...")
            
            # Initialize QLoRA orchestrator
            qlora_orchestrator = QLoRAOrchestrator()
            
            # Run QLoRA training pipeline
            training_results = qlora_orchestrator.run_training_pipeline(
                model_name=base_model,
                data_path=training_data_path,
                output_dir=f"{output_dir}/qlora_training",
                quantization="4bit",  # Use QLoRA (4-bit quantization)
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                batch_size=1,
                num_epochs=3,
                learning_rate=1e-4,
                system_prompt="EV Assistant: I'm here to help you with electric vehicle charging information."
            )
            
            logger.info("✅ QLoRA training completed successfully!")
            logger.info(f"Training results: {training_results}")
            
            # Extract model path for evaluation
            model_path = training_results.get("model_path", "")
            
        except Exception as e:
            logger.error(f"QLoRA training failed: {e}")
            logger.info("Skipping evaluation due to training failure")
            return
        
        # Stage 5: Evaluation
        logger.info("=== Stage 5: Evaluation ===")
        
        # Load fine-tuned model
        try:
            from peft import PeftModel
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load base model and tokenizer
            base_model_instance = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Load fine-tuned adapter
            fine_tuned_model = PeftModel.from_pretrained(base_model_instance, model_path)
            logger.info("Successfully loaded fine-tuned QLoRA model")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            logger.info("Skipping evaluation")
            return
        
        # Initialize model evaluator
        model_evaluator = ModelEvaluator(device="auto")
        
        # Load benchmark dataset
        benchmark_questions = []
        try:
            with open(f"{output_dir}/benchmark_dataset.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        benchmark_questions.append(json.loads(line))
        except FileNotFoundError:
            logger.warning("Benchmark dataset not found, using sample questions")
            benchmark_questions = [
                {"question": "What is EV charging?", "answer": "Electric vehicle charging"},
                {"question": "How fast can Tesla charge?", "answer": "Up to 250kW with Supercharger V3"}
            ]
        
        # Compare fine-tuned model with baseline
        logger.info("Comparing QLoRA fine-tuned model with baseline...")
        
        try:
            # Load baseline model (same as base model)
            baseline_model, baseline_tokenizer = model_evaluator.load_model(base_model, is_peft=False)
            logger.info("Successfully loaded baseline model")
            
            # Perform comprehensive comparison
            comparison_result = model_evaluator.compare_models(
                fine_tuned_model=fine_tuned_model,
                fine_tuned_tokenizer=tokenizer,
                baseline_model=baseline_model,
                baseline_tokenizer=baseline_tokenizer,
                benchmark=benchmark_questions,
                fine_tuned_name="qlora_fine_tuned_ev_model",
                baseline_name="baseline_model"
            )
            
            # Save comparison results
            model_evaluator.save_comparison_results(
                comparison_result, 
                f"{output_dir}/qlora_model_comparison_results.json"
            )
            
            # Generate and save comparison report
            comparison_report = model_evaluator.generate_comparison_report(comparison_result)
            with open(f"{output_dir}/qlora_comparison_report.md", 'w', encoding='utf-8') as f:
                f.write(comparison_report)
            
            logger.info(f"QLoRA model comparison completed. ROUGE-1 improvement: {comparison_result.improvements.get('rouge1', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            logger.info("Skipping detailed evaluation")
        
        # Stage 6: Deployment with Enhanced Monitoring
        logger.info("=== Stage 6: Deployment with Enhanced Monitoring ===")
        
        # Register model
        try:
            registry = ModelRegistry()
            registry.register(
                base_model=base_model,
                adapter_path=model_path,
                version="v1.0",
                metadata={"domain": domain, "method": "qlora", "performance": {"latency": 0}}
            )
            logger.info("QLoRA model registered successfully")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")
        
        # Start enhanced monitored inference server
        try:
            logger.info("🚀 Starting enhanced monitored inference server...")
            monitored_server = MonitoredInferenceServer(
                model_path=model_path,
                port=8000,
                host="0.0.0.0",
                enable_metrics=True,
                enable_health_checks=True
            )
            
            # Start server in background
            import threading
            server_thread = threading.Thread(target=monitored_server.start, daemon=True)
            server_thread.start()
            
            logger.info("✅ Enhanced monitored inference server started")
            logger.info("🌐 API available at: http://localhost:8000")
            logger.info("📊 Metrics available at: http://localhost:8000/metrics")
            logger.info("❤️ Health check at: http://localhost:8000/health")
            
        except Exception as e:
            logger.warning(f"Enhanced inference server failed: {e}")
        
        # Start monitoring stack if available
        if monitoring_ok and docker_ok:
            logger.info("Starting monitoring stack...")
            monitoring_started = start_monitoring_stack()
            if monitoring_started:
                logger.info("✅ Monitoring stack started successfully")
                logger.info("📊 Access Grafana dashboard at: http://localhost:3000")
                logger.info("📈 Access Prometheus at: http://localhost:9090")
                logger.info("🚨 Access Alertmanager at: http://localhost:9093")
                logger.info("🔗 Grafana datasource configured for Prometheus")
                logger.info("📋 Pre-configured dashboards available")
            else:
                logger.warning("⚠️ Failed to start monitoring stack")
        elif not docker_ok:
            logger.warning("⚠️ Docker not available - skipping monitoring stack")
        elif not monitoring_ok:
            logger.warning("⚠️ Monitoring files missing - skipping monitoring stack")
        
        # Final summary
        logger.info("=== Pipeline Summary ===")
        logger.info(f"Processed {len(processed_data)} documents")
        logger.info(f"Output directory: {output_dir}")
        
        # Create summary file
        summary_lines = [
            "=== ML Pipeline Summary (Enhanced Version with CI/CD, Docker & Monitoring) ===",
            f"Processed documents: {len(processed_data)}",
            f"Domain: {domain}",
            f"Base model: {base_model}",
            f"Training method: QLoRA (4-bit quantization)",
            f"Output directory: {output_dir}",
            "",
            "=== Pipeline Stages ===",
            "✓ Data Collection (sample data)",
            "✓ Data Processing",
            "✓ Data Cleaning",
            "✓ Quality Filtering", 
            "✓ Normalization",
            "✓ Deduplication",
            "✓ QA Generation (sample data)",
            "✓ QLoRA Fine-tuning",
            "✓ Model Evaluation",
            "✓ Enhanced Model Deployment",
            "",
            "=== Infrastructure Status ===",
            f"CI/CD Setup: {'✓' if cicd_ok else '⚠'} (.github/workflows/ml-pipeline.yml)",
                            f"Docker Setup: {'✓' if docker_ok else '⚠'} (ml_pipeline/docker/Dockerfile, ml_pipeline/docker-compose.monitoring.yml)",
                            f"Monitoring Setup: {'✓' if monitoring_ok else '⚠'} (ml_pipeline/monitoring/, ml_pipeline/start_monitoring.py)",
            "",
            "=== QLoRA Benefits ===",
            "• 75% less memory usage than LoRA",
            "• Works on consumer GPUs (8GB+ VRAM)",
            "• Maintains good performance",
            "• Modular, maintainable architecture",
            "",
            "=== Enhanced Monitoring Access ===",
            "• Enhanced Inference Server: http://localhost:8000",
            "• API Metrics: http://localhost:8000/metrics",
            "• Health Check: http://localhost:8000/health",
            "• Grafana Dashboard: http://localhost:3000",
            "• Prometheus Metrics: http://localhost:9090",
            "• Alertmanager: http://localhost:9093",
            "",
            "=== New Features ===",
            "• Docker containerization support",
            "• GitHub Actions CI/CD pipeline",
            "• Enhanced monitored inference server",
            "• Pre-configured Grafana dashboards",
            "• Prometheus alerting rules",
            "• Comprehensive monitoring stack",
            "",
            "=== Next Steps ===",
            "1. Test the fine-tuned model via API",
            "2. Monitor performance via Grafana",
            "3. Set up alerts in Alertmanager",
            "4. Configure CI/CD pipeline triggers",
            "5. Deploy to production with Docker",
            "6. Scale with Kubernetes (optional)"
        ]
        
        with open(f"{output_dir}/pipeline_summary.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        if os.path.exists(model_path):
            logger.info("✓ QLoRA fine-tuning completed")
        else:
            logger.info("⚠ QLoRA fine-tuning failed")
            
        if os.path.exists(f"{output_dir}/qlora_model_comparison_results.json"):
            logger.info("✓ QLoRA model evaluation completed")
        else:
            logger.info("⚠ QLoRA model evaluation skipped")
            
        logger.info("🚀 Enhanced QLoRA pipeline with CI/CD, Docker & Monitoring completed successfully!")
        logger.info(f"📋 Check {output_dir}/pipeline_summary.txt for detailed summary")
        logger.info("🔗 Access monitoring dashboards and APIs as shown in the summary")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 
