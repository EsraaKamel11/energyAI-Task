#!/usr/bin/env python3
"""
Start the complete monitoring stack for the ML pipeline
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_monitoring_stack():
    """Start the monitoring stack using Docker Compose"""
    try:
        # Change to the ml_pipeline directory
        os.chdir(Path(__file__).parent)
        
        # Check if model exists
        model_path = Path("pipeline_output/qlora_training/final_model")
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.info("Please run the pipeline first to generate the model")
            return False
        
        # Start the monitoring stack
        logger.info("Starting monitoring stack...")
        subprocess.run([
            "docker-compose", "-f", "docker-compose.monitoring.yml", "up", "-d"
        ], check=True)
        
        logger.info("✅ Monitoring stack started successfully!")
        logger.info("📊 Prometheus: http://localhost:9090")
        logger.info("📈 Grafana: http://localhost:3000 (admin/admin123)")
        logger.info("🚨 Alertmanager: http://localhost:9093")
        logger.info("🤖 ML Inference Server: http://localhost:8000")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start monitoring stack: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def stop_monitoring_stack():
    """Stop the monitoring stack"""
    try:
        os.chdir(Path(__file__).parent)
        logger.info("Stopping monitoring stack...")
        subprocess.run([
            "docker-compose", "-f", "docker-compose.monitoring.yml", "down"
        ], check=True)
        logger.info("✅ Monitoring stack stopped successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop monitoring stack: {e}")
        return False

def check_services():
    """Check if all services are running"""
    services = {
        "Prometheus": "http://localhost:9090",
        "Grafana": "http://localhost:3000",
        "Alertmanager": "http://localhost:9093",
        "ML Inference Server": "http://localhost:8000/health"
    }
    
    import requests
    
    logger.info("Checking service status...")
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {service}: Running")
            else:
                logger.warning(f"⚠️ {service}: Unexpected status {response.status_code}")
        except requests.exceptions.RequestException:
            logger.error(f"❌ {service}: Not responding")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python start_monitoring.py [start|stop|status]")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        if not check_docker():
            logger.error("Docker is not installed or not available")
            return
        
        if not check_docker_compose():
            logger.error("Docker Compose is not installed or not available")
            return
        
        if start_monitoring_stack():
            logger.info("Waiting for services to start...")
            time.sleep(30)
            check_services()
    
    elif command == "stop":
        stop_monitoring_stack()
    
    elif command == "status":
        check_services()
    
    else:
        logger.error(f"Unknown command: {command}")
        print("Available commands: start, stop, status")

if __name__ == "__main__":
    main() 