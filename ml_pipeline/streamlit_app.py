#!/usr/bin/env python3
"""
Streamlit App for EV Charging Stations LLM Pipeline
Provides a user-friendly interface for the complete ML pipeline
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import requests
import time
from pathlib import Path
import subprocess
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import pipeline components
from config.settings import settings, logger
from src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager, MetadataHandler, Deduplicator
from src.training_qlora import QLoRAOrchestrator, QLoRAConfigurator, QLoRADataPreparer
from src.evaluation import ModelEvaluator
from src.deployment import ModelRegistry
from config.model_configs import get_model_config, list_available_models

# Page configuration
st.set_page_config(
    page_title="EV Charging LLM Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def check_infrastructure():
    """Check infrastructure components"""
    status = {}
    
    # Check CI/CD
    cicd_file = Path("../.github/workflows/ml-pipeline.yml")
    status["CI/CD"] = "✅ Available" if cicd_file.exists() else "⚠️ Missing"
    
    # Check Docker
    docker_files = [
        Path("docker/Dockerfile"),
        Path("docker-compose.monitoring.yml")
    ]
    status["Docker"] = "✅ Available" if all(f.exists() for f in docker_files) else "⚠️ Missing"
    
    # Check Monitoring
    monitoring_files = [
        Path("monitoring/prometheus.yml"),
        Path("monitoring/alertmanager.yml"),
        Path("start_monitoring.py")
    ]
    status["Monitoring"] = "✅ Available" if all(f.exists() for f in monitoring_files) else "⚠️ Missing"
    
    return status

def check_services():
    """Check if monitoring services are running"""
    services = {}
    
    # Check Prometheus
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=2)
        services["Prometheus"] = "🟢 Running" if response.status_code == 200 else "🔴 Error"
    except:
        services["Prometheus"] = "⚪ Stopped"
    
    # Check Grafana
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=2)
        services["Grafana"] = "🟢 Running" if response.status_code == 200 else "🔴 Error"
    except:
        services["Grafana"] = "⚪ Stopped"
    
    # Check Inference Server
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        services["Inference Server"] = "🟢 Running" if response.status_code == 200 else "🔴 Error"
    except:
        services["Inference Server"] = "⚪ Stopped"
    
    return services

def create_sample_data():
    """Create sample EV charging data"""
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
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ EV Charging LLM Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("### Complete ML Pipeline with CI/CD, Docker & Monitoring")
    
    # Sidebar
    st.sidebar.title("🎛️ Pipeline Controls")
    
    # Infrastructure status
    st.sidebar.subheader("🏗️ Infrastructure Status")
    infrastructure_status = check_infrastructure()
    for component, status in infrastructure_status.items():
        st.sidebar.text(f"{component}: {status}")
    
    # Service status
    st.sidebar.subheader("🔧 Service Status")
    service_status = check_services()
    for service, status in service_status.items():
        st.sidebar.text(f"{service}: {status}")
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    if st.sidebar.button("🚀 Start Monitoring Stack"):
        try:
            # Change to the correct directory first
            current_dir = os.getcwd()
            os.chdir(Path(__file__).parent)  # Change to ml_pipeline directory
            
            # Try docker compose (newer syntax) first, fallback to docker-compose
            try:
                result = subprocess.run([
                    "docker", "compose", "-f", "docker-compose.monitoring.yml", "up", "-d"
                ], capture_output=True, text=True)
            except FileNotFoundError:
                result = subprocess.run([
                    "docker-compose", "-f", "docker-compose.monitoring.yml", "up", "-d"
                ], capture_output=True, text=True)
            
            # Change back to original directory
            os.chdir(current_dir)
            
            if result.returncode == 0:
                st.sidebar.success("Monitoring stack started!")
            else:
                st.sidebar.error(f"Failed to start monitoring stack: {result.stderr}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    if st.sidebar.button("🛑 Stop Monitoring Stack"):
        try:
            # Change to the correct directory first
            current_dir = os.getcwd()
            os.chdir(Path(__file__).parent)  # Change to ml_pipeline directory
            
            # Try docker compose (newer syntax) first, fallback to docker-compose
            try:
                result = subprocess.run([
                    "docker", "compose", "-f", "docker-compose.monitoring.yml", "down"
                ], capture_output=True, text=True)
            except FileNotFoundError:
                result = subprocess.run([
                    "docker-compose", "-f", "docker-compose.monitoring.yml", "down"
                ], capture_output=True, text=True)
            
            # Change back to original directory
            os.chdir(current_dir)
            
            if result.returncode == 0:
                st.sidebar.success("Monitoring stack stopped!")
            else:
                st.sidebar.error(f"Failed to stop monitoring stack: {result.stderr}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Dashboard", 
        "📊 Data Processing", 
        "🤖 Model Training", 
        "📈 Evaluation", 
        "🚀 Deployment", 
        "💬 Chatbot", 
        "📋 Monitoring"
    ])
    
    # Dashboard Tab
    with tab1:
        st.header("📊 Pipeline Dashboard")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Infrastructure Components",
                value=sum(1 for status in infrastructure_status.values() if "✅" in status),
                delta=f"/{len(infrastructure_status)}"
            )
        
        with col2:
            st.metric(
                label="Services Running",
                value=sum(1 for status in service_status.values() if "🟢" in status),
                delta=f"/{len(service_status)}"
            )
        
        with col3:
            # Check if output directory exists
            output_dir = Path("pipeline_output")
            if output_dir.exists():
                files = list(output_dir.glob("*"))
                st.metric(
                    label="Pipeline Outputs",
                    value=len(files),
                    delta="files"
                )
            else:
                st.metric(
                    label="Pipeline Outputs",
                    value=0,
                    delta="No outputs yet"
                )
        
        with col4:
            # Check for trained models
            model_dirs = list(Path("pipeline_output").glob("*/final_model")) if Path("pipeline_output").exists() else []
            st.metric(
                label="Trained Models",
                value=len(model_dirs),
                delta="models"
            )
        
        # Quick access links
        st.subheader("🔗 Quick Access")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Grafana Dashboard", use_container_width=True):
                st.markdown("🌐 [Grafana Dashboard](http://localhost:3000)")
        
        with col2:
            if st.button("📈 Prometheus", use_container_width=True):
                st.markdown("🌐 [Prometheus](http://localhost:9090)")
        
        with col3:
            if st.button("🤖 Inference API", use_container_width=True):
                st.markdown("🌐 [Inference Server](http://localhost:8000)")
        
        # Recent activity
        st.subheader("📋 Recent Activity")
        if Path("pipeline_output").exists():
            output_files = list(Path("pipeline_output").glob("*"))
            if output_files:
                # Get file info
                file_info = []
                for file_path in output_files:
                    if file_path.is_file():
                        stat = file_path.stat()
                        file_info.append({
                            "name": file_path.name,
                            "size": f"{stat.st_size / 1024:.1f} KB",
                            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                        })
                
                if file_info:
                    df = pd.DataFrame(file_info)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent files found")
            else:
                st.info("No pipeline outputs found yet")
        else:
            st.info("Pipeline output directory not found")
    
    # Data Processing Tab
    with tab2:
        st.header("📊 Data Processing")
        st.info("Data processing functionality will be implemented here")
    
    # Model Training Tab
    with tab3:
        st.header("🤖 Model Training")
        st.info("Model training functionality will be implemented here")
    
    # Evaluation Tab
    with tab4:
        st.header("📈 Model Evaluation")
        st.info("Model evaluation functionality will be implemented here")
    
    # Deployment Tab
    with tab5:
        st.header("🚀 Model Deployment")
        st.info("Model deployment functionality will be implemented here")
    
    # Chatbot Tab
    with tab6:
        st.header("💬 EV Charging Assistant Chatbot")
        
        # Check for deployed models
        model_dirs = list(Path("pipeline_output").glob("*/final_model")) if Path("pipeline_output").exists() else []
        
        if not model_dirs:
            st.warning("⚠️ No trained models found. Please train a model first to use the chatbot.")
            st.info("💡 Go to the '🤖 Model Training' tab to train a model, then come back here!")
        else:
            st.success(f"✅ Found {len(model_dirs)} trained model(s)")
            
            # Model selection
            selected_model_path = st.selectbox(
                "Select model to use:",
                [str(path) for path in model_dirs],
                key="chatbot_model_select"
            )
            
            if selected_model_path:
                st.info(f"Using model: {selected_model_path}")
                
                # Initialize session state for chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                
                # Chat interface
                st.subheader("💬 Start a conversation")
                
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask about EV charging..."):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        try:
                            # Load model and generate response
                            from peft import PeftModel
                            from transformers import AutoTokenizer, AutoModelForCausalLM
                            import torch
                            
                            # Load base model and tokenizer
                            base_model = "microsoft/DialoGPT-medium"  # Default
                            base_model_instance = AutoModelForCausalLM.from_pretrained(
                                base_model,
                                torch_dtype=torch.float16,
                                device_map="auto"
                            )
                            tokenizer = AutoTokenizer.from_pretrained(base_model)
                            
                            # Load fine-tuned adapter
                            fine_tuned_model = PeftModel.from_pretrained(base_model_instance, selected_model_path)
                            
                            # Generate response
                            inputs = tokenizer.encode(prompt, return_tensors="pt")
                            # Move inputs to the same device as the model
                            device = next(fine_tuned_model.parameters()).device
                            inputs = inputs.to(device)
                            
                            with torch.no_grad():
                                outputs = fine_tuned_model.generate(
                                    inputs, 
                                    max_length=150, 
                                    num_return_sequences=1,
                                    temperature=0.7,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id
                                )
                            
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            
                            # Clean up response (remove input prompt)
                            if prompt in response:
                                response = response.replace(prompt, "").strip()
                            
                            # Display response
                            message_placeholder.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            error_msg = f"❌ Error generating response: {e}"
                            message_placeholder.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                # Clear chat button
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun()
                
                # Sample questions
                st.subheader("💡 Sample Questions")
                sample_questions = [
                    "What is Level 2 charging?",
                    "How fast is DC charging?",
                    "What are Tesla Superchargers?",
                    "How much does EV charging cost?",
                    "What are the different types of charging connectors?"
                ]
                
                cols = st.columns(2)
                for i, question in enumerate(sample_questions):
                    with cols[i % 2]:
                        if st.button(question, key=f"sample_{i}"):
                            # Simulate clicking the question
                            st.session_state.messages.append({"role": "user", "content": question})
                            st.rerun()
    
    # Monitoring Tab
    with tab7:
        st.header("📋 Monitoring Dashboard")
        
        # Service status
        st.subheader("🔧 Service Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "🟢 Running" in service_status.get("Prometheus", ""):
                st.success("📊 Prometheus: Running")
            else:
                st.error("📊 Prometheus: Stopped")
        
        with col2:
            if "🟢 Running" in service_status.get("Grafana", ""):
                st.success("📈 Grafana: Running")
            else:
                st.error("📈 Grafana: Stopped")
        
        with col3:
            if "🟢 Running" in service_status.get("Inference Server", ""):
                st.success("🤖 Inference Server: Running")
            else:
                st.error("🤖 Inference Server: Stopped")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start All Services", use_container_width=True):
                try:
                    # Change to the correct directory first
                    current_dir = os.getcwd()
                    os.chdir(Path(__file__).parent)  # Change to ml_pipeline directory
                    
                    # Start monitoring stack
                    try:
                        subprocess.run([
                            "docker", "compose", "-f", "docker-compose.monitoring.yml", "up", "-d"
                        ], check=True)
                    except FileNotFoundError:
                        subprocess.run([
                            "docker-compose", "-f", "docker-compose.monitoring.yml", "up", "-d"
                        ], check=True)
                    
                    # Change back to original directory
                    os.chdir(current_dir)
                    st.success("✅ All services started!")
                except Exception as e:
                    st.error(f"❌ Failed to start services: {e}")
        
        with col2:
            if st.button("🛑 Stop All Services", use_container_width=True):
                try:
                    # Change to the correct directory first
                    current_dir = os.getcwd()
                    os.chdir(Path(__file__).parent)  # Change to ml_pipeline directory
                    
                    # Stop monitoring stack
                    try:
                        subprocess.run([
                            "docker", "compose", "-f", "docker-compose.monitoring.yml", "down"
                        ], check=True)
                    except FileNotFoundError:
                        subprocess.run([
                            "docker-compose", "-f", "docker-compose.monitoring.yml", "down"
                        ], check=True)
                    
                    # Change back to original directory
                    os.chdir(current_dir)
                    st.success("✅ All services stopped!")
                except Exception as e:
                    st.error(f"❌ Failed to stop services: {e}")
        
        with col3:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        
        # Access links
        st.subheader("🔗 Access Links")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Grafana Dashboard**
            - URL: http://localhost:3000
            - Default credentials: admin/admin
            - Pre-configured dashboards available
            """)
        
        with col2:
            st.markdown("""
            **📈 Prometheus**
            - URL: http://localhost:9090
            - Metrics collection and storage
            - Alerting rules configured
            """)
        
        with col3:
            st.markdown("""
            **🚨 Alertmanager**
            - URL: http://localhost:9093
            - Alert routing and notification
            - Incident management
            """)

if __name__ == "__main__":
    main() 
