import logging
import time
import psutil
import torch
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import json
import os
from pathlib import Path

# Import our existing components
from .auth_handler import APIKeyAuth

# Prometheus metrics
REQUEST_COUNT = Counter(
    "ml_inference_requests_total", "Total inference requests", ["model", "status"]
)
REQUEST_LATENCY = Histogram(
    "ml_inference_latency_seconds", "Inference latency in seconds", ["model"]
)
MODEL_MEMORY_USAGE = Gauge(
    "ml_model_memory_usage_bytes", "Model memory usage in bytes", ["model"]
)
ERROR_COUNT = Counter(
    "ml_inference_errors_total", "Total inference errors", ["model", "error_type"]
)
ACTIVE_REQUESTS = Gauge("ml_active_requests", "Number of active requests", ["model"])
MODEL_ACCURACY = Gauge("ml_model_accuracy", "Model accuracy score", ["model"])


class MonitoredInferenceServer:
    """Enhanced inference server with comprehensive monitoring"""

    def __init__(self, model_path: str, port: int = 8000, host: str = "0.0.0.0"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.port = port
        self.host = host
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        # Initialize auth handler
        self.auth_handler = APIKeyAuth()

        # Initialize FastAPI app
        self.app = FastAPI(
            title="EV Charging LLM Inference Server",
            description="Monitored inference server for EV charging domain LLM",
            version="1.0.0",
        )

        # Setup Prometheus instrumentation
        self._setup_monitoring()

        # Setup routes
        self._setup_routes()

        # Track model metrics
        self._update_model_metrics()

    def _setup_monitoring(self):
        """Setup Prometheus monitoring"""
        try:
            # Instrument FastAPI app
            Instrumentator().instrument(self.app).expose(self.app)

            # Add custom metrics
            @self.app.middleware("http")
            async def monitor_requests(request: Request, call_next):
                start_time = time.time()

                # Track active requests
                ACTIVE_REQUESTS.labels(model="ev_charging_llm").inc()

                try:
                    response = await call_next(request)

                    # Record metrics
                    REQUEST_COUNT.labels(
                        model="ev_charging_llm", status=response.status_code
                    ).inc()

                    REQUEST_LATENCY.labels(model="ev_charging_llm").observe(
                        time.time() - start_time
                    )

                    return response

                except Exception as e:
                    ERROR_COUNT.labels(
                        model="ev_charging_llm", error_type=type(e).__name__
                    ).inc()
                    raise
                finally:
                    ACTIVE_REQUESTS.labels(model="ev_charging_llm").dec()

        except Exception as e:
            self.logger.warning(f"Prometheus monitoring setup failed: {e}")

    def _setup_routes(self):
        """Setup API routes"""

        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check model availability
                model_loaded = self.base_server.model is not None

                # Check system resources
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent

                health_status = {
                    "status": "healthy" if model_loaded else "unhealthy",
                    "timestamp": time.time(),
                    "model_loaded": model_loaded,
                    "system": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                    },
                }

                return health_status

            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")

        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint"""
            return generate_latest()

        # Model info endpoint
        @self.app.get("/model/info")
        async def model_info():
            """Get model information"""
            try:
                model_info = {
                    "model_path": self.model_path,
                    "model_type": "QLoRA Fine-tuned DialoGPT",
                    "parameters": self._get_model_parameters(),
                    "memory_usage": self._get_model_memory_usage(),
                    "device": (
                        str(self.base_server.device)
                        if hasattr(self.base_server, "device")
                        else "unknown"
                    ),
                }
                return model_info
            except Exception as e:
                self.logger.error(f"Failed to get model info: {e}")
                raise HTTPException(status_code=500, detail="Failed to get model info")

        # Enhanced inference endpoint
        @self.app.post("/predict")
        async def predict(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        ):
            """Enhanced prediction endpoint with monitoring"""
            try:
                # Authenticate request
                if not self.auth_handler.authenticate(credentials.credentials):
                    raise HTTPException(
                        status_code=401, detail="Invalid authentication"
                    )

                # Parse request
                body = await request.json()
                question = body.get("question", "")
                max_length = body.get("max_length", 100)
                temperature = body.get("temperature", 0.7)

                if not question:
                    raise HTTPException(status_code=400, detail="Question is required")

                # Generate response with monitoring
                start_time = time.time()

                try:
                    response = self.base_server.generate_response(
                        question, max_length, temperature
                    )

                    # Record successful inference
                    latency = time.time() - start_time
                    REQUEST_LATENCY.labels(model="ev_charging_llm").observe(latency)

                    return {
                        "response": response,
                        "latency": latency,
                        "model": "ev_charging_llm",
                        "timestamp": time.time(),
                    }

                except Exception as e:
                    # Record inference error
                    ERROR_COUNT.labels(
                        model="ev_charging_llm", error_type="inference_error"
                    ).inc()
                    raise HTTPException(
                        status_code=500, detail=f"Inference failed: {str(e)}"
                    )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Prediction endpoint error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        # Batch prediction endpoint
        @self.app.post("/predict/batch")
        async def predict_batch(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        ):
            """Batch prediction endpoint"""
            try:
                # Authenticate request
                if not self.auth_handler.authenticate(credentials.credentials):
                    raise HTTPException(
                        status_code=401, detail="Invalid authentication"
                    )

                # Parse request
                body = await request.json()
                questions = body.get("questions", [])
                max_length = body.get("max_length", 100)
                temperature = body.get("temperature", 0.7)

                if not questions or not isinstance(questions, list):
                    raise HTTPException(
                        status_code=400, detail="Questions list is required"
                    )

                # Process batch
                start_time = time.time()
                responses = []

                for question in questions:
                    try:
                        response = self.base_server.generate_response(
                            question, max_length, temperature
                        )
                        responses.append(
                            {
                                "question": question,
                                "response": response,
                                "status": "success",
                            }
                        )
                    except Exception as e:
                        responses.append(
                            {
                                "question": question,
                                "response": None,
                                "status": "error",
                                "error": str(e),
                            }
                        )

                batch_latency = time.time() - start_time

                return {
                    "responses": responses,
                    "batch_latency": batch_latency,
                    "total_questions": len(questions),
                    "successful": len(
                        [r for r in responses if r["status"] == "success"]
                    ),
                    "failed": len([r for r in responses if r["status"] == "error"]),
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        # Performance metrics endpoint
        @self.app.get("/metrics/performance")
        async def performance_metrics():
            """Get performance metrics"""
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                # Get model metrics
                model_memory = self._get_model_memory_usage()

                metrics = {
                    "system": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / (1024**3),
                        "disk_percent": disk.percent,
                        "disk_free_gb": disk.free / (1024**3),
                    },
                    "model": {
                        "memory_usage_bytes": model_memory,
                        "memory_usage_gb": model_memory / (1024**3),
                        "device": (
                            str(self.base_server.device)
                            if hasattr(self.base_server, "device")
                            else "unknown"
                        ),
                    },
                    "timestamp": time.time(),
                }

                return metrics

            except Exception as e:
                self.logger.error(f"Failed to get performance metrics: {e}")
                raise HTTPException(
                    status_code=500, detail="Failed to get performance metrics"
                )

    def _get_model_parameters(self) -> int:
        """Get number of model parameters"""
        try:
            if self.base_server.model:
                return sum(p.numel() for p in self.base_server.model.parameters())
            return 0
        except:
            return 0

    def _get_model_memory_usage(self) -> int:
        """Get model memory usage in bytes"""
        try:
            if self.base_server.model:
                # Estimate memory usage
                total_params = sum(
                    p.numel() for p in self.base_server.model.parameters()
                )
                # Assume float16 precision
                memory_bytes = total_params * 2  # 2 bytes per parameter for float16

                # Update Prometheus metric
                MODEL_MEMORY_USAGE.labels(model="ev_charging_llm").set(memory_bytes)

                return memory_bytes
            return 0
        except:
            return 0

    def _update_model_metrics(self):
        """Update model-specific metrics"""
        try:
            # Update memory usage
            self._get_model_memory_usage()

            # Set initial accuracy (this would be updated based on evaluation results)
            MODEL_ACCURACY.labels(model="ev_charging_llm").set(0.85)  # Example value

        except Exception as e:
            self.logger.warning(f"Failed to update model metrics: {e}")

    def start(self):
        """Start the monitored inference server"""
        self.logger.info(
            f"Starting monitored inference server on {self.host}:{self.port}"
        )

        try:
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start monitored inference server")
    parser.add_argument(
        "--model_path", required=True, help="Path to the fine-tuned model"
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Start server
    server = MonitoredInferenceServer(args.model_path, args.port, args.host)
    server.start()
