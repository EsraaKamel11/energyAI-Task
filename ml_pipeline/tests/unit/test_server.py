#!/usr/bin/env python3
"""
Mock inference server for testing monitoring functionality
"""

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import logging
from typing import Dict, Any
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Prometheus monitoring imports
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus monitoring not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock LLM Inference API", version="1.0.0")

# Add rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Standard metrics
    API_REQUESTS = Counter(
        'api_requests_total',
        'Total API requests',
        ['endpoint', 'status', 'model', 'version']
    )
    
    RESPONSE_TIME = Histogram(
        'api_response_time_seconds',
        'Response time in seconds',
        ['endpoint', 'model', 'version']
    )
    
    REQUEST_SIZE = Histogram(
        'api_request_size_bytes',
        'Request size in bytes',
        ['endpoint']
    )
    
    RESPONSE_SIZE = Histogram(
        'api_response_size_bytes',
        'Response size in bytes',
        ['endpoint']
    )
    
    TOKEN_GENERATION_TIME = Histogram(
        'token_generation_time_seconds',
        'Token generation time per token',
        ['model', 'version']
    )
    
    TOKENS_GENERATED = Counter(
        'tokens_generated_total',
        'Total tokens generated',
        ['model', 'version']
    )
    
    # Business metrics
    SUCCESSFUL_INFERENCES = Counter(
        'successful_inferences_total',
        'Total successful inferences',
        ['model', 'version', 'domain']
    )
    
    FAILED_INFERENCES = Counter(
        'failed_inferences_total',
        'Total failed inferences',
        ['model', 'version', 'error_type']
    )
    
    # Initialize Prometheus instrumentator
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, include_in_schema=True, should_gzip=True)

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    adapter_version: str
    base_model: str

from fastapi import Query

def verify_api_key(api_key: str = Query(..., description="API key for authentication")):
    """Mock API key verification"""
    # For testing, accept any API key
    return api_key

@app.on_event("startup")
async def startup_event():
    logger.info("Mock inference server started with monitoring.")

@app.post("/infer")
@limiter.limit("5/minute")
async def infer(request: Request, payload: InferenceRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    
    # Track request size
    if PROMETHEUS_AVAILABLE:
        request_size = len(payload.prompt.encode('utf-8'))
        REQUEST_SIZE.labels(endpoint='/infer').observe(request_size)
    
    try:
        # Simulate processing time
        time.sleep(0.1)
        
        # Mock response generation
        mock_responses = {
            "What is the capital of France?": "The capital of France is Paris.",
            "How does EV charging work?": "EV charging works by transferring electricity from the grid to your vehicle's battery.",
            "What is machine learning?": "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        }
        
        # Generate mock response
        if payload.prompt in mock_responses:
            result = mock_responses[payload.prompt]
        else:
            result = f"Mock response to: {payload.prompt}"
        
        # Calculate metrics
        response_time = time.time() - start_time
        response_size = len(result.encode('utf-8'))
        tokens_generated = len(result.split())  # Mock token count
        
        # Track Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            # Standard metrics
            API_REQUESTS.labels(
                endpoint='/infer', 
                status='success', 
                model=payload.base_model, 
                version=payload.adapter_version
            ).inc()
            
            RESPONSE_TIME.labels(
                endpoint='/infer', 
                model=payload.base_model, 
                version=payload.adapter_version
            ).observe(response_time)
            
            RESPONSE_SIZE.labels(endpoint='/infer').observe(response_size)
            
            # Token generation metrics
            TOKEN_GENERATION_TIME.labels(
                model=payload.base_model, 
                version=payload.adapter_version
            ).observe(response_time / max(tokens_generated, 1))
            
            TOKENS_GENERATED.labels(
                model=payload.base_model, 
                version=payload.adapter_version
            ).inc(tokens_generated)
            
            # Business metrics
            SUCCESSFUL_INFERENCES.labels(
                model=payload.base_model, 
                version=payload.adapter_version, 
                domain='general'
            ).inc()
        
        return JSONResponse({
            "result": result,
            "metadata": {
                "model": payload.base_model,
                "version": payload.adapter_version,
                "tokens_generated": tokens_generated,
                "response_time": response_time,
                "generation_time": response_time
            }
        })
        
    except Exception as e:
        # Track error metrics
        if PROMETHEUS_AVAILABLE:
            API_REQUESTS.labels(
                endpoint='/infer', 
                status='error', 
                model=payload.base_model, 
                version=payload.adapter_version
            ).inc()
            
            FAILED_INFERENCES.labels(
                model=payload.base_model, 
                version=payload.adapter_version, 
                error_type=type(e).__name__
            ).inc()
        
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/status")
async def status():
    """Get detailed service status"""
    status_info = {
        "service": "Mock LLM Inference API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": time.time(),
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "cached_models": 0,
        "device": "mock"
    }
    return status_info

@app.get("/models")
async def list_models():
    """List available models"""
    models = [
        {
            "base_model": "microsoft/DialoGPT-medium",
            "versions": ["v1.0", "v1.1"],
            "description": "Mock model for testing"
        }
    ]
    return {"models": models}

@app.get("/prometheus")
async def prometheus_metrics():
    """Get Prometheus metrics"""
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    else:
        raise HTTPException(status_code=503, detail="Prometheus monitoring not available")

def main():
    """Start the mock server"""
    logger.info("🚀 Starting Mock LLM Inference Server with Monitoring")
    logger.info("=" * 50)
    
    host = "localhost"
    port = 8000
    
    logger.info(f"📍 Server will run on: http://{host}:{port}")
    logger.info("📊 Monitoring endpoints:")
    logger.info(f"  Health: http://{host}:{port}/health")
    logger.info(f"  Status: http://{host}:{port}/status")
    logger.info(f"  Prometheus: http://{host}:{port}/prometheus")
    logger.info(f"  Models: http://{host}:{port}/models")
    logger.info(f"  Inference: http://{host}:{port}/infer")
    logger.info("=" * 50)
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 