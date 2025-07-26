from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import logging
import time
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from .model_registry import ModelRegistry
from .auth_handler import APIKeyAuth
from .monitoring import Monitoring
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Prometheus monitoring imports
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram, Gauge, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Prometheus monitoring not available. Install with: pip install prometheus-fastapi-instrumentator prometheus-client"
    )

app = FastAPI(title="LLM Inference API", version="1.0.0")
logger = logging.getLogger("inference_server")
registry = ModelRegistry()
auth = APIKeyAuth()
monitor = Monitoring()

# Add rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Standard metrics
    API_REQUESTS = Counter(
        "api_requests_total",
        "Total API requests",
        ["endpoint", "status", "model", "version"],
    )

    RESPONSE_TIME = Histogram(
        "api_response_time_seconds",
        "Response time in seconds",
        ["endpoint", "model", "version"],
    )

    REQUEST_SIZE = Histogram(
        "api_request_size_bytes", "Request size in bytes", ["endpoint"]
    )

    RESPONSE_SIZE = Histogram(
        "api_response_size_bytes", "Response size in bytes", ["endpoint"]
    )

    MODEL_LOAD_TIME = Histogram(
        "model_load_time_seconds", "Model loading time in seconds", ["model", "version"]
    )

    MODEL_MEMORY_USAGE = Gauge(
        "model_memory_usage_bytes", "Model memory usage in bytes", ["model", "version"]
    )

    TOKEN_GENERATION_TIME = Histogram(
        "token_generation_time_seconds",
        "Token generation time per token",
        ["model", "version"],
    )

    TOKENS_GENERATED = Counter(
        "tokens_generated_total", "Total tokens generated", ["model", "version"]
    )

    CACHE_HITS = Counter(
        "model_cache_hits_total", "Total model cache hits", ["model", "version"]
    )

    CACHE_MISSES = Counter(
        "model_cache_misses_total", "Total model cache misses", ["model", "version"]
    )

    # Business metrics
    SUCCESSFUL_INFERENCES = Counter(
        "successful_inferences_total",
        "Total successful inferences",
        ["model", "version", "domain"],
    )

    FAILED_INFERENCES = Counter(
        "failed_inferences_total",
        "Total failed inferences",
        ["model", "version", "error_type"],
    )

    # Initialize Prometheus instrumentator
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, include_in_schema=True, should_gzip=True)


class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    adapter_version: str
    base_model: str


class ModelCache:
    def __init__(self):
        self.cache = {}

    async def get_model(self, base_model, adapter_version):
        key = (base_model, adapter_version)

        # Track cache hits/misses
        if PROMETHEUS_AVAILABLE:
            if key in self.cache:
                CACHE_HITS.labels(model=base_model, version=adapter_version).inc()
            else:
                CACHE_MISSES.labels(model=base_model, version=adapter_version).inc()

        if key in self.cache:
            return self.cache[key]

        # Track model loading time
        start_time = time.time()

        try:
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load base model with quantization
            if device == "cpu":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            adapter_info = registry.get_adapter(base_model, adapter_version)
            if not adapter_info:
                raise HTTPException(status_code=404, detail="Adapter version not found")
            model = PeftModel.from_pretrained(model, adapter_info["adapter_path"])
            model.eval()

            # Track model memory usage
            if PROMETHEUS_AVAILABLE:
                model_size = sum(
                    p.numel() * p.element_size() for p in model.parameters()
                )
                MODEL_MEMORY_USAGE.labels(
                    model=base_model, version=adapter_version
                ).set(model_size)

            self.cache[key] = (model, tokenizer)

            # Warm-up
            _ = model.generate(
                **tokenizer("Hello", return_tensors="pt").to(model.device),
                max_new_tokens=1,
            )

            # Track loading time
            if PROMETHEUS_AVAILABLE:
                load_time = time.time() - start_time
                MODEL_LOAD_TIME.labels(
                    model=base_model, version=adapter_version
                ).observe(load_time)

            return model, tokenizer

        except Exception as e:
            # Track loading failures
            if PROMETHEUS_AVAILABLE:
                load_time = time.time() - start_time
                MODEL_LOAD_TIME.labels(
                    model=base_model, version=adapter_version
                ).observe(load_time)
            raise e


model_cache = ModelCache()


@app.on_event("startup")
async def startup_event():
    logger.info("Inference server started.")


@app.post("/infer")
@limiter.limit("5/minute")
async def infer(
    request: Request, payload: InferenceRequest, api_key: str = Depends(auth.verify_key)
):
    start_time = time.time()

    # Track request size
    if PROMETHEUS_AVAILABLE:
        request_size = len(payload.prompt.encode("utf-8"))
        REQUEST_SIZE.labels(endpoint="/infer").observe(request_size)

    try:
        # Log request
        monitor.log_request(request)

        # Get model with monitoring
        model, tokenizer = await model_cache.get_model(
            payload.base_model, payload.adapter_version
        )

        # Tokenize input
        inputs = tokenizer(payload.prompt, return_tensors="pt").to(model.device)

        # Generate response with timing
        generation_start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=payload.max_new_tokens)
        generation_time = time.time() - generation_start

        # Decode result
        result = tokenizer.decode(output[0], skip_special_tokens=True)

        # Calculate metrics
        response_time = time.time() - start_time
        response_size = len(result.encode("utf-8"))
        tokens_generated = len(output[0]) - len(inputs["input_ids"][0])

        # Track Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            # Standard metrics
            API_REQUESTS.labels(
                endpoint="/infer",
                status="success",
                model=payload.base_model,
                version=payload.adapter_version,
            ).inc()

            RESPONSE_TIME.labels(
                endpoint="/infer",
                model=payload.base_model,
                version=payload.adapter_version,
            ).observe(response_time)

            RESPONSE_SIZE.labels(endpoint="/infer").observe(response_size)

            # Token generation metrics
            TOKEN_GENERATION_TIME.labels(
                model=payload.base_model, version=payload.adapter_version
            ).observe(generation_time / max(tokens_generated, 1))

            TOKENS_GENERATED.labels(
                model=payload.base_model, version=payload.adapter_version
            ).inc(tokens_generated)

            # Business metrics
            SUCCESSFUL_INFERENCES.labels(
                model=payload.base_model,
                version=payload.adapter_version,
                domain="general",
            ).inc()

        # Log response time
        monitor.log_response_time(request)

        return JSONResponse(
            {
                "result": result,
                "metadata": {
                    "model": payload.base_model,
                    "version": payload.adapter_version,
                    "tokens_generated": tokens_generated,
                    "response_time": response_time,
                    "generation_time": generation_time,
                },
            }
        )

    except Exception as e:
        # Track error metrics
        if PROMETHEUS_AVAILABLE:
            API_REQUESTS.labels(
                endpoint="/infer",
                status="error",
                model=payload.base_model,
                version=payload.adapter_version,
            ).inc()

            FAILED_INFERENCES.labels(
                model=payload.base_model,
                version=payload.adapter_version,
                error_type=type(e).__name__,
            ).inc()

        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/metrics")
async def metrics():
    """Get monitoring metrics"""
    return monitor.get_metrics()


@app.get("/prometheus")
async def prometheus_metrics():
    """Get Prometheus metrics"""
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response

        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    else:
        raise HTTPException(
            status_code=503, detail="Prometheus monitoring not available"
        )


@app.get("/status")
async def status():
    """Get detailed service status"""
    status_info = {
        "service": "LLM Inference API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": time.time(),
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "cached_models": len(model_cache.cache),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Add model cache info
    if model_cache.cache:
        status_info["cached_model_info"] = [
            {"base_model": key[0], "version": key[1], "loaded": True}
            for key in model_cache.cache.keys()
        ]

    return status_info


@app.get("/models")
async def list_models():
    """List available models"""
    try:
        models = registry.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")
