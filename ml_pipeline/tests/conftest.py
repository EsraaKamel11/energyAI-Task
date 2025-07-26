"""
Shared pytest fixtures for EV Charging LLM Pipeline tests
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_ev_data():
    """Sample EV charging data for testing"""
    return [
        {
            "text": "Level 1 charging uses 120V and provides 2-5 miles per hour",
            "source": "test",
            "url": "https://example.com/level1",
            "timestamp": "2024-01-01T00:00:00Z"
        },
        {
            "text": "Level 2 charging uses 240V and provides 10-60 miles per hour",
            "source": "test",
            "url": "https://example.com/level2",
            "timestamp": "2024-01-01T00:00:00Z"
        },
        {
            "text": "DC fast charging can provide 60-80% charge in 20-30 minutes",
            "source": "test",
            "url": "https://example.com/dc-fast",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    ]

@pytest.fixture
def sample_qa_pairs():
    """Sample Q&A pairs for testing"""
    return [
        {
            "question": "What is Level 1 charging?",
            "answer": "Level 1 charging uses 120V and provides 2-5 miles per hour",
            "source": "test",
            "category": "charging_levels"
        },
        {
            "question": "How fast is Level 2 charging?",
            "answer": "Level 2 charging uses 240V and provides 10-60 miles per hour",
            "source": "test",
            "category": "charging_levels"
        },
        {
            "question": "What is DC fast charging?",
            "answer": "DC fast charging can provide 60-80% charge in 20-30 minutes",
            "source": "test",
            "category": "charging_levels"
        }
    ]

@pytest.fixture
def mock_api_keys(monkeypatch):
    """Mock API keys for testing"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("PIPELINE_ENV", "testing")

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from OpenAI API"
                }
            }
        ]
    }

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "pipeline": {
            "environment": "testing",
            "batch_size": 4,
            "max_length": 512,
            "learning_rate": 2e-5
        },
        "model": {
            "base_model": "meta-llama/Llama-2-7b-hf",
            "quantization": "4bit",
            "lora_rank": 16,
            "lora_alpha": 32
        },
        "data": {
            "input_dir": "./data/raw/",
            "output_dir": "./data/processed/",
            "cache_dir": "./cache/"
        }
    }

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return """
    Electric Vehicle Charging Guide
    
    Level 1 Charging
    Level 1 charging uses a standard 120-volt household outlet and provides 
    2-5 miles of range per hour of charging. This is the slowest charging option 
    but requires no special equipment installation.
    
    Level 2 Charging
    Level 2 charging uses a 240-volt outlet and provides 10-60 miles of range 
    per hour. This requires installation of a Level 2 charging station and is 
    the most common option for home charging.
    
    DC Fast Charging
    DC fast charging can provide 60-80% charge in 20-30 minutes. These stations 
    are typically found at public locations and along highways for long-distance travel.
    """

@pytest.fixture
def mock_web_response():
    """Mock web scraping response"""
    return {
        "url": "https://example.com/ev-charging",
        "content": """
        <html>
        <body>
            <h1>EV Charging Information</h1>
            <p>Level 1 charging uses 120V and provides 2-5 miles per hour.</p>
            <p>Level 2 charging uses 240V and provides 10-60 miles per hour.</p>
            <p>DC fast charging can provide 60-80% charge in 20-30 minutes.</p>
        </body>
        </html>
        """,
        "status_code": 200,
        "headers": {"content-type": "text/html"}
    }

@pytest.fixture
def sample_benchmark_questions():
    """Sample benchmark questions for testing"""
    return [
        {
            "question": "What is the difference between Level 1 and Level 2 charging?",
            "category": "charging_levels",
            "expected_keywords": ["120V", "240V", "miles per hour"]
        },
        {
            "question": "How long does DC fast charging take?",
            "category": "charging_speed",
            "expected_keywords": ["20-30 minutes", "60-80%"]
        },
        {
            "question": "What voltage does Level 1 charging use?",
            "category": "technical_specs",
            "expected_keywords": ["120V", "120-volt"]
        }
    ]

@pytest.fixture
def mock_model_response():
    """Mock model inference response"""
    return {
        "answer": "Level 1 charging uses 120V and provides 2-5 miles per hour of charging.",
        "confidence": 0.85,
        "tokens_used": 25,
        "response_time": 0.5
    }

@pytest.fixture
def sample_evaluation_metrics():
    """Sample evaluation metrics for testing"""
    return {
        "bert_score": {
            "precision": 0.85,
            "recall": 0.82,
            "f1": 0.83
        },
        "rouge": {
            "rouge1": 0.45,
            "rouge2": 0.32,
            "rougeL": 0.38
        },
        "bleu": 0.42,
        "exact_match": 0.15,
        "response_time": 0.5,
        "memory_usage": 2048
    }

@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    logger.debug = Mock()
    return logger

@pytest.fixture
def sample_error_log():
    """Sample error log for testing"""
    return {
        "timestamp": "2024-01-01T00:00:00Z",
        "level": "ERROR",
        "message": "Test error message",
        "module": "test_module",
        "function": "test_function",
        "line": 42,
        "exception": "TestException",
        "traceback": "Traceback (most recent call last):\n  File test.py, line 42, in test_function\n    raise TestException('Test error')\nTestException: Test error"
    }

@pytest.fixture
def mock_file_system(temp_data_dir):
    """Mock file system with test data"""
    # Create test directories
    (temp_data_dir / "raw").mkdir()
    (temp_data_dir / "processed").mkdir()
    (temp_data_dir / "models").mkdir()
    (temp_data_dir / "outputs").mkdir()
    
    # Create sample files
    sample_data = [
        {"text": "Test document 1", "source": "test1"},
        {"text": "Test document 2", "source": "test2"}
    ]
    
    with open(temp_data_dir / "raw" / "sample_data.json", "w") as f:
        json.dump(sample_data, f)
    
    with open(temp_data_dir / "processed" / "qa_pairs.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    return temp_data_dir

@pytest.fixture
def mock_pandas_dataframe():
    """Mock pandas DataFrame for testing"""
    return pd.DataFrame({
        "text": [
            "Level 1 charging uses 120V",
            "Level 2 charging uses 240V",
            "DC fast charging provides quick charging"
        ],
        "source": ["test1", "test2", "test3"],
        "category": ["charging", "charging", "charging"]
    })

@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model for testing"""
    model = Mock()
    model.parameters.return_value = [Mock(numel=Mock(return_value=1000))]
    model.eval.return_value = model
    model.train.return_value = model
    return model

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "Test decoded text"
    tokenizer.model_max_length = 512
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    return tokenizer

@pytest.fixture
def mock_dataset():
    """Mock dataset for testing"""
    dataset = Mock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [1, 2, 3, 4, 5]
    }
    return dataset

@pytest.fixture
def mock_training_args():
    """Mock training arguments for testing"""
    return Mock(
        output_dir="./test_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        warmup_steps=100,
        max_steps=1000,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None
    )

@pytest.fixture
def mock_metrics():
    """Mock training metrics for testing"""
    return {
        "train_loss": 2.5,
        "eval_loss": 2.3,
        "learning_rate": 2e-5,
        "epoch": 1.0,
        "global_step": 100
    }

@pytest.fixture
def mock_wandb_run():
    """Mock Weights & Biases run for testing"""
    run = Mock()
    run.log = Mock()
    run.finish = Mock()
    return run

@pytest.fixture
def sample_config_file(temp_data_dir):
    """Create sample configuration file for testing"""
    config = {
        "pipeline": {
            "environment": "testing",
            "batch_size": 4,
            "max_length": 512
        },
        "model": {
            "base_model": "test-model",
            "quantization": "4bit"
        }
    }
    
    config_file = temp_data_dir / "test_config.yaml"
    with open(config_file, "w") as f:
        import yaml
        yaml.dump(config, f)
    
    return config_file

@pytest.fixture
def mock_requests_response():
    """Mock requests response for testing"""
    response = Mock()
    response.status_code = 200
    response.text = "Test response content"
    response.json.return_value = {"status": "success"}
    response.headers = {"content-type": "application/json"}
    return response

@pytest.fixture
def mock_subprocess_result():
    """Mock subprocess result for testing"""
    result = Mock()
    result.returncode = 0
    result.stdout = "Test output"
    result.stderr = ""
    return result

@pytest.fixture
def sample_error_data():
    """Sample error data for testing error handling"""
    return {
        "error_type": "ValueError",
        "error_message": "Test error message",
        "error_location": "test_function",
        "error_context": {
            "input_data": "test_input",
            "parameters": {"param1": "value1"}
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }

@pytest.fixture
def mock_memory_info():
    """Mock memory information for testing"""
    return {
        "total": 8589934592,  # 8GB
        "available": 4294967296,  # 4GB
        "used": 4294967296,  # 4GB
        "percent": 50.0
    }

@pytest.fixture
def mock_cpu_info():
    """Mock CPU information for testing"""
    return {
        "count": 8,
        "percent": 25.0,
        "frequency": 2400000000  # 2.4GHz
    }

@pytest.fixture
def mock_gpu_info():
    """Mock GPU information for testing"""
    return {
        "name": "NVIDIA GeForce RTX 3080",
        "memory_total": 10737418240,  # 10GB
        "memory_used": 5368709120,  # 5GB
        "memory_free": 5368709120,  # 5GB
        "utilization": 60.0
    } 