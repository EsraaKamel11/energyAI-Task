import time
import logging
import torch


class PerformanceTester:
    def __init__(self, model, tokenizer, device=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_latency(self, prompt: str, n_runs: int = 10) -> float:
        self.logger.info(f"Testing latency for {n_runs} runs...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.time()
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=32)
            torch.cuda.synchronize() if self.device == "cuda" else None
            times.append(time.time() - start)
        avg_latency = sum(times) / len(times)
        self.logger.info(f"Average latency: {avg_latency:.4f}s")
        return avg_latency

    def test_throughput(self, prompts: list, batch_size: int = 4) -> float:
        self.logger.info(f"Testing throughput with batch size {batch_size}...")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        torch.cuda.synchronize() if self.device == "cuda" else None
        start = time.time()
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=32)
        torch.cuda.synchronize() if self.device == "cuda" else None
        elapsed = time.time() - start
        throughput = len(prompts) / elapsed
        self.logger.info(f"Throughput: {throughput:.2f} samples/sec")
        return throughput
