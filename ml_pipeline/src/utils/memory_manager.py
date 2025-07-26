"""
Memory Management Utilities for ML Pipeline
Provides chunked processing, memory monitoring, and optimization strategies
"""

import logging
import psutil
import gc
import time
from typing import Iterator, List, Dict, Any, Optional, Callable, TypeVar, Generic
from itertools import islice
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

T = TypeVar("T")


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percentage: float
    timestamp: float


class MemoryManager:
    """Centralized memory management for the pipeline"""

    def __init__(self, max_memory_mb: Optional[float] = None, chunk_size: int = 1000):
        """
        Initialize memory manager

        Args:
            max_memory_mb: Maximum memory usage in MB (None for auto-detect)
            chunk_size: Default chunk size for processing
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb or self._get_system_memory_limit()
        self.memory_history: List[MemoryStats] = []
        self.monitoring_enabled = True

        self.logger.info(
            f"Memory manager initialized with max_memory_mb={self.max_memory_mb}, chunk_size={self.chunk_size}"
        )

    def _get_system_memory_limit(self) -> float:
        """Get system memory limit in MB"""
        try:
            # Get available system memory
            memory = psutil.virtual_memory()
            # Use 80% of available memory as limit
            return memory.total / (1024 * 1024) * 0.8
        except Exception as e:
            self.logger.warning(f"Could not detect system memory: {e}")
            return 4096.0  # Default to 4GB

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics"""
        try:
            memory = psutil.virtual_memory()
            stats = MemoryStats(
                total_memory_mb=memory.total / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                memory_percentage=memory.percent,
                timestamp=time.time(),
            )

            if self.monitoring_enabled:
                self.memory_history.append(stats)
                # Keep only last 1000 entries
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]

            return stats
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, time.time())

    def is_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available"""
        stats = self.get_memory_stats()
        return stats.available_memory_mb >= required_mb

    def is_memory_critical(self, threshold_percentage: float = 90.0) -> bool:
        """Check if memory usage is critical"""
        stats = self.get_memory_stats()
        return stats.memory_percentage >= threshold_percentage

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory"""
        try:
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization"""
        optimization_results = {
            "garbage_collection": False,
            "memory_freed_mb": 0.0,
            "optimizations_applied": [],
        }

        try:
            # Force garbage collection
            before_stats = self.get_memory_stats()
            self.force_garbage_collection()
            after_stats = self.get_memory_stats()

            memory_freed = before_stats.used_memory_mb - after_stats.used_memory_mb
            optimization_results["garbage_collection"] = True
            optimization_results["memory_freed_mb"] = memory_freed
            optimization_results["optimizations_applied"].append("garbage_collection")

            self.logger.info(
                f"Memory optimization completed. Freed {memory_freed:.2f}MB"
            )

        except Exception as e:
            self.logger.error(f"Error during memory optimization: {e}")

        return optimization_results

    def chunked(
        self, iterable: Iterator[T], size: Optional[int] = None
    ) -> Iterator[List[T]]:
        """
        Split iterable into chunks of specified size

        Args:
            iterable: Input iterable
            size: Chunk size (uses default if None)

        Yields:
            Chunks of items
        """
        size = size or self.chunk_size
        iterator = iter(iterable)

        while chunk := list(islice(iterator, size)):
            yield chunk

    def process_in_chunks(
        self,
        items: List[T],
        processor: Callable[[List[T]], Any],
        chunk_size: Optional[int] = None,
        memory_check: bool = True,
    ) -> List[Any]:
        """
        Process items in chunks with memory management

        Args:
            items: List of items to process
            processor: Function to process each chunk
            chunk_size: Size of each chunk
            memory_check: Whether to check memory before processing

        Returns:
            List of results from processing
        """
        chunk_size = chunk_size or self.chunk_size
        results = []
        total_chunks = (len(items) + chunk_size - 1) // chunk_size

        self.logger.info(
            f"Processing {len(items)} items in {total_chunks} chunks of size {chunk_size}"
        )

        for i, chunk in enumerate(self.chunked(items, chunk_size)):
            try:
                # Check memory before processing
                if memory_check and self.is_memory_critical():
                    self.logger.warning(
                        f"Memory critical before chunk {i+1}/{total_chunks}"
                    )
                    self.optimize_memory()

                # Process chunk
                chunk_result = processor(chunk)
                results.append(chunk_result)

                # Log progress
                if (i + 1) % 10 == 0 or i + 1 == total_chunks:
                    self.logger.info(f"Processed chunk {i+1}/{total_chunks}")

                # Periodic memory optimization
                if (i + 1) % 50 == 0:
                    self.optimize_memory()

            except Exception as e:
                self.logger.error(f"Error processing chunk {i+1}: {e}")
                # Continue with next chunk
                continue

        return results

    def memory_safe_embedding(
        self,
        texts: List[str],
        embedder: Callable[[List[str]], np.ndarray],
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate embeddings with memory safety

        Args:
            texts: List of texts to embed
            embedder: Embedding function
            chunk_size: Chunk size for processing

        Returns:
            Combined embeddings array
        """
        chunk_size = chunk_size or self.chunk_size

        # Estimate memory requirements
        sample_size = min(10, len(texts))
        if sample_size > 0:
            sample_embeddings = embedder(texts[:sample_size])
            estimated_memory_per_text = sample_embeddings.nbytes / sample_size
            estimated_total_memory = estimated_memory_per_text * len(texts)

            # Adjust chunk size based on memory
            available_memory = (
                self.get_memory_stats().available_memory_mb * 1024 * 1024
            )  # Convert to bytes
            safe_chunk_size = min(
                chunk_size, int(available_memory * 0.5 / estimated_memory_per_text)
            )

            if safe_chunk_size < chunk_size:
                self.logger.info(
                    f"Adjusted chunk size from {chunk_size} to {safe_chunk_size} for memory safety"
                )
                chunk_size = safe_chunk_size

        # Process in chunks
        all_embeddings = []

        for chunk in self.chunked(texts, chunk_size):
            try:
                chunk_embeddings = embedder(chunk)
                all_embeddings.append(chunk_embeddings)

                # Force garbage collection after each chunk
                del chunk_embeddings
                self.force_garbage_collection()

            except Exception as e:
                self.logger.error(f"Error embedding chunk: {e}")
                # Create zero embeddings for failed chunk
                if all_embeddings:
                    zero_embeddings = np.zeros((len(chunk), all_embeddings[0].shape[1]))
                    all_embeddings.append(zero_embeddings)

        # Combine all embeddings
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])

    def save_to_disk(self, data: Any, filepath: str, method: str = "pickle") -> None:
        """
        Save data to disk to free memory

        Args:
            data: Data to save
            filepath: Path to save file
            method: Save method ('pickle', 'json', 'numpy')
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if method == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
            elif method == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif method == "numpy":
                np.save(filepath, data)
            else:
                raise ValueError(f"Unknown save method: {method}")

            self.logger.info(f"Saved data to {filepath} using {method}")

        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")

    def load_from_disk(self, filepath: str, method: str = "pickle") -> Any:
        """
        Load data from disk

        Args:
            filepath: Path to load file
            method: Load method ('pickle', 'json', 'numpy')

        Returns:
            Loaded data
        """
        try:
            filepath = Path(filepath)

            if method == "pickle":
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            elif method == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif method == "numpy":
                return np.load(filepath)
            else:
                raise ValueError(f"Unknown load method: {method}")

        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {e}")
            return None

    def streaming_processor(
        self,
        input_iterator: Iterator[T],
        processor: Callable[[T], Any],
        output_file: str,
        batch_size: int = 100,
    ) -> None:
        """
        Process data in streaming fashion to minimize memory usage

        Args:
            input_iterator: Iterator of input items
            processor: Function to process each item
            output_file: File to save results
            batch_size: Size of batches to process
        """
        results = []

        try:
            for item in input_iterator:
                try:
                    result = processor(item)
                    results.append(result)

                    # Save batch when it reaches batch_size
                    if len(results) >= batch_size:
                        self.save_to_disk(results, output_file, "json")
                        results = []
                        self.force_garbage_collection()

                except Exception as e:
                    self.logger.error(f"Error processing item: {e}")
                    continue

            # Save remaining results
            if results:
                self.save_to_disk(results, output_file, "json")

        except Exception as e:
            self.logger.error(f"Error in streaming processor: {e}")

    def parallel_processor(
        self,
        items: List[T],
        processor: Callable[[T], Any],
        max_workers: Optional[int] = None,
        use_processes: bool = False,
    ) -> List[Any]:
        """
        Process items in parallel with memory management

        Args:
            items: List of items to process
            processor: Function to process each item
            max_workers: Maximum number of workers
            use_processes: Whether to use processes instead of threads

        Returns:
            List of results
        """
        max_workers = max_workers or min(os.cpu_count() or 1, 4)

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(processor, item): item for item in items}

            results = []
            for future in future_to_item:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}")
                    results.append(None)

        return results

    def memory_monitor(self, interval: float = 60.0) -> None:
        """
        Start memory monitoring in background

        Args:
            interval: Monitoring interval in seconds
        """

        def monitor():
            while self.monitoring_enabled:
                try:
                    stats = self.get_memory_stats()

                    if stats.memory_percentage > 80:
                        self.logger.warning(
                            f"High memory usage: {stats.memory_percentage:.1f}%"
                        )
                        self.optimize_memory()

                    time.sleep(interval)

                except Exception as e:
                    self.logger.error(f"Error in memory monitoring: {e}")
                    time.sleep(interval)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info(f"Memory monitoring started with {interval}s interval")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        stats = self.get_memory_stats()

        report = {
            "current_usage": {
                "total_mb": stats.total_memory_mb,
                "used_mb": stats.used_memory_mb,
                "available_mb": stats.available_memory_mb,
                "percentage": stats.memory_percentage,
            },
            "limits": {
                "max_memory_mb": self.max_memory_mb,
                "chunk_size": self.chunk_size,
            },
            "optimization": {
                "garbage_collection_count": len(
                    [s for s in self.memory_history if s.memory_percentage > 80]
                ),
                "critical_memory_events": len(
                    [s for s in self.memory_history if s.memory_percentage > 90]
                ),
            },
            "history": {
                "total_measurements": len(self.memory_history),
                "average_usage_percentage": (
                    np.mean([s.memory_percentage for s in self.memory_history])
                    if self.memory_history
                    else 0
                ),
                "peak_usage_percentage": (
                    max([s.memory_percentage for s in self.memory_history])
                    if self.memory_history
                    else 0
                ),
            },
        }

        return report

    def cleanup(self) -> None:
        """Clean up memory manager resources"""
        self.monitoring_enabled = False
        self.memory_history.clear()
        self.force_garbage_collection()
        self.logger.info("Memory manager cleaned up")


# Global memory manager instance
memory_manager = MemoryManager()


def memory_safe(func: Callable) -> Callable:
    """Decorator to make functions memory safe"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check memory before execution
        if memory_manager.is_memory_critical():
            memory_manager.optimize_memory()

        try:
            result = func(*args, **kwargs)
            return result
        except MemoryError:
            memory_manager.logger.error("Memory error in function execution")
            memory_manager.optimize_memory()
            # Retry once after optimization
            return func(*args, **kwargs)
        finally:
            # Force garbage collection after function execution
            memory_manager.force_garbage_collection()

    return wrapper


def chunked_processing(chunk_size: Optional[int] = None):
    """Decorator for chunked processing"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, items: List[T], *args, **kwargs) -> List[Any]:
            return memory_manager.process_in_chunks(
                items, lambda chunk: func(self, chunk, *args, **kwargs), chunk_size
            )

        return wrapper

    return decorator
