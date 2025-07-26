import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import hashlib
import time
from collections import defaultdict
import json

# Add FAISS for efficient similarity search
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Import configuration manager
from src.utils.config_manager import get_config

# Import memory management
from src.utils.memory_manager import memory_manager, memory_safe, chunked_processing


class Deduplicator:
    def __init__(
        self, similarity_threshold: Optional[float] = None, method: Optional[str] = None
    ):
        """
        Initialize deduplicator with configurable threshold and method

        Args:
            similarity_threshold: Threshold for considering documents as duplicates (0.0-1.0)
                                 If None, uses configuration value
            method: Deduplication method ('levenshtein', 'semantic', 'hybrid', 'fast_levenshtein', 'faiss_semantic')
                   If None, uses configuration value
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self.config = get_config()
        dedup_config = self.config.get_deduplication_config()

        # Set parameters from config or arguments
        self.method = method or dedup_config.get("method", "levenshtein")
        self.similarity_threshold = similarity_threshold or dedup_config.get(
            "semantic_threshold", 0.95
        )

        # Load method-specific thresholds
        self.levenshtein_threshold = dedup_config.get("levenshtein_threshold", 0.97)
        self.fast_levenshtein_threshold = dedup_config.get(
            "fast_levenshtein_threshold", 0.95
        )

        # Load hybrid configuration
        self.hybrid_config = dedup_config.get("hybrid", {})
        self.semantic_first = self.hybrid_config.get("semantic_first", True)
        self.hybrid_semantic_threshold = self.hybrid_config.get(
            "semantic_threshold", 0.90
        )
        self.hybrid_levenshtein_threshold = self.hybrid_config.get(
            "levenshtein_threshold", 0.95
        )

        # Load FAISS configuration
        self.faiss_config = dedup_config.get("faiss_semantic", {})
        self.faiss_batch_size = self.faiss_config.get("batch_size", 256)
        self.faiss_search_k = self.faiss_config.get("search_k", 100)
        self.faiss_normalize = self.faiss_config.get("normalize_embeddings", True)

        # Load performance configuration
        self.performance_config = dedup_config.get("performance", {})
        self.enable_simhash = self.performance_config.get("enable_simhash", False)
        self.chunk_size = self.performance_config.get("chunk_size", 1000)
        self.memory_limit_mb = self.performance_config.get("memory_limit_mb", 2048)

        # Initialize semantic model if needed
        self.semantic_model = None
        if self.method in ["semantic", "hybrid", "faiss_semantic"]:
            try:
                model_config = self.config.get_model_config()
                semantic_config = model_config.get("semantic", {})
                model_name = semantic_config.get("model_name", "all-MiniLM-L6-v2")
                batch_size = semantic_config.get("batch_size", 256)
                device = semantic_config.get("device", "auto")

                self.semantic_model = SentenceTransformer(model_name)
                self.semantic_model.batch_size = batch_size

                if device != "auto":
                    self.semantic_model.device = device

                self.logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}")
                self.method = "levenshtein" if method == "semantic" else "levenshtein"

        # Check FAISS availability
        if self.method == "faiss_semantic" and not FAISS_AVAILABLE:
            self.logger.warning(
                "FAISS not available. Install with: pip install faiss-cpu or pip install faiss-gpu"
            )
            self.method = "semantic"  # Fallback to regular semantic

        # Check SimHash availability if enabled
        if self.enable_simhash:
            try:
                from simhash import Simhash

                self.SIMHASH_AVAILABLE = True
                self.Simhash = Simhash
            except ImportError:
                self.SIMHASH_AVAILABLE = False
                self.logger.warning(
                    "SimHash not available. Install with: pip install simhash-py"
                )

        self.logger.info(
            f"Initialized deduplicator with method: {self.method}, threshold: {self.similarity_threshold}"
        )

    @memory_safe
    def deduplicate(
        self, documents: List[Dict[str, Any]], text_column: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate documents using content similarity with memory management

        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content

        Returns:
            List of deduplicated documents
        """
        if not documents:
            self.logger.info("No documents to deduplicate")
            return []

        self.logger.info(
            f"Starting deduplication of {len(documents)} documents using {self.method} method"
        )
        start_time = time.time()

        # Check memory before processing
        if memory_manager.is_memory_critical():
            self.logger.warning("Memory critical before deduplication, optimizing...")
            memory_manager.optimize_memory()

        # Pre-filter exact duplicates using hash
        documents = self._remove_exact_duplicates(documents, text_column)

        # Apply SimHash pre-filter if enabled and available
        if (
            self.enable_simhash
            and hasattr(self, "SIMHASH_AVAILABLE")
            and self.SIMHASH_AVAILABLE
        ):
            documents = self._simhash_filter(documents, text_column)

        # Use chunked processing for large datasets
        if len(documents) > self.chunk_size:
            self.logger.info(
                f"Large dataset detected ({len(documents)} documents), using chunked processing"
            )
            result = self._chunked_deduplication(documents, text_column)
        else:
            # Process all at once for smaller datasets
            if self.method == "levenshtein":
                result = self._levenshtein_deduplication(documents, text_column)
            elif self.method == "fast_levenshtein":
                result = self._fast_levenshtein_deduplication(documents, text_column)
            elif self.method == "semantic":
                result = self._semantic_deduplication(documents, text_column)
            elif self.method == "faiss_semantic":
                result = self._faiss_semantic_deduplication(documents, text_column)
            elif self.method == "hybrid":
                result = self._hybrid_deduplication(documents, text_column)
            else:
                raise ValueError(f"Unknown deduplication method: {self.method}")

        end_time = time.time()
        self.logger.info(
            f"Deduplication completed in {end_time - start_time:.2f} seconds"
        )

        return result

    def _chunked_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """Deduplicate large datasets using chunked processing"""
        self.logger.info(
            f"Using chunked deduplication with chunk size {self.chunk_size}"
        )

        # Process documents in chunks
        unique_docs = []
        total_chunks = (len(documents) + self.chunk_size - 1) // self.chunk_size

        for i, chunk in enumerate(memory_manager.chunked(documents, self.chunk_size)):
            self.logger.info(
                f"Processing chunk {i+1}/{total_chunks} ({len(chunk)} documents)"
            )

            # Deduplicate within chunk
            if self.method == "levenshtein":
                chunk_result = self._levenshtein_deduplication(chunk, text_column)
            elif self.method == "fast_levenshtein":
                chunk_result = self._fast_levenshtein_deduplication(chunk, text_column)
            elif self.method == "semantic":
                chunk_result = self._semantic_deduplication(chunk, text_column)
            elif self.method == "faiss_semantic":
                chunk_result = self._faiss_semantic_deduplication(chunk, text_column)
            elif self.method == "hybrid":
                chunk_result = self._hybrid_deduplication(chunk, text_column)
            else:
                chunk_result = chunk

            # Add unique documents from this chunk
            unique_docs.extend(chunk_result)

            # Memory optimization between chunks
            memory_manager.force_garbage_collection()

            # Check memory after each chunk
            if memory_manager.is_memory_critical():
                self.logger.warning(f"Memory critical after chunk {i+1}, optimizing...")
                memory_manager.optimize_memory()

        # Final deduplication across all chunks (if needed)
        if len(unique_docs) > self.chunk_size:
            self.logger.info("Performing final cross-chunk deduplication")
            return self._final_cross_chunk_deduplication(unique_docs, text_column)

        return unique_docs

    def _final_cross_chunk_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """Perform final deduplication across chunks using sampling"""
        self.logger.info("Performing final cross-chunk deduplication using sampling")

        # Use sampling for very large datasets
        if len(documents) > 10000:
            sample_size = min(5000, len(documents) // 2)
            sample_docs = np.random.choice(
                documents, sample_size, replace=False
            ).tolist()
            self.logger.info(
                f"Using sampling with {sample_size} documents for cross-chunk deduplication"
            )
        else:
            sample_docs = documents

        # Create representative embeddings for sampling
        if self.semantic_model:
            sample_texts = [doc.get(text_column, "") for doc in sample_docs]
            sample_embeddings = memory_manager.memory_safe_embedding(
                sample_texts, self.semantic_model.encode, self.faiss_batch_size
            )

            # Find similar documents across chunks
            similar_groups = self._find_similar_groups(
                sample_embeddings, sample_docs, text_column
            )

            # Remove duplicates based on similarity groups
            unique_docs = self._remove_similar_groups(
                documents, similar_groups, text_column
            )
        else:
            # Fallback to Levenshtein for cross-chunk deduplication
            unique_docs = self._levenshtein_deduplication(documents, text_column)

        return unique_docs

    def _find_similar_groups(
        self, embeddings: np.ndarray, documents: List[Dict[str, Any]], text_column: str
    ) -> List[List[int]]:
        """Find groups of similar documents using embeddings"""
        similar_groups = []
        processed = set()

        for i in range(len(embeddings)):
            if i in processed:
                continue

            group = [i]
            processed.add(i)

            for j in range(i + 1, len(embeddings)):
                if j in processed:
                    continue

                # Calculate cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                if similarity > self.similarity_threshold:
                    group.append(j)
                    processed.add(j)

            if len(group) > 1:
                similar_groups.append(group)

        return similar_groups

    def _remove_similar_groups(
        self,
        documents: List[Dict[str, Any]],
        similar_groups: List[List[int]],
        text_column: str,
    ) -> List[Dict[str, Any]]:
        """Remove documents based on similarity groups"""
        to_remove = set()

        for group in similar_groups:
            # Keep the longest document in each group
            group_docs = [documents[i] for i in group]
            longest_doc = max(group_docs, key=lambda x: len(x.get(text_column, "")))
            longest_idx = group_docs.index(longest_doc)

            # Mark others for removal
            for i in group:
                if i != group[longest_idx]:
                    to_remove.add(i)

        # Return documents not marked for removal
        return [doc for i, doc in enumerate(documents) if i not in to_remove]

    def _simhash_filter(
        self, docs: List[Dict[str, Any]], text_column: str = "text"
    ) -> List[Dict[str, Any]]:
        """SimHash pre-filter to reduce search space"""
        if not hasattr(self, "SIMHASH_AVAILABLE") or not self.SIMHASH_AVAILABLE:
            return docs

        buckets, unique_docs = set(), []
        bucket_bits = 16  # 2^16 = 65,536 buckets
        shift = 64 - bucket_bits

        for doc in docs:
            h = self.Simhash(doc.get(text_column, "")).value >> shift
            if h not in buckets:
                buckets.add(h)
                unique_docs.append(doc)

        removed_count = len(docs) - len(unique_docs)
        self.logger.info(
            f"SimHash filter removed {removed_count} candidates before expensive comparisons"
        )
        return unique_docs

    @memory_safe
    def _faiss_semantic_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """
        FAISS-based semantic deduplication with O(n log n) complexity and memory management
        Much faster than O(n²) pairwise comparison
        """
        if not FAISS_AVAILABLE:
            self.logger.warning(
                "FAISS not available, falling back to regular semantic deduplication"
            )
            return self._semantic_deduplication(documents, text_column)

        if self.semantic_model is None:
            self.logger.warning(
                "Semantic model not available, falling back to Levenshtein"
            )
            return self._levenshtein_deduplication(documents, text_column)

        self.logger.info(
            "Using FAISS-based semantic deduplication for optimal performance"
        )

        # Extract texts
        texts = [doc.get(text_column, "") for doc in documents]

        # Compute embeddings with memory management
        self.logger.info("Computing semantic embeddings with memory management...")
        embeddings = memory_manager.memory_safe_embedding(
            texts, self.semantic_model.encode, self.faiss_batch_size
        )

        # Normalize embeddings for cosine similarity if configured
        if self.faiss_normalize:
            faiss.normalize_L2(embeddings)

        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        index.add(embeddings.astype("float32"))

        # Search for similar documents
        self.logger.info("Searching for similar documents using FAISS...")
        k = min(self.faiss_search_k, len(documents))  # Search top k similar documents

        # Search for each document
        unique_docs = []
        removed_count = 0
        duplicate_pairs = []
        processed = set()

        for i in tqdm(range(len(documents)), desc="FAISS semantic deduplication"):
            if i in processed:
                continue

            # Add current document to unique set
            unique_docs.append(documents[i])
            processed.add(i)

            # Search for similar documents
            query_embedding = embeddings[i : i + 1].astype("float32")
            similarities, indices = index.search(query_embedding, k)

            # Process similar documents
            for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == i or idx in processed:  # Skip self and already processed
                    continue

                if similarity > self.similarity_threshold:
                    processed.add(idx)
                    removed_count += 1
                    duplicate_pairs.append(
                        {
                            "original": documents[i].get("id", f"doc_{i}"),
                            "duplicate": documents[idx].get("id", f"doc_{idx}"),
                            "similarity": float(similarity),
                        }
                    )

            # Memory optimization every 1000 documents
            if i % 1000 == 0 and i > 0:
                memory_manager.force_garbage_collection()

        self.logger.info(
            f"FAISS semantic deduplication: removed {removed_count} duplicates"
        )
        self.duplicate_pairs = duplicate_pairs

        return unique_docs

    def _remove_exact_duplicates(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """Remove exact duplicates using content hash"""
        seen_hashes = set()
        unique_docs = []

        for doc in documents:
            content = doc.get(text_column, "")
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)

        removed = len(documents) - len(unique_docs)
        if removed > 0:
            self.logger.info(f"Removed {removed} exact duplicates")

        return unique_docs

    def _fast_levenshtein_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """
        Fast Levenshtein deduplication using the ratio approach
        """
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error(
                "Levenshtein package not installed. Install with: pip install python-Levenshtein"
            )
            return documents

        self.logger.info("Using fast Levenshtein ratio deduplication")

        # Sort by length to keep longer documents (more complete)
        sorted_docs = sorted(
            documents, key=lambda x: len(x.get(text_column, "")), reverse=True
        )

        unique = []
        removed_count = 0
        duplicate_pairs = []

        for doc in tqdm(sorted_docs, desc="Fast Levenshtein deduplication"):
            doc_content = doc.get(text_column, "")
            is_duplicate = False

            # Check against all unique documents
            for u_doc in unique:
                u_content = u_doc.get(text_column, "")

                # Calculate similarity ratio
                similarity = ratio(doc_content, u_content)

                if similarity > self.fast_levenshtein_threshold:
                    is_duplicate = True
                    removed_count += 1
                    duplicate_pairs.append(
                        {
                            "original": u_doc.get("id", "unknown"),
                            "duplicate": doc.get("id", "unknown"),
                            "similarity": similarity,
                        }
                    )
                    break

            if not is_duplicate:
                unique.append(doc)

        self.logger.info(
            f"Fast Levenshtein deduplication: removed {removed_count} duplicates"
        )
        self.duplicate_pairs = duplicate_pairs

        return unique

    def _levenshtein_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """Enhanced Levenshtein deduplication with optimizations"""
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error(
                "Levenshtein package not installed. Install with: pip install python-Levenshtein"
            )
            return documents

        # Sort by length to compare long vs short (keep longer documents)
        sorted_docs = sorted(
            documents, key=lambda x: len(x.get(text_column, "")), reverse=True
        )

        unique_docs = [sorted_docs[0]]
        removed_count = 0
        duplicate_pairs = []

        for doc in tqdm(sorted_docs[1:], desc="Deduplicating with Levenshtein"):
            is_duplicate = False
            doc_content = doc.get(text_column, "")

            for u_doc in unique_docs:
                u_content = u_doc.get(text_column, "")

                # Skip if lengths are too different (optimization)
                if (
                    abs(len(doc_content) - len(u_content))
                    / max(len(doc_content), len(u_content))
                    > 0.3
                ):
                    continue

                # Calculate similarity
                similarity = ratio(doc_content, u_content)
                if similarity > self.levenshtein_threshold:
                    is_duplicate = True
                    removed_count += 1
                    duplicate_pairs.append(
                        {
                            "original": u_doc.get("id", "unknown"),
                            "duplicate": doc.get("id", "unknown"),
                            "similarity": similarity,
                        }
                    )
                    break

            if not is_duplicate:
                unique_docs.append(doc)

        self.logger.info(
            f"Levenshtein deduplication: removed {removed_count} duplicates"
        )
        self.duplicate_pairs = duplicate_pairs

        return unique_docs

    @memory_safe
    def _semantic_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """Deduplicate using semantic similarity with chunk-wise encoding and memory management"""
        if self.semantic_model is None:
            self.logger.warning(
                "Semantic model not available, falling back to Levenshtein"
            )
            return self._levenshtein_deduplication(documents, text_column)

        # Extract texts
        texts = [doc.get(text_column, "") for doc in documents]

        # Get model configuration
        model_config = self.config.get_model_config()
        semantic_config = model_config.get("semantic", {})
        batch_size = semantic_config.get("batch_size", 256)

        # Compute embeddings with memory management
        self.logger.info("Computing semantic embeddings with memory management...")
        embeddings = memory_manager.memory_safe_embedding(
            texts, self.semantic_model.encode, batch_size
        )

        # Find duplicates using cosine similarity
        unique_docs = [documents[0]]
        removed_count = 0
        duplicate_pairs = []

        for i in tqdm(
            range(1, len(documents)), desc="Deduplicating with semantic similarity"
        ):
            is_duplicate = False
            doc_embedding = embeddings[i]

            for u_doc in unique_docs:
                u_idx = documents.index(u_doc)
                u_embedding = embeddings[u_idx]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(doc_embedding, u_embedding)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    duplicate_pairs.append(
                        {
                            "original": u_doc.get("id", "unknown"),
                            "duplicate": documents[i].get("id", "unknown"),
                            "similarity": similarity,
                        }
                    )
                    break

            if not is_duplicate:
                unique_docs.append(documents[i])

            # Memory optimization every 1000 documents
            if i % 1000 == 0:
                memory_manager.force_garbage_collection()

        self.logger.info(f"Semantic deduplication: removed {removed_count} duplicates")
        self.duplicate_pairs = duplicate_pairs

        return unique_docs

    def _hybrid_deduplication(
        self, documents: List[Dict[str, Any]], text_column: str
    ) -> List[Dict[str, Any]]:
        """Deduplicate using both Levenshtein and semantic similarity"""
        if self.semantic_first:
            # First pass: semantic deduplication
            self.logger.info("Hybrid deduplication: First pass (semantic)")
            # Temporarily set threshold for semantic pass
            original_threshold = self.similarity_threshold
            self.similarity_threshold = self.hybrid_semantic_threshold
            semantic_docs = self._semantic_deduplication(documents, text_column)
            self.similarity_threshold = original_threshold

            # Second pass: Levenshtein deduplication
            self.logger.info("Hybrid deduplication: Second pass (Levenshtein)")
            # Temporarily set threshold for Levenshtein pass
            original_levenshtein = self.levenshtein_threshold
            self.levenshtein_threshold = self.hybrid_levenshtein_threshold
            final_docs = self._levenshtein_deduplication(semantic_docs, text_column)
            self.levenshtein_threshold = original_levenshtein

            return final_docs
        else:
            # First pass: Levenshtein deduplication
            self.logger.info("Hybrid deduplication: First pass (Levenshtein)")
            original_levenshtein = self.levenshtein_threshold
            self.levenshtein_threshold = self.hybrid_levenshtein_threshold
            levenshtein_docs = self._levenshtein_deduplication(documents, text_column)
            self.levenshtein_threshold = original_levenshtein

            # Second pass: semantic deduplication
            self.logger.info("Hybrid deduplication: Second pass (semantic)")
            original_threshold = self.similarity_threshold
            self.similarity_threshold = self.hybrid_semantic_threshold
            final_docs = self._semantic_deduplication(levenshtein_docs, text_column)
            self.similarity_threshold = original_threshold

            return final_docs

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def deduplicate_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """Deduplicate pandas DataFrame with memory management"""
        documents = df.to_dict("records")
        deduplicated_docs = self.deduplicate(documents, text_column)
        return pd.DataFrame(deduplicated_docs)

    def get_deduplication_stats(
        self, original_count: int, final_count: int
    ) -> Dict[str, Any]:
        """Get deduplication statistics"""
        removed_count = original_count - final_count
        reduction_percentage = (
            (removed_count / original_count * 100) if original_count > 0 else 0
        )

        stats = {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": removed_count,
            "reduction_percentage": reduction_percentage,
            "method": self.method,
            "threshold": self.similarity_threshold,
            "config_source": "centralized",
            "memory_usage": memory_manager.get_memory_report(),
        }

        # Add method-specific thresholds
        if self.method == "levenshtein":
            stats["levenshtein_threshold"] = self.levenshtein_threshold
        elif self.method == "fast_levenshtein":
            stats["fast_levenshtein_threshold"] = self.fast_levenshtein_threshold
        elif self.method == "hybrid":
            stats["hybrid_config"] = {
                "semantic_first": self.semantic_first,
                "semantic_threshold": self.hybrid_semantic_threshold,
                "levenshtein_threshold": self.hybrid_levenshtein_threshold,
            }
        elif self.method == "faiss_semantic":
            stats["faiss_config"] = {
                "batch_size": self.faiss_batch_size,
                "search_k": self.faiss_search_k,
                "normalize_embeddings": self.faiss_normalize,
            }

        # Add duplicate pair analysis if available
        if hasattr(self, "duplicate_pairs") and self.duplicate_pairs:
            similarities = [pair["similarity"] for pair in self.duplicate_pairs]
            stats.update(
                {
                    "duplicate_pairs_count": len(self.duplicate_pairs),
                    "avg_similarity": np.mean(similarities),
                    "min_similarity": np.min(similarities),
                    "max_similarity": np.max(similarities),
                }
            )

        return stats

    def analyze_duplicates(
        self, documents: List[Dict[str, Any]], text_column: str = "text"
    ) -> Dict[str, Any]:
        """
        Analyze potential duplicates without removing them

        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content

        Returns:
            Dictionary with duplicate analysis
        """
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed")
            return {"error": "Levenshtein package not available"}

        analysis = {
            "total_documents": len(documents),
            "potential_duplicates": [],
            "similarity_distribution": defaultdict(int),
            "high_similarity_pairs": [],
            "config_used": {
                "method": self.method,
                "threshold": self.similarity_threshold,
            },
        }

        # Analyze all pairs
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i + 1 :], i + 1):
                content1 = doc1.get(text_column, "")
                content2 = doc2.get(text_column, "")

                similarity = ratio(content1, content2)

                # Categorize by similarity level
                if similarity > 0.9:
                    category = "very_high"
                elif similarity > 0.8:
                    category = "high"
                elif similarity > 0.7:
                    category = "medium"
                else:
                    category = "low"

                analysis["similarity_distribution"][category] += 1

                # Store high similarity pairs
                if similarity > 0.8:
                    analysis["high_similarity_pairs"].append(
                        {
                            "doc1_id": doc1.get("id", f"doc_{i}"),
                            "doc2_id": doc2.get("id", f"doc_{j}"),
                            "similarity": similarity,
                            "doc1_preview": (
                                content1[:100] + "..."
                                if len(content1) > 100
                                else content1
                            ),
                            "doc2_preview": (
                                content2[:100] + "..."
                                if len(content2) > 100
                                else content2
                            ),
                        }
                    )

        analysis["potential_duplicates"] = len(analysis["high_similarity_pairs"])

        return analysis

    def save_duplicate_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save duplicate analysis to file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Duplicate analysis saved to {output_path}")

    def load_duplicate_analysis(self, input_path: str) -> Dict[str, Any]:
        """Load duplicate analysis from file"""
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_duplicate_clusters(
        self, documents: List[Dict[str, Any]], text_column: str = "text"
    ) -> List[List[Dict[str, Any]]]:
        """
        Group documents into duplicate clusters

        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content

        Returns:
            List of duplicate clusters
        """
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed")
            return []

        clusters = []
        processed = set()

        for i, doc in enumerate(documents):
            if i in processed:
                continue

            cluster = [doc]
            processed.add(i)
            doc_content = doc.get(text_column, "")

            for j, other_doc in enumerate(documents[i + 1 :], i + 1):
                if j in processed:
                    continue

                other_content = other_doc.get(text_column, "")
                similarity = ratio(doc_content, other_content)

                if similarity > self.similarity_threshold:
                    cluster.append(other_doc)
                    processed.add(j)

            if len(cluster) > 1:  # Only return clusters with duplicates
                clusters.append(cluster)

        return clusters
