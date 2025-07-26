import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from collections import defaultdict, namedtuple
from dataclasses import dataclass

# Import error handling utilities
from src.utils.error_handling import (
    retry_with_fallback,
    circuit_breaker,
    robust_file_operation,
)
from src.utils.error_classification import (
    classify_and_handle_error,
    get_error_recovery_strategy,
    should_retry_operation,
    get_retry_delay_for_error,
)

# Import configuration manager
from src.utils.config_manager import get_config

# Import memory management
from src.utils.memory_manager import memory_manager, memory_safe, chunked_processing

# Import unstructured library for better layout preservation
try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Text, Title, NarrativeText, ListItem

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    partition = None


@dataclass
class TextBlock:
    """Represents a text block with position and content information."""

    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float
    font_name: str
    page_num: int
    block_type: str = "text"  # text, title, header, footer, table, image


@dataclass
class Column:
    """Represents a column in the document layout."""

    x_start: float
    x_end: float
    y_start: float
    y_end: float
    text_blocks: List[TextBlock]
    column_index: int


@dataclass
class Section:
    """Represents a document section with hierarchy information."""

    title: str
    level: int
    start_page: int
    end_page: int
    text_blocks: List[TextBlock]
    subsections: List["Section"]
    parent: Optional["Section"] = None


class LayoutAnalyzer:
    """Analyzes PDF layout to detect columns, sections, headers, and footers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Layout analysis settings
        self.column_detection_threshold = config.get("column_detection_threshold", 0.1)
        self.header_footer_margin = config.get("header_footer_margin", 0.1)
        self.min_section_length = config.get("min_section_length", 50)
        self.font_size_tolerance = config.get("font_size_tolerance", 0.2)

    def detect_columns(
        self, text_blocks: List[TextBlock], page_width: float, page_height: float
    ) -> List[Column]:
        """
        Detect columns based on text block positions.

        Args:
            text_blocks: List of text blocks on the page
            page_width: Width of the page
            page_height: Height of the page

        Returns:
            List of detected columns
        """
        if not text_blocks:
            return []

        # Extract x-positions of text blocks
        x_positions = [block.x for block in text_blocks]

        # Cluster x-positions to detect columns
        columns = self._cluster_x_positions(x_positions, page_width)

        # Create Column objects
        column_objects = []
        for i, (x_start, x_end) in enumerate(columns):
            # Find text blocks that belong to this column
            column_blocks = [
                block for block in text_blocks if x_start <= block.x <= x_end
            ]

            if column_blocks:
                y_start = min(block.y for block in column_blocks)
                y_end = max(block.y + block.height for block in column_blocks)

                column = Column(
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y_start,
                    y_end=y_end,
                    text_blocks=column_blocks,
                    column_index=i,
                )
                column_objects.append(column)

        return column_objects

    def _cluster_x_positions(
        self, x_positions: List[float], page_width: float
    ) -> List[Tuple[float, float]]:
        """
        Cluster x-positions to detect column boundaries.

        Args:
            x_positions: List of x-positions
            page_width: Width of the page

        Returns:
            List of (start_x, end_x) tuples for each column
        """
        if not x_positions:
            return []

        # Sort x-positions
        x_positions = sorted(set(x_positions))

        # Use density-based clustering to detect columns
        columns = []
        current_column_start = x_positions[0]
        current_column_end = x_positions[0]

        for i in range(1, len(x_positions)):
            x = x_positions[i]

            # If x is close to the current column, extend it
            if x - current_column_end <= page_width * self.column_detection_threshold:
                current_column_end = x
            else:
                # Start a new column
                columns.append((current_column_start, current_column_end))
                current_column_start = x
                current_column_end = x

        # Add the last column
        columns.append((current_column_start, current_column_end))

        return columns

    def detect_headers_footers(
        self, text_blocks: List[TextBlock], page_height: float
    ) -> Tuple[List[TextBlock], List[TextBlock]]:
        """
        Detect headers and footers based on position and repetition patterns.

        Args:
            text_blocks: List of text blocks
            page_height: Height of the page

        Returns:
            Tuple of (headers, footers) lists
        """
        headers = []
        footers = []

        # Define header and footer regions
        header_region = page_height * self.header_footer_margin
        footer_region = page_height * (1 - self.header_footer_margin)

        for block in text_blocks:
            # Check if block is in header region
            if block.y <= header_region:
                block.block_type = "header"
                headers.append(block)
            # Check if block is in footer region
            elif block.y >= footer_region:
                block.block_type = "footer"
                footers.append(block)

        return headers, footers

    def detect_sections(self, text_blocks: List[TextBlock]) -> List[Section]:
        """
        Detect document sections based on font size and position.

        Args:
            text_blocks: List of text blocks

        Returns:
            List of detected sections
        """
        sections = []
        current_section = None

        # Sort blocks by page and y-position
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.page_num, b.y))

        for block in sorted_blocks:
            # Determine if this block is a title/section header
            section_level = self._detect_section_level(block, text_blocks)

            if section_level > 0:
                # Start a new section
                if current_section:
                    sections.append(current_section)

                current_section = Section(
                    title=block.text,
                    level=section_level,
                    start_page=block.page_num,
                    end_page=block.page_num,
                    text_blocks=[block],
                    subsections=[],
                    parent=None,
                )
            elif current_section:
                # Add block to current section
                current_section.text_blocks.append(block)
                current_section.end_page = max(current_section.end_page, block.page_num)

        # Add the last section
        if current_section:
            sections.append(current_section)

        # Build section hierarchy
        return self._build_section_hierarchy(sections)

    def _detect_section_level(
        self, block: TextBlock, all_blocks: List[TextBlock]
    ) -> int:
        """
        Detect the section level based on font size and formatting.

        Args:
            block: Text block to analyze
            all_blocks: All text blocks for comparison

        Returns:
            Section level (0 = not a section, 1+ = section level)
        """
        # Calculate average font size
        font_sizes = [b.font_size for b in all_blocks if b.font_size > 0]
        if not font_sizes:
            return 0

        avg_font_size = np.mean(font_sizes)

        # Check if this block has significantly larger font (likely a title)
        if block.font_size > avg_font_size * (1 + self.font_size_tolerance):
            # Determine level based on font size
            if block.font_size > avg_font_size * 1.5:
                return 1  # Main section
            elif block.font_size > avg_font_size * 1.3:
                return 2  # Subsection
            elif block.font_size > avg_font_size * 1.1:
                return 3  # Sub-subsection

        # Check for common title patterns
        title_patterns = [
            r"^\d+\.\s+[A-Z]",  # 1. Title
            r"^[A-Z][A-Z\s]+$",  # ALL CAPS
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$",  # Title Case
        ]

        for pattern in title_patterns:
            if re.match(pattern, block.text.strip()):
                return 1

        return 0

    def _build_section_hierarchy(self, sections: List[Section]) -> List[Section]:
        """
        Build section hierarchy based on levels.

        Args:
            sections: List of sections

        Returns:
            List of top-level sections with subsections
        """
        if not sections:
            return []

        # Sort sections by level and position
        sorted_sections = sorted(
            sections, key=lambda s: (s.level, s.start_page, s.text_blocks[0].y)
        )

        # Build hierarchy
        root_sections = []
        section_stack = []

        for section in sorted_sections:
            # Find the appropriate parent
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()

            if section_stack:
                # Add as subsection
                section.parent = section_stack[-1]
                section_stack[-1].subsections.append(section)
            else:
                # Add as root section
                root_sections.append(section)

            section_stack.append(section)

        return root_sections

    def organize_text_by_columns(
        self, text_blocks: List[TextBlock], columns: List[Column]
    ) -> List[TextBlock]:
        """
        Organize text blocks by columns for proper reading order.

        Args:
            text_blocks: List of text blocks
            columns: List of detected columns

        Returns:
            List of text blocks in reading order
        """
        if not columns:
            return text_blocks

        # Group text blocks by column
        column_blocks = defaultdict(list)

        for block in text_blocks:
            # Find which column this block belongs to
            for column in columns:
                if column.x_start <= block.x <= column.x_end:
                    column_blocks[column.column_index].append(block)
                    break

        # Sort blocks within each column by y-position
        for column_idx in column_blocks:
            column_blocks[column_idx].sort(key=lambda b: (b.page_num, b.y))

        # Interleave blocks from different columns (reading order)
        organized_blocks = []
        column_indices = sorted(column_blocks.keys())

        # Simple interleaving: take blocks from each column in round-robin fashion
        max_blocks = max(len(column_blocks[idx]) for idx in column_indices)

        for i in range(max_blocks):
            for col_idx in column_indices:
                if i < len(column_blocks[col_idx]):
                    organized_blocks.append(column_blocks[col_idx][i])

        return organized_blocks


class PDFExtractor:
    def __init__(
        self, strategy: Optional[str] = None, max_workers: Optional[int] = None
    ):
        """
        Initialize PDF extractor with configurable strategy and memory management

        Args:
            strategy: Extraction strategy ('pymupdf', 'unstructured', 'hybrid')
                     If None, uses configuration value
            max_workers: Maximum number of worker threads (if None, uses config)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self.config = get_config()
        pdf_config = self.config.get_pdf_extraction_config()

        # Set parameters from config or arguments
        self.strategy = strategy or pdf_config.get("strategy", "pymupdf")
        self.max_workers = max_workers or pdf_config.get("max_workers", 4)

        # Load extraction settings
        self.extraction_config = pdf_config.get("extraction", {})
        self.extract_text = self.extraction_config.get("extract_text", True)
        self.extract_images = self.extraction_config.get("extract_images", False)
        self.extract_tables = self.extraction_config.get("extract_tables", True)
        self.extract_metadata = self.extraction_config.get("extract_metadata", True)
        self.preserve_layout = self.extraction_config.get("preserve_layout", True)
        self.min_text_length = self.extraction_config.get("min_text_length", 10)

        # Load memory management settings
        self.memory_config = pdf_config.get("memory_management", {})
        self.chunk_size = self.memory_config.get("chunk_size", 10)
        self.max_pages_per_chunk = self.memory_config.get("max_pages_per_chunk", 50)
        self.save_intermediate = self.memory_config.get("save_intermediate", True)
        self.intermediate_dir = Path(
            self.memory_config.get("intermediate_dir", "temp_pdf_data")
        )

        # Load text processing settings
        self.text_config = pdf_config.get("text_processing", {})
        self.clean_text = self.text_config.get("clean_text", True)
        self.remove_headers_footers = self.text_config.get(
            "remove_headers_footers", True
        )
        self.merge_lines = self.text_config.get("merge_lines", True)
        self.extract_ev_terms = self.text_config.get("extract_ev_terms", True)

        # Load EV-specific terminology for preservation
        self.ev_terms = self.text_config.get(
            "ev_terminology",
            [
                "electric vehicle",
                "EV",
                "battery electric vehicle",
                "BEV",
                "plug-in hybrid",
                "PHEV",
                "hybrid electric vehicle",
                "HEV",
                "charging station",
                "fast charging",
                "level 1 charging",
                "level 2 charging",
                "DC fast charging",
                "supercharger",
                "range anxiety",
                "kilowatt-hour",
                "kWh",
                "miles per gallon equivalent",
                "MPGe",
                "regenerative braking",
                "electric motor",
                "battery pack",
                "lithium-ion",
                "nickel-metal hydride",
                "charging infrastructure",
                "smart charging",
                "vehicle-to-grid",
                "V2G",
            ],
        )

        # Check unstructured library availability
        if self.strategy in ["unstructured", "hybrid"] and not UNSTRUCTURED_AVAILABLE:
            self.logger.warning(
                "Unstructured library not available. Install with: pip install unstructured"
            )
            self.strategy = "pymupdf"

        # Create intermediate directory
        if self.save_intermediate:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Initialize layout analyzer
        layout_config = pdf_config.get("layout_analysis", {})
        self.layout_analyzer = LayoutAnalyzer(layout_config)

        self.logger.info(
            f"PDF extractor initialized with strategy: {self.strategy}, max_workers: {self.max_workers}"
        )

    @memory_safe
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract content from a single PDF file with comprehensive error handling,
        memory management, and intelligent fallback strategies.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing extracted content or error information
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            error_info = classify_and_handle_error(
                FileNotFoundError(f"PDF file not found: {pdf_path}"),
                {"file_path": str(pdf_path)},
            )
            return self._create_pdf_error_result(str(pdf_path), error_info)

        # Check memory before processing
        if memory_manager.is_memory_critical():
            self.logger.warning(
                f"Memory critical before extracting {pdf_path}, optimizing..."
            )
            memory_manager.optimize_memory()

        # Track retry attempts and errors
        retry_count = 0
        max_retries = 3
        last_error_info = None

        try:
            self.logger.info(f"Extracting content from: {pdf_path}")

            # Get file info
            file_size = pdf_path.stat().st_size
            file_hash = self._calculate_file_hash(pdf_path)

            # Choose extraction strategy based on file size and configuration
            if file_size > 50 * 1024 * 1024:  # 50MB
                self.logger.info("Large PDF detected, using chunked processing")
                result = self._extract_large_pdf(pdf_path)
            else:
                result = self._extract_single_pdf(pdf_path)

            # Add file metadata
            result.update(
                {
                    "file_path": str(pdf_path),
                    "file_size": file_size,
                    "file_hash": file_hash,
                    "extraction_strategy": self.strategy,
                    "extracted_at": datetime.now().isoformat(),
                }
            )

            return result

        except Exception as e:
            # Classify the error
            error_info = classify_and_handle_error(e, {"file_path": str(pdf_path)})
            last_error_info = error_info

            # Log detailed error information
            self.logger.error(
                f"Error extracting from {pdf_path}: {error_info.category.value} ({error_info.severity.value}) - {error_info.message}"
            )

            # Check if we should retry
            if should_retry_operation(error_info, retry_count, max_retries):
                retry_count += 1
                delay = get_retry_delay_for_error(error_info, retry_count)

                self.logger.info(
                    f"Retrying {pdf_path} in {delay}s (attempt {retry_count}/{max_retries})"
                )
                time.sleep(delay)

                # Try fallback strategy
                return self._try_pdf_fallback_strategy(pdf_path, error_info)
            else:
                # Don't retry, return error result
                return self._create_pdf_error_result(str(pdf_path), error_info)
        finally:
            # Memory optimization after extraction
            memory_manager.force_garbage_collection()

    def _try_pdf_fallback_strategy(self, pdf_path: Path, error_info) -> Dict[str, Any]:
        """
        Try fallback strategies for PDF extraction based on error classification.

        Args:
            pdf_path: Path to the PDF file
            error_info: Classified error information

        Returns:
            Extracted data or error result
        """
        strategy = get_error_recovery_strategy(error_info)

        if not strategy:
            return self._create_pdf_error_result(str(pdf_path), error_info)

        self.logger.info(f"Trying PDF fallback strategy: {strategy}")

        try:
            if strategy == "try_different_extractor":
                # Try different extraction strategy
                return self._try_different_extraction_strategy(pdf_path)

            elif strategy == "use_chunked_extraction":
                # Force chunked extraction even for smaller files
                return self._extract_large_pdf(pdf_path)

            elif strategy == "try_different_strategy":
                # Try alternative extraction strategies
                return self._try_alternative_strategies(pdf_path)

            elif strategy == "skip_file":
                # Skip the file and return error result
                return self._create_pdf_error_result(str(pdf_path), error_info)

            elif strategy == "try_repair_tools":
                # Try to repair corrupted PDF
                return self._try_repair_pdf(pdf_path)

            elif strategy == "try_common_passwords":
                # Try common passwords for password-protected PDFs
                return self._try_common_passwords(pdf_path)

            else:
                # Unknown strategy, try different extractor as last resort
                self.logger.warning(
                    f"Unknown PDF fallback strategy: {strategy}, trying different extractor"
                )
                return self._try_different_extraction_strategy(pdf_path)

        except Exception as fallback_error:
            # Fallback also failed
            fallback_error_info = classify_and_handle_error(
                fallback_error, {"file_path": str(pdf_path)}
            )
            self.logger.error(
                f"PDF fallback strategy {strategy} failed: {fallback_error_info.message}"
            )
            return self._create_pdf_error_result(str(pdf_path), fallback_error_info)

    def _try_different_extraction_strategy(self, pdf_path: Path) -> Dict[str, Any]:
        """Try different extraction strategies in order of preference."""
        strategies = ["pymupdf", "unstructured", "hybrid"]

        # Remove current strategy from the list
        if self.strategy in strategies:
            strategies.remove(self.strategy)

        for strategy in strategies:
            try:
                self.logger.info(f"Trying extraction strategy: {strategy}")

                if strategy == "pymupdf":
                    result = self._extract_with_pymupdf(pdf_path)
                elif strategy == "unstructured":
                    result = self._extract_with_unstructured(pdf_path)
                elif strategy == "hybrid":
                    result = self._extract_hybrid(pdf_path)

                if result and not result.get("error"):
                    result["fallback_strategy_used"] = strategy
                    return result

            except Exception as e:
                self.logger.warning(f"Strategy {strategy} failed: {e}")
                continue

        # All strategies failed
        raise Exception("All PDF extraction strategies failed")

    def _try_alternative_strategies(self, pdf_path: Path) -> Dict[str, Any]:
        """Try alternative extraction approaches."""
        try:
            # Try with different PyMuPDF settings
            self.logger.info("Trying alternative PyMuPDF settings")

            doc = fitz.open(pdf_path)

            # Try with different text extraction methods
            all_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Try different text extraction methods
                try:
                    # Method 1: Standard text extraction
                    text = page.get_text()
                    if text.strip():
                        all_text.append(text)
                        continue
                except:
                    pass

                try:
                    # Method 2: Text extraction with layout preservation
                    text = page.get_text("text")
                    if text.strip():
                        all_text.append(text)
                        continue
                except:
                    pass

                try:
                    # Method 3: Raw text extraction
                    text = page.get_text("raw")
                    if text.strip():
                        all_text.append(text)
                        continue
                except:
                    pass

                try:
                    # Method 4: HTML extraction
                    text = page.get_text("html")
                    if text.strip():
                        # Extract text from HTML
                        from bs4 import BeautifulSoup

                        soup = BeautifulSoup(text, "html.parser")
                        text = soup.get_text()
                        if text.strip():
                            all_text.append(text)
                            continue
                except:
                    pass

            doc.close()

            if all_text:
                return {
                    "text": all_text,
                    "fallback_strategy_used": "alternative_pymupdf_methods",
                    "extraction_method": "alternative_pymupdf",
                }

        except Exception as e:
            self.logger.warning(f"Alternative strategies failed: {e}")

        # If all alternatives fail, raise exception
        raise Exception("All alternative extraction strategies failed")

    def _try_repair_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Try to repair corrupted PDF using various methods."""
        try:
            self.logger.info("Attempting PDF repair")

            # Method 1: Try to re-open with different settings
            try:
                doc = fitz.open(pdf_path, filetype="pdf")
                doc.close()
                # If successful, try normal extraction
                return self._extract_single_pdf(pdf_path)
            except:
                pass

            # Method 2: Try with different file opening modes
            try:
                doc = fitz.open(pdf_path, filetype="pdf", stream=True)
                doc.close()
                return self._extract_single_pdf(pdf_path)
            except:
                pass

            # Method 3: Try to extract partial content
            try:
                doc = fitz.open(pdf_path)
                partial_text = []

                # Try to extract from first few pages only
                for page_num in range(min(5, len(doc))):
                    try:
                        page = doc[page_num]
                        text = page.get_text()
                        if text.strip():
                            partial_text.append(text)
                    except:
                        continue

                doc.close()

                if partial_text:
                    return {
                        "text": partial_text,
                        "fallback_strategy_used": "partial_extraction",
                        "extraction_method": "partial_repair",
                        "warning": "Only partial content extracted due to PDF corruption",
                    }

            except Exception as e:
                self.logger.warning(f"PDF repair failed: {e}")

        except Exception as e:
            self.logger.warning(f"PDF repair attempt failed: {e}")

        raise Exception("PDF repair failed")

    def _try_common_passwords(self, pdf_path: Path) -> Dict[str, Any]:
        """Try common passwords for password-protected PDFs."""
        common_passwords = [
            "",  # No password
            "password",
            "123456",
            "admin",
            "user",
            "guest",
            "public",
            "readonly",
            "view",
            "access",
        ]

        for password in common_passwords:
            try:
                self.logger.info(
                    f"Trying password: {'<empty>' if password == '' else password}"
                )

                doc = fitz.open(pdf_path)
                if doc.needs_pass:
                    doc.authenticate(password)

                # Try to access first page to verify password
                if len(doc) > 0:
                    doc[0].get_text()

                doc.close()

                # Password worked, try normal extraction
                return self._extract_single_pdf(pdf_path)

            except Exception as e:
                self.logger.debug(f"Password '{password}' failed: {e}")
                continue

        raise Exception("All common passwords failed")

    def _create_pdf_error_result(self, file_path: str, error_info) -> Dict[str, Any]:
        """Create a structured error result for PDF extraction."""
        return {
            "file_path": file_path,
            "error": True,
            "error_category": error_info.category.value,
            "error_severity": error_info.severity.value,
            "error_message": error_info.message,
            "recovery_strategies": error_info.recovery_strategies,
            "timestamp": error_info.timestamp.isoformat(),
            "text": [],
            "tables": [],
            "images": [],
            "metadata": {
                "extraction_method": "error",
                "error_info": {
                    "category": error_info.category.value,
                    "severity": error_info.severity.value,
                    "message": error_info.message,
                },
            },
        }

    def _extract_large_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content from large PDF using chunked processing"""
        self.logger.info(f"Using chunked processing for large PDF: {pdf_path}")

        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # Process pages in chunks
        all_text = []
        all_tables = []
        all_images = []
        metadata = {}

        for chunk_start in range(0, total_pages, self.max_pages_per_chunk):
            chunk_end = min(chunk_start + self.max_pages_per_chunk, total_pages)
            chunk_pages = list(range(chunk_start, chunk_end))

            self.logger.info(
                f"Processing pages {chunk_start+1}-{chunk_end} of {total_pages}"
            )

            # Extract chunk
            chunk_result = self._extract_pdf_chunk(pdf_path, chunk_pages)

            # Collect results
            if chunk_result.get("text"):
                all_text.extend(chunk_result["text"])
            if chunk_result.get("tables"):
                all_tables.extend(chunk_result["tables"])
            if chunk_result.get("images"):
                all_images.extend(chunk_result["images"])
            if chunk_result.get("metadata") and not metadata:
                metadata = chunk_result["metadata"]

            # Save intermediate results
            if self.save_intermediate:
                self._save_intermediate_chunk(chunk_result, pdf_path.stem, chunk_start)

            # Memory optimization between chunks
            memory_manager.force_garbage_collection()

            # Check memory after each chunk
            if memory_manager.is_memory_critical():
                self.logger.warning(
                    f"Memory critical after chunk {chunk_start//self.max_pages_per_chunk + 1}, optimizing..."
                )
                memory_manager.optimize_memory()

        return {
            "text": all_text,
            "tables": all_tables,
            "images": all_images,
            "metadata": metadata,
            "total_pages": total_pages,
            "processing_method": "chunked",
        }

    def _extract_pdf_chunk(
        self, pdf_path: Path, page_numbers: List[int]
    ) -> Dict[str, Any]:
        """Extract content from a chunk of PDF pages"""
        try:
            doc = fitz.open(pdf_path)

            chunk_text = []
            chunk_tables = []
            chunk_images = []
            chunk_metadata = {}

            for page_num in page_numbers:
                if page_num >= len(doc):
                    continue

                page = doc[page_num]

                # Extract text
                if self.extract_text:
                    page_text = self._extract_page_text(page, page_num)
                    if page_text:
                        chunk_text.append(
                            {
                                "page": page_num + 1,
                                "content": page_text,
                                "length": len(page_text),
                            }
                        )

                # Extract tables
                if self.extract_tables:
                    page_tables = self._extract_page_tables(page, page_num)
                    chunk_tables.extend(page_tables)

                # Extract images
                if self.extract_images:
                    page_images = self._extract_page_images(page, page_num)
                    chunk_images.extend(page_images)

                # Extract metadata (only from first page)
                if self.extract_metadata and page_num == 0:
                    chunk_metadata = self._extract_metadata(doc)

            doc.close()

            return {
                "text": chunk_text,
                "tables": chunk_tables,
                "images": chunk_images,
                "metadata": chunk_metadata,
            }

        except Exception as e:
            self.logger.error(f"Error extracting PDF chunk: {e}")
            return {}

    def _extract_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content from a single PDF file"""
        if self.strategy == "unstructured":
            return self._extract_with_unstructured(pdf_path)
        elif self.strategy == "hybrid":
            return self._extract_hybrid(pdf_path)
        else:
            return self._extract_with_pymupdf(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)

            all_text = []
            all_tables = []
            all_images = []
            metadata = {}

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                if self.extract_text:
                    page_text = self._extract_page_text(page, page_num)
                    if page_text:
                        all_text.append(
                            {
                                "page": page_num + 1,
                                "content": page_text,
                                "length": len(page_text),
                            }
                        )

                # Extract tables
                if self.extract_tables:
                    page_tables = self._extract_page_tables(page, page_num)
                    all_tables.extend(page_tables)

                # Extract images
                if self.extract_images:
                    page_images = self._extract_page_images(page, page_num)
                    all_images.extend(page_images)

                # Extract metadata (only from first page)
                if self.extract_metadata and page_num == 0:
                    metadata = self._extract_metadata(doc)

            doc.close()

            return {
                "text": all_text,
                "tables": all_tables,
                "images": all_images,
                "metadata": metadata,
                "total_pages": len(doc),
                "processing_method": "pymupdf",
            }

        except Exception as e:
            self.logger.error(f"Error extracting with PyMuPDF: {e}")
            return {"error": str(e)}

    def _extract_with_unstructured(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content using unstructured library for better layout preservation"""
        if not UNSTRUCTURED_AVAILABLE:
            self.logger.warning(
                "Unstructured library not available, falling back to PyMuPDF"
            )
            return self._extract_with_pymupdf(pdf_path)

        try:
            # Extract elements using unstructured
            elements = partition(str(pdf_path))

            all_text = []
            current_page = 1
            current_content = ""

            for element in elements:
                if isinstance(element, (Text, Title, NarrativeText, ListItem)):
                    content = str(element)

                    # Check if this is a new page (simple heuristic)
                    if len(current_content) > 1000 and "\n" in content[:100]:
                        if current_content.strip():
                            all_text.append(
                                {
                                    "page": current_page,
                                    "content": current_content.strip(),
                                    "length": len(current_content),
                                }
                            )
                        current_content = content
                        current_page += 1
                    else:
                        current_content += "\n" + content

            # Add final content
            if current_content.strip():
                all_text.append(
                    {
                        "page": current_page,
                        "content": current_content.strip(),
                        "length": len(current_content),
                    }
                )

            return {
                "text": all_text,
                "tables": [],  # Unstructured doesn't extract tables by default
                "images": [],  # Unstructured doesn't extract images by default
                "metadata": {},
                "total_pages": len(all_text),
                "processing_method": "unstructured",
            }

        except Exception as e:
            self.logger.error(f"Error extracting with unstructured: {e}")
            return self._extract_with_pymupdf(pdf_path)  # Fallback

    def _extract_hybrid(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content using hybrid approach (unstructured + PyMuPDF)"""
        try:
            # Try unstructured first for better layout
            unstructured_result = self._extract_with_unstructured(pdf_path)

            # If unstructured failed or returned no content, use PyMuPDF
            if not unstructured_result.get("text") or "error" in unstructured_result:
                self.logger.info(
                    "Unstructured extraction failed, using PyMuPDF fallback"
                )
                return self._extract_with_pymupdf(pdf_path)

            # Use PyMuPDF for tables and images
            pymupdf_result = self._extract_with_pymupdf(pdf_path)

            # Combine results
            hybrid_result = {
                "text": unstructured_result["text"],  # Better layout from unstructured
                "tables": pymupdf_result.get("tables", []),
                "images": pymupdf_result.get("images", []),
                "metadata": pymupdf_result.get("metadata", {}),
                "total_pages": unstructured_result["total_pages"],
                "processing_method": "hybrid",
            }

            return hybrid_result

        except Exception as e:
            self.logger.error(f"Error in hybrid extraction: {e}")
            return self._extract_with_pymupdf(pdf_path)  # Fallback

    def _extract_page_text(self, page, page_num: int) -> str:
        """
        Extract text from a single page with advanced layout preservation

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            Extracted text content with layout preserved
        """
        try:
            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height

            # Extract text blocks with detailed position information
            text_blocks = self._extract_text_blocks_with_layout(page, page_num)

            if not text_blocks:
                return ""

            # Perform layout analysis
            layout_result = self._analyze_page_layout(
                text_blocks, page_width, page_height
            )

            # Organize text based on layout analysis
            organized_blocks = layout_result.get("organized_blocks", text_blocks)

            # Extract text content from organized blocks
            extracted_text = []
            for block in organized_blocks:
                if (
                    block.block_type not in ["header", "footer"]
                    or not self.remove_headers_footers
                ):
                    extracted_text.append(block.text)

            # Join text with proper spacing
            text_content = "\n".join(extracted_text)

            # Apply text processing
            if self.clean_text:
                text_content = self._clean_text_content(text_content)

            if self.extract_ev_terms:
                text_content = self._preserve_ev_terminology(text_content)

            if self.merge_lines:
                text_content = self._merge_text_lines(text_content)

            return text_content.strip()

        except Exception as e:
            self.logger.error(f"Error extracting text from page {page_num + 1}: {e}")
            return ""

    def _extract_text_blocks_with_layout(self, page, page_num: int) -> List[TextBlock]:
        """
        Extract text blocks with detailed layout information.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            List of TextBlock objects with position and formatting info
        """
        text_blocks = []

        try:
            # Get text dictionary with detailed information
            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_x = float("inf")
                        line_y = float("inf")
                        line_width = 0
                        line_height = 0
                        line_font_size = 0
                        line_font_name = ""

                        for span in line["spans"]:
                            span_text = span["text"]
                            line_text += span_text

                            # Update line position and size
                            span_x = span["bbox"][0]
                            span_y = span["bbox"][1]
                            span_width = span["bbox"][2] - span["bbox"][0]
                            span_height = span["bbox"][3] - span["bbox"][1]

                            line_x = min(line_x, span_x)
                            line_y = min(line_y, span_y)
                            line_width = max(line_width, span_x + span_width - line_x)
                            line_height = max(
                                line_height, span_y + span_height - line_y
                            )

                            # Use the most common font size and name for the line
                            if span.get("size", 0) > line_font_size:
                                line_font_size = span.get("size", 0)
                                line_font_name = span.get("font", "")

                        if line_text.strip():
                            text_block = TextBlock(
                                text=line_text.strip(),
                                x=line_x,
                                y=line_y,
                                width=line_width,
                                height=line_height,
                                font_size=line_font_size,
                                font_name=line_font_name,
                                page_num=page_num,
                                block_type="text",
                            )
                            text_blocks.append(text_block)

        except Exception as e:
            self.logger.error(f"Error extracting text blocks from page {page_num}: {e}")

        return text_blocks

    def _analyze_page_layout(
        self, text_blocks: List[TextBlock], page_width: float, page_height: float
    ) -> Dict[str, Any]:
        """
        Analyze page layout to detect columns, sections, headers, and footers.

        Args:
            text_blocks: List of text blocks on the page
            page_width: Width of the page
            page_height: Height of the page

        Returns:
            Dictionary containing layout analysis results
        """
        layout_result = {
            "columns": [],
            "headers": [],
            "footers": [],
            "sections": [],
            "organized_blocks": text_blocks,
        }

        try:
            # Detect columns
            columns = self.layout_analyzer.detect_columns(
                text_blocks, page_width, page_height
            )
            layout_result["columns"] = columns

            # Detect headers and footers
            headers, footers = self.layout_analyzer.detect_headers_footers(
                text_blocks, page_height
            )
            layout_result["headers"] = headers
            layout_result["footers"] = footers

            # Organize text by columns
            if columns:
                organized_blocks = self.layout_analyzer.organize_text_by_columns(
                    text_blocks, columns
                )
                layout_result["organized_blocks"] = organized_blocks

            # Detect sections (for multi-page documents, this would be done across all pages)
            # For now, we'll detect sections within the page
            sections = self.layout_analyzer.detect_sections(text_blocks)
            layout_result["sections"] = sections

        except Exception as e:
            self.logger.error(f"Error analyzing page layout: {e}")

        return layout_result

    def _extract_page_tables(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from a single page"""
        try:
            tables = page.get_tables()
            extracted_tables = []

            for i, table in enumerate(tables):
                if table and len(table) > 0:
                    extracted_tables.append(
                        {
                            "page": page_num + 1,
                            "table_index": i,
                            "rows": len(table),
                            "columns": len(table[0]) if table[0] else 0,
                            "data": table,
                        }
                    )

            return extracted_tables

        except Exception as e:
            self.logger.error(f"Error extracting tables from page {page_num + 1}: {e}")
            return []

    def _extract_page_images(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract image information from a single page"""
        try:
            images = page.get_images()
            extracted_images = []

            for i, img in enumerate(images):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)

                    extracted_images.append(
                        {
                            "page": page_num + 1,
                            "image_index": i,
                            "width": pix.width,
                            "height": pix.height,
                            "colorspace": (
                                pix.colorspace.name if pix.colorspace else "unknown"
                            ),
                            "size_bytes": len(pix.tobytes()),
                        }
                    )

                    pix = None  # Free memory

                except Exception as e:
                    self.logger.debug(
                        f"Error extracting image {i} from page {page_num + 1}: {e}"
                    )
                    continue

            return extracted_images

        except Exception as e:
            self.logger.error(f"Error extracting images from page {page_num + 1}: {e}")
            return []

    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            metadata = doc.metadata
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": len(doc),
            }
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {}

    def _clean_text_content(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters that might be artifacts
        text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]", "", text)

        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("–", "-").replace("—", "-")

        return text.strip()

    def _preserve_ev_terminology(self, text: str) -> str:
        """Preserve EV-specific terminology during text processing"""
        # Create a mapping of EV terms to preserved versions
        ev_mapping = {}

        for term in self.ev_terms:
            # Create a unique placeholder
            placeholder = f"__EV_TERM_{hashlib.md5(term.encode()).hexdigest()[:8]}__"
            ev_mapping[placeholder] = term

            # Replace term with placeholder (case insensitive)
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(placeholder, text)

        # Apply text cleaning
        text = self._clean_text_content(text)

        # Restore EV terms
        for placeholder, term in ev_mapping.items():
            text = text.replace(placeholder, term)

        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers"""
        lines = text.split("\n")
        filtered_lines = []

        for line in lines:
            line = line.strip()

            # Skip common header/footer patterns
            if (
                re.match(r"^\d+$", line)  # Page numbers
                or re.match(r"^Page \d+ of \d+$", line, re.IGNORECASE)
                or re.match(r"^Confidential|Internal|Draft$", line, re.IGNORECASE)
                or len(line) < 3
            ):  # Very short lines
                continue

            filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _merge_text_lines(self, text: str) -> str:
        """Merge broken text lines"""
        lines = text.split("\n")
        merged_lines = []
        current_line = ""

        for line in lines:
            line = line.strip()

            if not line:
                if current_line:
                    merged_lines.append(current_line)
                    current_line = ""
                continue

            # Check if line should be merged (ends with lowercase or no punctuation)
            if current_line and (line[0].islower() or not current_line[-1] in ".!?"):
                current_line += " " + line
            else:
                if current_line:
                    merged_lines.append(current_line)
                current_line = line

        if current_line:
            merged_lines.append(current_line)

        return "\n".join(merged_lines)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""

    def _save_intermediate_chunk(
        self, chunk_result: Dict[str, Any], pdf_name: str, chunk_start: int
    ) -> None:
        """Save intermediate chunk results to disk"""
        try:
            filename = f"{pdf_name}_chunk_{chunk_start}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.intermediate_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chunk_result, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved intermediate chunk to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving intermediate chunk: {e}")

    @chunked_processing(chunk_size=5)
    def extract_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract content from multiple PDF files with chunked processing

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            List of extracted content dictionaries
        """
        results = []

        for pdf_path in pdf_paths:
            try:
                result = self.extract_from_pdf(pdf_path)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error extracting {pdf_path}: {e}")
                results.append(
                    {
                        "file_path": pdf_path,
                        "error": str(e),
                        "extracted_at": datetime.now().isoformat(),
                    }
                )

        return results

    @memory_safe
    def extract_pdfs_parallel(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract content from multiple PDF files in parallel with memory management

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            List of extracted content dictionaries
        """
        self.logger.info(
            f"Extracting {len(pdf_paths)} PDFs in parallel with {self.max_workers} workers"
        )

        # Check memory before starting
        if memory_manager.is_memory_critical():
            self.logger.warning(
                "Memory critical before parallel extraction, optimizing..."
            )
            memory_manager.optimize_memory()

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all extraction tasks
            future_to_path = {
                executor.submit(self.extract_from_pdf, path): path for path in pdf_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Memory optimization every 10 extractions
                    if len(results) % 10 == 0:
                        memory_manager.force_garbage_collection()

                except Exception as e:
                    self.logger.error(f"Error extracting {pdf_path}: {e}")
                    results.append(
                        {
                            "file_path": pdf_path,
                            "error": str(e),
                            "extracted_at": datetime.now().isoformat(),
                        }
                    )

        self.logger.info(
            f"Parallel extraction completed: {len(results)} PDFs processed"
        )
        return results

    def save_extraction_results(
        self, results: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Save extraction results to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Save as CSV (flattened)
            csv_path = output_path.with_suffix(".csv")
            flattened_results = self._flatten_results_for_csv(results)
            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_path, index=False, encoding="utf-8")

            self.logger.info(f"Extraction results saved to {json_path} and {csv_path}")

        except Exception as e:
            self.logger.error(f"Error saving extraction results: {e}")

    def _flatten_results_for_csv(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Flatten results for CSV export"""
        flattened = []

        for result in results:
            if "error" in result:
                flattened.append(
                    {
                        "file_path": result.get("file_path", ""),
                        "error": result.get("error", ""),
                        "extracted_at": result.get("extracted_at", ""),
                    }
                )
                continue

            # Extract text content
            text_content = ""
            if result.get("text"):
                text_content = "\n".join([t.get("content", "") for t in result["text"]])

            # Extract table count
            table_count = len(result.get("tables", []))

            # Extract image count
            image_count = len(result.get("images", []))

            flattened.append(
                {
                    "file_path": result.get("file_path", ""),
                    "file_size": result.get("file_size", 0),
                    "total_pages": result.get("total_pages", 0),
                    "text_content": text_content,
                    "text_length": len(text_content),
                    "table_count": table_count,
                    "image_count": image_count,
                    "extraction_strategy": result.get("extraction_strategy", ""),
                    "processing_method": result.get("processing_method", ""),
                    "extracted_at": result.get("extracted_at", ""),
                }
            )

        return flattened

    def get_extraction_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get extraction statistics"""
        if not results:
            return {"error": "No results to analyze"}

        successful_extractions = [r for r in results if "error" not in r]
        failed_extractions = [r for r in results if "error" in r]

        stats = {
            "total_files": len(results),
            "successful_extractions": len(successful_extractions),
            "failed_extractions": len(failed_extractions),
            "success_rate": (
                len(successful_extractions) / len(results) if results else 0
            ),
            "total_pages": sum(r.get("total_pages", 0) for r in successful_extractions),
            "total_text_length": sum(
                len("\n".join([t.get("content", "") for t in r.get("text", [])]))
                for r in successful_extractions
            ),
            "total_tables": sum(
                len(r.get("tables", [])) for r in successful_extractions
            ),
            "total_images": sum(
                len(r.get("images", [])) for r in successful_extractions
            ),
            "file_size_stats": {
                "min": (
                    min(r.get("file_size", 0) for r in successful_extractions)
                    if successful_extractions
                    else 0
                ),
                "max": (
                    max(r.get("file_size", 0) for r in successful_extractions)
                    if successful_extractions
                    else 0
                ),
                "avg": (
                    sum(r.get("file_size", 0) for r in successful_extractions)
                    / len(successful_extractions)
                    if successful_extractions
                    else 0
                ),
            },
            "strategy_usage": {},
            "config_source": "centralized",
            "memory_usage": memory_manager.get_memory_report(),
        }

        # Analyze strategy usage
        for result in successful_extractions:
            strategy = result.get("extraction_strategy", "unknown")
            stats["strategy_usage"][strategy] = (
                stats["strategy_usage"].get(strategy, 0) + 1
            )

        return stats

    def cleanup(self) -> None:
        """Clean up extractor resources"""
        # Clean up intermediate files if requested
        if self.save_intermediate and self.intermediate_dir.exists():
            try:
                for filepath in self.intermediate_dir.glob("*_chunk_*.json"):
                    filepath.unlink()
                self.logger.info("Cleaned up intermediate files")
            except Exception as e:
                self.logger.error(f"Error cleaning up intermediate files: {e}")

        self.logger.info("PDF extractor cleaned up")
