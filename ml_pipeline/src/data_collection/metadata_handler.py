"""
Metadata Handler for ML Pipeline
Handles metadata extraction, validation, and integration with data sources
"""

import logging
import json
import hashlib
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Import configuration manager
from src.utils.config_manager import get_config

# Import memory management
from src.utils.memory_manager import memory_manager, memory_safe


@dataclass
class Metadata:
    """Standardized metadata structure"""

    # Basic identification
    source_id: str
    source_type: str  # 'web', 'pdf', 'database', 'api'
    url: Optional[str] = None
    file_path: Optional[str] = None

    # Content metadata
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    language: Optional[str] = None
    content_type: Optional[str] = None

    # Temporal metadata
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None

    # Source metadata
    author: Optional[str] = None
    publisher: Optional[str] = None
    domain: Optional[str] = None
    site_name: Optional[str] = None

    # Technical metadata
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    encoding: Optional[str] = None

    # Quality metrics
    quality_score: Optional[float] = None
    completeness_score: Optional[float] = None
    freshness_score: Optional[float] = None

    # Custom fields
    custom_fields: Optional[Dict[str, Any]] = None

    # Processing metadata
    processing_status: str = "pending"  # pending, processing, completed, failed
    processing_errors: Optional[List[str]] = None
    processing_notes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary"""
        result = asdict(self)

        # Convert datetime objects to ISO format strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, list) and value and isinstance(value[0], datetime):
                result[key] = [
                    v.isoformat() if isinstance(v, datetime) else v for v in value
                ]

        return result

    def to_json(self) -> str:
        """Convert metadata to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __post_init__(self):
        """Ensure all datetimes are UTC-aware"""
        datetime_fields = ["created_at", "modified_at", "scraped_at"]
        for field in datetime_fields:
            value = getattr(self, field)
            if value and value.tzinfo is None:
                setattr(self, field, value.replace(tzinfo=timezone.utc))


class MetadataHandler:
    """Centralized metadata handling and validation"""

    def __init__(self):
        # Set default timezone to UTC
        os.environ["TZ"] = "UTC"

        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self.config = get_config()
        metadata_config = self.config.get("metadata", {})

        # Initialize settings
        self.required_fields = metadata_config.get(
            "required_fields", ["source_id", "source_type"]
        )
        self.quality_thresholds = metadata_config.get(
            "quality_thresholds",
            {
                "min_title_length": 10,
                "min_description_length": 50,
                "min_content_length": 100,
            },
        )

        # Initialize extractors
        self.extractors = {
            "web": WebMetadataExtractor(),
            "pdf": PDFMetadataExtractor(),
            "api": APIMetadataExtractor(),
            "database": DatabaseMetadataExtractor(),
        }

        self.logger.info("Metadata handler initialized")

    def create_metadata(
        self, source_type: str, source_data: Dict[str, Any]
    ) -> Metadata:
        """
        Create standardized metadata from source data

        Args:
            source_type: Type of source ('web', 'pdf', 'api', 'database')
            source_data: Raw source data

        Returns:
            Metadata object
        """
        try:
            # Get appropriate extractor
            extractor = self.extractors.get(source_type)
            if not extractor:
                raise ValueError(f"Unsupported source type: {source_type}")

            # Extract metadata using source-specific extractor
            metadata_dict = extractor.extract(source_data)

            # Add common fields
            metadata_dict.update(
                {
                    "source_type": source_type,
                    "scraped_at": datetime.now(timezone.utc),
                    "processing_status": "pending",
                }
            )

            # Generate source ID if not provided
            if "source_id" not in metadata_dict:
                metadata_dict["source_id"] = self._generate_source_id(
                    source_type, source_data
                )

            # Create metadata object
            metadata = Metadata(**metadata_dict)

            # Validate metadata
            self._validate_metadata(metadata)

            # Calculate quality scores
            self._calculate_quality_scores(metadata)

            return metadata

        except Exception as e:
            self.logger.error(f"Error creating metadata: {e}")
            return self._create_error_metadata(source_type, source_data, str(e))

    def _generate_source_id(self, source_type: str, source_data: Dict[str, Any]) -> str:
        """Generate unique source ID"""
        if source_type == "web" and "url" in source_data:
            # Use URL hash for web sources
            url_hash = hashlib.md5(source_data["url"].encode()).hexdigest()[:12]
            return f"web_{url_hash}"
        elif source_type == "pdf" and "file_path" in source_data:
            # Use file path hash for PDF sources
            file_hash = hashlib.md5(str(source_data["file_path"]).encode()).hexdigest()[
                :12
            ]
            return f"pdf_{file_hash}"
        else:
            # Generate timestamp-based ID
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return f"{source_type}_{timestamp}"

    def _validate_metadata(self, metadata: Metadata) -> None:
        """Validate metadata completeness and quality"""
        errors = []

        # Check required fields
        for field in self.required_fields:
            if not getattr(metadata, field, None):
                errors.append(f"Missing required field: {field}")

        # Check quality thresholds
        if (
            metadata.title
            and len(metadata.title) < self.quality_thresholds["min_title_length"]
        ):
            errors.append(f"Title too short: {len(metadata.title)} characters")

        if (
            metadata.description
            and len(metadata.description)
            < self.quality_thresholds["min_description_length"]
        ):
            errors.append(
                f"Description too short: {len(metadata.description)} characters"
            )

        if (
            metadata.word_count
            and metadata.word_count < self.quality_thresholds["min_content_length"]
        ):
            errors.append(f"Content too short: {metadata.word_count} words")

        # Update metadata with errors
        if errors:
            metadata.processing_errors = errors
            metadata.processing_status = "failed"
            self.logger.warning(f"Metadata validation failed: {errors}")
        else:
            metadata.processing_status = "completed"

    def _calculate_quality_scores(self, metadata: Metadata) -> None:
        """Calculate quality scores for metadata"""
        scores = {}

        # Completeness score
        completeness_fields = [
            "title",
            "description",
            "author",
            "keywords",
            "created_at",
            "modified_at",
            "content_type",
        ]
        filled_fields = sum(
            1 for field in completeness_fields if getattr(metadata, field)
        )
        scores["completeness"] = filled_fields / len(completeness_fields)

        # Quality score based on content length
        if metadata.word_count:
            if metadata.word_count > 1000:
                scores["quality"] = 1.0
            elif metadata.word_count > 500:
                scores["quality"] = 0.8
            elif metadata.word_count > 100:
                scores["quality"] = 0.6
            else:
                scores["quality"] = 0.3
        else:
            scores["quality"] = 0.5

        # Freshness score - FIX APPLIED HERE
        if metadata.modified_at:
            # Ensure both datetimes are timezone-aware
            now = datetime.now(timezone.utc)

            # Convert modified_at to UTC if it's naive
            modified_at = metadata.modified_at
            if (
                modified_at.tzinfo is None
                or modified_at.tzinfo.utcoffset(modified_at) is None
            ):
                modified_at = modified_at.replace(tzinfo=timezone.utc)
            else:
                modified_at = modified_at.astimezone(timezone.utc)

            days_old = (now - modified_at).days
            if days_old < 30:
                scores["freshness"] = 1.0
            elif days_old < 90:
                scores["freshness"] = 0.8
            elif days_old < 365:
                scores["freshness"] = 0.6
            else:
                scores["freshness"] = 0.3
        else:
            scores["freshness"] = 0.5

        # Update metadata
        metadata.quality_score = scores.get("quality", 0.5)
        metadata.completeness_score = scores.get("completeness", 0.5)
        metadata.freshness_score = scores.get("freshness", 0.5)

    def _create_error_metadata(
        self, source_type: str, source_data: Dict[str, Any], error: str
    ) -> Metadata:
        """Create metadata object for failed extractions"""
        return Metadata(
            source_id=self._generate_source_id(source_type, source_data),
            source_type=source_type,
            processing_status="failed",
            processing_errors=[error],
            scraped_at=datetime.now(timezone.utc),
        )

    def save_metadata(self, metadata: Metadata, output_path: str) -> None:
        """Save metadata to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dictionary
            metadata_dict = asdict(metadata)

            # Convert datetime objects to ISO strings
            for key, value in metadata_dict.items():
                if isinstance(value, datetime):
                    metadata_dict[key] = value.isoformat()

            # Save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metadata saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def load_metadata(self, input_path: str) -> Metadata:
        """Load metadata from file"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            return self.load_metadata_from_dict(metadata_dict)

        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            raise

    def load_metadata_from_dict(self, metadata_dict: Dict[str, Any]) -> Metadata:
        """Load metadata from dictionary"""
        try:
            # Convert ISO strings back to datetime objects
            datetime_fields = ["created_at", "modified_at", "scraped_at"]
            for field in datetime_fields:
                if field in metadata_dict and metadata_dict[field]:
                    # FIX: Handle both naive and aware datetimes
                    dt_str = metadata_dict[field]
                    if dt_str.endswith("Z"):
                        dt_str = dt_str[:-1] + "+00:00"
                    metadata_dict[field] = datetime.fromisoformat(dt_str)

            return Metadata(**metadata_dict)

        except Exception as e:
            self.logger.error(f"Error loading metadata from dict: {e}")
            raise

    def batch_process_metadata(
        self, source_data_list: List[Dict[str, Any]]
    ) -> List[Metadata]:
        """Process multiple sources and create metadata"""
        metadata_list = []

        for source_data in source_data_list:
            try:
                source_type = source_data.get("source_type", "web")
                metadata = self.create_metadata(source_type, source_data)
                metadata_list.append(metadata)

            except Exception as e:
                self.logger.error(f"Error processing source: {e}")
                continue

        return metadata_list

    def export_metadata_report(
        self, metadata_list: List[Metadata], output_path: str
    ) -> None:
        """Export metadata as a comprehensive report"""
        try:
            # Convert to DataFrame
            metadata_dicts = [asdict(metadata) for metadata in metadata_list]

            # Convert datetime objects
            for md_dict in metadata_dicts:
                for key, value in md_dict.items():
                    if isinstance(value, datetime):
                        md_dict[key] = value.isoformat()
                    elif isinstance(value, list):
                        md_dict[key] = ", ".join(str(item) for item in value)

            df = pd.DataFrame(metadata_dicts)

            # Save as CSV
            csv_path = Path(output_path).with_suffix(".csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")

            # Generate summary report
            summary = self._generate_summary_report(metadata_list)
            summary_path = Path(output_path).with_suffix(".summary.json")

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Metadata report exported to {csv_path} and {summary_path}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting metadata report: {e}")

    def _generate_summary_report(self, metadata_list: List[Metadata]) -> Dict[str, Any]:
        """Generate summary statistics for metadata"""
        if not metadata_list:
            return {"error": "No metadata to summarize"}

        # Basic statistics
        total_sources = len(metadata_list)
        successful_sources = sum(
            1 for md in metadata_list if md.processing_status == "completed"
        )
        failed_sources = sum(
            1 for md in metadata_list if md.processing_status == "failed"
        )

        # Source type distribution
        source_types = {}
        for md in metadata_list:
            source_types[md.source_type] = source_types.get(md.source_type, 0) + 1

        # Quality statistics
        quality_scores = [
            md.quality_score for md in metadata_list if md.quality_score is not None
        ]
        completeness_scores = [
            md.completeness_score
            for md in metadata_list
            if md.completeness_score is not None
        ]
        freshness_scores = [
            md.freshness_score for md in metadata_list if md.freshness_score is not None
        ]

        # FIX: Convert timedelta to days for serialization
        processing_times = []
        for md in metadata_list:
            if md.created_at and md.scraped_at:
                # Ensure both are timezone-aware
                created = md.created_at
                scraped = md.scraped_at

                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if scraped.tzinfo is None:
                    scraped = scraped.replace(tzinfo=timezone.utc)

                processing_times.append((scraped - created).days)

        summary = {
            "total_sources": total_sources,
            "successful_sources": successful_sources,
            "failed_sources": failed_sources,
            "success_rate": (
                successful_sources / total_sources if total_sources > 0 else 0
            ),
            "source_type_distribution": source_types,
            "quality_statistics": {
                "average_quality_score": (
                    sum(quality_scores) / len(quality_scores) if quality_scores else 0
                ),
                "average_completeness_score": (
                    sum(completeness_scores) / len(completeness_scores)
                    if completeness_scores
                    else 0
                ),
                "average_freshness_score": (
                    sum(freshness_scores) / len(freshness_scores)
                    if freshness_scores
                    else 0
                ),
                "min_quality_score": min(quality_scores) if quality_scores else 0,
                "max_quality_score": max(quality_scores) if quality_scores else 0,
            },
            "processing_times": {
                "average_days": (
                    sum(processing_times) / len(processing_times)
                    if processing_times
                    else 0
                ),
                "min_days": min(processing_times) if processing_times else 0,
                "max_days": max(processing_times) if processing_times else 0,
            },
            "processing_errors": self._collect_processing_errors(metadata_list),
        }

        return summary

    def _collect_processing_errors(
        self, metadata_list: List[Metadata]
    ) -> Dict[str, int]:
        """Collect and count processing errors"""
        error_counts = {}

        for metadata in metadata_list:
            if metadata.processing_errors:
                for error in metadata.processing_errors:
                    error_counts[error] = error_counts.get(error, 0) + 1

        return error_counts


class WebMetadataExtractor:
    """Extract metadata from web sources"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @memory_safe
    def extract(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from web source data"""
        try:
            url = source_data.get("url")
            if not url:
                raise ValueError("URL is required for web metadata extraction")

            # Parse URL
            parsed_url = urlparse(url)

            # Extract basic metadata
            metadata = {
                "url": url,
                "domain": parsed_url.netloc,
                "source_id": self._generate_web_id(url),
            }

            # Extract content metadata if available
            if "content" in source_data:
                content = source_data["content"]
                metadata.update(self._extract_content_metadata(content))

            # Extract HTML metadata if available
            if "html" in source_data:
                html = source_data["html"]
                metadata.update(self._extract_html_metadata(html))

            # Extract response metadata if available
            if "response" in source_data:
                response = source_data["response"]
                metadata.update(self._extract_response_metadata(response))

            # Extract custom fields
            if "custom_fields" in source_data:
                metadata["custom_fields"] = source_data["custom_fields"]

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting web metadata: {e}")
            return {"error": str(e)}

    def _generate_web_id(self, url: str) -> str:
        """Generate unique ID for web source"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"web_{url_hash}"

    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        metadata = {}

        # Basic content analysis
        metadata["word_count"] = len(content.split())
        metadata["content_type"] = "text/html"

        # Extract title from content (simple heuristic)
        lines = content.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.startswith("http"):
                metadata["title"] = line
                break

        # Extract description (first paragraph)
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50 and len(para) < 500:
                metadata["description"] = para
                break

        return metadata

    def _extract_html_metadata(self, html: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {}

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text(strip=True)

            # Extract meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                metadata["description"] = meta_desc.get("content", "")

            # Extract meta keywords
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords:
                keywords = meta_keywords.get("content", "")
                metadata["keywords"] = [
                    kw.strip() for kw in keywords.split(",") if kw.strip()
                ]

            # Extract language
            html_tag = soup.find("html")
            if html_tag:
                metadata["language"] = html_tag.get("lang", "en")

            # Extract author
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author:
                metadata["author"] = meta_author.get("content", "")

            # Extract site name
            meta_site_name = soup.find("meta", attrs={"property": "og:site_name"})
            if meta_site_name:
                metadata["site_name"] = meta_site_name.get("content", "")

            # Extract creation/modification dates
            meta_created = soup.find("meta", attrs={"name": "article:published_time"})
            if meta_created:
                try:
                    dt_str = meta_created.get("content", "")
                    if dt_str.endswith("Z"):
                        dt_str = dt_str[:-1] + "+00:00"
                    metadata["created_at"] = datetime.fromisoformat(dt_str)
                except:
                    pass

            meta_modified = soup.find("meta", attrs={"name": "article:modified_time"})
            if meta_modified:
                try:
                    dt_str = meta_modified.get("content", "")
                    if dt_str.endswith("Z"):
                        dt_str = dt_str[:-1] + "+00:00"
                    metadata["modified_at"] = datetime.fromisoformat(dt_str)
                except:
                    pass

        except Exception as e:
            self.logger.warning(f"Error parsing HTML metadata: {e}")

        return metadata

    def _extract_response_metadata(self, response: requests.Response) -> Dict[str, Any]:
        """Extract metadata from HTTP response"""
        metadata = {}

        # Content type
        content_type = response.headers.get("content-type", "")
        if content_type:
            metadata["content_type"] = content_type.split(";")[0]

        # Content length
        content_length = response.headers.get("content-length")
        if content_length:
            metadata["file_size"] = int(content_length)

        # Last modified
        last_modified = response.headers.get("last-modified")
        if last_modified:
            try:
                # Parse the datetime and make it timezone-aware
                dt = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
                # Assume UTC if no timezone info
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC if it has timezone info
                    dt = dt.astimezone(timezone.utc)
                metadata["modified_at"] = dt
            except:
                pass

        # Encoding
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1]
            metadata["encoding"] = charset

        return metadata


class PDFMetadataExtractor:
    """Extract metadata from PDF sources"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from PDF source data"""
        try:
            file_path = source_data.get("file_path")
            if not file_path:
                raise ValueError("File path is required for PDF metadata extraction")

            metadata = {
                "file_path": str(file_path),
                "source_id": self._generate_pdf_id(file_path),
                "content_type": "application/pdf",
            }

            # Extract PDF metadata if available
            if "pdf_metadata" in source_data:
                pdf_metadata = source_data["pdf_metadata"]
                metadata.update(self._extract_pdf_metadata(pdf_metadata))

            # Extract content metadata if available
            if "content" in source_data:
                content = source_data["content"]
                metadata.update(self._extract_content_metadata(content))

            # Extract file metadata
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                metadata["file_size"] = file_path_obj.stat().st_size
                metadata["created_at"] = datetime.fromtimestamp(
                    file_path_obj.stat().st_ctime, tz=timezone.utc
                )
                metadata["modified_at"] = datetime.fromtimestamp(
                    file_path_obj.stat().st_mtime, tz=timezone.utc
                )

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting PDF metadata: {e}")
            return {"error": str(e)}

    def _generate_pdf_id(self, file_path: str) -> str:
        """Generate unique ID for PDF source"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
        return f"pdf_{file_hash}"

    def _extract_pdf_metadata(self, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from PDF metadata"""
        metadata = {}

        # Map PDF metadata to standard fields
        field_mapping = {
            "title": "title",
            "author": "author",
            "subject": "description",
            "creator": "publisher",
            "producer": "publisher",
            "creationDate": "created_at",
            "modDate": "modified_at",
        }

        for pdf_field, standard_field in field_mapping.items():
            if pdf_field in pdf_metadata and pdf_metadata[pdf_field]:
                metadata[standard_field] = pdf_metadata[pdf_field]

        return metadata

    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from PDF content"""
        metadata = {}

        # Basic content analysis
        metadata["word_count"] = len(content.split())

        # Extract title from first few lines
        lines = content.split("\n")
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                metadata["title"] = line
                break

        return metadata


class APIMetadataExtractor:
    """Extract metadata from API sources"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from API source data"""
        try:
            endpoint = source_data.get("endpoint")
            if not endpoint:
                raise ValueError("Endpoint is required for API metadata extraction")

            metadata = {
                "url": endpoint,
                "source_id": self._generate_api_id(endpoint),
                "source_type": "api",
            }

            # Extract response metadata if available
            if "response" in source_data:
                response = source_data["response"]
                metadata.update(self._extract_response_metadata(response))

            # Extract API-specific metadata
            if "api_info" in source_data:
                api_info = source_data["api_info"]
                metadata.update(self._extract_api_info(api_info))

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting API metadata: {e}")
            return {"error": str(e)}

    def _generate_api_id(self, endpoint: str) -> str:
        """Generate unique ID for API source"""
        endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:12]
        return f"api_{endpoint_hash}"

    def _extract_response_metadata(self, response: requests.Response) -> Dict[str, Any]:
        """Extract metadata from API response"""
        metadata = {}

        # Content type
        content_type = response.headers.get("content-type", "")
        if content_type:
            metadata["content_type"] = content_type.split(";")[0]

        # Response size
        metadata["file_size"] = len(response.content)

        # Timestamp
        metadata["created_at"] = datetime.now(timezone.utc)

        return metadata

    def _extract_api_info(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from API information"""
        metadata = {}

        # Map API info to standard fields
        field_mapping = {
            "name": "title",
            "description": "description",
            "version": "custom_fields.version",
            "base_url": "custom_fields.base_url",
        }

        for api_field, standard_field in field_mapping.items():
            if api_field in api_info and api_info[api_field]:
                if "." in standard_field:
                    # Handle nested fields
                    main_field, sub_field = standard_field.split(".")
                    if main_field not in metadata:
                        metadata[main_field] = {}
                    metadata[main_field][sub_field] = api_info[api_field]
                else:
                    metadata[standard_field] = api_info[api_field]

        return metadata


class DatabaseMetadataExtractor:
    """Extract metadata from database sources"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from database source data"""
        try:
            table_name = source_data.get("table_name")
            if not table_name:
                raise ValueError(
                    "Table name is required for database metadata extraction"
                )

            metadata = {
                "source_id": self._generate_db_id(table_name),
                "source_type": "database",
                "title": f"Database Table: {table_name}",
            }

            # Extract database-specific metadata
            if "db_info" in source_data:
                db_info = source_data["db_info"]
                metadata.update(self._extract_db_info(db_info))

            # Extract schema metadata if available
            if "schema" in source_data:
                schema = source_data["schema"]
                metadata.update(self._extract_schema_metadata(schema))

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting database metadata: {e}")
            return {"error": str(e)}

    def _generate_db_id(self, table_name: str) -> str:
        """Generate unique ID for database source"""
        table_hash = hashlib.md5(table_name.encode()).hexdigest()[:12]
        return f"db_{table_hash}"

    def _extract_db_info(self, db_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from database information"""
        metadata = {}

        # Map database info to standard fields
        field_mapping = {
            "database_name": "custom_fields.database_name",
            "table_name": "custom_fields.table_name",
            "row_count": "custom_fields.row_count",
            "column_count": "custom_fields.column_count",
        }

        for db_field, standard_field in field_mapping.items():
            if db_field in db_info and db_info[db_field]:
                main_field, sub_field = standard_field.split(".")
                if main_field not in metadata:
                    metadata[main_field] = {}
                metadata[main_field][sub_field] = db_info[db_field]

        return metadata

    def _extract_schema_metadata(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from database schema"""
        metadata = {}

        # Extract column information
        if "columns" in schema:
            columns = schema["columns"]
            metadata["custom_fields"] = metadata.get("custom_fields", {})
            metadata["custom_fields"]["columns"] = columns

        return metadata
