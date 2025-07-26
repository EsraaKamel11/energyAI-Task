import os
from unstructured.partition.pdf import partition_pdf
from typing import List, Dict, Any
from .base_collector import BaseCollector


class UnstructuredExtractor(BaseCollector):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)

    def collect(self, pdf_paths: List[str]) -> None:
        for pdf_path in pdf_paths:
            self.logger.info(f"Extracting PDF with unstructured: {pdf_path}")
            try:
                elements = partition_pdf(filename=pdf_path)
                # Extract text, tables, and sections
                data = [el.to_dict() for el in elements]
                metadata = self._extract_metadata(pdf_path)
                filename = self._sanitize_filename(pdf_path) + ".json"
                self.save_data(data, metadata, filename)
            except Exception as e:
                self.logger.error(f"Error extracting {pdf_path} with unstructured: {e}")

    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        return {"source": pdf_path}

    def _sanitize_filename(self, path: str) -> str:
        import re

        return re.sub(r"[^a-zA-Z0-9]", "_", os.path.basename(path))

    def resume(self, pdf_paths: List[str]) -> None:
        extracted = set()
        for fname in os.listdir(self.output_dir):
            if fname.endswith(".json"):
                extracted.add(fname.replace(".json", ""))
        to_extract = [
            p for p in pdf_paths if self._sanitize_filename(p) not in extracted
        ]
        self.logger.info(
            f"Resuming unstructured extraction. {len(to_extract)} PDFs left."
        )
        self.collect(to_extract)
