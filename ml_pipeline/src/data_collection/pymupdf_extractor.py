import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from .base_collector import BaseCollector


class PyMuPDFExtractor(BaseCollector):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)

    def collect(self, pdf_paths: List[str]) -> None:
        for pdf_path in pdf_paths:
            self.logger.info(f"Extracting PDF with PyMuPDF: {pdf_path}")
            try:
                doc = fitz.open(pdf_path)
                pages = []
                for page in doc:
                    text = page.get_text("blocks")  # Layout-preserving
                    pages.append(text)
                metadata = self._extract_metadata(doc, pdf_path)
                filename = self._sanitize_filename(pdf_path) + ".json"
                self.save_data(pages, metadata, filename)
            except Exception as e:
                self.logger.error(f"Error extracting {pdf_path} with PyMuPDF: {e}")

    def _extract_metadata(self, doc, pdf_path: str) -> Dict[str, Any]:
        meta = {"source": pdf_path}
        try:
            meta.update(doc.metadata)
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        return meta

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
        self.logger.info(f"Resuming PyMuPDF extraction. {len(to_extract)} PDFs left.")
        self.collect(to_extract)
