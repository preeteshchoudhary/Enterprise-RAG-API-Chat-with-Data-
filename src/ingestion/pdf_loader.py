"""
PDF Ingestion Engine for complex financial documents.
Parses page-by-page content and extracts document section headers.
"""

import io
import re
import hashlib
from typing import List, Optional
from dataclasses import dataclass
import pypdf


@dataclass
class ParsedPage:
    page_number: int
    text: str
    detected_header: str
    doc_id: str
    filename: str


class PDFLoader:
    def __init__(self) -> None:
        # Regex patterns for common financial section headers (10-K, 10-Q, Annual Reports)
        self.header_pattern = re.compile(
            r"^(?:ITEM\s+\d+[A-Z]?[\.\:]?|NOTE\s+\d+[\.\:]?|SECTION\s+\d+|EXECUTIVE\s+SUMMARY|MANAGEMENT'?S\s+DISCUSSION|FINANCIAL\s+STATEMENTS|RISK\s+FACTORS|CONSOLIDATED\s+BALANCE\s+SHEETS)",
            re.IGNORECASE,
        )

    def generate_doc_id(self, file_content: bytes, filename: str) -> str:
        """Generates a deterministic SHA256 document hash."""
        return hashlib.sha256(file_content + filename.encode()).hexdigest()[:16]

    def load_pdf_bytes(self, file_content: bytes, filename: str) -> List[ParsedPage]:
        """Parses a PDF byte stream into structured ParsedPage objects."""
        doc_id = self.generate_doc_id(file_content, filename)
        pdf_file = io.BytesIO(file_content)
        reader = pypdf.PdfReader(pdf_file)
        
        parsed_pages: List[ParsedPage] = []
        current_header = "General Document Overview"

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            extracted_text = page.extract_text() or ""
            
            # Exclusion filter: skip pages that contain practice questions or data dictionary tests
            lower_text = extracted_text.lower()
            if "data dictionary & analysis test" in lower_text or "practice questions" in lower_text:
                continue
            
            # Detect section headers within the page text
            lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]
            for line in lines[:5]:  # Inspect first few lines of page
                if self.header_pattern.search(line) or (line.isupper() and len(line) < 80):
                    current_header = line.strip()
                    break

            parsed_pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=extracted_text,
                    detected_header=current_header,
                    doc_id=doc_id,
                    filename=filename,
                )
            )

        return parsed_pages

    def load_pdf_file(self, file_path: str) -> List[ParsedPage]:
        """Loads a PDF directly from a file path."""
        with open(file_path, "rb") as f:
            content = f.read()
        filename = file_path.split("/")[-1]
        return self.load_pdf_bytes(content, filename)
