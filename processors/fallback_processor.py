#!/usr/bin/env python3
"""
Fallback Processor Implementation
--------------------------------
A simple fallback processor that extracts basic text from documents
when other processors are not available.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger("FallbackProcessor")


class SimpleFallbackProcessor:
    """A simple fallback processor for when other processors are not available."""

    def __init__(self):
        """Initialize the fallback processor."""
        self.name = "fallback"
        logger.info("Initialized fallback processor")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get processor capabilities.

        Returns:
            Dictionary with supported content types and file types
        """
        return {"content_types": ["text"], "file_types": ["pdf", "text", "document"]}

    def process(
        self, file_path: str, page_indices: Optional[List[int]] = None, progress_callback = None
    ) -> Dict[str, Any]:
        """Process a document file with basic text extraction.

        Args:
            file_path: Path to the document file
            page_indices: Optional list of page indices to process

        Returns:
            Dictionary with processing results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {}

        results = {}

        # Handle different file types
        file_extension = file_path.suffix.lower()
        
        if progress_callback:
            progress_callback("processing_start")

        try:
            if file_extension == ".pdf":
                results = self._process_pdf(file_path, page_indices, progress_callback)
            elif file_extension in [".txt", ".md", ".rst"]:
                results = self._process_text_file(file_path)
            elif file_extension in [".docx", ".doc"]:
                results = self._process_doc_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                results = {
                    "1": {
                        "page_number": 1,
                        "content": f"Unsupported file type: {file_extension}",
                    }
                }
        except Exception as e:
            logger.error(f"Error processing document with fallback processor: {e}")
            results = {
                "1": {
                    "page_number": 1,
                    "content": f"Error processing document: {str(e)}",
                }
            }

        return results

    def _process_pdf(
        self, pdf_path: Path, page_indices: Optional[List[int]] = None, progress_callback = None
    ) -> Dict[str, Any]:
        """Process a PDF file with basic text extraction.

        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process

        Returns:
            Dictionary with processing results
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))

            # Determine which pages to process
            if page_indices:
                pages_to_process = [
                    p - 1 for p in page_indices if 0 < p <= len(doc)
                ]  # Convert to 0-based
            else:
                pages_to_process = range(len(doc))

            results = {}
            total_pages = len(pages_to_process)
            for i, page_idx in enumerate(pages_to_process):
                if progress_callback:
                    progress_callback("processing_page", i+1, total_pages)
                try:
                    page = doc[page_idx]
                    page_num = page_idx + 1  # Convert to 1-based

                    # Extract text
                    text = page.get_text()

                    # Store results
                    results[str(page_num)] = {
                        "page_number": page_num,
                        "text": text,
                        "content": text,
                    }

                    logger.info(f"Extracted {len(text)} chars from page {page_num}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_idx+1}: {e}")

            return results
        except ImportError:
            logger.error("PyMuPDF (fitz) not available for PDF processing")
            return {
                "1": {
                    "page_number": 1,
                    "content": "PyMuPDF not available for PDF processing",
                }
            }

    def _process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a text file.

        Args:
            file_path: Path to the text file

        Returns:
            Dictionary with processing results
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            return {"1": {"page_number": 1, "text": text, "content": text}}
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return {
                "1": {"page_number": 1, "content": f"Error reading text file: {str(e)}"}
            }

    def _process_doc_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a Word document.

        Args:
            file_path: Path to the Word document

        Returns:
            Dictionary with processing results
        """
        try:
            from docx import Document

            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

            return {"1": {"page_number": 1, "text": text, "content": text}}
        except ImportError:
            logger.error("python-docx not available for Word document processing")
            return {
                "1": {
                    "page_number": 1,
                    "content": "python-docx not available for Word document processing",
                }
            }
        except Exception as e:
            logger.error(f"Error reading Word document: {e}")
            return {
                "1": {
                    "page_number": 1,
                    "content": f"Error reading Word document: {str(e)}",
                }
            }

    def save_results(
        self, results: Dict[str, Any], output_dir: str, base_name: str
    ) -> None:
        """Save processing results.

        Args:
            results: Processing results
            output_dir: Directory to save results
            base_name: Base name for output files
        """
        # Create output directory
        processor_dir = Path(output_dir) / f"{base_name}_{self.name}"
        processor_dir.mkdir(parents=True, exist_ok=True)

        # Save combined results
        combined_md = processor_dir / f"{base_name}_combined.md"
        combined_json = processor_dir / f"{base_name}_combined.json"

        # Save markdown
        with open(combined_md, "w", encoding="utf-8") as f:
            f.write(f"# {base_name} - {self.name} Results\n\n")
            for page_num, page_data in sorted(
                results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]
            ):
                if isinstance(page_data, dict) and "content" in page_data:
                    f.write(f"## Page {page_num}\n\n")
                    f.write(page_data["content"])
                    f.write("\n\n---\n\n")

        # Save JSON
        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save individual page results
        pages_dir = processor_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        for page_num, page_data in results.items():
            if isinstance(page_data, dict) and "content" in page_data:
                # Save as markdown
                md_path = pages_dir / f"{base_name}_page{page_num}_content.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"# Page {page_num}\n\n")
                    f.write(page_data["content"])

                # Save as JSON
                json_path = pages_dir / f"{base_name}_page{page_num}_content.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(page_data, f, indent=2, ensure_ascii=False)
