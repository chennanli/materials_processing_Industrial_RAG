#!/usr/bin/env python3
"""
Gemini Processor Implementation
------------------------------
Process PDFs using Google's Gemini Vision API.
"""

import logging

# Import from parent directory
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from config import GEMINI_API_KEY, GEMINI_MODEL

# Configure logging
logger = logging.getLogger("GeminiProcessor")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in config.py")


class GeminiProcessor:
    """Process PDFs using Google's Gemini Vision API."""

    def __init__(self, model_name=GEMINI_MODEL):
        """Initialize the Gemini processor.

        Args:
            model_name: Gemini model name to use
        """
        self.name = "gemini"
        self.model_name = model_name

        # Check if API key is configured
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in config.py")

    def process(
        self, pdf_path: str, page_indices: Optional[List[int]] = None, progress_callback = None
    ) -> Dict[str, Any]:
        """Process a PDF file with Gemini Vision API.

        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)

        Returns:
            Dictionary with processing results
        """
        # Extract pages as images
        page_images = self._extract_pages_to_images(pdf_path, page_indices)

        if not page_images:
            logger.warning(f"No valid pages extracted from {pdf_path}")
            return {}

        # Process each page
        results = {}
        total_pages = len(page_images)
        for i, (page_num, image) in enumerate(page_images):
            logger.info(f"Processing page {page_num} with Gemini ({i+1}/{total_pages})")
            if progress_callback:
                progress_callback("processing_page", i+1, total_pages)

            # Process with Gemini using optimized prompt
            prompt = self._get_optimized_gemini_prompt(page_num)
            text_content = self._process_image_with_gemini(image, prompt)

            # Store results
            results[str(page_num)] = {"page_number": page_num, "content": text_content}

        return results

    def _get_optimized_gemini_prompt(self, page_num: int) -> str:
        """Get optimized prompt for Gemini Vision API.

        Args:
            page_num: Page number being processed

        Returns:
            Optimized prompt for Gemini
        """
        return f"""You are an expert document analyst with advanced vision capabilities. Analyze this document image (page {page_num}) comprehensively using Google's state-of-the-art vision understanding.

**Primary Task: Complete Document Analysis**
- Extract ALL text content with perfect accuracy
- Preserve exact formatting, spacing, and document structure
- Maintain reading order and hierarchy

**Advanced Capabilities:**
- Convert tables to well-formatted markdown with proper alignment
- Identify and describe charts, graphs, diagrams, and visual elements
- Recognize document structure (headers, sections, footnotes, captions)
- Preserve mathematical formulas, equations, and special notation
- Identify stamps, signatures, handwritten annotations, or markings
- Note color coding, highlighting, or visual emphasis
- Describe images, logos, and graphical elements

**Visual Understanding:**
- Analyze document layout and design elements
- Identify relationships between text and visual components
- Recognize document type and purpose from visual cues
- Extract visible metadata (dates, page numbers, watermarks)
- Note any quality issues or scanning artifacts

**Output Format:**
```markdown
# Page {page_num} Analysis

## Text Content
[Complete text extraction in reading order with preserved formatting]

## Tables
[All tables converted to clean markdown format]

## Visual Elements
[Detailed descriptions of charts, graphs, images, diagrams]

## Document Structure
[Headers, sections, layout observations, formatting notes]

## Additional Observations
[Any other relevant details, quality notes, or special elements]
```

Leverage Google Gemini's advanced multimodal understanding to provide the most comprehensive and accurate document analysis possible. Focus on both precision in text extraction and rich understanding of visual elements."""

    def _extract_pages_to_images(
        self, pdf_path: str, page_indices: Optional[List[int]] = None, dpi: int = 300
    ) -> List[tuple]:
        """Extract pages from PDF as images.

        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)
            dpi: DPI for image extraction

        Returns:
            List of tuples (page_number, PIL.Image)
        """
        doc = fitz.open(pdf_path)

        # Determine which pages to process
        if page_indices:
            pages_to_process = [
                p - 1 for p in page_indices if 0 < p <= len(doc)
            ]  # Convert to 0-based
        else:
            pages_to_process = range(len(doc))

        if not pages_to_process:
            logger.warning(f"No valid pages to process in {pdf_path}")
            return []

        images = []
        for page_idx in pages_to_process:
            try:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((page_idx + 1, img))  # Store as (page_number, image)
            except Exception as e:
                logger.error(f"Error extracting page {page_idx+1}: {e}")

        return images

    def _process_image_with_gemini(self, image: Image.Image, prompt: str) -> str:
        """Process an image with Gemini Vision API.

        Args:
            image: PIL Image to process
            prompt: Prompt to send to Gemini

        Returns:
            Extracted text content
        """
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {e}")
            return f"Error: {str(e)}"

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
            for page_num, page_data in sorted(results.items()):
                if isinstance(page_data, dict) and "content" in page_data:
                    f.write(f"## Page {page_num}\n\n")
                    f.write(page_data["content"])
                    f.write("\n\n---\n\n")

        # Save JSON
        with open(combined_json, "w", encoding="utf-8") as f:
            import json

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
                    import json

                    json.dump(page_data, f, indent=2, ensure_ascii=False)
