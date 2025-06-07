#!/usr/bin/env python3
"""
LM Studio Processor Implementation
---------------------------------
Process PDFs using LM Studio for text extraction and table enhancement.
"""

import base64
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
import requests
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from config import LMSTUDIO_MODEL, LMSTUDIO_URL

# Configure logging
logger = logging.getLogger("LMStudioProcessor")


class LMStudioClient:
    """Client for interacting with LM Studio API."""

    def __init__(self, api_url=LMSTUDIO_URL, model=LMSTUDIO_MODEL):
        """Initialize the LM Studio client.

        Args:
            api_url: LM Studio API URL
            model: LM Studio model name
        """
        self.api_url = api_url
        self.model = model

    def generate(self, prompt, max_tokens=1024, temperature=0.1):
        """Generate text using LM Studio.

        Args:
            prompt: Prompt to send to LM Studio
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation

        Returns:
            Generated text
        """
        try:
            # First, check available models to confirm our model exists
            try:
                models_response = requests.get(f"{self.api_url}/v1/models", timeout=5)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [
                        model["id"] for model in models_data.get("data", [])
                    ]
                    logger.info(f"Available LMStudio models: {available_models}")

                    if self.model not in available_models:
                        logger.warning(
                            f"Model '{self.model}' not found in available models. Using first available model instead."
                        )
                        if available_models:
                            self.model = available_models[0]
                            logger.info(f"Using model: {self.model}")
                        else:
                            logger.error("No models available in LMStudio")
                            return None
                else:
                    logger.warning(
                        f"Could not get available models: {models_response.status_code} - {models_response.text}"
                    )
            except Exception as e:
                logger.warning(f"Error checking available models: {e}")

            # Use the chat completions endpoint with the correct format
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts and enhances text from documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            logger.info(f"Sending request to LMStudio with model: {self.model}")

            # Use the correct chat completions endpoint
            response = requests.post(
                f"{self.api_url}/v1/chat/completions", json=payload, timeout=60
            )

            if response.status_code != 200:
                logger.error(
                    f"LM Studio API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()
            logger.info(f"Received response from LMStudio: {result.keys()}")

            if "choices" in result and len(result["choices"]) > 0:
                # Extract content from the chat response format
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(
                    f"Successfully generated content with LMStudio (length: {len(content)})"
                )
                return content
            else:
                logger.error(f"Unexpected response format: {result}")
                return None

        except Exception as e:
            logger.error(f"Error calling LM Studio API: {e}")
            return None


class LMStudioProcessor:
    """Process PDFs using LM Studio for text extraction and table enhancement."""

    def __init__(self, api_url=LMSTUDIO_URL, model=LMSTUDIO_MODEL):
        """Initialize the LM Studio processor.

        Args:
            api_url: LM Studio API URL
            model: LM Studio model name
        """
        self.name = "lmstudio"
        self.api_url = api_url
        self.model = model
        self.lm_client = LMStudioClient(api_url, model)

        # Check connection
        if not self._check_connection():
            logger.warning(f"Could not connect to LM Studio at {api_url}")

    def _check_connection(self):
        """Check if LM Studio is running properly."""
        try:
            response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Could not connect to LM Studio: {e}")
            return False

    def process(
        self, pdf_path: str, page_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Process a PDF file with LM Studio.

        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)

        Returns:
            Dictionary with processing results
        """
        logger.info(f"LMStudio processor starting on {pdf_path}")

        # Check connection before proceeding
        connection_status = self._check_connection()
        logger.info(
            f"LMStudio connection status: {'Connected' if connection_status else 'Not connected'}"
        )

        if not connection_status:
            logger.warning(
                "LMStudio is not connected, falling back to basic text extraction"
            )
            # Fall back to basic text extraction if LMStudio is not available
            return self._extract_content_from_pdf(pdf_path, page_indices)

        # Extract pages as images (similar to Gemini approach)
        page_images = self._extract_pages_to_images(pdf_path, page_indices)

        if not page_images:
            logger.warning(f"No valid pages extracted from {pdf_path}")
            return {}

        logger.info(f"Extracted {len(page_images)} page images for OCR processing")

        # Process each page with LMStudio
        results = {}
        for page_num, image in page_images:
            logger.info(f"Processing page {page_num} with LMStudio")

            # Process image with LMStudio for OCR
            text_content = self._process_image_with_lmstudio(image, page_num)

            if text_content:
                # Store results
                results[str(page_num)] = {
                    "page_number": page_num,
                    "content": text_content,
                    "text": text_content,  # Also store as text for compatibility
                }
            else:
                logger.warning(
                    f"Failed to extract content from page {page_num}, using basic extraction"
                )
                # Fall back to basic extraction for this page
                basic_results = self._extract_single_page_content(pdf_path, page_num)
                if basic_results:
                    results[str(page_num)] = basic_results

        logger.info(
            f"LMStudio processor completed processing {pdf_path} with {len(results)} pages"
        )
        return results

    def _extract_content_from_pdf(
        self, pdf_path: str, page_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Extract text and tables from PDF.

        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)

        Returns:
            Dictionary with extracted content
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
            return {}

        results = {}
        for page_idx in pages_to_process:
            try:
                page = doc[page_idx]
                page_num = page_idx + 1  # Convert to 1-based

                # Extract text
                text = page.get_text()

                # Extract tables
                tables = []
                table_finder = page.find_tables()
                if table_finder and table_finder.tables:
                    for i, table in enumerate(table_finder.tables):
                        table_data = table.extract()
                        tables.append(
                            {
                                "table_id": f"page{page_num}_table{i+1}",
                                "data": table_data,
                                "markdown": self._table_to_markdown(table_data),
                            }
                        )

                # Store results
                results[str(page_num)] = {
                    "page_number": page_num,
                    "text": text,
                    "tables": tables,
                }

                logger.info(
                    f"Extracted content from page {page_num}: {len(text)} chars, {len(tables)} tables"
                )
            except Exception as e:
                logger.error(f"Error extracting content from page {page_idx+1}: {e}")

        return results

    def _table_to_markdown(self, table_data):
        """Convert table data to markdown format.

        Args:
            table_data: Table data as list of lists

        Returns:
            Table in markdown format
        """
        if not table_data:
            return ""

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Convert to markdown
        md = "| " + " | ".join(str(col) for col in df.iloc[0]) + " |\n"
        md += "| " + " | ".join("---" for _ in df.iloc[0]) + " |\n"

        for _, row in df.iloc[1:].iterrows():
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        return md

    def _enhance_table(self, table):
        """Enhance table with LM Studio.

        Args:
            table: Table data

        Returns:
            Enhanced table or original table if enhancement fails
        """
        if "markdown" not in table:
            logger.warning(
                f"Table {table.get('table_id', 'unknown')} has no markdown content to enhance"
            )
            return None

        if not self._check_connection():
            logger.warning("LMStudio connection failed, skipping table enhancement")
            return None

        prompt = f"""
        You are an expert at analyzing tables. Below is a table extracted from a PDF document.
        Please analyze this table and provide a cleaned, well-formatted version.
        Fix any alignment issues, merged cells, or extraction errors.

        Table:
        {table['markdown']}

        Cleaned table (output in markdown format):
        <answer>
        """

        try:
            logger.info(
                f"Attempting to enhance table {table.get('table_id', 'unknown')}"
            )
            result = self.lm_client.generate(prompt, max_tokens=2048)
            if result:
                logger.info(
                    f"Successfully enhanced table {table.get('table_id', 'unknown')}"
                )
                return {
                    "table_id": table["table_id"],
                    "markdown": result,
                    "enhanced": True,
                }
            logger.warning(
                f"Failed to enhance table {table.get('table_id', 'unknown')}: empty result"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error enhancing table {table.get('table_id', 'unknown')}: {e}"
            )
            return None

    def _enhance_text(self, text, page_num):
        """Enhance text content with LM Studio.

        Args:
            text: Text content to enhance
            page_num: Page number for logging

        Returns:
            Enhanced text or None if enhancement fails
        """
        if not text or not text.strip():
            logger.warning(f"No text content to enhance for page {page_num}")
            return None

        # Truncate text if it's too long to avoid token limits
        max_text_length = 4000  # Adjust based on model's context window
        if len(text) > max_text_length:
            logger.warning(
                f"Text for page {page_num} is too long ({len(text)} chars), truncating to {max_text_length} chars"
            )
            text = text[:max_text_length] + "\n[Content truncated due to length]\n"

        prompt = f"""
        You are an expert at OCR post-processing. Below is text extracted from a PDF document page {page_num}.
        Please clean and enhance this text by:
        1. Fixing OCR errors and typos
        2. Preserving the original formatting as much as possible
        3. Ensuring paragraphs are properly separated
        4. Maintaining any bullet points or numbered lists
        5. Preserving important data like dates, numbers, and measurements

        Original text:
        {text}

        Enhanced text:
        """

        try:
            logger.info(
                f"Sending text from page {page_num} to LMStudio for enhancement"
            )
            result = self.lm_client.generate(prompt, max_tokens=4096)
            if result:
                logger.info(f"Successfully enhanced text for page {page_num}")
                return result
            logger.warning(f"Failed to enhance text for page {page_num}: empty result")
            return None
        except Exception as e:
            logger.error(f"Error enhancing text for page {page_num}: {e}")
            return None

    def _extract_pages_to_images(
        self, pdf_path: str, page_indices: Optional[List[int]] = None, dpi: int = 300
    ) -> List[Tuple[int, Image.Image]]:
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

    def _process_image_with_lmstudio(self, image: Image.Image, page_num: int) -> str:
        """Process an image with LMStudio for OCR.

        Args:
            image: PIL Image to process
            page_num: Page number for logging

        Returns:
            Extracted text content
        """
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Create prompt for OCR
            prompt = f"""
            You are an expert at OCR and document analysis. Please extract all text and tables from this image.
            This is page {page_num} of a document. Format tables properly using markdown.
            Preserve the original layout as much as possible.
            """

            # Prepare the payload for LMStudio with the image
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts text and tables from images.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            }

            logger.info(
                f"Sending image for page {page_num} to LMStudio for OCR processing"
            )
            response = requests.post(
                f"{self.api_url}/v1/chat/completions", json=payload, timeout=120
            )

            if response.status_code != 200:
                logger.error(
                    f"LMStudio API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(
                    f"Successfully extracted content from page {page_num} image (length: {len(content)})"
                )
                return content
            else:
                logger.error(f"Unexpected response format from LMStudio: {result}")
                return None

        except Exception as e:
            logger.error(f"Error processing image with LMStudio: {e}")
            return None

    def _extract_single_page_content(
        self, pdf_path: str, page_num: int
    ) -> Dict[str, Any]:
        """Extract content from a single page as fallback.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract (1-based)

        Returns:
            Dictionary with page content
        """
        try:
            doc = fitz.open(pdf_path)
            if 0 < page_num <= len(doc):
                page = doc[page_num - 1]  # Convert to 0-based

                # Extract text
                text = page.get_text()

                # Extract tables
                tables = []
                table_finder = page.find_tables()
                if table_finder and table_finder.tables:
                    for i, table in enumerate(table_finder.tables):
                        table_data = table.extract()
                        tables.append(
                            {
                                "table_id": f"page{page_num}_table{i+1}",
                                "data": table_data,
                                "markdown": self._table_to_markdown(table_data),
                            }
                        )

                # Format content
                content = text
                for table in tables:
                    if "markdown" in table:
                        content += f"\n\n**Table:**\n\n{table['markdown']}"

                return {
                    "page_number": page_num,
                    "text": text,
                    "tables": tables,
                    "content": content,
                }
            return None
        except Exception as e:
            logger.error(f"Error extracting content from page {page_num}: {e}")
            return None

    def _format_page_content(self, page_data):
        """Format page content for output.

        Args:
            page_data: Page data

        Returns:
            Formatted content
        """
        # If content is already formatted (from OCR), return it directly
        if "content" in page_data and page_data["content"]:
            return page_data["content"]

        content = []

        # Prioritize enhanced text if available
        if "enhanced_text" in page_data and page_data["enhanced_text"]:
            content.append(page_data["enhanced_text"])
        elif "text" in page_data and page_data["text"]:
            content.append(page_data["text"])

        # Add tables
        if "tables" in page_data and page_data["tables"]:
            for table in page_data["tables"]:
                if "markdown" in table:
                    content.append("\n\n**Table:**\n\n" + table["markdown"])

        # Add enhanced tables
        if "enhanced_tables" in page_data and page_data["enhanced_tables"]:
            for table in page_data["enhanced_tables"]:
                if "markdown" in table:
                    content.append("\n\n**Enhanced Table:**\n\n" + table["markdown"])

        return "\n\n".join(content)

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
