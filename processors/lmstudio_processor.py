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
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
import requests
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LMSTUDIO_MODEL,
    LMSTUDIO_PREFERRED_OCR_MODEL,
    LMSTUDIO_PREFERRED_VISION_MODEL,
    LMSTUDIO_URL,
)

# Import modular components
from .lmstudio_modules.table_formatter import TableFormatter
from .lmstudio_modules.parallel_processor import ParallelProcessor
from .lmstudio_modules.image_processor import ImageProcessor

# Configure logging
logger = logging.getLogger("LMStudioProcessor")


class LMStudioClient:
    """Client for interacting with LM Studio API."""

    def __init__(self, api_url=LMSTUDIO_URL, model=LMSTUDIO_MODEL):
        """Initialize the LM Studio client.

        Args:
            api_url: LM Studio API URL
            model: LM Studio model name (fallback only)
        """
        self.api_url = api_url
        self.model = model  # Fallback model only
        self.current_model = None
        self.model_capabilities = {}
        self._connection_cache = None
        self._cache_time = 0
        self._cache_duration = 30  # Cache connection status for 30 seconds
        self._processed_pages = set()  # Track processed pages to avoid duplicates
        self._lock = threading.Lock()  # Thread safety for concurrent access

    def _is_connection_cached(self):
        """Check if we have a valid cached connection status."""
        import time

        return (
            self._connection_cache is not None
            and time.time() - self._cache_time < self._cache_duration
        )

    def _check_connection_fast(self):
        """Fast connection check with caching."""
        import time

        # Use cached result if available
        if self._is_connection_cached():
            return self._connection_cache

        # Perform actual connection check
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.api_url}/v1/models", timeout=3
            )  # Reduced timeout
            end_time = time.time()

            is_connected = response.status_code == 200

            # Cache the result
            self._connection_cache = is_connected
            self._cache_time = time.time()

            logger.debug(
                f"LMStudio connection check: {end_time - start_time:.2f}s, status: {is_connected}"
            )
            return is_connected

        except Exception as e:
            logger.debug(f"LMStudio connection failed: {e}")
            # Cache the failure for a shorter time
            self._connection_cache = False
            self._cache_time = time.time()
            return False

    def detect_current_model(self):
        """Detect which model is currently loaded in LM Studio.

        Returns:
            str: Current model name or None if no model detected
        """
        # Quick connection check first
        if not self._check_connection_fast():
            logger.debug("LMStudio not connected, skipping model detection")
            return None

        try:
            import time

            start_time = time.time()
            models_response = requests.get(
                f"{self.api_url}/v1/models", timeout=2
            )  # Further reduced timeout for speed
            end_time = time.time()

            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = models_data.get("data", [])
                if available_models:
                    # Get the first (and usually only) loaded model
                    current_model = available_models[0]["id"]
                    self.current_model = current_model
                    logger.info(
                        f"✅ Detected LMStudio model: {current_model} ({end_time - start_time:.2f}s)"
                    )
                    return current_model
                else:
                    logger.warning("No models currently loaded in LMStudio")
                    return None
            else:
                logger.warning(
                    f"Failed to get models from LMStudio: {models_response.status_code}"
                )
                return None
        except Exception as e:
            logger.warning(f"Could not detect current LMStudio model: {e}")
            return None

    def is_ocr_specialized_model(self, model_name=None):
        """Check if the current or specified model is specialized for OCR.

        Args:
            model_name: Model name to check, or None to check current model

        Returns:
            bool: True if model is OCR-specialized
        """
        if model_name is None:
            model_name = self.current_model or self.model

        if not model_name:
            return False

        # Check for OCR-specialized model indicators
        ocr_indicators = [
            "monkeyocr",
            "ocrflux",
            "ocr",
            "recognition",
            "text-recognition",
            "document-ai",
            "textract",
            "paddle-ocr",
            "flux",
        ]

        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in ocr_indicators)

    def is_vision_language_model(self, model_name=None):
        """Check if the model is a vision-language model.

        Args:
            model_name: Model name to check (optional)

        Returns:
            bool: True if model is vision-language specialized
        """
        if model_name is None:
            model_name = self.current_model or self.model

        if not model_name:
            return False

        # Check for vision-language model indicators
        vision_indicators = [
            'internvl', 'qwen', 'llava', 'blip', 'instructblip', 'minigpt',
            'vision', 'multimodal', 'vl-', 'visual', 'cogvlm', 'yi-vl',
            'deepseek-vl', 'phi-3-vision', 'pixtral', 'molmo', 'vl'
        ]

        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in vision_indicators)

    def get_model_type(self, model_name=None):
        """Determine the type/specialization of the model.

        Args:
            model_name: Model name to check, or None to check current model

        Returns:
            str: 'ocr', 'vision', or 'general'
        """
        if model_name is None:
            model_name = self.current_model or self.model

        if not model_name:
            return "general"

        if self.is_ocr_specialized_model(model_name):
            return "ocr"
        elif self.is_vision_language_model(model_name):
            return "vision"
        else:
            return "general"

    def get_optimized_ocr_prompt(self, page_num, model_name=None):
        """Get OCR prompt optimized for the detected model.

        Args:
            page_num: Page number being processed
            model_name: Model name to optimize for (uses current if None)

        Returns:
            str: Optimized prompt for OCR task
        """
        if model_name is None:
            model_name = self.current_model or self.model
        
        if not model_name:
            model_name = "generic"
        
        model_lower = model_name.lower()
        
        # Determine model capabilities dynamically
        model_capabilities = self._analyze_model_capabilities(model_name)

        if "ocr-specialized" in model_capabilities:
            # OCR-specialized models - focus on accuracy and structure
            return f"""Extract all text from this document image with maximum accuracy. This is page {page_num}.

This document contains a chemical data table with columns such as:
- NO (number index)
- FORMULA (chemical formula like C4H8)
- NAME (chemical name like CIS-2-BUTENE)
- MOLWT (molecular weight like 56.108)
- TFP, TB, TC, PC, VC, ZC, OMEGA, LIQDEN, TDEN, DIPM (numerical values with decimals)

CRITICAL REQUIREMENTS:
1. Extract EVERY visible character, number, and symbol
2. Preserve exact spacing, alignment, and formatting
3. Maintain table structures with proper column alignment
4. Include headers, footers, and any marginal text
5. Preserve chemical formulas, mathematical expressions, and special characters exactly as shown
6. Do not skip any text, even if partially visible or unclear

NUMERICAL ACCURACY:
- Preserve ALL digit values exactly - do not transform or round any numbers
- Maintain exact decimal point placement (e.g., 56.108, 0.272)
- DO NOT add any 'x6' or similar notation to numeric values
- DO NOT modify any numerical values
- For values like 0.272 or 0.202, preserve ALL digits and decimal points

OUTPUT FORMAT:
For tables, use this format with proper column alignment:
```
NO   FORMULA    NAME                    MOLWT    TFP     TB      TC
41   CCL3F      TRICHLOROFLUOROMETHANE  137.368  162.0   297.0
42   CCL4       CARBON TETRACHLORIDE    153.823  250.    349.7
```

- Use consistent spacing between columns (at least 2-3 spaces)
- Align column headers with their data
- Preserve line breaks and paragraph structure
- Maintain proper column alignment throughout the table

Focus on accuracy of numeric values and chemical data above all else."""

        elif "vision" in model_capabilities:
            # Vision-language models - leverage advanced capabilities
            if "instruction-following" in model_capabilities:
                # Advanced vision models with good instruction following
                return f"""You are an expert document analyst with advanced vision capabilities. Analyze this document image (page {page_num}) comprehensively.

**Primary Task: Complete Text Extraction**
- Extract ALL text content with perfect accuracy
- Preserve exact formatting, spacing, and structure
- Maintain reading order and document hierarchy

**Critical for Chemical Data:**
- Preserve ALL numerical values exactly (e.g., 56.108, 0.272, 134.3)
- Keep chemical formulas intact (e.g., C4H8, CCL3F)
- Maintain chemical names in uppercase (e.g., CIS-2-BUTENE)
- DO NOT add any 'x6' or similar notation to numeric values
- Preserve decimal points in exact positions

**Table Processing:**
- Extract tables maintaining column structure with clear alignment
- Use consistent spacing between columns (at least 2-3 spaces)
- Include column headers on the first line
- Align data under appropriate column headers
- Preserve empty cells as blank spaces
- Format as space-separated columns for easy reading

**Output Format:**
For tables, use this format:
```
NO   FORMULA    NAME                    MOLWT    TFP     TB      TC
41   CCL3F      TRICHLOROFLUOROMETHANE  137.368  162.0   297.0
42   CCL4       CARBON TETRACHLORIDE    153.823  250.    349.7
```

Provide clean, structured text that maintains proper column alignment.

Focus on numerical accuracy above all else."""

            else:
                # Basic vision models - simpler approach
                return f"""Extract all text from this document image (page {page_num}) with high accuracy.

**Requirements:**
- Extract every visible character, number, and symbol
- Preserve exact formatting and spacing
- Maintain table structures with proper column alignment
- Keep chemical formulas and names exactly as shown
- Preserve all numerical values without modification

**Table Format:**
Use this format for tables:
```
NO   FORMULA    NAME                    MOLWT    TFP     TB
41   CCL3F      TRICHLOROFLUOROMETHANE  137.368  162.0   297.0
```

**Critical:**
- DO NOT add any formatting markers like 'x6' or similar
- DO NOT modify any numerical values
- Preserve decimal points exactly (e.g., 56.108, 0.272)
- Use consistent spacing between columns

Focus on accuracy of numeric values and chemical data."""

        else:
            # General model - basic approach
            return f"""Extract all text from this document image (page {page_num}).

**Requirements:**
- Extract every visible character and number
- Preserve formatting and spacing
- Maintain table structures with column alignment
- Keep numerical values exactly as shown
- Do not add any extra formatting markers

**Table Format:**
Format tables like this:
```
NO   FORMULA    NAME                    MOLWT    TFP
41   CCL3F      TRICHLOROFLUOROMETHANE  137.368  162.0
```

Provide clean, accurate text extraction with proper table formatting."""

    def get_optimized_text_enhancement_prompt(self, text, page_num, model_type=None):
        """Get text enhancement prompt optimized for the current model type.

        Args:
            text: Text to enhance
            page_num: Page number
            model_type: Model type or None for auto-detect

        Returns:
            str: Optimized prompt for text enhancement
        """
        if model_type is None:
            model_type = self.get_model_type()

        if model_type == "ocr":
            # OCR model - focus on error correction
            return f"""Clean and correct this OCR text from page {page_num}:

{text}

Tasks:
- Fix OCR recognition errors
- Correct character substitutions (e.g., 'rn' → 'm', '0' → 'O')
- Maintain original formatting and structure
- Preserve technical terms and numbers exactly
- Keep table formatting intact

Output only the corrected text."""

        else:
            # Vision/General model - comprehensive enhancement
            return f"""You are an expert at OCR post-processing. Below is text extracted from a PDF document page {page_num}.
Please clean and enhance this text by:
1. Fixing OCR errors and typos
2. Improving formatting and structure
3. Ensuring proper spacing and line breaks
4. Preserving tables and lists
5. Maintaining technical accuracy

Original text:
{text}

Please provide the cleaned and enhanced version:"""

    def generate(self, prompt, max_tokens=1024, temperature=0.1):
        """Generate text using LM Studio.

        Args:
            prompt: Prompt to send to LM Studio
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation

        Returns:
            Generated text
        """
        import time

        start_time = time.time()

        try:
            # Quick connection check (cached)
            if not self._check_connection_fast():
                logger.error("LMStudio not available")
                return None

            # Use current model if detected, otherwise use configured model
            model_to_use = self.current_model or self.model

            # Use the chat completions endpoint with the correct format
            payload = {
                "model": model_to_use,
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

            logger.debug(f"Sending request to LMStudio with model: {model_to_use}")

            # Use optimized API call
            api_start_time = time.time()
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=60,  # Further reduced timeout for speed
            )
            api_end_time = time.time()

            if response.status_code != 200:
                logger.error(
                    f"LM Studio API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                # Extract content from the chat response format
                content = result["choices"][0]["message"]["content"].strip()
                total_time = time.time() - start_time
                api_time = api_end_time - api_start_time

                logger.info(
                    f"✅ LMStudio generation complete: {len(content)} chars in {total_time:.2f}s (API: {api_time:.2f}s)"
                )
                return content
            else:
                logger.error(f"Unexpected response format: {result}")
                return None

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ LMStudio API error after {total_time:.2f}s: {e}")
            return None


class LMStudioProcessor:
    """Process PDFs using LM Studio for text extraction and table enhancement."""

    def __init__(self, api_url=LMSTUDIO_URL, model=LMSTUDIO_MODEL, force_model=None):
        """Initialize the LM Studio processor.

        Args:
            api_url: LM Studio API URL
            model: LM Studio model name (fallback only)
            force_model: DEPRECATED - processor always uses the currently loaded model
        """
        self.name = "lmstudio"
        self.api_url = api_url
        self.model = model  # Fallback only
        # Always use the actual model loaded in LMStudio
        self.force_model = None  # Deprecated parameter
        self.lm_client = LMStudioClient(api_url, model)
        self.current_model = None
        self.model_type = None
        self._page_cache = {}  # Cache processed pages to avoid duplicates
        self._processing_times = []  # Track processing times for optimization
        self._partial_results = {}  # Store partial results for recovery

        # Check connection and detect current model (using fast cached method)
        if not self.lm_client._check_connection_fast():
            logger.warning(f"Could not connect to LM Studio at {api_url}")
        else:
            self._detect_and_optimize_for_current_model()

    def _detect_and_optimize_for_current_model(self):
        """Detect current model and optimize processing accordingly."""
        # Always use the actually loaded model - no forcing or preferences
        detected_model = self.lm_client.detect_current_model()
        self.current_model = detected_model
        
        if detected_model:
            logger.info(f"🔍 Detected and using model: {self.current_model}")
        else:
            logger.warning("⚠️  No model detected - will attempt to use fallback")

        if self.current_model:
            # Dynamically determine model capabilities
            self.model_type = self.lm_client.get_model_type(self.current_model)
            capabilities = self._analyze_model_capabilities(self.current_model)

            logger.info(f"📋 Model: {self.current_model}")
            logger.info(f"🏷️  Type: {self.model_type}")
            logger.info(f"🔧 Capabilities: {', '.join(capabilities)}")

            # Show what we detected
            self._show_model_analysis(capabilities)
        else:
            logger.warning("⚠️  No model detected in LMStudio - please load a model")
            logger.info("💡 Load any vision-capable model in LMStudio to continue")

    def _analyze_model_capabilities(self, model_name):
        """Analyze what capabilities a model has based on its name and type.

        Args:
            model_name: Name of the model to analyze

        Returns:
            List of capability strings
        """
        if not model_name:
            return ["unknown"]

        capabilities = []
        model_lower = model_name.lower()

        # Check for vision capabilities
        vision_indicators = ["vl", "vision", "visual", "multimodal", "internvl", "qwen", "llava", "clip", "image"]
        if any(indicator in model_lower for indicator in vision_indicators):
            capabilities.append("vision")

        # Check for OCR specialization
        ocr_indicators = ["ocr", "text", "document", "recognition", "monkey", "flux"]
        if any(indicator in model_lower for indicator in ocr_indicators):
            capabilities.append("ocr-specialized")

        # Check for instruction following
        instruction_indicators = ["instruct", "chat", "assistant"]
        if any(indicator in model_lower for indicator in instruction_indicators):
            capabilities.append("instruction-following")

        # If no specific capabilities detected, assume general
        if not capabilities:
            capabilities.append("general")

        return capabilities

    def _show_model_analysis(self, capabilities):
        """Show analysis of the detected model capabilities."""
        if "vision" in capabilities:
            logger.info("✅ Vision capabilities detected - can process images")
        else:
            logger.warning("⚠️  No vision capabilities detected - may not work with images")

        if "ocr-specialized" in capabilities:
            logger.info("✅ OCR specialization detected - optimized for text extraction")
        else:
            logger.info("ℹ️  General model - basic text processing capabilities")

        if "instruction-following" in capabilities:
            logger.info("✅ Instruction-following detected - good prompt adherence expected")
        else:
            logger.info("ℹ️  Basic model - simple prompts recommended")



    def _convert_to_html_table(self, text):
        """Convert extracted text to HTML table format for better display.

        Args:
            text: Raw extracted text

        Returns:
            Text with tables converted to HTML format
        """
        lines = text.strip().split('\n')
        html_parts = []
        current_table = []
        in_table = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_table and current_table:
                    # End of table
                    html_parts.append(self._format_table_as_html(current_table))
                    current_table = []
                    in_table = False
                html_parts.append('')
                continue

            # Check if this looks like a table row (has multiple columns separated by spaces)
            # Look for patterns like: number, formula, name, numbers
            if re.match(r'^\s*\d+\s+[A-Z0-9]+\s+[A-Z\-\s]+\s+[\d\.,\s]+', line) or \
               re.match(r'^\s*NO\s+FORMULA\s+NAME', line.upper()):
                # This looks like a table row
                if not in_table:
                    in_table = True
                current_table.append(line)
            else:
                # Not a table row
                if in_table and current_table:
                    # End of table
                    html_parts.append(self._format_table_as_html(current_table))
                    current_table = []
                    in_table = False
                html_parts.append(line)

        # Handle table at end of text
        if in_table and current_table:
            html_parts.append(self._format_table_as_html(current_table))

        return '\n'.join(html_parts)

    def _format_table_as_html(self, table_lines):
        """Format table lines as HTML table.

        Args:
            table_lines: List of table row strings

        Returns:
            HTML table string
        """
        if not table_lines:
            return ""

        html = ['<table class="chemical-data-table" style="border-collapse: collapse; width: 100%; margin: 10px 0;">']

        for i, line in enumerate(table_lines):
            # Split line into columns (handle multiple spaces)
            cols = re.split(r'\s{2,}', line.strip())

            if i == 0 or 'NO' in line.upper() and 'FORMULA' in line.upper():
                # Header row
                html.append('  <tr style="background-color: #f0f0f0; font-weight: bold;">')
                for col in cols:
                    html.append(f'    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">{col}</th>')
                html.append('  </tr>')
            else:
                # Data row
                html.append('  <tr>')
                for col in cols:
                    html.append(f'    <td style="border: 1px solid #ddd; padding: 8px;">{col}</td>')
                html.append('  </tr>')

        html.append('</table>')
        return '\n'.join(html)

    def get_model_info(self):
        """Get information about the current model setup.

        Returns:
            dict: Model information and recommendations
        """
        return {
            "current_model": self.current_model,
            "model_type": self.model_type,
            "is_ocr_specialized": self.lm_client.is_ocr_specialized_model(
                self.current_model
            ),
            "preferred_ocr_model": LMSTUDIO_PREFERRED_OCR_MODEL,
            "preferred_vision_model": LMSTUDIO_PREFERRED_VISION_MODEL,
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self):
        """Get model recommendations based on current setup."""
        if not self.current_model:
            return ["Load a model in LMStudio to enable processing"]

        recommendations = []

        if self.model_type == "ocr":
            recommendations.append("✅ Current model is optimized for OCR tasks")
        elif "monkeyocr" in self.current_model.lower():
            recommendations.append(
                "🎯 Perfect! MonkeyOCR provides excellent text recognition"
            )
        else:
            recommendations.append(
                "💡 For better OCR: Load MonkeyOCR-Recognition model"
            )

        return recommendations

    def _check_connection(self):
        """Check if LM Studio is running properly (uses fast cached method)."""
        return self.lm_client._check_connection_fast()

    def process(
        self, pdf_path: str, page_indices: Optional[List[int]] = None, progress_callback = None
    ) -> Dict[str, Any]:
        """Process a PDF file with LM Studio.

        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with processing results
        """
        import time
        start_time = time.time()
        logger.info(f"LMStudio processor starting on {pdf_path}")

        # Clear page cache for new document
        self._page_cache.clear()
        self.lm_client._processed_pages.clear()

        # Check connection and re-detect model for this session
        connection_status = self._check_connection()
        if connection_status:
            # Re-detect model to ensure we're using the current one
            detected_model = self.lm_client.detect_current_model()
            if detected_model and detected_model != self.current_model:
                logger.info(f"🔄 Model changed to: {detected_model}")
                self.current_model = detected_model
                self.model_type = self.lm_client.get_model_type(detected_model)

        logger.info(
            f"LMStudio connection status: {'Connected' if connection_status else 'Not connected'}"
        )

        if not connection_status:
            logger.warning(
                "LMStudio is not connected, falling back to basic text extraction"
            )
            # Fall back to basic text extraction if LMStudio is not available
            return self._extract_content_from_pdf(pdf_path, page_indices)

        # Extract pages as images with optimized DPI
        page_images = self._extract_pages_to_images(pdf_path, page_indices, dpi=200)  # Reduced DPI for speed

        if not page_images:
            logger.warning(f"No valid pages extracted from {pdf_path}")
            return {}

        logger.info(f"Extracted {len(page_images)} page images for OCR processing")

        # Process pages with LMStudio using parallel processing
        results = self._process_pages_parallel(page_images, pdf_path, progress_callback)

        total_time = time.time() - start_time
        avg_page_time = sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        
        logger.info(
            f"LMStudio processor completed processing {pdf_path} with {len(results)} pages in {total_time:.2f}s (avg {avg_page_time:.2f}s/page)"
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

        if not self.lm_client._check_connection_fast():
            logger.debug("LMStudio not available, skipping table enhancement")
            return None

        prompt = f"""
        You are an expert at analyzing tables. Below is a table extracted from a PDF document.
        Please analyze this table and provide a cleaned, well-formatted version.
        Fix any alignment issues, merged cells, or extraction errors.
        
        Critical requirements:
        1. Preserve all numeric values EXACTLY as shown (e.g., 56.108, 134.3, etc.)
        2. Maintain exact column headers and alignment
        3. Render decimal points correctly
        4. Do not insert formatting markers like 'x6' or similar notations
        5. Preserve chemical formulas exactly (e.g., C4H8)
        6. Identify and repair any OCR errors in numbers or text

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

        # Use optimized prompt based on current model type
        prompt = self.lm_client.get_optimized_text_enhancement_prompt(
            text, page_num, self.model_type
        )

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
        self, pdf_path: str, page_indices: Optional[List[int]] = None, dpi: int = 200
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
                # Use optimized rendering settings for speed
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=matrix, alpha=False)  # No alpha for speed
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Optimize image size for processing speed
                max_dimension = 1600
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
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
            import time
            process_start = time.time()
            
            # Use cached model info (don't re-detect for every page)
            model_to_use = self.current_model or self.model
            logger.debug(f"🎯 Processing page {page_num} with model: {model_to_use}")

            # Optimize image size for faster processing
            max_size = (1200, 1600)  # Reduced from default for speed
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image to {image.size} for faster processing")

            # Convert image to base64 with optimized quality
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True)  # Reduced quality for speed
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Create consistent prompt for all pages (avoid per-page variations)
            prompt = self._get_consistent_ocr_prompt(page_num)
            logger.debug(f"📝 Using consistent OCR prompt for page {page_num}")

            # Use pre-determined model (no dynamic detection per page)
            logger.debug(f"🔧 Using cached model: {model_to_use}")

            # Use cached vision capability check (done once at init)
            is_vision_capable = self._is_vision_capable_model(model_to_use)
            if not is_vision_capable and page_num == 1:  # Only warn once
                logger.warning(f"Model {model_to_use} may not support vision input!")
            
            # Use consistent system message for all pages (cached)
            system_message = self._get_system_message_for_model(model_to_use)

            # Use pre-cached vision capability
            
            if is_vision_capable:
                # Prepare the payload for LMStudio with the image (for vision models)
                payload = {
                    "model": model_to_use,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message,
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
            else:
                # For non-vision models, try using base64 directly in the prompt (fallback)
                # This is less likely to work but might help with some models
                logger.warning(f"Model {model_to_use} is not vision-capable. Trying alternate approach.")
                payload = {
                    "model": model_to_use,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message,
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\n[Image data not shown - model does not support vision]"
                        },
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                }

            logger.debug(f"Sending page {page_num} image to LMStudio")
            api_start = time.time()
            # Use session for connection reuse in parallel processing
            current_thread = threading.current_thread()
            session = getattr(current_thread, 'session', None)
            if session is None:
                session = requests.Session()
                session.headers.update({
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                })
                current_thread.session = session
            
            response = session.post(
                f"{self.api_url}/v1/chat/completions", 
                json=payload, 
                timeout=180  # Longer timeout for parallel processing
            )
            api_time = time.time() - api_start

            if response.status_code != 200:
                logger.error(
                    f"LMStudio API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()

                # Process the content to ensure consistent formatting
                # This is moved after checking for generic responses

                # Check for common error responses indicating vision capability issues
                generic_responses = [
                    "I'm unable to process images directly",
                    "I cannot view or process images",
                    "I don't have the ability to see images",
                    "I can't see the image you",
                    "As an AI text model",
                    "I'm unable to directly view or analyze images"
                ]
                
                # Check if response is generic/error
                is_generic_response = any(phrase in content for phrase in generic_responses)
                
                if is_generic_response:
                    logger.error(f"Detected generic 'cannot process image' response from model {model_to_use}. This model likely doesn't support vision/image input!")
                    # Try basic fallback extraction
                    logger.info(f"Falling back to basic extraction for page {page_num}")
                    fallback_content = self._extract_text_with_pytesseract(image, page_num)
                    if fallback_content:
                        logger.info(f"Retrieved fallback content using tesseract ({len(fallback_content)} chars)")
                        # Apply fixes to the fallback content
                        fixed_fallback = self._fix_chemical_table_formatting(fallback_content, page_num)
                        return fixed_fallback
                    return None
                
                # Handle OCRFlux JSON format if not a generic response
                if model_to_use and ("ocrflux" in model_to_use.lower() or "flux" in model_to_use.lower()):
                    content = self._parse_ocrflux_json(content)
                
                # Apply universal table converter for consistent formatting
                content = self._universal_table_converter(content)

                logger.info(
                    f"Successfully extracted content from page {page_num} image (length: {len(content)})"
                )
                total_time = time.time() - process_start
                logger.info(f"Page {page_num}: extracted {len(content)} chars in {total_time:.2f}s (API: {api_time:.2f}s)")
                
                # Apply consistent post-processing for all pages
                processed_content = self._post_process_content(content, page_num)
                
                if processed_content and processed_content.strip():
                    return processed_content
                else:
                    logger.error(f"No valid content for page {page_num}")
                    return None
            else:
                logger.error(f"Unexpected response format from LMStudio: {result}")
                return None

        except Exception as e:
            logger.error(f"Error processing image with LMStudio: {e}")
            return None
    
    def _get_consistent_ocr_prompt(self, page_num: int) -> str:
        """Get a consistent OCR prompt for all pages to ensure uniform results.
        
        Args:
            page_num: Page number being processed
            
        Returns:
            str: Consistent OCR prompt
        """
        return f"""Extract all text from this document page {page_num} with maximum accuracy.

This document contains a chemical data table with columns like:
- NO (number index)
- FORMULA (chemical formula like C4H8, CCL3F)
- NAME (chemical name like CIS-2-BUTENE, TRICHLOROFLUOROMETHANE)
- MOLWT (molecular weight like 56.108, 137.368)
- TFP, TB, TC, PC, VC, ZC, OMEGA (numerical properties with decimals)

IMPORTANT REQUIREMENTS:
1. Extract EVERY visible character, number, and symbol exactly as shown
2. Preserve ALL decimal values precisely (e.g., 56.108, 0.272, 137.368)
3. DO NOT add formatting markers like 'x6' or similar notations
4. Maintain table structure with proper column alignment
5. Keep chemical formulas intact (e.g., C4H8, CCL3F)
6. Chemical names should be in uppercase
7. Use consistent spacing between columns

Output the table as plain text with proper spacing:
```
NO   FORMULA    NAME                    MOLWT    TFP     TB      TC
41   CCL3F      TRICHLOROFLUOROMETHANE  137.368  162.0   297.0
42   CCL4       CARBON TETRACHLORIDE    153.823  250.    349.7
```

Focus on numerical accuracy above all else."""
    
    def _get_system_message_for_model(self, model_name: str) -> str:
        """Get optimized system message for the given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Optimized system message
        """
        if not model_name:
            model_name = "generic"
            
        model_lower = model_name.lower()
        
        if "ocrflux" in model_lower or "flux" in model_lower:
            return "Extract text from chemical data tables with extreme precision. Preserve ALL numerical values exactly as shown, including decimal points (e.g., 56.108, 0.272). Maintain chemical formulas (C4H8) and chemical names. DO NOT add formatting markers like 'x6'. Output plain text with proper spacing and alignment."
        elif "monkeyocr" in model_lower or "monkey" in model_lower:
            return "You are an advanced OCR system specialized in chemical data tables and scientific documents. Extract text with perfect precision, especially numerical values with decimal points (e.g., 56.108, 0.272, 134.3). Preserve chemical formulas (C4H8) exactly. Maintain proper column alignment. DO NOT add ANY formatting markers like 'x6' to the output. Keep chemical names in uppercase. Render the table in plain text with proper spacing."
        elif "internvl" in model_lower:
            return "You are an advanced multimodal AI with expert vision and scientific document analysis capabilities. For this chemical reference table: 1) Extract ALL numerical values with perfect precision (e.g., 56.108, 0.272, 134.3), 2) Preserve chemical formulas (C4H8) exactly, 3) Maintain proper column alignment with headers like NO, FORMULA, NAME, MOLWT, TFP, TB, etc., 4) Keep chemical names in uppercase, 5) DO NOT add any formatting markers like 'x6', 6) Render the table as plain text with proper spacing."
        elif "qwen" in model_lower and "vl" in model_lower:
            return "You are a state-of-the-art multimodal AI specialized in scientific and chemical data extraction. This document contains a reference table with columns for NO, FORMULA (e.g., C4H8), NAME (e.g., CIS-2-BUTENE), MOLWT (e.g., 56.108), and other properties with decimal values (e.g., 0.272, 134.3). Extract ALL numerical values with perfect precision. Preserve chemical formulas exactly. Maintain column alignment. DO NOT add formatting markers like 'x6'. Keep chemical names in uppercase. Render the table as plain text with proper spacing."
        else:
            return "You are a specialized document extractor focused on tabular data. Extract tables with these requirements: 1) ALL numerical values must be preserved with exact precision and decimal placement (e.g., 56.108, 0.272, 134.3), 2) Text content must be preserved exactly as shown, 3) Column headers and structure must be maintained, 4) DO NOT add any formatting markers or extra symbols, 5) Render the table as plain text with proper spacing and alignment, 6) Preserve the original data organization and hierarchy."
    
    def _post_process_content(self, content: str, page_num: int) -> str:
        """Apply consistent post-processing to all pages.
        
        Args:
            content: Raw OCR content
            page_num: Page number
            
        Returns:
            Processed content with consistent formatting
        """
        logger.debug(f"Post-processing page {page_num} content ({len(content)} chars)")
        
        # Step 1: Fix chemical table formatting issues
        fixed = self._fix_chemical_table_formatting(content, page_num)

        # Step 2: Convert all tables to markdown using the modular table formatter
        markdown_formatted = TableFormatter.detect_and_convert_all_tables_to_markdown(fixed)

        logger.debug(f"Post-processing complete for page {page_num} ({len(markdown_formatted)} chars)")
        return markdown_formatted

    def _extract_text_with_pytesseract(self, image: Image.Image, page_num: int) -> str:
        """Extract text from image using pytesseract as a fallback.
        
        Args:
            image: PIL Image to process
            page_num: Page number for logging
            
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            # Try to import pytesseract
            import pytesseract
            logger.info(f"Using pytesseract as fallback for page {page_num}")
            
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
                
            # Enhance image for better OCR
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
            
            # Extract text
            text = pytesseract.image_to_string(image)
            
            if text and len(text) > 50:  # Ensure we got meaningful content
                logger.info(f"Successfully extracted {len(text)} chars with pytesseract")
                return text
            else:
                logger.warning(f"Pytesseract extraction returned minimal text ({len(text) if text else 0} chars)")
                return None
        except ImportError:
            logger.warning("Pytesseract not available for fallback OCR")
            return None
        except Exception as e:
            logger.error(f"Error in pytesseract fallback: {e}")
            return None
        
    def _fix_chemical_table_formatting(self, content: str, page_num: int) -> str:
        """Apply specialized post-processing to fix common issues in chemical table OCR results.
        
        Args:
            content: Raw text content from OCR
            page_num: Page number for logging
            
        Returns:
            Fixed content with proper formatting
        """
        import re
        
        # Apply table formatting more broadly - look for any tabular data patterns
        # Check for chemical table headers OR numerical data patterns OR chemical formulas
        has_table_headers = re.search(r'(NO|FORMULA|NAME|MOLWT|TFP|TB|TC|PC|VC|ZC|OMEGA)', content, re.IGNORECASE)
        has_chemical_formulas = re.search(r'[A-Z][A-Za-z0-9]*[0-9]+|C[0-9]+H[0-9]+', content)
        has_numerical_data = re.search(r'\d+\.\d+|\d+,\d+|\d+\s+\d+', content)
        has_compound_numbers = re.search(r'^\s*\d{1,3}\s*$', content, re.MULTILINE)

        # Check for generic model responses that indicate processing issues
        generic_responses = [
            "unable to extract text",
            "can't extract text",
            "cannot extract text",
            "appears to be blank",
            "not properly loaded",
            "provide the actual text"
        ]

        is_generic_response = any(phrase in content.lower() for phrase in generic_responses)

        # Apply formatting if any tabular indicators are found OR if it's a generic response
        if not (has_table_headers or has_chemical_formulas or has_numerical_data or has_compound_numbers or is_generic_response):
            logger.info(f"Page {page_num} doesn't appear to contain tabular data, skipping formatting fixes")
            return content

        if is_generic_response:
            logger.warning(f"Page {page_num} has generic model response, providing informative message")
            # For generic responses, provide a helpful message instead of the generic one
            fallback_message = f"""# Page {page_num} Content

## Processing Note
LMStudio's vision model was unable to extract clear text from this page. This may be due to:
- Image quality or resolution issues
- Complex table formatting
- Unclear text in the source document

## Recommendation
Try using other processors (Docling, Camelot) which may handle this page better, or check the original document quality.

## Original Response
{content}"""
            return fallback_message

        logger.info(f"Applying chemical table formatting fixes to page {page_num}")

        # Fix common OCR errors
        fixed = content
        
        # Fix 'x6' pattern - common OCR error for numerical values
        fixed = re.sub(r'(\d+)x(\d+)', r'\1.\2', fixed)
        
        # Fix missing decimal points in numbers that should have them
        # Match patterns like 0272, 0202, etc. and add decimal point
        fixed = re.sub(r'(\s|^)(0)(\d{3})(\s|$)', r'\1\2.\3\4', fixed)
        
        # Fix chemical formulas with incorrect spacing
        fixed = re.sub(r'C\s+(\d)\s+H\s+(\d)', r'C\1H\2', fixed)
        
        # Convert markdown table format to plain text if needed
        if '|' in fixed and '---' in fixed:
            lines = fixed.split('\n')
            cleaned_lines = []
            for line in lines:
                # Skip markdown separator lines
                if re.match(r'^\s*\|\s*[-:\s]+\|\s*$', line):
                    continue
                # Remove leading/trailing | and extra whitespace
                cleaned = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', line)
                # Replace remaining | with spaces
                cleaned = re.sub(r'\s*\|\s*', '  ', cleaned)
                cleaned_lines.append(cleaned)
            fixed = '\n'.join(cleaned_lines)
        
        # Fix any missed decimal formats using expected patterns
        # These columns usually have decimal points: MOLWT, ZC, OMEGA, etc.
        # Convert patterns like '56 108' to '56.108'
        fixed = re.sub(r'(\d+)\s+(\d{3})(?=\s)', r'\1.\2', fixed)
        
        if fixed != content:
            logger.info(f"Fixed formatting issues in chemical table on page {page_num}")
        
        return fixed

    def _is_vision_capable_model(self, model_name: str) -> bool:
        """Check if a model is capable of processing vision/image inputs.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model can process images, False otherwise
        """
        if not model_name:
            return False

        model_lower = model_name.lower()

        # Vision-capable model indicators
        vision_indicators = [
            'internvl', 'qwen', 'llava', 'blip', 'instructblip', 'minigpt',
            'vision', 'multimodal', 'vl-', 'visual', 'cogvlm', 'yi-vl',
            'deepseek-vl', 'phi-3-vision', 'pixtral', 'molmo', 'vl',
            'ocrflux', 'monkeyocr', 'ocr'  # OCR models are vision-capable
        ]

        return any(indicator in model_lower for indicator in vision_indicators)

    def _apply_advanced_formatting(self, content: str, page_num: int) -> str:
        """Apply advanced formatting to match other processors' output quality.

        Args:
            content: Raw OCR content
            page_num: Page number for logging

        Returns:
            Formatted content with proper structure and HTML tables
        """
        import re

        # If content is too short, return as-is
        if len(content.strip()) < 50:
            return content

        try:
            # Don't add page header here - it will be added by _format_page_content
            formatted_lines = []

            # Use the new robust table detection
            table_formatted_content = self._detect_and_convert_all_tables(content)

            if table_formatted_content and table_formatted_content.strip():
                # Add the formatted content with HTML tables
                formatted_lines.append(table_formatted_content)
                final_content = '\n'.join(formatted_lines)
                logger.debug(f"Applied HTML table formatting to page {page_num}")
                return final_content
            else:
                # Return content as-is for further processing
                return content

        except Exception as e:
            logger.warning(f"Error in advanced formatting for page {page_num}: {e}")
            # Return original content
            return content

    def _detect_and_format_tables_as_html(self, content: str) -> str:
        """Detect tables in OCR content and format them as HTML tables.

        Args:
            content: Raw OCR content

        Returns:
            Content with properly formatted HTML tables
        """
        import re

        lines = content.split('\n')
        formatted_lines = []
        current_table = []
        in_table = False

        # Patterns to detect table-like content
        table_patterns = [
            # Chemical table header pattern: NO FORMULA NAME MOLWT ...
            r'^(NO|FORMULA|NAME|MOLWT|TFP|TB|TC|PC|VC|ZC|OMEGA)',
            # Data row pattern: number followed by chemical data
            r'^\s*\d+\s+[A-Z0-9]+\s+[A-Z\-,\s]+',
            # Header pattern with multiple columns (3+ words)
            r'^[A-Z\s]{3,}\s+[A-Z\s]{3,}\s+[A-Z\s]{3,}',
            # Row starting with number and containing multiple data fields
            r'^\s*\d+\s+\w+\s+[\w\s\-]+\s+[\d.x]+',
        ]

        for line in lines:
            line_stripped = line.strip()

            # Check if this line looks like a table
            is_table_line = any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in table_patterns)

            if is_table_line and len(line_stripped.split()) >= 3:
                # This looks like a table line
                if not in_table:
                    # Starting a new table
                    in_table = True
                    current_table = []
                current_table.append(line_stripped)
            else:
                # Not a table line
                if in_table and current_table:
                    # End of table - format it as HTML
                    formatted_table = self._format_table_as_html(current_table)
                    if formatted_table:
                        formatted_lines.append(formatted_table)
                        formatted_lines.append("")
                    current_table = []
                    in_table = False

                # Add regular line
                if line_stripped:  # Skip empty lines in between
                    formatted_lines.append(line)

        # Handle table at end of content
        if in_table and current_table:
            formatted_table = self._format_table_as_html(current_table)
            if formatted_table:
                formatted_lines.append(formatted_table)

        return '\n'.join(formatted_lines)

    def _universal_table_converter(self, content: str) -> str:
        """Convert ANY extracted content to consistent markdown table format.

        Args:
            content: Any extracted text content

        Returns:
            Content with consistent markdown table formatting (like Docling)
        """
        if not content or not content.strip():
            return content

        import re

        # First, try to detect and convert existing tabular data to markdown using modular formatter
        markdown_content = TableFormatter.detect_and_convert_all_tables_to_markdown(content)

        return markdown_content
    
    def _process_pages_parallel(self, page_images, pdf_path, progress_callback=None):
        """Process multiple pages in parallel for better performance.
        
        Args:
            page_images: List of (page_num, image) tuples
            pdf_path: Path to PDF file for fallback
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary of processing results
        """
        import concurrent.futures
        import threading
        
        results = {}
        total_pages = len(page_images)
        completed_pages = 0
        lock = threading.Lock()
        
        def process_single_page(page_data):
            """Process a single page (thread-safe)."""
            nonlocal completed_pages
            
            page_num, image = page_data
            
            # Skip if already processed
            with lock:
                if page_num in self.lm_client._processed_pages:
                    logger.warning(f"Page {page_num} already processed, skipping duplicate")
                    return None
            
            page_start_time = time.time()
            logger.info(f"Starting parallel processing of page {page_num}")
            
            # Process image with LMStudio
            text_content = self._process_image_with_lmstudio(image, page_num)
            
            page_process_time = time.time() - page_start_time
            
            with lock:
                self._processing_times.append(page_process_time)
                completed_pages += 1
                logger.info(f"Page {page_num} completed in {page_process_time:.2f}s ({completed_pages}/{total_pages})")
                
                if progress_callback:
                    progress_callback("processing_page", completed_pages, total_pages)
            
            if text_content:
                # Mark page as processed
                with lock:
                    self.lm_client._processed_pages.add(page_num)
                
                # Format content for web interface
                formatted_content = self._format_page_content({
                    "page_number": page_num,
                    "content": text_content
                })
                
                return {
                    "page_num": page_num,
                    "success": True,
                    "result": {
                        "page_number": page_num,
                        "content": formatted_content,
                        "text": formatted_content,
                    }
                }
            else:
                # Return failure indicator for fallback processing
                return {
                    "page_num": page_num,
                    "success": False,
                    "result": None
                }
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(5, total_pages)  # Limit concurrent requests to avoid overwhelming LMStudio
        logger.info(f"Starting parallel processing with {max_workers} workers for {total_pages} pages")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages for processing
            future_to_page = {executor.submit(process_single_page, page_data): page_data[0]
                            for page_data in page_images}

            # Collect results as they complete
            try:
                for future in concurrent.futures.as_completed(future_to_page, timeout=300):  # 5 minute total timeout
                    page_num = future_to_page[future]
                    try:
                        result = future.result(timeout=10)  # Quick timeout for getting result
                        if result:
                            if result["success"]:
                                results[str(result["page_num"])] = result["result"]
                            else:
                                # Handle failed page with fallback
                                logger.warning(f"Page {result['page_num']} failed LMStudio processing, using fallback")
                                fallback_result = self._handle_failed_page(pdf_path, result["page_num"])
                                if fallback_result:
                                    results[str(result["page_num"])] = fallback_result
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Page {page_num} timed out during parallel processing")
                        # Handle timeout with fallback
                        fallback_result = self._handle_failed_page(pdf_path, page_num)
                        if fallback_result:
                            results[str(page_num)] = fallback_result
                    except Exception as e:
                        logger.error(f"Page {page_num} failed with error: {e}")
                        # Handle error with fallback
                        fallback_result = self._handle_failed_page(pdf_path, page_num)
                        if fallback_result:
                            results[str(page_num)] = fallback_result
            except concurrent.futures.TimeoutError:
                # Handle case where as_completed() times out with unfinished futures
                unfinished_futures = [f for f in future_to_page.keys() if not f.done()]
                logger.warning(f"Parallel processing timed out with {len(unfinished_futures)} unfinished futures")

                # Handle unfinished pages with fallback
                for future in unfinished_futures:
                    page_num = future_to_page[future]
                    logger.warning(f"Page {page_num} never completed, using fallback processing")
                    fallback_result = self._handle_failed_page(pdf_path, page_num)
                    if fallback_result:
                        results[str(page_num)] = fallback_result
        
        logger.info(f"Parallel processing completed: {len(results)}/{total_pages} pages successful")

        # Store partial results for potential recovery
        self._partial_results = results

        # Check if we have significantly fewer results than expected
        if len(results) < total_pages * 0.5:  # Less than 50% success rate
            logger.warning(f"Low success rate: {len(results)}/{total_pages} pages processed successfully")
            # Still return results - don't raise exception to allow partial results to be saved

        return results
    
    def _handle_failed_page(self, pdf_path, page_num):
        """Handle a page that failed LMStudio processing.
        
        Args:
            pdf_path: PDF file path
            page_num: Page number that failed
            
        Returns:
            Fallback processing result
        """
        try:
            # Use basic extraction with table formatting
            basic_results = self._extract_single_page_content(pdf_path, page_num)
            if basic_results:
                raw_content = basic_results.get('text', '') or basic_results.get('content', '')
                if raw_content:
                    # Apply consistent post-processing like successful pages
                    processed_content = self._post_process_content(raw_content, page_num)
                    # Format for web interface
                    formatted_content = self._format_page_content({
                        "page_number": page_num,
                        "content": processed_content
                    })
                    basic_results['content'] = formatted_content
                    basic_results['text'] = formatted_content
                return basic_results
        except Exception as e:
            logger.error(f"Fallback processing failed for page {page_num}: {e}")
        
        return None

    def _get_partial_results(self):
        """Get partial results from failed processing."""
        return getattr(self, '_partial_results', {})

    def _detect_and_convert_all_tables(self, content: str) -> str:
        """Robust table detection and conversion for all types of table data.
        
        Args:
            content: Raw OCR content
            
        Returns:
            Content with all tables converted to HTML
        """
        import re
        
        # Check if this looks like vertical table data (common issue)
        if self._is_vertical_table_data(content):
            return self._convert_vertical_table_data(content)
        
        # Otherwise use line-by-line detection
        lines = content.split('\n')
        result_lines = []
        current_table_lines = []
        in_table = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines, but end table if we were in one
            if not line_stripped:
                if in_table and current_table_lines:
                    # Convert accumulated table
                    table_html = self._convert_lines_to_html_table(current_table_lines)
                    if table_html:
                        result_lines.append(table_html)
                    current_table_lines = []
                    in_table = False
                result_lines.append('')
                continue
            
            # Check if this line looks like table content
            is_table_line = self._is_table_line(line_stripped)
            
            if is_table_line:
                if not in_table:
                    in_table = True
                current_table_lines.append(line_stripped)
            else:
                # Not a table line
                if in_table and current_table_lines:
                    # Convert accumulated table
                    table_html = self._convert_lines_to_html_table(current_table_lines)
                    if table_html:
                        result_lines.append(table_html)
                    current_table_lines = []
                    in_table = False
                
                # Add regular line
                result_lines.append(line_stripped)
        
        # Handle table at end of content
        if in_table and current_table_lines:
            table_html = self._convert_lines_to_html_table(current_table_lines)
            if table_html:
                result_lines.append(table_html)
        
        return '\n'.join(result_lines)

    # Table formatting methods moved to lmstudio_modules/table_formatter.py
    # Use TableFormatter.detect_and_convert_all_tables_to_markdown() instead

    # Removed: _extract_table_columns - now in TableFormatter module
    # Removed: _create_unified_markdown_table - now in TableFormatter module
    # Removed: _is_table_line - now in TableFormatter module
    # Removed: _convert_lines_to_html_table - now in TableFormatter module
    # Removed: _convert_lines_to_markdown_table - now in TableFormatter module
    # Removed: _split_table_line - now in TableFormatter module
    # Removed: _is_vertical_table_data - now in TableFormatter module
    # Removed: _convert_vertical_table_data - now in TableFormatter module
    # Removed: _convert_vertical_table_data_to_markdown - now in TableFormatter module
    # Removed: _create_simple_data_table - now in TableFormatter module
    # Removed: _create_simple_data_table_markdown - now in TableFormatter module
    # Removed: _convert_data_to_markdown_table - now in TableFormatter module
    # Removed: _looks_like_existing_table - now in TableFormatter module
    # Removed: _existing_table_to_html - now in TableFormatter module
    # Removed: _existing_table_to_markdown - now in TableFormatter module
    # Removed: _vertical_list_to_html - now in TableFormatter module
    # Removed: _vertical_list_to_markdown - now in TableFormatter module
    # Removed: _format_detected_table - now in TableFormatter module

    # All duplicated table methods removed - using modular TableFormatter instead

    def _parse_ocrflux_json(self, content: str) -> str:
        """Parse OCRFlux JSON response to extract clean text.

        Args:
            content: Raw OCRFlux JSON response

        Returns:
            Clean text content extracted from JSON
        """
        try:
            import json

            # Try to parse as JSON
            if content.startswith('{') and content.endswith('}'):
                data = json.loads(content)

                # Extract text from various possible JSON structures
                if 'text' in data:
                    return data['text']
                elif 'content' in data:
                    return data['content']
                elif 'result' in data:
                    return data['result']
                elif 'output' in data:
                    return data['output']

            # If not JSON or no recognizable structure, return as-is
            return content
        except json.JSONDecodeError:
            # Not valid JSON, return original content
            return content
        except Exception as e:
            logger.error(f"Error parsing OCRFlux JSON: {e}")
            return content

    def _extract_single_page_content(
        self, pdf_path: str, page_num: int
    ) -> Dict[str, Any]:
        """Extract content from a single page as fallback.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract (1-based)

        Returns:
            Dictionary with page content or None if failed
        """
        try:
            # Extract page as image
            images = self._extract_pages_to_images(pdf_path, [page_num - 1])
            if not images:
                return None

            page_idx, image = images[0]

            # Process with LMStudio
            content = self._process_image_with_lmstudio(image, page_num)
            if not content:
                return None

            return {
                "page": page_num,
                "content": content,
                "source": "lmstudio_single_page"
            }
        except Exception as e:
            logger.error(f"Error extracting content from page {page_num}: {e}")
            return None

    def _format_page_content(self, page_data):
        """Format page content for output with consistent structure.

        Args:
            page_data: Page data

        Returns:
            Formatted page content
        """
        if not page_data or "content" not in page_data:
            return ""

        content = page_data["content"]
        page_num = page_data.get("page", "unknown")

        # Apply consistent formatting
        formatted_content = f"# Page {page_num}\n\n{content}"

        return formatted_content

    def save_results(
        self, results: Dict[str, Any], output_dir: str, base_name: str
    ) -> None:
        """Save processing results.

        Args:
            results: Processing results
            output_dir: Output directory
            base_name: Base name for output files
        """
        from pathlib import Path
        import json

        # Create output directory structure
        output_path = Path(output_dir)
        processor_dir = output_path / f"{base_name}_lmstudio"
        processor_dir.mkdir(parents=True, exist_ok=True)

        # Save combined results
        combined_content = []
        combined_json = {}

        for page_num in sorted(results.keys()):
            page_data = results[page_num]
            if isinstance(page_data, dict) and "content" in page_data:
                combined_content.append(f"# Page {page_num}\n\n{page_data['content']}")
                combined_json[str(page_num)] = page_data

        # Save combined markdown
        combined_md_path = processor_dir / f"{base_name}_combined.md"
        with open(combined_md_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(combined_content))

        # Save combined JSON
        combined_json_path = processor_dir / f"{base_name}_combined.json"
        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_json, f, indent=2, ensure_ascii=False)

        # Save individual pages
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
