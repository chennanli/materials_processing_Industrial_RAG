#!/usr/bin/env python3
"""
Image Processor Implementation
-----------------------------
Process PDF images with LM Studio.
"""

import base64
import io
import logging
import re
import threading
import time
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF
import requests
from PIL import Image

from .table_formatter import TableFormatter

# Configure logging
logger = logging.getLogger("ImageProcessor")

class ImageProcessor:
    """Process PDF images with LM Studio."""

    def __init__(self, api_url, lm_client):
        """Initialize the image processor.

        Args:
            api_url: LM Studio API URL
            lm_client: LM Studio client instance
        """
        self.api_url = api_url
        self.lm_client = lm_client
        self._processing_times = []  # Track processing times for optimization

    def extract_pages_to_images(
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

    def process_image_with_lmstudio(self, image: Image.Image, page_num: int) -> str:
        """Process an image with LMStudio for OCR.

        Args:
            image: PIL Image to process
            page_num: Page number for logging

        Returns:
            Extracted text content
        """
        try:
            process_start = time.time()
            
            # Use cached model info (don't re-detect for every page)
            model_to_use = self.lm_client.current_model or self.lm_client.model
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
            prompt = self.lm_client.get_consistent_ocr_prompt(page_num)
            logger.debug(f"📝 Using consistent OCR prompt for page {page_num}")

            # Use pre-determined model (no dynamic detection per page)
            logger.debug(f"🔧 Using cached model: {model_to_use}")

            # Use cached vision capability check (done once at init)
            is_vision_capable = self._is_vision_capable_model(model_to_use)
            if not is_vision_capable and page_num == 1:  # Only warn once
                logger.warning(f"Model {model_to_use} may not support vision input!")
            
            # Use consistent system message for all pages (cached)
            system_message = self.lm_client.get_system_message_for_model(model_to_use)
            
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
            session = getattr(threading.current_thread(), 'session', None)
            if session is None:
                session = requests.Session()
                session.headers.update({
                    'Content-Type': 'application/json',
                    'Connection': 'keep-alive'
                })
                threading.current_thread().session = session
            
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
                        fixed_fallback = TableFormatter.fix_chemical_table_formatting(fallback_content, page_num)
                        return fixed_fallback
                    return None
                
                # Handle OCRFlux JSON format if not a generic response
                if model_to_use and ("ocrflux" in model_to_use.lower() or "flux" in model_to_use.lower()):
                    content = TableFormatter.parse_ocrflux_json(content)
                
                # For LMStudio output, just clean up the content without attempting table formatting
                if '|' in content and (content.count('|') > 20 or content.count('\n|') > 10):
                    logger.debug("Detected LMStudio output pattern - preserving raw output as suggested")
                    # Clean up any markdown code blocks or excessive whitespace only
                    content = re.sub(r'```\w*\n|```', '', content)
                    content = re.sub(r'\n{3,}', '\n\n', content)
                    # Don't attempt to format tables - user prefers raw output over formatting attempts
                else:
                    # For non-LMStudio patterns, still try the universal converter
                    content = TableFormatter.detect_and_convert_all_tables_to_markdown(content)

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
        fixed = TableFormatter.fix_chemical_table_formatting(content, page_num)

        # Step 2: Only convert tables if this isn't LMStudio's vertical format
        if '|' in fixed and (fixed.count('|') > 20 or fixed.count('\n|') > 10):
            # For LMStudio output, preserve the raw output format
            logger.debug("Preserving raw LMStudio output without table formatting")
            markdown_formatted = fixed
        else:
            # For other formats, still try to convert tables
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

    def format_page_content(self, page_data):
        """Format page content for output with consistent structure.

        Args:
            page_data: Page data

        Returns:
            Formatted content matching other processors' output quality
        """
        # If content is already formatted (from OCR), ensure it has proper structure
        if "content" in page_data and page_data["content"]:
            content = page_data["content"]

            # Always add page header for consistency
            page_num = page_data.get("page_number", "Unknown")
            if not content.startswith("# Page"):
                content = f"# Page {page_num} Content\n\n## Document Content\n\n{content}"

            return content

        content = []
        page_num = page_data.get("page_number", "Unknown")

        # Add page header for consistency
        content.append(f"# Page {page_num} Content")

        # Prioritize enhanced text if available
        if "enhanced_text" in page_data and page_data["enhanced_text"]:
            content.append("## Text Content\n")
            # Apply universal table converter to enhanced text
            enhanced_text = TableFormatter.detect_and_convert_all_tables_to_markdown(page_data["enhanced_text"])
            content.append(enhanced_text)
        elif "text" in page_data and page_data["text"]:
            content.append("## Text Content\n")
            # Apply universal table converter to basic text
            basic_text = TableFormatter.detect_and_convert_all_tables_to_markdown(page_data["text"])
            content.append(basic_text)

        # Add tables with proper formatting
        if "tables" in page_data and page_data["tables"]:
            content.append("## Tables\n")
            for i, table in enumerate(page_data["tables"], 1):
                if "markdown" in table:
                    table_id = table.get("table_id", f"table_{i}")
                    content.append(f"**Table {i} ({table_id}):**\n")
                    content.append(table["markdown"])

        # Add enhanced tables
        if "enhanced_tables" in page_data and page_data["enhanced_tables"]:
            if "tables" not in page_data or not page_data["tables"]:
                content.append("## Tables\n")
            for i, table in enumerate(page_data["enhanced_tables"], 1):
                if "markdown" in table:
                    table_id = table.get("table_id", f"enhanced_table_{i}")
                    content.append(f"**Enhanced Table {i} ({table_id}):**\n")
                    content.append(table["markdown"])

        return "\n\n".join(content)