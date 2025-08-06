#!/usr/bin/env python3
"""
Parallel Processor Implementation
--------------------------------
Process pages in parallel for better performance.
"""

import concurrent.futures
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

import re
from .table_formatter import TableFormatter

# Configure logging
logger = logging.getLogger("ParallelProcessor")

class ParallelProcessor:
    """Process pages in parallel for better performance."""

    def __init__(self, api_url, lm_client, image_processor):
        """Initialize the parallel processor.

        Args:
            api_url: LM Studio API URL
            lm_client: LM Studio client instance
            image_processor: Image processor instance
        """
        self.api_url = api_url
        self.lm_client = lm_client
        self.image_processor = image_processor
        self._processing_times = []  # Track processing times for optimization

    def process_pages_parallel(self, page_images, pdf_path, progress_callback=None):
        """Process multiple pages in parallel for better performance.
        
        Args:
            page_images: List of (page_num, image) tuples
            pdf_path: Path to PDF file for fallback
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary of processing results
        """
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
            text_content = self.image_processor.process_image_with_lmstudio(image, page_num)
            
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
                formatted_content = self.image_processor.format_page_content({
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
                    formatted_content = self.image_processor.format_page_content({
                        "page_number": page_num,
                        "content": processed_content
                    })
                    basic_results['content'] = formatted_content
                    basic_results['text'] = formatted_content
                return basic_results
        except Exception as e:
            logger.error(f"Fallback processing failed for page {page_num}: {e}")
        
        return None

    def _extract_single_page_content(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
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

    def _table_to_markdown(self, table_data):
        """Convert table data to markdown format.

        Args:
            table_data: Table data as list of lists

        Returns:
            Table in markdown format
        """
        if not table_data:
            return ""

        try:
            import pandas as pd

            # Create DataFrame
            df = pd.DataFrame(table_data)

            # Convert to markdown
            md = "| " + " | ".join(str(col) for col in df.iloc[0]) + " |\n"
            md += "| " + " | ".join("---" for _ in df.iloc[0]) + " |\n"

            for _, row in df.iloc[1:].iterrows():
                md += "| " + " | ".join(str(cell) for cell in row) + " |\n"

            return md
        except Exception as e:
            logger.error(f"Error converting table to markdown: {e}")
            return ""

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
        
        # For LMStudio output, keep the original format without trying to convert to tables
        if '|' in fixed and (fixed.count('|') > 20 or fixed.count('\n|') > 10):
            logger.debug(f"Detected LMStudio output on page {page_num} - preserving raw output as suggested")
            # Just clean up any markdown code blocks or excessive whitespace
            markdown_formatted = re.sub(r'```\w*\n|```', '', fixed)
            markdown_formatted = re.sub(r'\n{3,}', '\n\n', markdown_formatted)
            # No further formatting - user prefers raw output over table formatting attempts
        else:
            # For non-LMStudio patterns, still try the universal converter
            markdown_formatted = TableFormatter.detect_and_convert_all_tables_to_markdown(fixed)

        logger.debug(f"Post-processing complete for page {page_num} ({len(markdown_formatted)} chars)")
        return markdown_formatted