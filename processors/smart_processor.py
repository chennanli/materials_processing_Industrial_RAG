#!/usr/bin/env python3
"""
Smart Document Processor Implementation
--------------------------------------
Intelligently routes pages to the best processor based on content analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

# Import all processors with error handling
try:
    from .docling_processor import DoclingProcessor
except ImportError:
    DoclingProcessor = None

try:
    from .camelot_processor import CamelotProcessor
except ImportError:
    CamelotProcessor = None

try:
    from .gemini_processor import GeminiProcessor
except ImportError:
    GeminiProcessor = None

try:
    from .lmstudio_processor import LMStudioProcessor
except ImportError:
    LMStudioProcessor = None

try:
    from .fallback_processor import FallbackProcessor
except ImportError:
    FallbackProcessor = None

# Configure logging
logger = logging.getLogger("SmartProcessor")


class SmartDocumentProcessor:
    """Intelligently combines multiple processors for optimal results."""

    def __init__(self, use_gemini_for_complex: bool = False):
        """Initialize the smart processor.
        
        Args:
            use_gemini_for_complex: Whether to use Gemini for complex visual documents
        """
        self.name = "smart"
        self.use_gemini_for_complex = use_gemini_for_complex
        
        # Initialize processors
        self.processors = {}
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Initialize available processors."""
        # Always try to initialize Docling (primary processor)
        if DoclingProcessor is not None:
            try:
                self.processors['docling'] = DoclingProcessor()
                logger.info("✅ Docling processor initialized")
            except Exception as e:
                logger.warning(f"❌ Docling not available: {e}")
        else:
            logger.warning("❌ Docling not available (not installed)")
            
        # Initialize Camelot for tables
        if CamelotProcessor is not None:
            try:
                self.processors['camelot'] = CamelotProcessor()
                logger.info("✅ Camelot processor initialized")
            except Exception as e:
                logger.warning(f"❌ Camelot not available: {e}")
        else:
            logger.warning("❌ Camelot not available (not installed)")
            
        # Initialize LMStudio for OCR
        if LMStudioProcessor is not None:
            try:
                self.processors['lmstudio'] = LMStudioProcessor()
                logger.info("✅ LMStudio processor initialized (using MonkeyOCR by default)")
            except Exception as e:
                logger.warning(f"❌ LMStudio not available: {e}")
        else:
            logger.warning("❌ LMStudio not available (not installed)")
            
        # Initialize Gemini if requested
        if self.use_gemini_for_complex and GeminiProcessor is not None:
            try:
                self.processors['gemini'] = GeminiProcessor()
                logger.info("✅ Gemini processor initialized")
            except Exception as e:
                logger.warning(f"❌ Gemini not available: {e}")
        elif self.use_gemini_for_complex:
            logger.warning("❌ Gemini not available (not installed)")
                
        # Always have fallback
        if FallbackProcessor is not None:
            self.processors['fallback'] = FallbackProcessor()
            logger.info("✅ Fallback processor initialized")
        else:
            logger.error("❌ No processors available - even fallback failed!")
        
    def analyze_page_content(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze page content to determine best processing strategy.
        
        Args:
            page_data: Page data from initial Docling analysis
            
        Returns:
            Classification of page content
        """
        text = page_data.get('text', '')
        tables = page_data.get('tables', [])
        
        # Calculate metrics
        text_length = len(text)
        num_tables = len(tables)
        
        # Detect if page might be scanned/image-based
        # Heuristic: very little text but page exists
        is_likely_scanned = text_length < 100
        
        # Detect if tables might need better extraction
        needs_better_tables = num_tables > 0 and any(
            len(table.get('data', [])) < 3 for table in tables
        )
        
        # Check text quality
        if text_length > 500:
            text_quality = 'good'
        elif text_length > 100:
            text_quality = 'medium'
        else:
            text_quality = 'poor'
            
        return {
            'text_length': text_length,
            'text_quality': text_quality,
            'has_tables': num_tables > 0,
            'num_tables': num_tables,
            'needs_better_tables': needs_better_tables,
            'is_likely_scanned': is_likely_scanned,
            'has_content': text_length > 0 or num_tables > 0
        }
        
    def process(self, pdf_path: str, page_indices: Optional[List[int]] = None, progress_callback = None) -> Dict[str, Any]:
        """Process a PDF file using intelligent routing.
        
        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"🧠 Smart processor starting on {pdf_path}")
        start_time = time.time()
        
        # Phase 1: Initial analysis with Docling (if available)
        if progress_callback:
            progress_callback("analyzing")
            
        if 'docling' in self.processors:
            logger.info("📄 Phase 1: Analyzing document structure with Docling...")
            docling_results = self.processors['docling'].process(pdf_path, page_indices, progress_callback)
            base_results = docling_results.copy()
        else:
            logger.info("📄 Phase 1: Using fallback for initial analysis...")
            base_results = self.processors['fallback'].process(pdf_path, page_indices, progress_callback)
            
        # Phase 2: Analyze each page and determine best processors
        logger.info("🔍 Phase 2: Analyzing pages and routing to specialized processors...")
        
        if progress_callback:
            progress_callback("selecting_processors")
        pages_needing_ocr = []
        pages_needing_tables = []
        pages_needing_vision = []
        
        for page_num, page_data in base_results.items():
            classification = self.analyze_page_content(page_data)
            
            # Determine which processors to use
            if classification['is_likely_scanned'] or classification['text_quality'] == 'poor':
                pages_needing_ocr.append(int(page_num))
                logger.info(f"   Page {page_num}: Needs OCR (text quality: {classification['text_quality']})")
                
            if classification['needs_better_tables'] and 'camelot' in self.processors:
                pages_needing_tables.append(int(page_num))
                logger.info(f"   Page {page_num}: Needs better table extraction")
                
            # Only use Gemini if explicitly enabled and page is complex
            if (self.use_gemini_for_complex and 
                classification['text_quality'] == 'poor' and 
                'gemini' in self.processors):
                pages_needing_vision.append(int(page_num))
                logger.info(f"   Page {page_num}: Complex content, using Gemini")
                
        # Phase 3: Process with specialized processors
        logger.info("🔧 Phase 3: Processing with specialized extractors...")
        
        if progress_callback:
            progress_callback("processing_start")
        
        # Process OCR pages with LMStudio
        if pages_needing_ocr and 'lmstudio' in self.processors:
            logger.info(f"   🖼️ Processing {len(pages_needing_ocr)} pages with LMStudio (MonkeyOCR)...")
            ocr_results = self.processors['lmstudio'].process(pdf_path, pages_needing_ocr, progress_callback)
            
            # Merge OCR results (OCR takes precedence for these pages)
            for page_num, ocr_data in ocr_results.items():
                if page_num in base_results:
                    base_results[page_num] = ocr_data
                    base_results[page_num]['processor_used'] = 'lmstudio'
                    
        # Process table pages with Camelot
        if pages_needing_tables and 'camelot' in self.processors:
            logger.info(f"   📊 Processing {len(pages_needing_tables)} pages with Camelot...")
            table_results = self.processors['camelot'].process(pdf_path, pages_needing_tables, progress_callback)
            
            # Merge table results (keep text from base, update tables from Camelot)
            for page_num, table_data in table_results.items():
                if page_num in base_results and table_data.get('tables'):
                    base_results[page_num]['tables'] = table_data['tables']
                    base_results[page_num]['camelot_tables'] = True
                    
        # Process complex pages with Gemini (if enabled)
        if pages_needing_vision and 'gemini' in self.processors:
            logger.info(f"   🌟 Processing {len(pages_needing_vision)} pages with Gemini...")
            gemini_results = self.processors['gemini'].process(pdf_path, pages_needing_vision, progress_callback)
            
            # Merge Gemini results for complex visual analysis
            for page_num, gemini_data in gemini_results.items():
                if page_num in base_results:
                    # Keep Gemini's comprehensive analysis
                    base_results[page_num] = gemini_data
                    base_results[page_num]['processor_used'] = 'gemini'
                    
        # Phase 4: Quality assurance
        logger.info("✅ Phase 4: Quality assurance and final formatting...")
        
        if progress_callback:
            progress_callback("combining_results")
            
        final_results = self._quality_assurance(base_results, pdf_path, progress_callback)
        
        # Add metadata
        elapsed_time = time.time() - start_time
        logger.info(f"🎯 Smart processing completed in {elapsed_time:.2f} seconds")
        
        # Add processing summary
        processing_summary = {
            'total_pages': len(final_results),
            'processors_used': list(set(
                page_data.get('processor_used', 'docling') 
                for page_data in final_results.values()
            )),
            'pages_with_ocr': len(pages_needing_ocr),
            'pages_with_tables': len(pages_needing_tables),
            'processing_time': elapsed_time
        }
        
        logger.info(f"📊 Summary: {processing_summary}")
        
        return final_results
        
    def _quality_assurance(self, results: Dict[str, Any], pdf_path: str, progress_callback = None) -> Dict[str, Any]:
        """Perform quality checks and ensure all pages have content.
        
        Args:
            results: Current results
            pdf_path: Path to PDF file
            
        Returns:
            Quality assured results
        """
        for page_num, page_data in results.items():
            # Ensure content field exists
            if 'content' not in page_data:
                page_data['content'] = self._format_page_content(page_data)
                
            # Check if page has meaningful content
            content_length = len(page_data.get('content', ''))
            if content_length < 50:
                logger.warning(f"⚠️ Page {page_num} has minimal content ({content_length} chars)")
                
                # Last resort: try fallback if not already used
                if page_data.get('processor_used') != 'fallback':
                    logger.info(f"   Attempting fallback extraction for page {page_num}...")
                    fallback_results = self.processors['fallback'].process(
                        pdf_path, [int(page_num)], progress_callback
                    )
                    if fallback_results.get(page_num):
                        fallback_text = fallback_results[page_num].get('text', '')
                        if len(fallback_text) > content_length:
                            page_data['fallback_text'] = fallback_text
                            page_data['content'] = fallback_text
                            
        return results
        
    def _format_page_content(self, page_data: Dict[str, Any]) -> str:
        """Format page content for output.
        
        Args:
            page_data: Page data dictionary
            
        Returns:
            Formatted content string
        """
        content_parts = []
        
        # Add main text
        if page_data.get('text'):
            content_parts.append(page_data['text'])
            
        # Add tables
        if page_data.get('tables'):
            for i, table in enumerate(page_data['tables']):
                content_parts.append(f"\n\n**Table {i+1}:**\n")
                if 'markdown' in table:
                    content_parts.append(table['markdown'])
                elif 'data' in table:
                    # Convert data to simple text representation
                    for row in table['data']:
                        content_parts.append(' | '.join(str(cell) for cell in row))
                        
        # Add any fallback text if main content is minimal
        if len(''.join(content_parts)) < 50 and page_data.get('fallback_text'):
            content_parts.append(f"\n\n**Fallback extraction:**\n{page_data['fallback_text']}")
            
        return '\n'.join(content_parts)
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about the smart processor configuration.
        
        Returns:
            Processor information
        """
        return {
            'name': 'smart',
            'description': 'Intelligent document processor that routes pages to optimal extractors',
            'available_processors': list(self.processors.keys()),
            'routing_strategy': {
                'primary': 'docling' if 'docling' in self.processors else 'fallback',
                'ocr': 'lmstudio' if 'lmstudio' in self.processors else None,
                'tables': 'camelot' if 'camelot' in self.processors else None,
                'complex': 'gemini' if 'gemini' in self.processors and self.use_gemini_for_complex else None
            },
            'features': [
                'Intelligent page routing',
                'Multi-processor fusion',
                'Quality assurance',
                'Automatic fallback'
            ]
        }