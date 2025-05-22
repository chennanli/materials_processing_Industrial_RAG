#!/usr/bin/env python3
"""
Document Processor - Flexible Document Processing Framework
----------------------------------------------------------
Process documents using multiple processors based on content capabilities
with the ability to select which processors to use.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docling_ocr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DocumentProcessor")

# Import configuration
from config import PDF_DIR, OUTPUT_DIR

# Content type constants
class ContentType:
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    FORM = "form"
    CODE = "code"
    MATH = "math"
    DIAGRAM = "diagram"

# File type constants
class FileType:
    PDF = "pdf"
    IMAGE = "image"
    DOCUMENT = "document"  # Word, etc.
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    TEXT = "text"
    HTML = "html"
    UNKNOWN = "unknown"

class BaseProcessor:
    """Base class for all document processors."""
    
    def __init__(self, name: str):
        self.name = name
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get processor capabilities.
        
        Returns:
            Dictionary with supported content types and file types
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def process(self, file_path: str, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process a document file.
        
        Args:
            file_path: Path to the document file
            page_indices: Optional list of page indices to process
            
        Returns:
            Dictionary with processing results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_results(self, results: Dict[str, Any], output_dir: str, base_name: str) -> None:
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

# Import processor implementations
try:
    from processors.docling_processor import DoclingProcessor
except ImportError:
    logger.warning("Docling processor not available")
    DoclingProcessor = None

try:
    from processors.camelot_processor import CamelotProcessor
except ImportError:
    logger.warning("Camelot processor not available. Install with: pip install camelot-py[cv]")
    CamelotProcessor = None

try:
    from processors.gemini_processor import GeminiProcessor
except ImportError:
    logger.warning("Gemini processor not available")
    GeminiProcessor = None

try:
    from processors.lmstudio_processor import LMStudioProcessor
except ImportError:
    logger.warning("LMStudio processor not available")
    LMStudioProcessor = None
    
# Import fallback processor
from processors.fallback_processor import SimpleFallbackProcessor

class DocumentProcessor:
    """Main document processor that coordinates multiple processing backends based on capabilities."""
    
    def __init__(self, enabled_processors=None):
        """Initialize with selected processors.
        
        Args:
            enabled_processors: List of processor names to enable
                              ('docling', 'camelot', 'gemini', 'lmstudio', or 'all')
        """
        self.processors = {}
        self.analyzer = None
        
        # Initialize selected processors
        if enabled_processors is None or 'all' in enabled_processors:
            self._init_all_processors()
        else:
            # Always initialize Docling as the analyzer if available
            if DoclingProcessor is not None:
                self.analyzer = DoclingProcessor()
                
            # Initialize other processors if requested
            if 'docling' in enabled_processors and DoclingProcessor is not None:
                self.processors['docling'] = DoclingProcessor()
            if 'camelot' in enabled_processors and CamelotProcessor is not None:
                self.processors['camelot'] = CamelotProcessor()
            if 'gemini' in enabled_processors and GeminiProcessor is not None:
                self.processors['gemini'] = GeminiProcessor()
            if 'lmstudio' in enabled_processors and LMStudioProcessor is not None:
                self.processors['lmstudio'] = LMStudioProcessor()
        
        if not self.processors:
            logger.warning("No processors were enabled or available")
    
    def _init_all_processors(self):
        """Initialize all available processors."""
        # Always initialize Docling as the analyzer if available
        if DoclingProcessor is not None:
            try:
                self.analyzer = DoclingProcessor()
                self.processors['docling'] = DoclingProcessor()
                logger.info("Initialized Docling processor")
            except Exception as e:
                logger.warning(f"Failed to initialize Docling processor: {e}")
            
        if CamelotProcessor is not None:
            try:
                self.processors['camelot'] = CamelotProcessor()
                logger.info("Initialized Camelot processor")
            except Exception as e:
                logger.warning(f"Failed to initialize Camelot processor: {e}")
                
        if GeminiProcessor is not None:
            try:
                self.processors['gemini'] = GeminiProcessor()
                logger.info("Initialized Gemini processor")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini processor: {e}")
                
        if LMStudioProcessor is not None:
            try:
                self.processors['lmstudio'] = LMStudioProcessor()
                logger.info("Initialized LMStudio processor")
            except Exception as e:
                logger.warning(f"Failed to initialize LMStudio processor: {e}")
                
        # If no processors were initialized, use a simple fallback processor
        if not self.processors:
            logger.warning("No processors were available, using simple fallback processor")
            self.processors['fallback'] = SimpleFallbackProcessor()
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type (one of FileType constants)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Simple extension-based detection
        if extension in ['.pdf']:
            return FileType.PDF
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            return FileType.IMAGE
        elif extension in ['.doc', '.docx', '.odt', '.rtf']:
            return FileType.DOCUMENT
        elif extension in ['.xls', '.xlsx', '.ods', '.csv']:
            return FileType.SPREADSHEET
        elif extension in ['.ppt', '.pptx', '.odp']:
            return FileType.PRESENTATION
        elif extension in ['.txt', '.md', '.rst']:
            return FileType.TEXT
        elif extension in ['.html', '.htm']:
            return FileType.HTML
        else:
            return FileType.UNKNOWN
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Analyze document to determine content types.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with content analysis
        """
        # Use Docling for initial analysis if available
        if self.analyzer:
            try:
                # Quick analysis to determine content types
                analysis = self.analyzer.process(file_path)
                
                # Extract content types
                content_types = set()
                for page_num, page_data in analysis.items():
                    if "text_blocks" in page_data and page_data["text_blocks"]:
                        content_types.add(ContentType.TEXT)
                    if "tables" in page_data and page_data["tables"]:
                        content_types.add(ContentType.TABLE)
                    if "images" in page_data and page_data["images"]:
                        content_types.add(ContentType.IMAGE)
                
                return {
                    "file_type": self.detect_file_type(file_path),
                    "content_types": list(content_types),
                    "page_count": len(analysis),
                    "analysis": analysis
                }
            except Exception as e:
                logger.error(f"Error analyzing document with Docling: {e}")
        
        # Fallback to simple file type detection and basic content type inference
        file_type = self.detect_file_type(file_path)
        content_types = []
        
        # Infer content types based on file type
        if file_type == FileType.PDF:
            # PDFs typically contain text and might contain tables and images
            content_types = [ContentType.TEXT, ContentType.TABLE, ContentType.IMAGE]
        elif file_type == FileType.IMAGE:
            # Images obviously contain image content
            content_types = [ContentType.IMAGE]
        elif file_type in [FileType.DOCUMENT, FileType.TEXT]:
            # Documents and text files typically contain text
            content_types = [ContentType.TEXT]
        elif file_type == FileType.SPREADSHEET:
            # Spreadsheets typically contain tables
            content_types = [ContentType.TABLE, ContentType.TEXT]
        
        # Try to determine page count for PDFs
        page_count = 0
        if file_type == FileType.PDF:
            try:
                import fitz
                doc = fitz.open(file_path)
                page_count = len(doc)
            except Exception as e:
                logger.error(f"Error determining PDF page count: {e}")
        else:
            # Non-PDF files are treated as single-page documents
            page_count = 1
            
        return {
            "file_type": file_type,
            "content_types": content_types,
            "page_count": page_count,
            "analysis": {}
        }
    
    def select_processors_for_document(self, document_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Select appropriate processors based on document analysis.
        
        Args:
            document_analysis: Document analysis from analyze_document
            
        Returns:
            Dictionary mapping content types to processor names
        """
        file_type = document_analysis["file_type"]
        content_types = document_analysis["content_types"]
        
        # Map content types to processors
        content_processors = {}
        
        # Always use Docling for general document processing if available
        if 'docling' in self.processors:
            for content_type in content_types:
                content_processors.setdefault(content_type, []).append('docling')
        
        # Add specialized processors based on content and file types
        if file_type == FileType.PDF:
            # For tables in PDFs, prefer Camelot
            if ContentType.TABLE in content_types and 'camelot' in self.processors:
                content_processors.setdefault(ContentType.TABLE, []).append('camelot')
            
            # For text in PDFs, use LMStudio or Gemini as additional processors
            if ContentType.TEXT in content_types:
                if 'lmstudio' in self.processors:
                    content_processors.setdefault(ContentType.TEXT, []).append('lmstudio')
                if 'gemini' in self.processors:
                    content_processors.setdefault(ContentType.TEXT, []).append('gemini')
            
            # For images in PDFs, prefer Gemini or LMStudio
            if ContentType.IMAGE in content_types:
                if 'gemini' in self.processors:
                    content_processors.setdefault(ContentType.IMAGE, []).append('gemini')
                elif 'lmstudio' in self.processors:
                    content_processors.setdefault(ContentType.IMAGE, []).append('lmstudio')
        
        elif file_type == FileType.IMAGE:
            # For images, prefer Gemini or LMStudio
            if 'gemini' in self.processors:
                content_processors.setdefault(ContentType.IMAGE, []).append('gemini')
            elif 'lmstudio' in self.processors:
                content_processors.setdefault(ContentType.IMAGE, []).append('lmstudio')
        
        # For other file types, rely on Docling
        
        return content_processors
    
    def process_document(self, file_path: str, page_indices: Optional[List[int]] = None, output_dir: str = None) -> Dict[str, Any]:
        """Process document with enabled processors.
        
        Args:
            file_path: Path to the document file
            page_indices: Optional list of page indices to process
            output_dir: Directory to save results
            
        Returns:
            Dictionary with results from each processor
        """
        file_path = Path(file_path)
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base name for output files
        base_name = file_path.stem
        
        # Analyze document
        logger.info(f"Analyzing document: {file_path}")
        document_analysis = self.analyze_document(str(file_path))
        
        # Select processors based on analysis
        processor_mapping = self.select_processors_for_document(document_analysis)
        
        # Log selected processors
        for content_type, processors in processor_mapping.items():
            logger.info(f"Selected processors for {content_type}: {', '.join(processors)}")
        
        # Process with selected processors
        processor_results = {}
        
        # If we have no processors selected for any content type, use all available processors
        if not any(processors for processors in processor_mapping.values()):
            logger.warning("No processors selected based on content types, using all available processors")
            for processor_name, processor in self.processors.items():
                logger.info(f"Processing with {processor_name}")
                try:
                    # Process the document
                    results = processor.process(str(file_path), page_indices)
                    
                    # Save results
                    processor.save_results(results, str(output_dir), base_name)
                    
                    # Store results
                    processor_results[processor_name] = results
                    
                    logger.info(f"Completed processing with {processor_name}")
                except Exception as e:
                    logger.error(f"Error processing with {processor_name}: {e}")
        else:
            # Process with selected processors based on content types
            for processor_name, processor in self.processors.items():
                # Skip processors not selected for any content type
                if not any(processor_name in processors for processors in processor_mapping.values()):
                    continue
                    
                logger.info(f"Processing with {processor_name}")
                try:
                    # Process the document
                    results = processor.process(str(file_path), page_indices)
                    
                    # Save results
                    processor.save_results(results, str(output_dir), base_name)
                    
                    # Store results
                    processor_results[processor_name] = results
                    
                    logger.info(f"Completed processing with {processor_name}")
                except Exception as e:
                    logger.error(f"Error processing with {processor_name}: {e}")
        
        # Combine results from all processors
        combined_results = self._combine_results(processor_results, processor_mapping)
        
        # Save combined results
        self._save_combined_results(combined_results, str(output_dir), base_name)
        
        return {
            "processor_results": processor_results,
            "combined_results": combined_results,
            "document_analysis": document_analysis
        }
    
    def _combine_results(self, processor_results: Dict[str, Any], processor_mapping: Dict[str, List[str]]) -> Dict[str, Any]:
        """Combine results from multiple processors based on content type mapping.
        
        Args:
            processor_results: Results from each processor
            processor_mapping: Mapping of content types to processors
            
        Returns:
            Combined results
        """
        combined_results = {}
        
        # Get all page numbers from all processors
        all_pages = set()
        for processor_name, results in processor_results.items():
            all_pages.update(results.keys())
        
        # Combine results for each page
        for page_num in sorted(all_pages, key=lambda x: int(x)):
            combined_page = {
                "page_number": int(page_num),
                "content_sections": []
            }
            
            # Process each content type with its preferred processors
            for content_type, processors in processor_mapping.items():
                for processor_name in processors:
                    if processor_name in processor_results and page_num in processor_results[processor_name]:
                        processor_page = processor_results[processor_name][page_num]
                        
                        # Extract content based on content type
                        if content_type == ContentType.TABLE and "tables" in processor_page:
                            for table in processor_page["tables"]:
                                combined_page["content_sections"].append({
                                    "type": ContentType.TABLE,
                                    "content": table["markdown"] if "markdown" in table else "",
                                    "source": processor_name,
                                    "data": table
                                })
                        
                        elif content_type == ContentType.TEXT and "text" in processor_page:
                            combined_page["content_sections"].append({
                                "type": ContentType.TEXT,
                                "content": processor_page["text"],
                                "source": processor_name
                            })
                        
                        elif content_type == ContentType.IMAGE and "images" in processor_page:
                            for image in processor_page["images"]:
                                combined_page["content_sections"].append({
                                    "type": ContentType.IMAGE,
                                    "content": f"![Image]({image['image_id']})",
                                    "source": processor_name,
                                    "data": image
                                })
            
            # Format combined content
            combined_page["content"] = self._format_combined_content(combined_page["content_sections"])
            
            # Add to combined results
            combined_results[page_num] = combined_page
        
        return combined_results
    
    def _format_combined_content(self, content_sections: List[Dict[str, Any]]) -> str:
        """Format combined content sections.
        
        Args:
            content_sections: List of content sections
            
        Returns:
            Formatted content
        """
        formatted_sections = []
        
        for section in content_sections:
            if section["type"] == ContentType.TABLE:
                formatted_sections.append(f"\n\n**Table (from {section['source']}):**\n\n{section['content']}")
            elif section["type"] == ContentType.TEXT:
                formatted_sections.append(f"{section['content']}")
            elif section["type"] == ContentType.IMAGE:
                formatted_sections.append(f"\n\n{section['content']} (from {section['source']})\n\n")
            else:
                formatted_sections.append(f"\n\n{section['content']}\n\n")
        
        return "\n\n".join(formatted_sections)
    
    def _save_combined_results(self, combined_results: Dict[str, Any], output_dir: str, base_name: str) -> None:
        """Save combined results.
        
        Args:
            combined_results: Combined results
            output_dir: Directory to save results
            base_name: Base name for output files
        """
        # Create output directory
        combined_dir = Path(output_dir) / f"{base_name}_combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined markdown
        combined_md = combined_dir / f"{base_name}_combined.md"
        with open(combined_md, "w", encoding="utf-8") as f:
            f.write(f"# {base_name} - Combined Results\n\n")
            for page_num, page_data in sorted(combined_results.items(), key=lambda x: int(x[0])):
                f.write(f"## Page {page_num}\n\n")
                f.write(page_data["content"])
                f.write("\n\n---\n\n")
        
        # Save combined JSON
        combined_json = combined_dir / f"{base_name}_combined.json"
        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        # Save individual page results
        pages_dir = combined_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        
        for page_num, page_data in combined_results.items():
            # Save as markdown
            md_path = pages_dir / f"{base_name}_page{page_num}_content.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Page {page_num}\n\n")
                f.write(page_data["content"])
            
            # Save as JSON
            json_path = pages_dir / f"{base_name}_page{page_num}_content.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)

def parse_page_indices(pages_str: Optional[str]) -> Optional[List[int]]:
    """Parse page indices from string like '1-3,5,7-9'.
    
    Args:
        pages_str: String representation of page ranges
        
    Returns:
        List of page indices (1-based)
    """
    if not pages_str:
        return None
        
    page_indices = []
    parts = pages_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle page range (e.g., '1-3')
            try:
                start, end = map(int, part.split('-'))
                page_indices.extend(range(start, end + 1))
            except ValueError:
                logger.warning(f"Invalid page range: {part}")
        else:
            # Handle single page (e.g., '5')
            try:
                page_indices.append(int(part))
            except ValueError:
                logger.warning(f"Invalid page number: {part}")
    
    return page_indices if page_indices else None

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process documents using multiple processors based on content capabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "input", 
        nargs="?",
        help="Path to the source document file or directory containing documents"
    )
    
    # Processor selection
    parser.add_argument(
        "--processors", 
        choices=["all", "docling", "gemini", "lmstudio", "camelot"],
        nargs="+", 
        default=["all"],
        help="Which processors to use for processing"
    )
    
    # Page selection
    parser.add_argument(
        "--pages", 
        help="Pages to process (e.g., '1-3,5,7-9'). Default: all pages"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        help=f"Output directory for results. Default: {OUTPUT_DIR}"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        # List available documents
        pdfs = list(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            logger.error(f"No documents found in {PDF_DIR}")
            return 1
            
        print("Available documents:")
        for i, pdf in enumerate(pdfs):
            print(f"{i+1}. {pdf.name}")
            
        # Ask user to select a document
        selection = input("Select a document (number): ")
        try:
            index = int(selection) - 1
            if 0 <= index < len(pdfs):
                input_path = pdfs[index]
            else:
                logger.error(f"Invalid selection: {selection}")
                return 1
        except ValueError:
            logger.error(f"Invalid selection: {selection}")
            return 1
    
    # Determine output directory
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse page indices
    page_indices = parse_page_indices(args.pages) if args.pages else None
    
    # Process documents
    if input_path.is_file():
        # Process single document
        process_single_document(input_path, args.processors, page_indices, output_dir)
    elif input_path.is_dir():
        # Process all documents in directory
        process_document_directory(input_path, args.processors, page_indices, output_dir)
    else:
        logger.error(f"Invalid input: {input_path} (not a file or directory)")
        return 1
    
    return 0

def process_single_document(file_path, processors, page_indices, output_dir):
    """Process a single document file.
    
    Args:
        file_path: Path to the document file
        processors: List of processors to use
        page_indices: List of page indices to process
        output_dir: Output directory
    """
    logger.info(f"Processing document: {file_path}")
    logger.info(f"Using processors: {', '.join(processors)}")
    
    # Initialize processor
    doc_processor = DocumentProcessor(enabled_processors=processors)
    
    # Process document
    results = doc_processor.process_document(
        file_path=str(file_path),
        page_indices=page_indices,
        output_dir=str(output_dir)
    )
    
    # Print summary
    logger.info(f"Processing complete. Results saved to: {output_dir}")
    for processor_name in results["processor_results"]:
        num_pages = len(results["processor_results"][processor_name])
        logger.info(f"  - {processor_name}: Processed {num_pages} pages")
    
    # Print combined results
    num_pages = len(results["combined_results"])
    logger.info(f"  - Combined: {num_pages} pages")
    
    # Print content types found
    content_types = results["document_analysis"].get("content_types", [])
    if content_types:
        logger.info(f"Content types found: {', '.join(content_types)}")

def process_document_directory(dir_path, processors, page_indices, output_dir):
    """Process all documents in a directory.
    
    Args:
        dir_path: Path to the directory containing documents
        processors: List of processors to use
        page_indices: List of page indices to process
        output_dir: Output directory
    """
    # Find all documents (currently just PDFs, but could be expanded)
    documents = list(dir_path.glob("*.pdf"))
    documents.extend(dir_path.glob("*.docx"))
    documents.extend(dir_path.glob("*.jpg"))
    documents.extend(dir_path.glob("*.png"))
    
    if not documents:
        logger.warning(f"No supported documents found in {dir_path}")
        return
    
    logger.info(f"Found {len(documents)} documents in {dir_path}")
    
    # Process each document
    for doc_path in documents:
        process_single_document(doc_path, processors, page_indices, output_dir)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error processing documents: {e}")
        sys.exit(1)
