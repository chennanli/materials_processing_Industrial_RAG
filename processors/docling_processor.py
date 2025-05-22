#!/usr/bin/env python3
"""
Docling Processor Implementation
-------------------------------
Process documents using Docling's comprehensive document understanding capabilities.
This processor can handle multiple document types, not just PDFs.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from docling.document_converter import DocumentConverter
from docling.document import Document

# Configure logging
logger = logging.getLogger("DoclingProcessor")

class DoclingProcessor:
    """Process documents using Docling's document understanding capabilities."""
    
    def __init__(self, extract_tables=True, extract_images=True):
        """Initialize the Docling processor.
        
        Args:
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
        """
        self.name = "docling"
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        
        # Initialize Docling converter
        self.converter = DocumentConverter()
    
    def process(self, file_path: str, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process a document file with Docling.
        
        Args:
            file_path: Path to the document file
            page_indices: Optional list of page indices to process (0-based for Docling)
            
        Returns:
            Dictionary with processing results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {}
            
        file_extension = file_path.suffix.lower()
        
        # Process the document
        try:
            # Convert the document using Docling
            logger.info(f"Processing {file_path} with Docling")
            
            # If page indices are specified and it's a PDF, extract those pages first
            if page_indices and file_extension == '.pdf':
                temp_file = self._extract_pages_to_temp_file(file_path, page_indices)
                result = self.converter.convert(temp_file)
                # Clean up temp file
                os.unlink(temp_file)
            else:
                result = self.converter.convert(str(file_path))
            
            # Extract document content
            doc = result.document
            
            # Process document structure
            return self._process_document(doc)
            
        except Exception as e:
            logger.error(f"Error processing document with Docling: {e}")
            return {}
    
    def _extract_pages_to_temp_file(self, pdf_path: str, page_indices: List[int]) -> str:
        """Extract specific pages from a PDF to a temporary file.
        
        Args:
            pdf_path: Path to the PDF file
            page_indices: List of page indices to extract (0-based)
            
        Returns:
            Path to the temporary file
        """
        from PyPDF2 import PdfReader, PdfWriter
        
        logger.info(f"Extracting pages {page_indices} from {pdf_path}")
        
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        for idx in page_indices:
            if 0 <= idx < len(reader.pages):
                writer.add_page(reader.pages[idx])
            else:
                logger.warning(f"Page index {idx} out of range (0-{len(reader.pages)-1})")
        
        # Create a temporary file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        with open(temp.name, 'wb') as f:
            writer.write(f)
        
        return temp.name
    
    def _process_document(self, doc: Document) -> Dict[str, Any]:
        """Process a Docling Document object.
        
        Args:
            doc: Docling Document object
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        # Process each page
        for i, page in enumerate(doc.pages):
            page_num = i + 1  # Convert to 1-based for consistency with other processors
            
            # Extract text content
            text_blocks = []
            for block in page.blocks:
                text_blocks.append({
                    "text": block.text,
                    "bbox": block.bbox.to_dict() if hasattr(block, "bbox") else None,
                    "type": block.type
                })
            
            # Extract tables if enabled
            tables = []
            if self.extract_tables:
                for table in page.find_tables():
                    table_data = []
                    for row in table.rows:
                        table_row = []
                        for cell in row.cells:
                            table_row.append(cell.text)
                        table_data.append(table_row)
                    
                    tables.append({
                        "table_id": f"page{page_num}_table{len(tables)+1}",
                        "data": table_data,
                        "markdown": self._table_to_markdown(table_data)
                    })
            
            # Extract images if enabled
            images = []
            if self.extract_images:
                for img in page.images:
                    images.append({
                        "image_id": f"page{page_num}_image{len(images)+1}",
                        "bbox": img.bbox.to_dict() if hasattr(img, "bbox") else None,
                        "alt_text": img.alt_text if hasattr(img, "alt_text") else ""
                    })
            
            # Store page results
            results[str(page_num)] = {
                "page_number": page_num,
                "text_blocks": text_blocks,
                "tables": tables,
                "images": images,
                "content": self._format_page_content(text_blocks, tables, images)
            }
        
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
        
        # Create markdown table
        md = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |\n"
        md += "| " + " | ".join("---" for _ in table_data[0]) + " |\n"
        
        for row in table_data[1:]:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return md
    
    def _format_page_content(self, text_blocks, tables, images):
        """Format page content for output.
        
        Args:
            text_blocks: List of text blocks
            tables: List of tables
            images: List of images
            
        Returns:
            Formatted content
        """
        content = []
        
        # Add text blocks
        for block in text_blocks:
            content.append(block["text"])
        
        # Add tables
        for table in tables:
            content.append("\n\n**Table:**\n\n" + table["markdown"])
        
        # Add image references
        for image in images:
            alt_text = image["alt_text"] if image["alt_text"] else f"Image {image['image_id']}"
            content.append(f"\n\n![{alt_text}]({image['image_id']})\n\n")
        
        return "\n\n".join(content)
    
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
