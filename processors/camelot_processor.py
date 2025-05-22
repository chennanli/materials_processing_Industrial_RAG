#!/usr/bin/env python3
"""
Camelot Processor Implementation
-------------------------------
Process PDFs using Camelot for specialized table extraction.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import fitz  # PyMuPDF
import pandas as pd
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not installed. Install with: pip install camelot-py[cv]")

# Import from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logger = logging.getLogger("CamelotProcessor")

class CamelotProcessor:
    """Process PDFs using Camelot for specialized table extraction."""
    
    def __init__(self):
        """Initialize the Camelot processor."""
        self.name = "camelot"
        
        if not CAMELOT_AVAILABLE:
            raise ImportError("Camelot is not installed. Install with: pip install camelot-py[cv]")
    
    def process(self, pdf_path: str, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process a PDF file with Camelot.
        
        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)
            
        Returns:
            Dictionary with processing results
        """
        # Determine which pages to process
        if page_indices:
            pages = ','.join(str(p) for p in page_indices)
        else:
            # Get total pages
            doc = fitz.open(pdf_path)
            pages = ','.join(str(p+1) for p in range(len(doc)))
        
        # Extract tables with Camelot
        results = {}
        
        try:
            # Try lattice mode first (for tables with borders)
            lattice_tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor='lattice',
                process_background=True
            )
            
            # Then try stream mode (for tables without borders)
            stream_tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor='stream',
                edge_tol=100,
                process_background=True
            )
            
            # Combine and process tables
            all_tables = self._process_tables(lattice_tables, stream_tables)
            
            # Organize by page
            for table_info in all_tables:
                page_num = str(table_info["page"])
                if page_num not in results:
                    results[page_num] = {
                        "page_number": int(page_num),
                        "tables": [],
                        "text": ""  # Camelot doesn't extract general text
                    }
                
                results[page_num]["tables"].append(table_info)
            
            # Format content for each page
            for page_num, page_data in results.items():
                page_data["content"] = self._format_page_content(page_data)
            
            logger.info(f"Extracted {len(all_tables)} tables from {len(results)} pages")
            
        except Exception as e:
            logger.error(f"Error extracting tables with Camelot: {e}")
        
        return results
    
    def _process_tables(self, lattice_tables, stream_tables):
        """Process tables from both lattice and stream mode.
        
        Args:
            lattice_tables: Tables extracted with lattice mode
            stream_tables: Tables extracted with stream mode
            
        Returns:
            List of processed tables
        """
        all_tables = []
        
        # Process lattice tables
        for i, table in enumerate(lattice_tables):
            if table.df.empty:
                continue
                
            # Calculate accuracy score
            accuracy = table.accuracy
            
            # Only include if accuracy is reasonable
            if accuracy > 50:  # Adjust threshold as needed
                all_tables.append({
                    "table_id": f"lattice_table_{i+1}",
                    "page": table.page,
                    "data": table.df.values.tolist(),
                    "accuracy": accuracy,
                    "flavor": "lattice",
                    "markdown": table.df.to_markdown(index=False),
                    "html": table.df.to_html(index=False)
                })
        
        # Process stream tables
        for i, table in enumerate(stream_tables):
            if table.df.empty:
                continue
                
            # Calculate accuracy score
            accuracy = table.accuracy
            
            # Only include if accuracy is reasonable and not duplicate
            if accuracy > 50:  # Adjust threshold as needed
                # Check if this table overlaps significantly with any lattice table
                is_duplicate = False
                for lt in all_tables:
                    if lt["page"] == table.page:
                        # Simple duplicate check - could be improved
                        if len(lt["data"]) == len(table.df.values.tolist()):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    all_tables.append({
                        "table_id": f"stream_table_{i+1}",
                        "page": table.page,
                        "data": table.df.values.tolist(),
                        "accuracy": accuracy,
                        "flavor": "stream",
                        "markdown": table.df.to_markdown(index=False),
                        "html": table.df.to_html(index=False)
                    })
        
        return all_tables
    
    def _format_page_content(self, page_data):
        """Format page content for output.
        
        Args:
            page_data: Page data
            
        Returns:
            Formatted content
        """
        content = []
        
        # Add text (Camelot doesn't extract general text)
        if "text" in page_data and page_data["text"]:
            content.append(page_data["text"])
        
        # Add tables
        if "tables" in page_data and page_data["tables"]:
            for table in page_data["tables"]:
                if "markdown" in table:
                    content.append(f"\n\n**Table {table['table_id']} (Accuracy: {table['accuracy']:.1f}%, Method: {table['flavor']}):**\n\n{table['markdown']}")
        
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
                    
        # Save HTML tables
        html_dir = processor_dir / "html"
        html_dir.mkdir(exist_ok=True)
        
        for page_num, page_data in results.items():
            if "tables" in page_data and page_data["tables"]:
                for table in page_data["tables"]:
                    if "html" in table:
                        html_path = html_dir / f"{base_name}_page{page_num}_{table['table_id']}.html"
                        with open(html_path, "w", encoding="utf-8") as f:
                            f.write(f"<h1>Table {table['table_id']}</h1>\n")
                            f.write(f"<p>Accuracy: {table['accuracy']:.1f}%, Method: {table['flavor']}</p>\n")
                            f.write(table["html"])
