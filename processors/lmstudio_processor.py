#!/usr/bin/env python3
"""
LM Studio Processor Implementation
---------------------------------
Process PDFs using LM Studio for text extraction and table enhancement.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import fitz  # PyMuPDF
import pandas as pd
import requests

# Import from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import LMSTUDIO_URL, LMSTUDIO_MODEL

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
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["</answer>"]
            }
            
            response = requests.post(
                f"{self.api_url}/v1/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"].strip()
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
    
    def process(self, pdf_path: str, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process a PDF file with LM Studio.
        
        Args:
            pdf_path: Path to the PDF file
            page_indices: Optional list of page indices to process (1-based)
            
        Returns:
            Dictionary with processing results
        """
        # Extract text and tables from PDF
        results = self._extract_content_from_pdf(pdf_path, page_indices)
        
        # Enhance tables if available
        for page_num, page_data in results.items():
            if "tables" in page_data and page_data["tables"]:
                enhanced_tables = []
                for table in page_data["tables"]:
                    enhanced = self._enhance_table(table)
                    if enhanced:
                        enhanced_tables.append(enhanced)
                
                if enhanced_tables:
                    page_data["enhanced_tables"] = enhanced_tables
            
            # Format content for output
            page_data["content"] = self._format_page_content(page_data)
        
        return results
    
    def _extract_content_from_pdf(self, pdf_path: str, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
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
            pages_to_process = [p-1 for p in page_indices if 0 < p <= len(doc)]  # Convert to 0-based
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
                        tables.append({
                            "table_id": f"page{page_num}_table{i+1}",
                            "data": table_data,
                            "markdown": self._table_to_markdown(table_data)
                        })
                
                # Store results
                results[str(page_num)] = {
                    "page_number": page_num,
                    "text": text,
                    "tables": tables
                }
                
                logger.info(f"Extracted content from page {page_num}: {len(text)} chars, {len(tables)} tables")
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
            Enhanced table
        """
        if not self.lm_client or "markdown" not in table:
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
            result = self.lm_client.generate(prompt, max_tokens=2048)
            if result:
                return {
                    "table_id": table["table_id"],
                    "markdown": result,
                    "enhanced": True
                }
            return None
        except Exception as e:
            logger.error(f"Error enhancing table: {e}")
            return None
    
    def _format_page_content(self, page_data):
        """Format page content for output.
        
        Args:
            page_data: Page data
            
        Returns:
            Formatted content
        """
        content = []
        
        # Add text
        if "text" in page_data and page_data["text"]:
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
