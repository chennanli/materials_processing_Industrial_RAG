#!/usr/bin/env python3
"""
Compare Results Tool
-------------------
Compare results from different processors to identify differences and similarities.
"""

import argparse
import json
import os
from pathlib import Path
import difflib
import sys
from typing import Dict, List, Any

def load_results(output_dir: str, base_name: str) -> Dict[str, Dict[str, Any]]:
    """Load results from all processor directories.
    
    Args:
        output_dir: Output directory
        base_name: Base name of the document
        
    Returns:
        Dictionary mapping processor names to their results
    """
    output_dir = Path(output_dir)
    results = {}
    
    # Find all processor directories
    for processor_dir in output_dir.glob(f"{base_name}_*"):
        if not processor_dir.is_dir() or processor_dir.name == f"{base_name}_combined":
            continue
            
        processor_name = processor_dir.name.replace(f"{base_name}_", "")
        json_file = processor_dir / f"{base_name}_combined.json"
        
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    results[processor_name] = json.load(f)
                print(f"Loaded results from {processor_name}")
            except Exception as e:
                print(f"Error loading results from {processor_name}: {e}")
    
    return results

def extract_page_content(results: Dict[str, Dict[str, Any]], page_num: str) -> Dict[str, str]:
    """Extract content for a specific page from all processors.
    
    Args:
        results: Results from all processors
        page_num: Page number to extract
        
    Returns:
        Dictionary mapping processor names to their content for the page
    """
    page_content = {}
    
    for processor_name, processor_results in results.items():
        if page_num in processor_results:
            page_data = processor_results[page_num]
            if isinstance(page_data, dict) and "content" in page_data:
                page_content[processor_name] = page_data["content"]
            elif isinstance(page_data, dict) and "text" in page_data:
                page_content[processor_name] = page_data["text"]
    
    return page_content

def compare_page_content(page_content: Dict[str, str]) -> None:
    """Compare content for a page across processors.
    
    Args:
        page_content: Dictionary mapping processor names to their content
    """
    processors = list(page_content.keys())
    
    if len(processors) < 2:
        print("Not enough processors to compare")
        return
    
    # Compare each pair of processors
    for i in range(len(processors)):
        for j in range(i+1, len(processors)):
            proc1 = processors[i]
            proc2 = processors[j]
            
            content1 = page_content[proc1]
            content2 = page_content[proc2]
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            
            print(f"\nComparing {proc1} vs {proc2}:")
            print(f"Similarity: {similarity:.2%}")
            
            # Show differences if content is not too large
            if len(content1) < 1000 and len(content2) < 1000:
                diff = difflib.ndiff(content1.splitlines(), content2.splitlines())
                print("\nDifferences:")
                for line in diff:
                    if line.startswith("- ") or line.startswith("+ "):
                        print(line)

def generate_combined_content(page_content: Dict[str, str]) -> str:
    """Generate combined content using OR logic.
    
    Args:
        page_content: Dictionary mapping processor names to their content
        
    Returns:
        Combined content
    """
    # If no content, return empty string
    if not page_content:
        return ""
    
    # If only one processor, use its content
    if len(page_content) == 1:
        return next(iter(page_content.values()))
    
    # Prioritize processors (you can adjust this order)
    priority_order = ["gemini", "lmstudio", "camelot", "docling", "fallback"]
    
    # Find the processor with highest priority that has content
    for processor in priority_order:
        if processor in page_content and page_content[processor].strip():
            return page_content[processor]
    
    # If no prioritized processor found, use the first one with content
    for content in page_content.values():
        if content.strip():
            return content
    
    return ""

def main():
    parser = argparse.ArgumentParser(description="Compare results from different processors")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--base-name", required=True, help="Base name of the document")
    parser.add_argument("--page", help="Specific page to compare")
    parser.add_argument("--fix-combined", action="store_true", help="Fix combined results using OR logic")
    args = parser.parse_args()
    
    # Load results from all processors
    results = load_results(args.output, args.base_name)
    
    if not results:
        print("No results found")
        return 1
    
    # Get all page numbers
    all_pages = set()
    for processor_results in results.values():
        all_pages.update(processor_results.keys())
    
    # Sort page numbers
    all_pages = sorted(all_pages, key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    if args.page:
        # Compare specific page
        if args.page not in all_pages:
            print(f"Page {args.page} not found")
            return 1
            
        page_content = extract_page_content(results, args.page)
        print(f"\nContent for page {args.page}:")
        for processor, content in page_content.items():
            print(f"\n--- {processor} ---")
            print(content[:500] + ("..." if len(content) > 500 else ""))
        
        compare_page_content(page_content)
        
        if args.fix_combined:
            combined = generate_combined_content(page_content)
            print(f"\nFixed combined content for page {args.page}:")
            print(combined[:500] + ("..." if len(combined) > 500 else ""))
    else:
        # Compare all pages
        for page in all_pages:
            page_content = extract_page_content(results, page)
            print(f"\n=== Page {page} ===")
            print(f"Processors with content: {', '.join(page_content.keys())}")
            
            if args.fix_combined:
                combined = generate_combined_content(page_content)
                output_dir = Path(args.output) / f"{args.base_name}_fixed_combined"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Append to combined markdown
                md_file = output_dir / f"{args.base_name}_combined.md"
                with open(md_file, "a" if md_file.exists() else "w", encoding="utf-8") as f:
                    if not md_file.exists() or md_file.stat().st_size == 0:
                        f.write(f"# {args.base_name} - Fixed Combined Results\n\n")
                    f.write(f"## Page {page}\n\n")
                    f.write(combined)
                    f.write("\n\n---\n\n")
        
        if args.fix_combined:
            print(f"\nFixed combined results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
