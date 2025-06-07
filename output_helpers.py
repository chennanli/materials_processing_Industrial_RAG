#!/usr/bin/env python3
"""
Output Helper Functions for Docling OCR
--------------------------------------
Helper functions to improve the output organization of the Docling OCR project.
"""

import json
from pathlib import Path


def save_processor_results(processor, results, output_dir, base_name):
    """Save processor results in a better organized structure.

    Args:
        processor: The processor instance
        results: Processing results
        output_dir: Base output directory
        base_name: Document base name

    Returns:
        Path to the processor directory
    """
    # Create document-specific directory
    doc_dir = Path(output_dir) / base_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Create processor-specific directory
    processor_dir = doc_dir / processor.name
    processor_dir.mkdir(parents=True, exist_ok=True)

    # Save combined results
    combined_md = processor_dir / "combined.md"
    combined_json = processor_dir / "combined.json"

    # Save markdown
    with open(combined_md, "w", encoding="utf-8") as f:
        f.write(f"# {base_name} - {processor.name} Results\n\n")
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
            md_path = pages_dir / f"page{page_num}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Page {page_num}\n\n")
                f.write(page_data["content"])

            # Save as JSON
            json_path = pages_dir / f"page{page_num}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)

    # IMPORTANT: Also save in the old directory structure for backward compatibility with the UI
    old_processor_dir = Path(output_dir) / f"{base_name}_{processor.name}"
    old_processor_dir.mkdir(parents=True, exist_ok=True)

    old_combined_md = old_processor_dir / f"{base_name}_combined.md"
    old_combined_json = old_processor_dir / f"{base_name}_combined.json"

    # Save markdown in old location
    with open(old_combined_md, "w", encoding="utf-8") as f:
        f.write(f"# {base_name} - {processor.name} Results\n\n")
        for page_num, page_data in sorted(results.items()):
            if isinstance(page_data, dict) and "content" in page_data:
                f.write(f"## Page {page_num}\n\n")
                f.write(page_data["content"])
                f.write("\n\n---\n\n")

    # Save JSON in old location
    with open(old_combined_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return processor_dir


def save_combined_results(combined_results, output_dir, base_name):
    """Save combined results from all processors.

    Args:
        combined_results: Combined results from all processors
        output_dir: Base output directory
        base_name: Document base name

    Returns:
        Path to the combined directory
    """
    # Create document-specific directory
    doc_dir = Path(output_dir) / base_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Create combined directory
    combined_dir = doc_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Save combined results
    combined_md = combined_dir / "combined.md"
    combined_json = combined_dir / "combined.json"

    # Save markdown
    with open(combined_md, "w", encoding="utf-8") as f:
        f.write(f"# {base_name} - Combined Results\n\n")
        for page_num, page_data in sorted(combined_results.items()):
            if isinstance(page_data, dict) and "content" in page_data:
                f.write(f"## Page {page_num}\n\n")
                f.write(page_data["content"])
                f.write("\n\n---\n\n")

    # Save JSON
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    # Save individual page results
    pages_dir = combined_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    for page_num, page_data in combined_results.items():
        if isinstance(page_data, dict) and "content" in page_data:
            # Save as markdown
            md_path = pages_dir / f"page{page_num}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Page {page_num}\n\n")
                f.write(page_data["content"])

            # Save as JSON
            json_path = pages_dir / f"page{page_num}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)

    # IMPORTANT: Also save in the original locations for backward compatibility with the UI
    # This ensures the UI can still find the results

    # 1. Save in the root output directory
    original_md = Path(output_dir) / f"{base_name}.md"
    original_json = Path(output_dir) / f"{base_name}.json"

    # Save markdown in original location
    with open(original_md, "w", encoding="utf-8") as f:
        f.write(f"# {base_name} - Combined Results\n\n")
        for page_num, page_data in sorted(combined_results.items()):
            if isinstance(page_data, dict) and "content" in page_data:
                f.write(f"## Page {page_num}\n\n")
                f.write(page_data["content"])
                f.write("\n\n---\n\n")

    # Save JSON in original location
    with open(original_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    # 2. Save in the old combined directory structure
    old_combined_dir = Path(output_dir) / f"{base_name}_combined"
    old_combined_dir.mkdir(parents=True, exist_ok=True)

    old_combined_md = old_combined_dir / f"{base_name}_combined.md"
    old_combined_json = old_combined_dir / f"{base_name}_combined.json"

    # Save markdown in old combined location
    with open(old_combined_md, "w", encoding="utf-8") as f:
        f.write(f"# {base_name} - Combined Results\n\n")
        for page_num, page_data in sorted(combined_results.items()):
            if isinstance(page_data, dict) and "content" in page_data:
                f.write(f"## Page {page_num}\n\n")
                f.write(page_data["content"])
                f.write("\n\n---\n\n")

    # Save JSON in old combined location
    with open(old_combined_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    return combined_dir
