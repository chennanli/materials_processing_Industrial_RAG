import argparse
import os
from docling.document_converter import DocumentConverter

def main():
    parser = argparse.ArgumentParser(description='Convert PDF documents using Docling OCR')
    parser.add_argument('--source', type=str, default="./pdf/11919255_02.pdf",
                        help='Path to the source PDF file (default: ./pdf/11919255_02.pdf)')
    parser.add_argument('--output', type=str, default="./output/docling.md",
                        help='Path to save the Markdown output (default: ./output/docling.md)')
    parser.add_argument('--json-output', type=str, default="./output/docling.json",
                        help='Path to save the lossless JSON output (default: ./output/docling.json)')
    parser.add_argument('--pages', type=str, default=None,
                        help='Pages to extract before OCR (e.g. "2-4", "0,2,5"). If omitted, process all pages.')
    args = parser.parse_args()

    source = args.source
    output_path = args.output
    json_output_path = args.json_output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # If you want to process only certain pages, use --pages argument like '2-4' or '0,2,5'.
    # This will extract those pages to a temporary PDF and process only them.
    # Otherwise, the whole PDF will be processed.
import argparse
import os
import tempfile
import re
import json
from PyPDF2 import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter

def extract_pages_to_temp_pdf(input_pdf, page_indices):
    """Extract specified pages from a PDF into a temporary file. Returns the temp filename."""
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    for idx in page_indices:
        writer.add_page(reader.pages[idx])
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    with open(temp.name, 'wb') as f:
        writer.write(f)
    return temp.name

def parse_page_indices(page_arg):
    """Parse a string like '2-4' or '0,2,5' into a list of page indices (ints)."""
    if not page_arg:
        return None
    if ',' in page_arg:
        try:
            return [int(x.strip()) for x in page_arg.split(',')]
        except Exception:
            return None
    elif '-' in page_arg:
        m = re.match(r'^(\d+)-(\d+)$', page_arg.strip())
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            return list(range(start, end+1))
        return None
    else:
        try:
            return [int(page_arg.strip())]
        except Exception:
            return None

def docling_process_pdf(source, output_path, json_output_path, page_indices=None):
    """Extract pages if needed, run Docling, and save Markdown and JSON outputs."""
    if page_indices:
        print(f"Extracting pages {page_indices} from {source} using PyPDF2 (no information loss for page content)...")
        pdf_to_process = extract_pages_to_temp_pdf(source, page_indices)
    else:
        print("Converting all pages (no page selection or invalid input, processing entire PDF)...")
        pdf_to_process = source

    converter = DocumentConverter()
    result = converter.convert(pdf_to_process)

    markdown_text = result.document.export_to_markdown()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"Markdown has been saved to: {output_path}")

    json_content = result.document.export_to_dict()
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)
    print(f"Lossless JSON has been saved to: {json_output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PDF documents using Docling OCR')
    parser.add_argument('--source', type=str, default="./pdf/11919255_02.pdf",
                        help='Path to the source PDF file (default: ./pdf/11919255_02.pdf)')
    parser.add_argument('--output', type=str, default="./output/docling.md",
                        help='Path to save the Markdown output (default: ./output/docling.md)')
    parser.add_argument('--json-output', type=str, default="./output/docling.json",
                        help='Path to save the lossless JSON output (default: ./output/docling.json)')
    parser.add_argument('--pages', type=str, default=None,
                        help='Pages to extract before OCR (e.g. "2-4", "0,2,5"). If omitted, process all pages.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)

    page_indices = parse_page_indices(args.pages)
    docling_process_pdf(args.source, args.output, args.json_output, page_indices)

if __name__ == "__main__":
    main()
# pages_to_convert = "2-4"  # Convert pages 3-5 (0-indexed) (转换第3-5页，从0开始索引)
