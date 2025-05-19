"""
pdf_page_processor.py

A modular utility for memory-efficient, information-preserving, page-by-page PDF processing.
Detects image-only pages and processes them as images; otherwise, processes as single-page PDFs.

Usage:
    from pdf_page_processor import process_pdf_by_page

    def process_image_page(image, page_number):
        ...  # your logic here
    def process_pdf_page(pdf_bytes_io, page_number):
        ...  # your logic here
    process_pdf_by_page('yourfile.pdf', process_image_page, process_pdf_page)
"""

from typing import Callable, BinaryIO
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import io
import gc


def is_image_only_page(pdf_reader: PdfReader, page_num: int) -> bool:
    """
    Returns True if the page contains only images (no extractable text or vector objects).
    """
    page = pdf_reader.pages[page_num]
    # If the page has extractable text, it's not image-only
    text = page.extract_text()
    if text and text.strip():
        return False
    # Check for XObject images in the page's resources
    try:
        xobjects = page['/Resources'].get('/XObject', None)
        if not xobjects:
            return False
        xobjects = xobjects.get_object()
        for obj in xobjects.values():
            if obj.get('/Subtype', None) == '/Image':
                continue  # Found image
            else:
                return False  # Found non-image object
        return True if xobjects else False
    except Exception:
        # If no XObject or error, treat as not image-only
        return False


def process_pdf_by_page(
    pdf_path: str,
    process_image_page_fn: Callable[[object, int], None],
    process_pdf_page_fn: Callable[[BinaryIO, int], None],
    selected_pages: list = None
) -> None:
    """
    For each page in the PDF, detects its type and processes as image (in memory) or as single-page PDF (in memory).
    Calls the user-supplied processing functions for each page.
    Releases memory after each page.

    Args:
        pdf_path: Path to the PDF file.
        process_image_page_fn: Function to process image-only pages. Args: (PIL.Image, page_number)
        process_pdf_page_fn: Function to process non-image pages. Args: (BytesIO, page_number)
        first_page: First page to process (1-based, default: 1)
        last_page: Last page to process (1-based, default: all pages)
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    if selected_pages is None:
        pages_to_process = range(num_pages)
    else:
        # Convert to 0-based indices and filter out-of-range
        pages_to_process = [p-1 if p > 0 else p for p in selected_pages if 0 <= (p-1 if p > 0 else p) < num_pages]
    for i in pages_to_process:
        page_number = i + 1
        if is_image_only_page(reader, i):
            images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
            process_image_page_fn(images[0], page_number=page_number)
            del images
        else:
            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            pdf_bytes = io.BytesIO()
            writer.write(pdf_bytes)
            pdf_bytes.seek(0)
            process_pdf_page_fn(pdf_bytes, page_number=page_number)
            pdf_bytes.close()
            del writer
        gc.collect()

def process_pdf_by_batch_gemini(
    pdf_path: str,
    process_batch_fn: Callable[[list, list], None],
    selected_pages: list = None,
    batch_size: int = 1
) -> None:
    """
    Process PDF pages in batches for Gemini. This allows sending multiple pages at once
    to take advantage of Gemini's multimodal capabilities while respecting context limits.
    
    Args:
        pdf_path: Path to the PDF file.
        process_batch_fn: Function to process batches of pages. Args: (list_of_page_data, list_of_page_numbers)
        selected_pages: List of page numbers to process (1-based, default: all pages)
        batch_size: Number of pages to process in each batch (default: 1)
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    if selected_pages is None:
        pages_to_process = list(range(num_pages))
    else:
        # Convert to 0-based indices and filter out-of-range
        pages_to_process = [p-1 if p > 0 else p for p in selected_pages if 0 <= (p-1 if p > 0 else p) < num_pages]
    
    # Process pages in batches
    for i in range(0, len(pages_to_process), batch_size):
        batch_indices = pages_to_process[i:i+batch_size]
        batch_page_numbers = [idx + 1 for idx in batch_indices]  # Convert to 1-based page numbers
        
        # Extract each page in the batch as a PDF
        batch_pdfs = []
        for page_idx in batch_indices:
            writer = PdfWriter()
            writer.add_page(reader.pages[page_idx])
            pdf_bytes = io.BytesIO()
            writer.write(pdf_bytes)
            pdf_bytes.seek(0)
            batch_pdfs.append(pdf_bytes)
        
        # Process the batch
        process_batch_fn(batch_pdfs, batch_page_numbers)
        
        # Clean up
        for pdf_bytes in batch_pdfs:
            pdf_bytes.close()
        gc.collect()

# Main entry for CLI usage and integration with LMstudio/Gemini
if __name__ == "__main__":
    import argparse
    import sys
    def parse_page_arg(page_arg, num_pages):
        if not page_arg:
            return None
        result = set()
        for part in page_arg.split(','):
            if '-' in part:
                start, end = part.split('-')
                result.update(range(int(start), int(end)+1))
            else:
                result.add(int(part))
        return sorted([p for p in result if 1 <= p <= num_pages])

    parser = argparse.ArgumentParser(description='Process PDF by page with hybrid text/OCR extraction')
    parser.add_argument('--pdf', required=True, help='PDF file path')
    parser.add_argument('--pages', default=None, help='Pages to process (e.g. "1-3,5,7")')
    args = parser.parse_args()
    reader = PdfReader(args.pdf)
    num_pages = len(reader.pages)
    selected_pages = parse_page_arg(args.pages, num_pages) if args.pages else None
    print(f"Total pages in PDF: {num_pages}")
    print(f"Processing pages: {selected_pages if selected_pages else 'ALL'}")
    def stub_image_fn(image, page_number):
        print(f"[DEMO] Would OCR image page {page_number}")
    def stub_pdf_fn(pdf_bytes, page_number):
        print(f"[DEMO] Would extract text from PDF page {page_number}")
    process_pdf_by_page(args.pdf, stub_image_fn, stub_pdf_fn, selected_pages=selected_pages)
