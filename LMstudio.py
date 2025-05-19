import argparse
import os
import tempfile
import re
import json
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
import requests
from docling.document_converter import DocumentConverter

# =================== CONFIGURATION SETTINGS ===================
API_TIMEOUT = 1800  # Timeout in seconds (30 minutes = 1800 seconds)
MODEL_NAME = "internvl3-14b-instruct"  # Model name in LM Studio
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
# =============================================================

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

def check_lm_studio_connection(url=LMSTUDIO_URL, timeout=5):
    """Check if LM Studio is running properly"""
    try:
        print(f"Attempting to connect to LM Studio at {url}...")
        response = requests.get(url.replace('/v1/chat/completions', '/v1/models'), timeout=timeout)
        if response.status_code == 200:
            models = response.json()
            print(f"\u2705 LM Studio connection successful! Found models: {models}")
            return True, models
        else:
            print(f"\u274c LM Studio response error. Status code: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"\u274c Failed to connect to LM Studio: {e}")
        return False, None

def lm_studio_request(prompt, response_format, model=MODEL_NAME, timeout=API_TIMEOUT):
    """Send a request to LM Studio API with the given prompt and response format."""
    headers = {"Content-Type": "application/json"}
    
    # Basic request data
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 8192,
        "temperature": 0.1
    }
    
    # Add response format if JSON is requested
    if response_format == "json":
        data["response_format"] = {"type": "json_object"}
    # For markdown, we don't need to specify response_format
    try:
        response = requests.post(LMSTUDIO_URL, headers=headers, data=json.dumps(data), timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in LM Studio request: {e}")
        return None

def get_markdown_prompt():
    return ("OCR the full page to markdown, preserving all text content, diagrams descriptions, and spatial relationships.")

def get_json_prompt():
    return (
        """
Extract all information from this engineering drawing into a structured JSON format.\n"
Include:\n"
1. Title and drawing number\n"
2. Any text annotations and their positions\n"
3. Dimensions and measurements with units\n"
4. Component parts and labels\n"
5. Any symbols, callouts, or reference marks\n"
6. Scale information\n"
7. Revision information\n"
8. Notes or special instructions\n"
\n"
Your response MUST be a valid JSON object. Do NOT include any markdown code block or explanation, only the JSON object itself.\n"
"""
    )

def docling_process_pdf(source, output_dir, page_indices=None):
    """
    Process selected PDF pages robustly: if a page is image-only, use OCR; otherwise, use Docling for text extraction.
    """
    from pdf_page_processor import process_pdf_by_page
    os.makedirs(output_dir, exist_ok=True)
    pdf_stem = Path(source).stem
    def process_image_page(image, page_number):
        # Stub: Replace with actual OCR logic if needed
        print(f"[OCR] Page {page_number} is image-only. Run OCR here.")
        image.save(os.path.join(output_dir, f"{pdf_stem}_page{page_number}_image.png"))
    def process_pdf_page(pdf_bytes, page_number):
        # Create a temporary file since DocumentConverter can't handle BytesIO directly
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_bytes.read())
            temp_path = temp_pdf.name
        
        try:
            # Use the temporary file path instead of BytesIO
            converter = DocumentConverter()
            result = converter.convert(temp_path)
            markdown_text = result.document.export_to_markdown()
            json_content = result.document.export_to_dict()
            
            # Save both markdown and JSON results
            md_file = os.path.join(output_dir, f"{pdf_stem}_page{page_number}_content.md")
            json_file = os.path.join(output_dir, f"{pdf_stem}_page{page_number}_content.json")
            
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(markdown_text)
                
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
                
            print(f"Page {page_number} content saved to: {md_file}")
            print(f"Page {page_number} JSON data saved to: {json_file}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        # Send to LM Studio for processing
        md_prompt = get_markdown_prompt()
        lmstudio_result = lm_studio_request(md_prompt + "\n" + markdown_text, response_format="markdown")
        
        # If LM Studio returned a result, append it to the markdown file
        if lmstudio_result:
            with open(md_file, "a", encoding="utf-8") as f:
                f.write("\n\n## LM Studio Enhanced Content\n\n")
                f.write(lmstudio_result)
            print(f"LM Studio enhanced content added to: {md_file}")
        else:
            print("LM Studio processing failed - using Docling content only")
    # Use page_indices as 1-based page numbers if provided
    selected_pages = [p+1 if p >= 0 else p for p in page_indices] if page_indices else None
    process_pdf_by_page(source, process_image_page, process_pdf_page, selected_pages=selected_pages)

def main():
    parser = argparse.ArgumentParser(description='Convert PDF using Docling and LM Studio (Markdown)')
    parser.add_argument('--input', '--pdf_path', dest='source', type=str, required=True, 
                      help='Path to the source PDF file')
    parser.add_argument('--output', dest='output_dir', type=str, default="./output", 
                      help='Directory to save all outputs')
    parser.add_argument('--pages', type=str, default=None, 
                      help='Pages to extract (e.g. "2-4", "0,2,5"). If omitted, process all pages.')
    parser.add_argument('--batch-size', type=int, default=1, 
                      help='Batch size for processing (included for compatibility with gemini_OCR.py)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    page_indices = parse_page_indices(args.pages)
    docling_process_pdf(args.source, args.output_dir, page_indices)

if __name__ == "__main__":
    main()
