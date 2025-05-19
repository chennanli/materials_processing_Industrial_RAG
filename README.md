# Docling OCR

A modular document processing system that supports multiple VLM (Vision-Language Model) backends for OCR and document understanding.

## Project Structure

```
Docling_OCR/
├── .gitignore              # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── basic.py               # Basic Docling OCR script (direct Docling usage)
├── gemini_OCR.py         # Google Gemini-based OCR processor
├── LMstudio.py           # Local LM Studio-based processor
├── output/               # Output directory for processed files
│   └── .gitkeep         # Keep the directory in Git
├── output_from_cloud/    # Output from cloud-based processing
├── archieved/            # Archived files (not tracked by Git)
├── future_use_pdf/       # PDFs for future use (not tracked by Git)
└── pdf/                  # Sample PDFs for processing
    └── 11919255_02.pdf   # Example PDF file
```

## Basic Usage with Docling (basic.py)

The `basic.py` script demonstrates the most basic usage of Docling for document processing. It provides a simple command-line interface to convert PDFs to Markdown and JSON formats while preserving the document structure.

### Features
- Convert PDFs to Markdown and JSON formats
- Select specific pages to process
- Preserve document structure and formatting
- Simple command-line interface

### Installation

1. First, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

Process a PDF file with default settings (outputs to ./output/):
```bash
python basic.py
```

Process a specific PDF file:
```bash
python basic.py --source path/to/your/document.pdf
```

Specify output paths:
```bash
python basic.py --output ./output/markdown.md --json-output ./output/output.json
```

Process specific pages (0-indexed):
```bash
# Process pages 1-3 (0,1,2)
python basic.py --pages "0-2"

# Process specific pages (first, third, and fifth pages)
python basic.py --pages "0,2,4"
```

### Output
- **Markdown (.md)**: Human-readable formatted text
- **JSON (.json)**: Structured document data including text, layout, and formatting information

### Notes
- The script creates necessary output directories if they don't exist
- Page numbers are 0-indexed (first page is 0)
- For large documents, processing specific pages can significantly improve performance

---

## Advanced Usage

For more advanced usage with different backends, see the documentation for the specific processor scripts below.

A modular document processing system that supports multiple VLM (Vision-Language Model) backends for OCR and document understanding. This system provides two main scripts for processing PDF documents:

1. **gemini_OCR.py** - Uses Google's Gemini API for cloud-based processing
2. **LMstudio.py** - Uses locally running LM Studio for processing

## Features

- Support for multiple VLM backends (LM Studio, Google Gemini)
- Processing of PDF documents with mixed content (text, images, tables)
- Individual page processing with combined output files
- Dual output formats (Markdown for readability, JSON for structural information)
- Configurable page selection
- Consistent command-line interface across scripts
- Comprehensive logging and error handling

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Docling_OCR.git
   cd Docling_OCR
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv docling_OCR
   source docling_OCR/bin/activate  # On Windows: .\docling_OCR\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your Google Gemini API key:
     ```
     GEMINI_API_KEY=your-api-key-here
     ```
   - For LM Studio, uncomment and set the appropriate values if needed

   **Important:** Never commit the `.env` file to version control. It's already in `.gitignore` for security.

## Configuration

1. **LM Studio** (Local):
   - Download and run [LM Studio](https://lmstudio.ai/)
   - Make sure the local API server is running (default: http://localhost:1234)

2. **Google Gemini** (Cloud):
   - Get an API key from [Google AI Studio](https://aistudio.google.com/)
   - Set the API key as an environment variable:
     ```bash
     export GEMINI_API_KEY='your-api-key-here'
     ```

## Usage

The project provides two main scripts for processing PDF documents:

### 1. Using Gemini (Cloud-Based Processing)

`gemini_OCR.py` uses Google's Gemini API for cloud-based processing of PDF documents.

#### Setup

Before using Gemini, set your API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

#### Basic Usage

```bash
python gemini_OCR.py
```

This will process the default PDF (`./pdf/11919255_02.pdf`) and save the results to `./output_from_cloud/`.

#### Specifying PDF and Pages

```bash
python gemini_OCR.py --input pdf/your_document.pdf --pages "2,3,5" --output ./your_output_folder
```

You can specify pages using individual numbers separated by commas or ranges:

```bash
python gemini_OCR.py --pages "1-3,5,7-9"
```

#### How Gemini Processing Works

1. The script extracts individual pages from the PDF
2. Each page is sent to Gemini's API through Docling's DocumentConverter
3. Gemini processes the page, handling both text and visual elements
4. Results are saved as:
   - A single combined Markdown file with all pages' content
   - Individual JSON files for each page with structural information

### 2. Using LM Studio (Local Processing)

`LMstudio.py` uses a locally running LM Studio instance for processing PDF documents.

#### Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Open LM Studio and load a model (default: internvl3-14b-instruct)
3. Start the local server (typically at http://127.0.0.1:1234)

#### Basic Usage

```bash
python LMstudio.py
```

This will process the default PDF (`./pdf/11919255_02.pdf`) and save the results to `./output/`.

#### Specifying PDF and Pages

```bash
python LMstudio.py --pdf_path pdf/your_document.pdf --pages "2,3,5" --output ./your_output_folder
```

#### How LM Studio Processing Works

1. The script extracts individual pages from the PDF
2. Each page is processed by Docling's DocumentConverter
3. The extracted content is sent to the local LM Studio server
4. LM Studio enhances the content with additional understanding
5. Results are saved as:
   - Markdown files with both original and enhanced content
   - JSON files with structural information

### Common Command-Line Arguments

Both scripts support the same command-line arguments for consistency:

| Argument | Description | Default |
|----------|-------------|--------|
| `--input`, `--pdf_path` | Path to the PDF file | `./pdf/11919255_02.pdf` |
| `--output` | Output directory | `./output_from_cloud` (Gemini) or `./output` (LM Studio) |
| `--pages` | Pages to process (e.g., "1,3,5" or "1-5") | All pages |
| `--batch-size` | Batch size for processing (currently fixed at 1) | 1 |

## Examples

### Gemini Examples

1. Process specific pages of a PDF with Gemini:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   python gemini_OCR.py --input pdf/11919255_02.pdf --pages "2-4" --output ./output_from_cloud
   ```

2. Process all pages of a PDF with Gemini:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   python gemini_OCR.py --input pdf/your_document.pdf
   ```

### LM Studio Examples

1. Process specific pages with LM Studio:
   ```bash
   python LMstudio.py --pdf_path pdf/11919255_02.pdf --pages "1,3,5" --output ./output
   ```

2. Process a single page with LM Studio:
   ```bash
   python LMstudio.py --pdf_path pdf/your_document.pdf --pages "5"
   ```

### Output Files

After processing, you'll find these files in the output directory:

- **Markdown files**: `[filename]_pages[page-numbers]_content.md`
  - Contains human-readable text extracted from the PDF
  - For LM Studio, includes both original and enhanced content

- **JSON files**: `[filename]_page[page-number]_content.json`
  - Contains structured data with element positions and relationships
  - Useful for applications that need to understand document structure
```

### Using the API

```python
from document_processing import PDFProcessor, config

# Configure the backend
config.update_backend('lmstudio', base_url='http://localhost:1234')

# Create a processor
processor = PDFProcessor(backend='lmstudio')

# Process a single PDF
result = processor.process(
    input_path='path/to/document.pdf',
    output_dir='output',
    output_format='markdown',
    max_pages=2
)

# Batch process multiple PDFs
results = processor.batch_process(
    input_dir='path/to/pdf/folder',
    output_dir='output',
    output_format='json',
    max_pages=2
)
```

## Project Structure

```
Docling_OCR/
├── document_processing/     # Core package
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── clients/            # API client implementations
│   │   ├── __init__.py
│   │   ├── base_client.py
│   │   ├── lmstudio_client.py
│   │   └── gemini_client.py
│   ├── converters/         # Document converters
│   │   ├── __init__.py
│   │   └── pdf_processor.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── file_utils.py
│       └── logging_utils.py
├── examples/               # Example scripts
│   ├── process_with_lmstudio.py
│   └── process_with_gemini.py
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Adding a New Backend

1. Create a new client class in `document_processing/clients/` that inherits from `BaseClient`
2. Implement the required methods (primarily `process_document`)
3. Update `document_processing/config.py` to include default settings for your backend
4. Add your backend to the `__init__.py` files as needed

## License

MIT License - See [LICENSE](LICENSE) for details.
