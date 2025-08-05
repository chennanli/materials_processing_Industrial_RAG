# Document Processing System

A comprehensive document processing system with a web interface that supports multiple processing engines (Docling, Gemini, LMStudio, Camelot) for OCR, table extraction, and document understanding.

## Project Structure

```
Docling_OCR/
├── .gitignore              # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── app.py                # Flask web application
├── document_processor.py # Document processing core
├── compare_results.py    # Result comparison tool
├── config.py             # Configuration settings
├── processors/           # Processor implementations
│   ├── docling_processor.py
│   ├── camelot_processor.py
│   ├── gemini_processor.py
│   ├── lmstudio_processor.py
│   └── fallback_processor.py
├── static/               # Static web assets
│   ├── style.css         # Custom CSS styles
│   └── script.js         # Custom JavaScript
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── results.html      # Results page
│   └── compare.html      # Comparison page
├── uploads/              # Uploaded documents
├── output/               # Output directory for processed files
│   └── .gitkeep         # Keep the directory in Git
├── architecture_consolidated.md  # System architecture documentation
└── modularization_plan.md        # Plan for code modularization
```

## Web Interface

The system now includes a user-friendly web interface for document processing, allowing users to:

1. Upload documents (PDF, DOCX, images)
2. Select which processors to use
3. Specify pages to process
4. View and compare results from different processors

### Running the Web Interface

```bash
python app.py
```

Then open your browser to http://127.0.0.1:5000/

### Web Interface Features

- **Document Upload**: Upload documents through a drag-and-drop interface
- **Processor Selection**: Choose which processors to use for each document
- **Page Selection**: Specify which pages to process
- **Processing Status**: Real-time status updates during processing
- **Result Viewing**: View results from each processor
- **Result Comparison**: Compare outputs from different processors
  - Side-by-side view
  - Diff view with highlighted differences

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

## System Architecture

The Document Processing System uses a modular architecture with the following components:

1. **Document Processor**: Core engine that analyzes documents and coordinates processing
2. **Processor Registry**: Collection of document processors with different capabilities
3. **Web Interface**: Flask-based web application for user interaction
4. **Result Comparison**: Tools for comparing outputs from different processors

For detailed architecture information, see `architecture_consolidated.md`.

## Features

- **Multiple Processing Engines** (Parallel Processing):
  - **Docling**: Document structure analysis, layout understanding, text extraction
  - **Gemini**: Fast cloud-based OCR, vision understanding, image processing
  - **LMStudio**: Private local OCR with custom models (internvl3-14b-instruct)
  - **Camelot**: Specialized table extraction from PDFs
  - **Fallback**: Basic text extraction when other processors fail

- **Smart Processing Architecture**:
  - **Parallel Execution**: All selected processors run simultaneously (not sequentially)
  - **Individual Selection**: Choose specific processors (Gemini only, LMStudio only, etc.)
  - **Intelligent Combination**: Two-pass "OR" logic with preferences for maximum information
  - **Content Type Specialization**: Best processor for each content type (TEXT, TABLES, IMAGES)

- **Advanced Features**:
  - **Content Analysis**: Optional automatic detection of document content types
  - **Processor Preferences**: TEXT (Gemini→LMStudio→Docling), TABLES (Camelot→Docling), IMAGES (Gemini→LMStudio)
  - **Web Interface**: User-friendly interface with processor selection
  - **Comparison Tools**: Side-by-side and diff views for comparing processor outputs
  - **Processing Status**: Real-time status updates during document processing
  - **Error Handling**: Graceful handling of processing errors with fallbacks

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-processing-system.git
   cd document-processing-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root:
     ```bash
     touch .env
     ```
   - Add your API keys and configuration:
     ```
     GEMINI_API_KEY=your-gemini-api-key-here
     LMSTUDIO_API_URL=http://localhost:1234/v1
     ```

   **Important:** Never commit the `.env` file to version control. It's already in `.gitignore` for security.

## Configuration

### Required Services

1. **LM Studio** (Local, Optional):
   - Download and run [LM Studio](https://lmstudio.ai/)
   - Start the local API server (default: http://localhost:1234)
   - Load a model (recommended: internvl3-14b-instruct)

2. **Google Gemini** (Cloud, Optional):
   - Get an API key from [Google AI Studio](https://aistudio.google.com/)
   - Add to your `.env` file or set as an environment variable:
     ```bash
     export GEMINI_API_KEY='your-api-key-here'
     ```

3. **Docling** (Local, Optional):
   - Installed automatically with requirements.txt

4. **Camelot** (Local, Optional):
   - Installed automatically with requirements.txt
   - Requires Ghostscript for PDF processing

### Note on Processors

The system is designed to work even if some processors are unavailable. It will use whatever processors are available and fall back to basic text extraction if needed.

## Usage

### Web Interface (Recommended)

1. Start the web server:
   ```bash
   python app.py
   ```

2. Open your browser to http://127.0.0.1:5000/

3. Upload a document and select processing options

4. View and compare results

### Command Line Usage

For advanced users, the system can also be used from the command line:

```bash
python -c "from document_processor import DocumentProcessor; \
          processor = DocumentProcessor(); \
          processor.process_document('path/to/document.pdf', \
                                    page_indices=[0,1,2], \
                                    output_dir='./output')"
```

### API Usage

The document processor can be imported and used in your own Python code:

```python
from document_processor import DocumentProcessor

# Initialize with specific processors
processor = DocumentProcessor(enabled_processors=['gemini', 'camelot'])

# Process a document
results = processor.process_document(
    file_path='path/to/document.pdf',
    page_indices=[0, 1, 2],  # Optional: specific pages to process
    output_dir='./output'     # Optional: where to save results
)

# Access results
for page_num, page_data in results.items():
    print(f"Page {page_num}: {page_data['content'][:100]}...")
```

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

## Modular PDF Processing Framework

The project now includes a modular PDF processing framework that allows you to use multiple processors simultaneously or choose specific ones based on your needs.

### Key Components

1. **pdf_processor.py** - Main coordinator that manages multiple processors
2. **processors/** - Directory containing individual processor implementations:
   - **gemini_processor.py** - Google Gemini-based processor
   - **lmstudio_processor.py** - Local LM Studio-based processor
   - **camelot_processor.py** - Specialized table extraction processor

### Using the Modular Framework

The `process_pdf_example.py` script demonstrates how to use the modular framework:

```bash
# Process with all available processors
python process_pdf_example.py path/to/your/document.pdf

# Process with specific processors
python process_pdf_example.py path/to/your/document.pdf --processors gemini lmstudio

# Process specific pages
python process_pdf_example.py path/to/your/document.pdf --pages "1-3,5,7-9"

# Specify output directory
python process_pdf_example.py path/to/your/document.pdf --output ./custom_output
```

### Benefits of the Modular Framework

- **Flexibility**: Choose which processors to use for each run
- **Parallel Processing**: Use multiple processors simultaneously
- **Extensibility**: Easily add new processors by implementing the processor interface
- **Consistent Output**: All processors follow the same output format
- **Combined Results**: Option to combine results from multiple processors

## Adding a New Backend

1. Create a new client class in `document_processing/clients/` that inherits from `BaseClient`
2. Implement the required methods (primarily `process_document`)
3. Update `document_processing/config.py` to include default settings for your backend
4. Add your backend to the `__init__.py` files as needed

## Future Development

See `modularization_plan.md` for planned improvements to the codebase structure.

Future enhancements may include:

1. **Authentication**: User accounts and authentication
2. **Caching**: Faster processing for previously seen documents
3. **Batch Processing**: Processing multiple documents at once
4. **Advanced Visualization**: Better visualization for tables and images
5. **API Endpoints**: RESTful API for programmatic access

## License

This project is licensed under the MIT License - see the LICENSE file for details.
