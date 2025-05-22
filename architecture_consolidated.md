# Document Processing System Architecture

## Overview

The Document Processing System is designed to analyze and extract information from various document types using multiple processing engines. The system combines the strengths of different processors (Docling, Gemini, LMStudio, Camelot) to provide comprehensive document understanding.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Document Processing System                          │
│                                                                             │
│  ┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐  │
│  │ Document        │      │ Processor        │      │ Result             │  │
│  │ Analyzer        │─────►│ Selection        │─────►│ Combination        │  │
│  └─────────────────┘      └──────────────────┘      └────────────────────┘  │
│           │                        │                          │              │
│           ▼                        ▼                          ▼              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          Processor Registry                             │ │
│  │                                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │   Docling   │  │   Camelot   │  │   Gemini    │  │  LMStudio   │    │ │
│  │  │  Processor  │  │  Processor  │  │  Processor  │  │  Processor  │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Web Interface Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            User's Web Browser                               │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Flask Web Server                                │
│                                                                             │
│  ┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐  │
│  │   Web Routes    │      │  API Endpoints   │      │   Static Files     │  │
│  │  (app.py)       │◄────►│  (app.py)        │      │   (CSS/JS)         │  │
│  └────────┬────────┘      └──────────────────┘      └────────────────────┘  │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐  │
│  │  HTML Templates │      │ Document         │      │  Result            │  │
│  │  (Jinja2)       │◄────►│ Processor        │◄────►│  Comparison        │  │
│  └─────────────────┘      └──────────────────┘      └────────────────────┘  │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Document Processing System                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Document Upload**: User uploads a document through the web interface
2. **Document Analysis**: System analyzes the document to determine content types
3. **Processor Selection**: Based on content types, appropriate processors are selected
4. **Parallel Processing**: Document is processed by selected processors in parallel
5. **Result Combination**: Results from all processors are combined using "OR" logic
6. **Result Display**: Combined results are displayed to the user through the web interface

## Component Details

### 1. Document Analyzer

The Document Analyzer examines the uploaded document to determine:
- File type (PDF, DOCX, image, etc.)
- Content types (text, tables, images, etc.)
- Page count and structure

### 2. Processor Registry

The Processor Registry maintains a list of available processors and their capabilities:

| Processor | Capabilities |
|-----------|--------------|
| Docling   | Document structure, text extraction |
| Camelot   | Table extraction from PDFs |
| Gemini    | Image processing, complex text understanding |
| LMStudio  | Text extraction, document understanding |
| Fallback  | Basic text extraction when others fail |

### 3. Processor Selection

Based on document analysis, the system selects the most appropriate processors:
- For PDFs with tables: Camelot + Docling/LMStudio
- For image-heavy documents: Gemini + LMStudio
- For text-heavy documents: Docling + LMStudio

### 4. Result Combination

Results from all processors are combined using "OR" logic to ensure comprehensive coverage:
- Text content from all processors is merged
- Table content is preserved
- Conflicting information is flagged for review

### 5. Web Interface

The web interface provides:
- Document upload and processor selection
- Processing status tracking
- Result viewing and comparison
- Side-by-side and diff views for comparing processor outputs

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | Flask | Lightweight web server |
| Frontend | HTML/CSS/JS | User interface |
| Templates | Jinja2 | Dynamic HTML generation |
| Styling | Bootstrap | Responsive design |
| Diff Generation | difflib | Compare document outputs |
| Document Processing | Python | Backend processing |
| File Handling | Werkzeug | Secure file uploads |

## File Structure

```
Docling_OCR/
├── app.py                 # Flask web application
├── document_processor.py  # Document processing logic
├── compare_results.py     # Result comparison tool
├── config.py              # Configuration settings
├── processors/            # Processor implementations
│   ├── docling_processor.py
│   ├── camelot_processor.py
│   ├── gemini_processor.py
│   ├── lmstudio_processor.py
│   └── fallback_processor.py
├── static/                # Static web assets
│   ├── style.css          # Custom CSS styles
│   └── script.js          # Custom JavaScript
├── templates/             # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Home page
│   ├── results.html       # Results page
│   └── compare.html       # Comparison page
├── uploads/               # Uploaded documents
└── output/                # Processing results
```

## Modularization Plan

The following files need to be modularized due to their size:

### 1. document_processor.py (783 lines)

Should be split into:
- `document_analyzer.py`: Document analysis functionality
- `processor_manager.py`: Processor selection and management
- `result_combiner.py`: Result combination logic
- `document_processor.py`: Main orchestration (reduced size)

### 2. LMstudio.py (844 lines)

Should be archived as we're using the modular `processors/lmstudio_processor.py`

## Future Improvements

1. **Authentication**: Add user authentication for secure access
2. **Caching**: Implement caching for faster processing of previously seen documents
3. **Batch Processing**: Add support for processing multiple documents at once
4. **Advanced Visualization**: Enhance visualization for tables and images
5. **API Endpoints**: Create RESTful API endpoints for programmatic access
