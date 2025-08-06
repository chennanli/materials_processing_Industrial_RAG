# 🧠 Smart Document Processor Guide

## Overview

The **Smart Document Processor** is an intelligent system that automatically routes different pages of your documents to the best extraction method based on content analysis. Instead of running all processors on every page, it intelligently decides which processor to use for each page.

## 🎯 How It Works

### Phase 1: Document Analysis
- Uses **Docling** to analyze document structure and content
- Classifies each page based on:
  - Text quality (good/medium/poor)
  - Table presence and complexity
  - Image content
  - Whether it's likely scanned/image-based

### Phase 2: Intelligent Routing
Based on the analysis, pages are routed to specialized processors:

| Page Type | Primary Processor | Reason |
|-----------|------------------|---------|
| **Text-heavy with good structure** | Docling | Fast, accurate text extraction |
| **Poor text/scanned pages** | LMStudio (MonkeyOCR) | OCR for image-based content |
| **Complex tables** | Camelot | Specialized table extraction |
| **Visual/complex content** | Gemini (optional) | Advanced vision analysis |
| **Fallback** | PyMuPDF | Simple text extraction |

### Phase 3: Quality Assurance
- Validates extraction results
- Ensures all pages have meaningful content
- Uses fallback methods if needed

## 🚀 Usage

### Via Web Interface
1. Start the web app: `python app.py`
2. Select **"Smart (Recommended)"** processor
3. Upload your document
4. The system automatically handles everything!

### Via Command Line
```bash
# Process with smart processor
python -c "
from document_processor import DocumentProcessor
processor = DocumentProcessor(['smart'])
results = processor.process_document('your_file.pdf')
"
```

### Via Python Code
```python
from processors.smart_processor import SmartDocumentProcessor

# Initialize
processor = SmartDocumentProcessor()

# Process document
results = processor.process('document.pdf')

# Results include processing metadata
for page_num, page_data in results.items():
    print(f"Page {page_num}: {page_data.get('processor_used', 'docling')}")
```

## 🎛️ Configuration Options

### Enable Gemini for Complex Documents
```python
# Enable Gemini for visually complex documents
processor = SmartDocumentProcessor(use_gemini_for_complex=True)
```

### Force Specific Processors
You can still use individual processors if needed:
```python
# Use only LMStudio
processor = DocumentProcessor(['lmstudio'])

# Use Docling + Camelot
processor = DocumentProcessor(['docling', 'camelot'])
```

## 📊 Benefits vs Traditional Approach

### ❌ Old Way (All processors on all pages):
- **Slow**: Every page processed by every method
- **Expensive**: Unnecessary API calls to Gemini
- **Redundant**: Same work done multiple times
- **Complex**: User has to choose processors

### ✅ Smart Way:
- **Fast**: Only use what's needed per page
- **Efficient**: Minimize expensive vision model calls
- **Intelligent**: Best tool for each job
- **Simple**: Just select "Smart" and let it decide

## 🔧 Advanced Features

### Processing Summary
The smart processor provides detailed processing metadata:
```json
{
  "total_pages": 5,
  "processors_used": ["docling", "lmstudio", "camelot"],
  "pages_with_ocr": 2,
  "pages_with_tables": 1,
  "processing_time": 15.3
}
```

### Per-Page Metadata
Each page includes information about how it was processed:
```json
{
  "page_number": 1,
  "content": "...",
  "processor_used": "lmstudio",
  "text_quality": "poor",
  "has_tables": false,
  "camelot_tables": false
}
```

## 🛠️ Troubleshooting

### "Smart processor not available"
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### No processors available
Check that at least one processor is working:
- **Docling**: `pip install docling`
- **LMStudio**: Start LMStudio with a model loaded
- **Camelot**: `pip install camelot-py[cv]`
- **Gemini**: Set `GEMINI_API_KEY` in `.env`

### Poor results on specific pages
The smart processor logs which processor was used for each page. Check the logs to see the routing decisions and manually test problematic pages with different processors.

## 📈 Performance Tips

1. **For text-heavy documents**: Smart processor will primarily use Docling (fastest)
2. **For scanned documents**: Will automatically route to LMStudio OCR
3. **For table-heavy documents**: Will use Camelot for better table extraction
4. **For mixed documents**: Will intelligently combine multiple processors

## 🎯 When to Use Each Approach

### Use Smart Processor When:
- ✅ You want the best results with minimal setup
- ✅ You have mixed document types
- ✅ You want to minimize processing time
- ✅ You don't want to manually choose processors

### Use Individual Processors When:
- 🔧 You know exactly which processor works best for your documents
- 🔧 You want to compare results from different processors
- 🔧 You're testing or debugging specific extraction methods
- 🔧 You have very specific requirements

---

The Smart Document Processor represents the next evolution of document processing - intelligent, efficient, and automatic. Just upload your document and let the AI decide the best approach! 🚀