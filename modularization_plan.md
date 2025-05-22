# Modularization Plan for Document Processing System

## Overview

This document outlines the plan to modularize the document processing system, focusing on breaking down large files into more maintainable components. The goal is to improve code organization, readability, and maintainability.

## Files to Modularize

### 1. document_processor.py (783 lines)

This file should be split into four separate modules:

#### a. document_analyzer.py
```python
"""
Document Analyzer Module
------------------------
Analyzes documents to determine content types and structure.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import other necessary modules

class DocumentAnalyzer:
    """Analyzes documents to determine content types and structure."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a document to determine its content types and structure.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing analysis results
        """
        # Implementation from document_processor.py
        pass
    
    def _analyze_with_docling(self, file_path: str) -> Dict[str, Any]:
        """Use Docling for document analysis if available."""
        # Implementation from document_processor.py
        pass
    
    def _analyze_basic(self, file_path: str) -> Dict[str, Any]:
        """Basic document analysis without Docling."""
        # Implementation from document_processor.py
        pass
    
    def _detect_content_types(self, file_path: str) -> List[str]:
        """Detect content types based on file extension and basic analysis."""
        # Implementation from document_processor.py
        pass
```

#### b. processor_manager.py
```python
"""
Processor Manager Module
-----------------------
Manages processor selection and execution.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import importlib

class ProcessorManager:
    """Manages processor selection and execution."""
    
    def __init__(self, enabled_processors=None):
        self.logger = logging.getLogger(__name__)
        self.enabled_processors = enabled_processors or ["all"]
        self.available_processors = self._discover_processors()
    
    def _discover_processors(self) -> Dict[str, Any]:
        """Discover available processors in the processors directory."""
        # Implementation from document_processor.py
        pass
    
    def select_processors(self, document_analysis: Dict[str, Any]) -> List[str]:
        """
        Select appropriate processors based on document analysis.
        
        Args:
            document_analysis: Results from document analysis
            
        Returns:
            List of processor names to use
        """
        # Implementation from document_processor.py
        pass
    
    def get_processor_instance(self, processor_name: str) -> Any:
        """Get an instance of a processor by name."""
        # Implementation from document_processor.py
        pass
    
    def execute_processor(self, processor, file_path: str, page_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Execute a processor on a document."""
        # Implementation from document_processor.py
        pass
```

#### c. result_combiner.py
```python
"""
Result Combiner Module
---------------------
Combines results from multiple processors.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class ResultCombiner:
    """Combines results from multiple processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def combine_results(self, processor_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple processors.
        
        Args:
            processor_results: Dictionary of results from each processor
            
        Returns:
            Combined results
        """
        # Implementation from document_processor.py
        pass
    
    def save_results(self, results: Dict[str, Any], base_name: str, output_dir: str) -> None:
        """
        Save combined results to output directory.
        
        Args:
            results: Combined results to save
            base_name: Base name for output files
            output_dir: Directory to save results in
        """
        # Implementation from document_processor.py
        pass
```

#### d. document_processor.py (refactored)
```python
"""
Document Processor Module
------------------------
Main orchestration module for document processing.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from document_analyzer import DocumentAnalyzer
from processor_manager import ProcessorManager
from result_combiner import ResultCombiner

class DocumentProcessor:
    """Processes documents using multiple processors based on content analysis."""
    
    def __init__(self, enabled_processors=None):
        self.logger = logging.getLogger(__name__)
        self.analyzer = DocumentAnalyzer()
        self.processor_manager = ProcessorManager(enabled_processors)
        self.result_combiner = ResultCombiner()
    
    def process_document(self, file_path: str, page_indices: Optional[List[int]] = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Process a document using selected processors based on content analysis.
        
        Args:
            file_path: Path to the document file
            page_indices: Optional list of page indices to process
            output_dir: Directory to save results in
            
        Returns:
            Dictionary of processing results
        """
        # Analyze document
        analysis = self.analyzer.analyze_document(file_path)
        
        # Select processors
        selected_processors = self.processor_manager.select_processors(analysis)
        
        # Process with each processor
        processor_results = {}
        for processor_name in selected_processors:
            processor = self.processor_manager.get_processor_instance(processor_name)
            if processor:
                try:
                    results = self.processor_manager.execute_processor(processor, file_path, page_indices)
                    processor_results[processor_name] = results
                except Exception as e:
                    self.logger.error(f"Error processing with {processor_name}: {e}")
        
        # Combine results
        combined_results = self.result_combiner.combine_results(processor_results)
        
        # Save results
        if output_dir:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            self.result_combiner.save_results(combined_results, base_name, output_dir)
        
        return combined_results

# Helper functions
def parse_page_indices(page_str: str) -> List[int]:
    """Parse page indices from a string."""
    # Implementation from document_processor.py
    pass
```

### 2. app.py (312 lines)

This file should be split into:

#### a. app.py (main Flask application)
```python
"""
Flask Web Application
--------------------
Main web application for document processing.
"""

import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from web_routes import register_routes
from api_endpoints import register_api_endpoints
from config import OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("docling_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DoclingWeb")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register routes and API endpoints
register_routes(app)
register_api_endpoints(app)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

#### b. web_routes.py
```python
"""
Web Routes Module
---------------
Web routes for the document processing application.
"""

import os
import uuid
import threading
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from document_processor import DocumentProcessor, parse_page_indices
from processing_tasks import processing_tasks, process_document_task
from utils import allowed_file, get_processor_list, get_recent_documents, get_processor_results
from config import OUTPUT_DIR

def register_routes(app):
    """Register web routes with the Flask application."""
    
    @app.route('/')
    def index():
        """Render the main page."""
        processors = get_processor_list()
        recent_documents = get_recent_documents()
        return render_template('index.html', processors=processors, recent_documents=recent_documents)
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file upload and processing."""
        # Implementation from app.py
        pass
    
    @app.route('/results/<base_name>')
    def view_results(base_name):
        """View results for a document."""
        # Implementation from app.py
        pass
    
    @app.route('/compare/<base_name>')
    def compare_results(base_name):
        """Compare results from different processors."""
        # Implementation from app.py
        pass
```

#### c. api_endpoints.py
```python
"""
API Endpoints Module
------------------
API endpoints for the document processing application.
"""

from flask import jsonify
from utils import get_document_pages, get_page_content
from processing_tasks import processing_tasks, get_processor_results

def register_api_endpoints(app):
    """Register API endpoints with the Flask application."""
    
    @app.route('/api/pages/<base_name>/<processor>')
    def api_get_pages(base_name, processor):
        """API endpoint to get pages for a document from a specific processor."""
        # Implementation from app.py
        pass
    
    @app.route('/api/content/<base_name>/<processor>/<page>')
    def api_get_content(base_name, processor, page):
        """API endpoint to get content for a specific page from a processor."""
        # Implementation from app.py
        pass
    
    @app.route('/api/status/<task_id>')
    def api_task_status(task_id):
        """API endpoint to get task status."""
        # Implementation from app.py
        pass
```

#### d. processing_tasks.py
```python
"""
Processing Tasks Module
---------------------
Background processing tasks for document processing.
"""

import threading
import logging
from typing import Dict, List, Any, Optional

from document_processor import DocumentProcessor
from utils import get_processor_results

# Dictionary to track processing tasks
processing_tasks = {}

logger = logging.getLogger("DoclingWeb")

def process_document_task(task_id, file_path, processors, page_indices, output_dir):
    """Background task to process a document."""
    # Implementation from app.py
    pass
```

#### e. utils.py
```python
"""
Utilities Module
--------------
Utility functions for the document processing application.
"""

import os
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

from config import OUTPUT_DIR

logger = logging.getLogger("DoclingWeb")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    # Implementation from app.py
    pass

def get_processor_list():
    """Get list of available processors."""
    # Implementation from app.py
    pass

def get_recent_documents():
    """Get list of recently processed documents."""
    # Implementation from app.py
    pass

def get_processor_results(base_name):
    """Get list of processor results for a document."""
    # Implementation from app.py
    pass

def get_document_pages(base_name, processor):
    """Get list of pages for a document from a specific processor."""
    # Implementation from app.py
    pass

def get_page_content(base_name, processor, page):
    """Get content for a specific page from a processor."""
    # Implementation from app.py
    pass

def compare_content(content1, content2):
    """Compare two content strings and return HTML with differences highlighted."""
    # Implementation from app.py
    pass
```

## Files to Archive

The following files should be moved to the `archived` folder:

1. `LMstudio.py` (844 lines) - Replaced by modular processor implementation
2. `gemini_OCR.py` (447 lines) - Replaced by modular processor implementation
3. `gemini_OCR_fixed.py` (217 lines) - Replaced by modular processor implementation
4. `pdf_processor.py` (429 lines) - Functionality now in document_processor.py
5. `pdf_page_processor.py` (168 lines) - Functionality now in processor modules
6. `basic.py` (110 lines) - Just a basic example, not needed for core functionality
7. `process_pdf_example.py` (171 lines) - Example code, not needed for core functionality

## Implementation Strategy

1. Create the new module files with proper docstrings and class definitions
2. Move code from the original files to the new modules
3. Update imports in all affected files
4. Test each module individually
5. Test the entire system to ensure functionality is preserved
6. Archive the old files once the new structure is working correctly

## Benefits of Modularization

1. **Improved Readability**: Smaller files are easier to understand
2. **Better Maintainability**: Each module has a single responsibility
3. **Easier Testing**: Modules can be tested independently
4. **Simplified Debugging**: Issues are isolated to specific modules
5. **Enhanced Collaboration**: Multiple developers can work on different modules simultaneously
6. **Clearer Documentation**: Each module has focused documentation
