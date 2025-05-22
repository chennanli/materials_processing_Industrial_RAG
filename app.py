#!/usr/bin/env python3
"""
Document Processing Web UI
-------------------------
A web interface for processing documents with multiple processors
and comparing results side by side.
"""

import os
import json
import tempfile
import difflib
from pathlib import Path
import logging
import sys
from typing import Dict, List, Any, Optional
import uuid
import shutil
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Import document processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from document_processor import DocumentProcessor, parse_page_indices
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

# Dictionary to track processing tasks
processing_tasks = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_processor_list():
    """Get list of available processors."""
    return ['docling', 'gemini', 'lmstudio', 'camelot']

def get_recent_documents():
    """Get list of recently processed documents."""
    documents = []
    for doc_dir in Path(OUTPUT_DIR).glob("*_combined"):
        base_name = doc_dir.name.replace("_combined", "")
        documents.append(base_name)
    return sorted(documents)

def get_processor_results(base_name):
    """Get list of processor results for a document."""
    processors = []
    for proc_dir in Path(OUTPUT_DIR).glob(f"{base_name}_*"):
        if proc_dir.is_dir() and not proc_dir.name.endswith("_combined"):
            processor_name = proc_dir.name.replace(f"{base_name}_", "")
            processors.append(processor_name)
    return sorted(processors)

def get_document_pages(base_name, processor):
    """Get list of pages for a document from a specific processor."""
    pages = []
    json_file = Path(OUTPUT_DIR) / f"{base_name}_{processor}" / f"{base_name}_combined.json"
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                pages = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        except Exception as e:
            logger.error(f"Error loading pages from {json_file}: {e}")
    return pages

def get_page_content(base_name, processor, page):
    """Get content for a specific page from a processor."""
    json_file = Path(OUTPUT_DIR) / f"{base_name}_{processor}" / f"{base_name}_combined.json"
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if page in data and "content" in data[page]:
                    return data[page]["content"]
                elif page in data and "text" in data[page]:
                    return data[page]["text"]
        except Exception as e:
            logger.error(f"Error loading content from {json_file}: {e}")
    return ""

def compare_content(content1, content2):
    """Compare two content strings and return HTML with differences highlighted."""
    if not content1 and not content2:
        return ""
    
    if not content1:
        return f'<div class="added">{content2}</div>'
    
    if not content2:
        return f'<div class="removed">{content1}</div>'
    
    # Split content into lines
    lines1 = content1.splitlines()
    lines2 = content2.splitlines()
    
    # Get diff
    diff = difflib.HtmlDiff()
    html_diff = diff.make_table(lines1, lines2, 'Processor 1', 'Processor 2', context=True)
    
    return html_diff

@app.route('/')
def index():
    """Render the main page."""
    processors = get_processor_list()
    recent_documents = get_recent_documents()
    return render_template('index.html', processors=processors, recent_documents=recent_documents)

def process_document_task(task_id, file_path, processors, page_indices):
    """Background task to process a document."""
    try:
        # Update task status
        processing_tasks[task_id] = {
            'status': 'processing',
            'progress': 10,
            'message': 'Starting document processing'
        }
        
        # Initialize processor
        doc_processor = DocumentProcessor(enabled_processors=processors)
        
        # Update status
        processing_tasks[task_id]['progress'] = 20
        processing_tasks[task_id]['message'] = 'Analyzing document'
        
        # Process document
        results = doc_processor.process_document(
            file_path=file_path,
            page_indices=page_indices,
            output_dir=OUTPUT_DIR
        )
        
        # Update status
        processing_tasks[task_id]['status'] = 'complete'
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['message'] = 'Processing complete'
        processing_tasks[task_id]['processors'] = get_processor_results(task_id)
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        processing_tasks[task_id] = {
            'status': 'error',
            'message': f"Error processing document: {str(e)}"
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save file with secure filename
        filename = secure_filename(file.filename)
        base_name = os.path.splitext(filename)[0]
        
        # Add unique ID to avoid overwriting
        unique_id = uuid.uuid4().hex[:8]
        task_id = f"{base_name}_{unique_id}"
        unique_filename = f"{task_id}{os.path.splitext(filename)[1]}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Get selected processors
        processors = request.form.getlist('processors')
        if not processors:
            processors = ['all']
        
        # Get page indices
        pages = request.form.get('pages', '')
        page_indices = parse_page_indices(pages) if pages else None
        
        # Initialize task status
        processing_tasks[task_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Document queued for processing'
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_document_task,
            args=(task_id, file_path, processors, page_indices)
        )
        thread.daemon = True
        thread.start()
        
        # Redirect to results page with task ID
        return redirect(url_for('view_results', base_name=task_id))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/results/<base_name>')
def view_results(base_name):
    """View results for a document."""
    # Check if task is still processing
    task_status = None
    if base_name in processing_tasks:
        task_status = processing_tasks[base_name]
    
    # Get available processors (even if still processing)
    processors = get_processor_results(base_name)
    
    return render_template('results.html', 
                           base_name=base_name, 
                           processors=processors, 
                           task_status=task_status)

@app.route('/compare/<base_name>')
def compare_results(base_name):
    """Compare results from different processors."""
    processors = get_processor_results(base_name)
    
    # Get selected processors
    proc1 = request.args.get('proc1', processors[0] if processors else None)
    proc2 = request.args.get('proc2', processors[1] if len(processors) > 1 else proc1)
    
    # Get pages for the first processor
    pages = get_document_pages(base_name, proc1) if proc1 else []
    
    # Get selected page
    page = request.args.get('page', pages[0] if pages else None)
    
    # Get content for selected page from both processors
    content1 = get_page_content(base_name, proc1, page) if proc1 and page else ""
    content2 = get_page_content(base_name, proc2, page) if proc2 and page else ""
    
    # Compare content
    diff_html = compare_content(content1, content2)
    
    return render_template(
        'compare.html', 
        base_name=base_name, 
        processors=processors, 
        pages=pages, 
        selected_proc1=proc1, 
        selected_proc2=proc2, 
        selected_page=page,
        content1=content1,
        content2=content2,
        diff_html=diff_html
    )

@app.route('/api/pages/<base_name>/<processor>')
def api_get_pages(base_name, processor):
    """API endpoint to get pages for a document from a specific processor."""
    pages = get_document_pages(base_name, processor)
    return jsonify(pages)

@app.route('/api/content/<base_name>/<processor>/<page>')
def api_get_content(base_name, processor, page):
    """API endpoint to get content for a specific page from a processor."""
    content = get_page_content(base_name, processor, page)
    return jsonify({"content": content})

@app.route('/api/status/<task_id>')
def api_task_status(task_id):
    """API endpoint to get task status."""
    if task_id in processing_tasks:
        return jsonify(processing_tasks[task_id])
    else:
        # Check if results exist even if task is not in memory
        processors = get_processor_results(task_id)
        if processors:
            return jsonify({
                'status': 'complete',
                'progress': 100,
                'message': 'Processing complete',
                'processors': processors
            })
        return jsonify({
            'status': 'unknown',
            'message': 'Task not found'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
