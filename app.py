#!/usr/bin/env python3
"""
Document Processing Web UI
-------------------------
A web interface for processing documents with multiple processors
and comparing results side by side.
"""

import difflib
import json
import logging
import os
import sys
import threading
import uuid
import warnings
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

# Import document processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR
from document_processor import DocumentProcessor, parse_page_indices

# Suppress MPS pin_memory warnings (cosmetic only, doesn't affect performance)
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("docling_web.log"), logging.StreamHandler()],
)
logger = logging.getLogger("DoclingWeb")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size


# Custom template filter for proper processor name formatting
@app.template_filter("processor_name")
def processor_name_filter(processor):
    """Format processor names for display."""
    name_mapping = {
        "smart": "Smart (Recommended)",
        "lmstudio": "LMStudio",
        "docling": "Docling",
        "camelot": "Camelot",
        "gemini": "Gemini",
    }
    return name_mapping.get(processor, processor.capitalize())


# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Dictionary to track processing tasks
processing_tasks = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "docx", "jpg", "jpeg", "png"}


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_processor_list():
    """Get list of available processors."""
    return ["smart", "docling", "gemini", "lmstudio", "camelot"]


def get_recent_documents():
    """Get list of recently processed documents."""
    documents = []

    # Check in the new directory structure first
    for doc_dir in Path(OUTPUT_DIR).glob("*"):
        if doc_dir.is_dir() and (doc_dir / "combined").exists():
            documents.append(doc_dir.name)

    # Also check in the old directory structure for backward compatibility
    for doc_dir in Path(OUTPUT_DIR).glob("*_combined"):
        base_name = doc_dir.name.replace("_combined", "")
        if base_name not in documents:  # Avoid duplicates
            documents.append(base_name)

    return sorted(documents)


def get_processor_results(base_name):
    """Get list of processor results for a document."""
    processors = []

    # Check in the new directory structure
    doc_dir = Path(OUTPUT_DIR) / base_name
    if doc_dir.exists() and doc_dir.is_dir():
        for proc_dir in doc_dir.glob("*"):
            if proc_dir.is_dir() and proc_dir.name != "combined":
                processors.append(proc_dir.name)

    # Also check in the old directory structure for backward compatibility
    if not processors:
        for proc_dir in Path(OUTPUT_DIR).glob(f"{base_name}_*"):
            if proc_dir.is_dir() and not proc_dir.name.endswith("_combined"):
                processor_name = proc_dir.name.replace(f"{base_name}_", "")
                processors.append(processor_name)

    return sorted(processors)


def get_document_pages(base_name, processor):
    """Get list of pages for a document from a specific processor."""
    pages = []

    # Check in the new directory structure first
    new_json_file = Path(OUTPUT_DIR) / base_name / processor / "combined.json"

    # Fall back to the old directory structure if needed
    old_json_file = (
        Path(OUTPUT_DIR) / f"{base_name}_{processor}" / f"{base_name}_combined.json"
    )

    # Try the new structure first
    if new_json_file.exists():
        try:
            with open(new_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                pages = sorted(
                    data.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
                )
        except Exception as e:
            logger.error(f"Error loading pages from {new_json_file}: {e}")

    # Fall back to old structure if needed
    elif old_json_file.exists():
        try:
            with open(old_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                pages = sorted(
                    data.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
                )
        except Exception as e:
            logger.error(f"Error loading pages from {old_json_file}: {e}")

    return pages


def get_page_content(base_name, processor, page):
    """Get content for a specific page from a processor."""
    content = ""

    # Check in the new directory structure first
    new_json_file = Path(OUTPUT_DIR) / base_name / processor / "combined.json"

    # Fall back to the old directory structure if needed
    old_json_file = (
        Path(OUTPUT_DIR) / f"{base_name}_{processor}" / f"{base_name}_combined.json"
    )

    # Try the new structure first
    if new_json_file.exists():
        try:
            with open(new_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if page in data and "content" in data[page]:
                    content = data[page]["content"]
                elif page in data and "text" in data[page]:
                    content = data[page]["text"]
        except Exception as e:
            logger.error(f"Error loading content from {new_json_file}: {e}")

    # Fall back to old structure if needed and content is still empty
    if not content and old_json_file.exists():
        try:
            with open(old_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if page in data and "content" in data[page]:
                    content = data[page]["content"]
                elif page in data and "text" in data[page]:
                    content = data[page]["text"]
        except Exception as e:
            logger.error(f"Error loading content from {old_json_file}: {e}")

    return content


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
    html_diff = diff.make_table(
        lines1, lines2, "Processor 1", "Processor 2", context=True
    )

    return html_diff


@app.route("/")
def index():
    """Render the main page."""
    processors = get_processor_list()
    recent_documents = get_recent_documents()
    return render_template(
        "index.html", processors=processors, recent_documents=recent_documents
    )


def process_document_task(task_id, file_path, processors, page_indices):
    """Background task to process a document."""
    try:
        # Update task status
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 5,
            "message": "Starting document processing",
        }

        # Initialize processor
        doc_processor = DocumentProcessor(enabled_processors=processors)
        
        # Update status
        processing_tasks[task_id]["progress"] = 10
        processing_tasks[task_id]["message"] = "Analyzing document"
        
        # First determine page count (to calculate progress)
        import fitz
        total_pages = 0
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()
        except Exception as e:
            logger.error(f"Error determining page count: {e}")
            total_pages = 10  # Default fallback
            
        # Create a callback for progress updates
        def progress_callback(stage, current_page=None, total=None):
            if stage == "analyzing":
                processing_tasks[task_id]["progress"] = 15
                processing_tasks[task_id]["message"] = "Analyzing document content"
            elif stage == "selecting_processors":
                processing_tasks[task_id]["progress"] = 20
                processing_tasks[task_id]["message"] = "Selecting processors"
            elif stage == "processing_start":
                processing_tasks[task_id]["progress"] = 25
                processing_tasks[task_id]["message"] = "Starting document processing"
            elif stage == "processing_page" and current_page is not None and total is not None:
                # Calculate progress between 25% and 90% based on page processing
                page_progress = 65 * (current_page / total)
                processing_tasks[task_id]["progress"] = 25 + page_progress
                processing_tasks[task_id]["message"] = f"Processing page {current_page} of {total}"
            elif stage == "combining_results":
                processing_tasks[task_id]["progress"] = 90
                processing_tasks[task_id]["message"] = "Combining results"
            elif stage == "saving_results":
                processing_tasks[task_id]["progress"] = 95
                processing_tasks[task_id]["message"] = "Saving results"

        # Process document with progress callback
        # Note: We'll need to modify document_processor.py to accept and use this callback
        # For now, we'll just call the regular method and update progress manually
        
        # Update to analyzing stage
        progress_callback("analyzing")
        
        # Call process_document with the progress callback
        doc_processor.process_document(
            file_path=file_path, page_indices=page_indices, output_dir=OUTPUT_DIR,
            progress_callback=progress_callback
        )

        # Update status
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["progress"] = 100
        processing_tasks[task_id]["message"] = "Processing complete"
        processing_tasks[task_id]["processors"] = get_processor_results(task_id)

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        processing_tasks[task_id] = {
            "status": "error",
            "message": f"Error processing document: {str(e)}",
        }


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and processing."""
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save file with secure filename
        filename = secure_filename(file.filename)
        base_name = os.path.splitext(filename)[0]

        # Add unique ID to avoid overwriting
        unique_id = uuid.uuid4().hex[:8]
        task_id = f"{base_name}_{unique_id}"
        unique_filename = f"{task_id}{os.path.splitext(filename)[1]}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(file_path)

        # Get selected processors
        processors = request.form.getlist("processors")
        if not processors:
            processors = ["all"]

        # Get page indices
        pages = request.form.get("pages", "")
        page_indices = parse_page_indices(pages) if pages else None

        # Initialize task status
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Document queued for processing",
        }

        # Start processing in background thread
        thread = threading.Thread(
            target=process_document_task,
            args=(task_id, file_path, processors, page_indices),
        )
        thread.daemon = True
        thread.start()

        # Redirect to results page with task ID
        return redirect(url_for("view_results", base_name=task_id))

    flash("Invalid file type")
    return redirect(url_for("index"))


@app.route("/results/<base_name>")
def view_results(base_name):
    """View results for a document."""
    # Check if task is still processing
    task_status = None
    if base_name in processing_tasks:
        task_status = processing_tasks[base_name]

    # Get available processors (even if still processing)
    processors = get_processor_results(base_name)

    return render_template(
        "results.html",
        base_name=base_name,
        processors=processors,
        task_status=task_status,
    )


@app.route("/compare/<base_name>")
def compare_results(base_name):
    """Compare results from different processors."""
    processors = get_processor_results(base_name)

    # Get selected processors
    proc1 = request.args.get("proc1", processors[0] if processors else None)
    proc2 = request.args.get("proc2", processors[1] if len(processors) > 1 else proc1)

    # Get pages for the first processor
    pages = get_document_pages(base_name, proc1) if proc1 else []

    # Get selected page
    page = request.args.get("page", pages[0] if pages else None)

    # Get content for selected page from both processors
    content1 = get_page_content(base_name, proc1, page) if proc1 and page else ""
    content2 = get_page_content(base_name, proc2, page) if proc2 and page else ""

    # Compare content
    diff_html = compare_content(content1, content2)

    return render_template(
        "compare.html",
        base_name=base_name,
        processors=processors,
        pages=pages,
        selected_proc1=proc1,
        selected_proc2=proc2,
        selected_page=page,
        content1=content1,
        content2=content2,
        diff_html=diff_html,
    )


@app.route("/api/pages/<base_name>/<processor>")
def api_get_pages(base_name, processor):
    """API endpoint to get pages for a document from a specific processor."""
    pages = get_document_pages(base_name, processor)
    return jsonify(pages)


@app.route("/api/content/<base_name>/<processor>/<page>")
def api_get_content(base_name, processor, page):
    """API endpoint to get content for a specific page from a processor."""
    content = get_page_content(base_name, processor, page)
    return jsonify({"content": content})


@app.route("/api/status/<task_id>")
def api_task_status(task_id):
    """API endpoint to get task status."""
    if task_id in processing_tasks:
        return jsonify(processing_tasks[task_id])
    else:
        # Check if results exist even if task is not in memory
        processors = get_processor_results(task_id)
        if processors:
            return jsonify(
                {
                    "status": "complete",
                    "progress": 100,
                    "message": "Processing complete",
                    "processors": processors,
                }
            )
        return jsonify({"status": "unknown", "message": "Task not found"})


@app.route("/api/model-info")
def api_model_info():
    """API endpoint to get current model information."""
    try:
        # Get model info from LMStudio processor
        from processors.lmstudio_processor import LMStudioProcessor

        lm_processor = LMStudioProcessor()
        model_info = lm_processor.get_model_info()

        return jsonify({"status": "success", "model_info": model_info})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    # Use a dynamic port to avoid conflicts
    import socket
    import webbrowser

    def find_free_port():
        """Find a free port on localhost"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    print("=" * 80)
    print(f"\n🚀 Document Processing System is running at: {url}")
    print(f"\n👉 Open this URL in your browser: {url}")
    print("\nPress CTRL+C to quit the server")
    print("=" * 80)

    # Attempt to open the browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Disable auto-reloader to avoid port confusion
    app.run(debug=True, host="127.0.0.1", port=port, use_reloader=False)
