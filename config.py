from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdf"  # Input PDFs
OUTPUT_DIR = BASE_DIR / "output"  # Output files

# Ensure directories exist
PDF_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemini Configuration
GEMINI_API_KEY = "Your Gemini API key"  # Your Gemini API key
GEMINI_MODEL = "gemini-1.5-pro"

# LM Studio Configuration
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_MODEL = "local-model"
