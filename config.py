import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdf"  # Input PDFs
OUTPUT_DIR = BASE_DIR / "output"  # Output files

# Ensure directories exist
PDF_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemini Configuration
# GEMINI_API_KEY is loaded from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "Your Gemini API key")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# LM Studio Configuration
# Use base URL only, endpoints will be appended in the processor
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "internvl3-14b-instruct")
