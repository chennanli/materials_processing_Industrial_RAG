#!/usr/bin/env python3
"""
LM Studio Client Implementation
-------------------------------
Client for interacting with LM Studio API.
"""

import logging
import time
from typing import Dict, List, Optional, Any

import requests

# Configure logging
logger = logging.getLogger("LMStudioClient")

class LMStudioClient:
    """Client for interacting with LM Studio API."""

    def __init__(self, api_url, model):
        """Initialize the LM Studio client.

        Args:
            api_url: LM Studio API URL
            model: LM Studio model name (fallback only)
        """
        self.api_url = api_url
        self.model = model  # Fallback model only
        self.current_model = None
        self.model_capabilities = {}
        self._connection_cache = None
        self._cache_time = 0
        self._cache_duration = 30  # Cache connection status for 30 seconds
        self._processed_pages = set()  # Track processed pages to avoid duplicates
        import threading
        self._lock = threading.Lock()  # Thread safety for concurrent access

    def _is_connection_cached(self):
        """Check if we have a valid cached connection status."""
        return (
            self._connection_cache is not None
            and time.time() - self._cache_time < self._cache_duration
        )

    def _check_connection_fast(self):
        """Fast connection check with caching."""
        # Use cached result if available
        if self._is_connection_cached():
            return self._connection_cache

        # Perform actual connection check
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.api_url}/v1/models", timeout=3
            )  # Reduced timeout
            end_time = time.time()

            is_connected = response.status_code == 200

            # Cache the result
            self._connection_cache = is_connected
            self._cache_time = time.time()

            logger.debug(
                f"LMStudio connection check: {end_time - start_time:.2f}s, status: {is_connected}"
            )
            return is_connected

        except Exception as e:
            logger.debug(f"LMStudio connection failed: {e}")
            # Cache the failure for a shorter time
            self._connection_cache = False
            self._cache_time = time.time()
            return False

    def detect_current_model(self):
        """Detect which model is currently loaded in LM Studio.

        Returns:
            str: Current model name or None if no model detected
        """
        # Quick connection check first
        if not self._check_connection_fast():
            logger.debug("LMStudio not connected, skipping model detection")
            return None

        try:
            start_time = time.time()
            models_response = requests.get(
                f"{self.api_url}/v1/models", timeout=2
            )  # Further reduced timeout for speed
            end_time = time.time()

            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = models_data.get("data", [])
                if available_models:
                    # Get the first (and usually only) loaded model
                    current_model = available_models[0]["id"]
                    self.current_model = current_model
                    logger.info(
                        f"✅ Detected LMStudio model: {current_model} ({end_time - start_time:.2f}s)"
                    )
                    return current_model
                else:
                    logger.warning("No models currently loaded in LMStudio")
                    return None
            else:
                logger.warning(
                    f"Failed to get models from LMStudio: {models_response.status_code}"
                )
                return None
        except Exception as e:
            logger.warning(f"Could not detect current LMStudio model: {e}")
            return None

    def is_ocr_specialized_model(self, model_name=None):
        """Check if the current or specified model is specialized for OCR.

        Args:
            model_name: Model name to check, or None to check current model

        Returns:
            bool: True if model is OCR-specialized
        """
        if model_name is None:
            model_name = self.current_model or self.model

        if not model_name:
            return False

        # Check for OCR-specialized model indicators
        ocr_indicators = [
            "monkeyocr",
            "ocrflux",
            "ocr",
            "recognition",
            "text-recognition",
            "document-ai",
            "textract",
            "paddle-ocr",
            "flux",
        ]

        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in ocr_indicators)

    def is_vision_language_model(self, model_name=None):
        """Check if the model is a vision-language model.

        Args:
            model_name: Model name to check (optional)

        Returns:
            bool: True if model is vision-language specialized
        """
        if model_name is None:
            model_name = self.current_model or self.model

        if not model_name:
            return False

        # Check for vision-language model indicators
        vision_indicators = [
            'internvl', 'qwen', 'llava', 'blip', 'instructblip', 'minigpt',
            'vision', 'multimodal', 'vl-', 'visual', 'cogvlm', 'yi-vl',
            'deepseek-vl', 'phi-3-vision', 'pixtral', 'molmo', 'vl'
        ]

        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in vision_indicators)

    def get_model_type(self, model_name=None):
        """Determine the type/specialization of the model.

        Args:
            model_name: Model name to check, or None to check current model

        Returns:
            str: 'ocr', 'vision', or 'general'
        """
        if model_name is None:
            model_name = self.current_model or self.model

        if not model_name:
            return "general"

        if self.is_ocr_specialized_model(model_name):
            return "ocr"
        elif self.is_vision_language_model(model_name):
            return "vision"
        else:
            return "general"

    def analyze_model_capabilities(self, model_name):
        """Analyze what capabilities a model has based on its name and type.

        Args:
            model_name: Name of the model to analyze

        Returns:
            List of capability strings
        """
        if not model_name:
            return ["unknown"]

        capabilities = []
        model_lower = model_name.lower()

        # Check for vision capabilities
        vision_indicators = ["vl", "vision", "visual", "multimodal", "internvl", "qwen", "llava", "clip", "image"]
        if any(indicator in model_lower for indicator in vision_indicators):
            capabilities.append("vision")

        # Check for OCR specialization
        ocr_indicators = ["ocr", "text", "document", "recognition", "monkey", "flux"]
        if any(indicator in model_lower for indicator in ocr_indicators):
            capabilities.append("ocr-specialized")

        # Check for instruction following
        instruction_indicators = ["instruct", "chat", "assistant"]
        if any(indicator in model_lower for indicator in instruction_indicators):
            capabilities.append("instruction-following")

        # If no specific capabilities detected, assume general
        if not capabilities:
            capabilities.append("general")

        return capabilities

    def generate(self, prompt, max_tokens=1024, temperature=0.1):
        """Generate text using LM Studio.

        Args:
            prompt: Prompt to send to LM Studio
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation

        Returns:
            Generated text
        """
        start_time = time.time()

        try:
            # Quick connection check (cached)
            if not self._check_connection_fast():
                logger.error("LMStudio not available")
                return None

            # Use current model if detected, otherwise use configured model
            model_to_use = self.current_model or self.model

            # Use the chat completions endpoint with the correct format
            payload = {
                "model": model_to_use,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts and enhances text from documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            logger.debug(f"Sending request to LMStudio with model: {model_to_use}")

            # Use optimized API call
            api_start_time = time.time()
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=60,  # Further reduced timeout for speed
            )
            api_end_time = time.time()

            if response.status_code != 200:
                logger.error(
                    f"LM Studio API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                # Extract content from the chat response format
                content = result["choices"][0]["message"]["content"].strip()
                total_time = time.time() - start_time
                api_time = api_end_time - api_start_time

                logger.info(
                    f"✅ LMStudio generation complete: {len(content)} chars in {total_time:.2f}s (API: {api_time:.2f}s)"
                )
                return content
            else:
                logger.error(f"Unexpected response format: {result}")
                return None

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ LMStudio API error after {total_time:.2f}s: {e}")
            return None

    def get_system_message_for_model(self, model_name: str) -> str:
        """Get optimized system message for the given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Optimized system message
        """
        if not model_name:
            model_name = "generic"
            
        model_lower = model_name.lower()
        
        if "ocrflux" in model_lower or "flux" in model_lower:
            return "Extract text from chemical data tables with extreme precision. Preserve ALL numerical values exactly as shown, including decimal points (e.g., 56.108, 0.272). Maintain chemical formulas (C4H8) and chemical names. DO NOT add formatting markers like 'x6'. MOST IMPORTANT: DO NOT use markdown table format with vertical bars. Use plain text format with spaces between columns. Each compound should be on its own line with all its properties. Format like this:\n\nNO   FORMULA   NAME                     MOLWT     TFP      TB\n1    AR        ARGON                    39.948    83.8     87.3\n2    BCL3      BORON TRI CHLORIDE       117.169   165.9    285.7\n\nOutput as plain text with spacing. NO vertical bars or markdown formatting."
        elif "monkeyocr" in model_lower or "monkey" in model_lower:
            return "You are an advanced OCR system specialized in chemical data tables and scientific documents. Extract text with perfect precision, especially numerical values with decimal points (e.g., 56.108, 0.272, 134.3). Preserve chemical formulas (C4H8) exactly. CRITICAL REQUIREMENT: DO NOT use markdown table format with vertical bars. Use plain text format with spaces between columns. Each compound should be on a single line with all its properties. Format like this:\n\nNO   FORMULA   NAME                     MOLWT     TFP      TB\n1    AR        ARGON                    39.948    83.8     87.3\n2    BCL3      BORON TRI CHLORIDE       117.169   165.9    285.7\n\nOutput as plain text with spacing only. NO vertical bars or markdown formatting."
        elif "internvl" in model_lower:
            return "You are an advanced multimodal AI with expert vision and scientific document analysis capabilities. For this chemical reference table: 1) Extract ALL numerical values with perfect precision (e.g., 56.108, 0.272, 134.3), 2) Preserve chemical formulas (C4H8) exactly, 3) Maintain proper column alignment, 4) Keep chemical names in uppercase, 5) DO NOT add any formatting markers like 'x6'. ABSOLUTELY CRITICAL: DO NOT use markdown table format with vertical bars. Use plain text format with spaces between columns. Each compound should be on its own line with all its properties. Format your output as plain text like this:\n\nNO   FORMULA   NAME                     MOLWT     TFP      TB\n1    AR        ARGON                    39.948    83.8     87.3\n2    BCL3      BORON TRI CHLORIDE       117.169   165.9    285.7\n\nNEVER use vertical bars or markdown formatting in your output."
        elif "qwen" in model_lower and "vl" in model_lower:
            return "You are a state-of-the-art multimodal AI specialized in scientific and chemical data extraction. This document contains a reference table with columns for NO, FORMULA (e.g., C4H8), NAME (e.g., CIS-2-BUTENE), MOLWT (e.g., 56.108), and other properties with decimal values (e.g., 0.272, 134.3). Extract ALL numerical values with perfect precision. Preserve chemical formulas exactly. Maintain column alignment. DO NOT add formatting markers like 'x6'. Keep chemical names in uppercase. CRITICAL REQUIREMENT: DO NOT output in markdown table format with vertical bars. Format as plain text with spaces to separate columns. Each compound should be on its own line with all properties. Format like this:\n\nNO   FORMULA   NAME                     MOLWT     TFP      TB\n1    AR        ARGON                    39.948    83.8     87.3\n2    BCL3      BORON TRI CHLORIDE       117.169   165.9    285.7\n\nPlease output in this plain text format with NO vertical bars or markdown formatting."
        else:
            return "You are a specialized scientific document extractor focused on chemical data tables. Extract tables with these requirements: 1) ALL numerical values must be preserved with exact precision and decimal placement (e.g., 56.108, 0.272, 134.3), 2) Chemical formulas (e.g., C4H8) must be preserved exactly, 3) Column headers like NO, FORMULA, NAME, MOLWT must be maintained, 4) Chemical names should be in uppercase, 5) DO NOT add any formatting markers like 'x6'. ABSOLUTELY CRITICAL REQUIREMENT: DO NOT format as a markdown table with vertical bars. Instead, output as plain text with proper spacing. Format each compound on its own line like this:\n\nNO   FORMULA   NAME                     MOLWT     TFP      TB\n1    AR        ARGON                    39.948    83.8     87.3\n2    BCL3      BORON TRI CHLORIDE       117.169   165.9    285.7\n\nOutput as plain text with spacing, NOT markdown or any other formatting. NEVER use vertical bars."

    def get_consistent_ocr_prompt(self, page_num: int) -> str:
        """Get a consistent OCR prompt for all pages to ensure uniform results.
        
        Args:
            page_num: Page number being processed
            
        Returns:
            str: Consistent OCR prompt
        """
        return f"""Extract all text from this document page {page_num} with maximum accuracy.

This document contains a chemical data table with columns like:
- NO (number index)
- FORMULA (chemical formula like C4H8, CCL3F)
- NAME (chemical name like CIS-2-BUTENE, TRICHLOROFLUOROMETHANE)
- MOLWT (molecular weight like 56.108, 137.368)
- TFP, TB, TC, PC, VC, ZC, OMEGA (numerical properties with decimals)

ABSOLUTELY CRITICAL REQUIREMENTS:
1. Extract EVERY visible character, number, and symbol exactly as shown
2. Preserve ALL decimal values precisely (e.g., 56.108, 0.272, 137.368)
3. DO NOT use markdown table format with vertical bars (|)
4. DO NOT output in a table format at all
5. Instead, output each compound on a single line with its properties separated by tabs or multiple spaces
6. Keep chemical formulas intact (e.g., C4H8, CCL3F)
7. Chemical names should be in uppercase

Output the data in a simple plain text format like this:

NO   FORMULA   NAME                      MOLWT     TFP      TB
1    AR        ARGON                     39.948    83.8     87.3
2    BCL3      BORON TRI CHLORIDE        117.169   165.9    285.7
3    BF3       BORON TRIFLUORIDE         67.805    146.5    -

Output as plain text with spacing, NOT markdown or any other formatting.
Do NOT use vertical bars in your output."""