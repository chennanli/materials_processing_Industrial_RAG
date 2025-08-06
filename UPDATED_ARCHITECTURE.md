# Docling OCR - Updated Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Document Processor                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐     ┌───────────────┐    ┌───────────────┐   │
│  │ Document      │     │ Processor     │    │ Result        │   │
│  │ Analysis      │────▶│ Selection     │───▶│ Aggregation   │   │
│  │ (10-20%)      │     │ (20-25%)      │    │ (90-100%)     │   │
│  └───────────────┘     └───────────────┘    └───────────────┘   │
│          │                     │                    │            │
└──────────┼─────────────────────┼────────────────────┼────────────┘
           │                     │                    │
           ▼                     ▼                    ▼
┌──────────────────┐  ┌─────────────────────────────────────────────┐
│                  │  │          Processor Registry                  │
│  File/Content    │  ├─────────────┬─────────────┬─────────────────┤
│  Type Detection  │  │ Docling     │ Camelot     │ Gemini/LMStudio │
│                  │  │ Processor   │ Processor   │ Processor       │
└──────────────────┘  └─────────────┴─────────────┴─────────────────┘
                                                      │
                                                      │
                                                      ▼
                       ┌───────────────────────────────────────────┐
                       │ Page-by-Page Processing (25-90%)          │
                       │ (Progress tracking per page)              │
                       └───────────────────────────────────────────┘
```

## Enhanced Data Flow

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Input         │     │ Document      │     │ Content       │     │ Processor     │
│ Document      │────▶│ Analysis      │────▶│ Type Mapping  │────▶│ Selection     │
│ (5-10%)       │     │ (10-20%)      │     │               │     │ (20-25%)      │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                                                          │
                                                                          ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Combined      │     │ Result        │     │ Parallel      │     │ Processor     │
│ Output        │◀────│ Aggregation   │◀────│ Processing    │◀────│ Execution     │
│ (95-100%)     │     │ (90-95%)      │     │ (Per Page)    │     │ (25-90%)      │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

## LMStudio Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LMStudio Processor                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐     ┌───────────────┐      ┌────────────────────────┐    │
│  │ Extract Pages │     │ Detect Current│      │ Select Best Available  │    │
│  │ as Images     │────▶│ Model Type    │─────▶│ Model                  │    │
│  │               │     │               │      │                        │    │
│  └───────────────┘     └───────────────┘      └────────────────────────┘    │
│          │                                                │                  │
│          └────────────────────┬───────────────────────────┘                  │
│                               │                                              │
│                               ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Model-Specific Processing                         │  │
│  ├───────────────┬───────────────────┬─────────────────┬─────────────────┤  │
│  │ OCRFlux       │ MonkeyOCR         │ InternVL        │ Qwen-VL         │  │
│  │ Processor     │ Processor         │ Processor       │ Processor       │  │
│  └───────────────┴───────────────────┴─────────────────┴─────────────────┘  │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────┐                         │
│  │ Specialized Chemical Table Prompt Generation    │                         │
│  │ * Table-specific prompts with column details    │                         │
│  │ * Decimal precision requirements                │                         │
│  │ * Chemical formula preservation                 │                         │
│  └────────────────────────────────────────────────┘                         │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────┐                         │
│  │ Process Image with LMStudio API                │                         │
│  │ * Send image with specialized prompt           │                         │
│  │ * Use model-specific system message            │                         │
│  │ * Set low temperature (0.1)                    │                         │
│  └────────────────────────────────────────────────┘                         │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────┐                         │
│  │ Post-Processing & Formatting Fixes             │                         │
│  │ * Fix 'x6' patterns to decimal values          │                         │
│  │ * Correct chemical formulas                    │                         │
│  │ * Fix missing decimal points                   │                         │
│  │ * Clean up formatting                          │                         │
│  └────────────────────────────────────────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Smart Processor Decision Flow

```
┌───────────────┐     ┌───────────────────────┐     ┌───────────────┐
│ Input PDF     │     │ Initial Analysis      │     │ Page Content  │
│ or Image      │────▶│ with Docling/Fallback │────▶│ Classification │
│               │     │                       │     │               │
└───────────────┘     └───────────────────────┘     └───────────────┘
                                                            │
                                                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Content-Based Routing                              │
├────────────────┬────────────────────┬────────────────┬─────────────────┤
│ Poor Text      │ Contains Tables    │ Contains       │ Complex Visual  │
│ Quality Pages  │                    │ Images         │ Content         │
│ ↓              │ ↓                  │ ↓              │ ↓               │
│ LMStudio       │ Camelot            │ Gemini         │ Gemini/LMStudio │
│ (OCR)          │ (Table Extractor)  │ (Vision API)   │ (Vision Models) │
└────────────────┴────────────────────┴────────────────┴─────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      Quality Assurance                                 │
├───────────────────────────────────────────────────────────────────────┤
│ • Check content length and quality                                     │
│ • Try fallback extraction for minimal content                          │
│ • Merge results from multiple processors                               │
│ • Apply post-processing fixes                                          │
└───────────────────────────────────────────────────────────────────────┘
```

## Progress Tracking System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Progress Tracking Flow                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Start (0%) → Initial Setup (5%) → Document Analysis (10-15%)           │
│                                                                         │
│  → Processor Selection (20%) → Processing Start (25%)                   │
│                                                                         │
│  → Page Processing (25-90%, based on page count)                        │
│    [Each page updates progress: current_page/total_pages * 65% + 25%]   │
│                                                                         │
│  → Combining Results (90%) → Saving Results (95%) → Complete (100%)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prompt Selection for Chemical Table OCR

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Table OCR Prompt Selection Logic                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐      │
│  │ Detect Current │     │ Check Model    │     │ Select Model-  │      │
│  │ LMStudio Model │────▶│ Type           │────▶│ Specific Prompt│      │
│  │                │     │ (OCR/Vision)   │     │                │      │
│  └────────────────┘     └────────────────┘     └────────────────┘      │
│                                                        │                │
│                                                        ▼                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Chemical Table-Specific Prompt Components                        │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ • Explicit column definitions (NO, FORMULA, NAME, MOLWT, etc.)   │   │
│  │ • Decimal precision requirements (e.g., 56.108, 0.272)           │   │
│  │ • Chemical formula preservation (e.g., C4H8)                     │   │
│  │ • Chemical name preservation in uppercase                        │   │
│  │ • Warning against 'x6' and similar formatting markers            │   │
│  │ • Instructions for proper spacing and alignment                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Model-Specific System Message                                    │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ OCRFlux:     Specialized for chemical data tables with precise   │   │
│  │              numerical extraction                                │   │
│  │                                                                  │   │
│  │ MonkeyOCR:   Advanced OCR system specialized in chemical data    │   │
│  │              with decimal value precision                        │   │
│  │                                                                  │   │
│  │ InternVL:    Multimodal AI with expert scientific document       │   │
│  │              analysis capabilities                               │   │
│  │                                                                  │   │
│  │ Qwen-VL:     State-of-the-art multimodal AI specialized in       │   │
│  │              scientific and chemical data extraction             │   │
│  │                                                                  │   │
│  │ Others:      Specialized scientific document extractor for       │   │
│  │              chemical data tables                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Post-Processing for Chemical Tables

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Chemical Table Post-Processing Flow                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐      │
│  │ Detect if      │     │ Apply Regex    │     │ Fix Common     │      │
│  │ Content is a   │────▶│ Pattern Fixes  │────▶│ OCR Errors     │      │
│  │ Chemical Table │     │                │     │                │      │
│  └────────────────┘     └────────────────┘     └────────────────┘      │
│                                                        │                │
│                                                        ▼                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Fix Common OCR Issues                                            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ • Convert 'NxM' patterns to 'N.M' decimal values                 │   │
│  │ • Fix missing decimal points in values like '0272' to '0.272'    │   │
│  │ • Correct chemical formula spacing (C 4 H 8 → C4H8)              │   │
│  │ • Convert markdown tables to plain text if needed                │   │
│  │ • Fix spacing in numeric values like '56 108' to '56.108'        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Optimization Strategies for Any LMStudio Model

### Model Selection and Configuration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LMStudio Model Optimization                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Model Selection Priority                                            │
│     - OCR-specialized models: MonkeyOCR, OCRFlux                        │
│     - Vision-language models: InternVL, Qwen-VL, LLaVA                  │
│     - General models: Mistral, Llama, etc.                              │
│                                                                         │
│  2. Image Quality Enhancement                                           │
│     - Increase resolution (300 DPI → 600 DPI)                           │
│     - Convert to grayscale                                              │
│     - Enhance contrast                                                  │
│     - Apply sharpening                                                  │
│     - Scale up small images                                             │
│                                                                         │
│  3. Model-Specific Optimizations                                        │
│     - OCRFlux: Parse JSON output format                                 │
│     - General models: Provide more explicit formatting instructions      │
│     - All models: Use low temperature (0.1)                             │
│                                                                         │
│  4. Additional Processing Strategies                                    │
│     - Multi-pass OCR with different models                              │
│     - Confidence checks for expected patterns                           │
│     - Retry with different settings for low-quality output              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Overall Process Map

```
┌─────────────────────────────────┐
│ User uploads document to web UI │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│ app.py creates background task  │────▶│ process_document_task() updates │
│ with 5-10% initial progress     │     │ progress to 10%                 │
└─────────────────┬───────────────┘     └─────────────────┬───────────────┘
                  │                                       │
                  ▼                                       ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│ DocumentProcessor analyzes      │     │ Progress updated to 15-20%      │
│ document with Docling           │────▶│ as analysis completes           │
└─────────────────┬───────────────┘     └─────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│ Processor selection based on    │     │ Progress updated to 20-25%      │
│ content type mapping            │────▶│ entering processing phase       │
└─────────────────┬───────────────┘     └─────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Parallel Processing of Pages                         │
├─────────────────┬───────────────┬─────────────────────┬─────────────────┤
│ Docling         │ Camelot       │ LMStudio            │ Gemini          │
│ Processor       │ Processor     │ Processor           │ Processor       │
│ (Text/Structure)│ (Tables)      │ (OCR/Vision)        │ (Vision API)    │
└─────────────────┴───────────────┴─────────────────────┴─────────────────┘
                  │
                  │ Progress updates per-page (25-90%)
                  ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│ Results combined from all       │     │ Progress updated to 90-95%      │
│ processors                      │────▶│ for combination phase           │
└─────────────────┬───────────────┘     └─────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│ Combined results saved to       │     │ Progress updated to 100%        │
│ output directory                │────▶│ task complete                   │
└─────────────────┬───────────────┘     └─────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│ Results displayed to user       │
│ in web UI                       │
└─────────────────────────────────┘
```