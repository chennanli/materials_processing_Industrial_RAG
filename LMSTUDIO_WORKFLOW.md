# LMStudio OCR Workflow Details

## Model Selection and Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  LMStudio OCR Processing Workflow                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐      │
│  │ Initialize     │     │ Check          │     │ Auto-detect    │      │
│  │ LMStudioClient │────▶│ Connection     │────▶│ Current Model  │      │
│  │ & Processor    │     │                │     │                │      │
│  └────────────────┘     └────────────────┘     └───────┬────────┘      │
│                                                        │                │
│          ┌───────────────────────────────────────────┐ │                │
│          │ force_model parameter provided?           │ │                │
│          └───────────────────┬───────────────────────┘ │                │
│                              │                         │                │
│                              ▼                         ▼                │
│          ┌───────────────────────────────┐ ┌────────────────────────┐  │
│          │ Use forced model              │ │ _select_best_available  │  │
│          │ and get model type            │ │ _model()                │  │
│          └───────────────────┬───────────┘ └────────────┬───────────┘  │
│                              │                          │               │
│                              └────────────┬─────────────┘               │
│                                           │                             │
│                                           ▼                             │
│                             ┌────────────────────────────┐              │
│                             │ _detect_and_optimize_for_  │              │
│                             │ current_model()            │              │
│                             └────────────────┬───────────┘              │
│                                              │                          │
│  ┌─────────────────────────────────────────┐ ▼                          │
│  │ Page Processing Loop                     │                           │
│  │                                          │                           │
│  │ ┌────────────────┐   ┌────────────────┐ │                           │
│  │ │ Extract Page   │   │ Update Progress │ │                           │
│  │ │ as Image       │──▶│ Callback       │ │                           │
│  │ │                │   │                │ │                           │
│  │ └────────────────┘   └───────┬────────┘ │                           │
│  │                              │          │                           │
│  │                              ▼          │                           │
│  │               ┌─────────────────────────┐                           │
│  │               │ get_optimized_ocr_prompt│                           │
│  │               └─────────────┬───────────┘                           │
│  │                             │                                        │
│  │                             ▼                                        │
│  │               ┌─────────────────────────┐                           │
│  │               │ _process_image_with_    │                           │
│  │               │ lmstudio()              │                           │
│  │               └─────────────┬───────────┘                           │
│  │                             │                                        │
│  │                             ▼                                        │
│  │               ┌─────────────────────────┐                           │
│  │               │ _fix_chemical_table_    │                           │
│  │               │ formatting()            │                           │
│  │               └─────────────┬───────────┘                           │
│  │                             │                                        │
│  └─────────────────────────────┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prompt Selection Logic

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OCR Prompt Selection Logic                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐     ┌────────────────────────────────────────────┐  │
│  │ get_model_type │     │ model_type == "ocr"?                       │  │
│  └────────┬───────┘     └──────────────┬─────────────────────────────┘  │
│           │                            │                                │
│           │             ┌──────────────▼─────────────────────────────┐  │
│           │             │ Check specific OCR model                    │  │
│           │             │ ┌─────────────┐   ┌─────────────┐          │  │
│           │             │ │ OCRFlux?    │   │ MonkeyOCR?  │          │  │
│           │             │ └─────┬───────┘   └──────┬──────┘          │  │
│           │             │       │                  │                  │  │
│           │             │ ┌─────▼───────┐   ┌──────▼──────┐          │  │
│           │             │ │ OCRFlux     │   │ MonkeyOCR   │          │  │
│           │             │ │ prompt with │   │ prompt with │          │  │
│           │             │ │ chemical    │   │ chemical    │          │  │
│           │             │ │ table focus │   │ table focus │          │  │
│           │             │ └─────────────┘   └─────────────┘          │  │
│           │             └───────────────────────────────────────────┬┘  │
│           │                                                         │    │
│           └─────────────────┐                                       │    │
│                             ▼                                       ▼    │
│                    ┌────────────────────┐             ┌───────────────┐ │
│                    │ model_type ==      │             │ Default OCR   │ │
│                    │ "vision"?          │             │ model prompt  │ │
│                    └─────────┬──────────┘             └───────────────┘ │
│                              │                                          │
│              ┌───────────────▼───────────────┐                          │
│              │ Check specific vision model    │                         │
│              │ ┌─────────────┐  ┌────────────┐                         │
│              │ │ InternVL?   │  │ Qwen-VL?   │                         │
│              │ └─────┬───────┘  └─────┬──────┘                         │
│              │       │                │                                 │
│              │ ┌─────▼───────┐  ┌─────▼──────┐                         │
│              │ │ InternVL    │  │ Qwen-VL    │                         │
│              │ │ prompt with │  │ prompt with│                         │
│              │ │ chemical    │  │ chemical   │                         │
│              │ │ table focus │  │ table focus│                         │
│              │ └─────────────┘  └────────────┘                         │
│              │                                                          │
│              └─────────────────────┬────────────────────────────────┐  │
│                                    │                                 │  │
│                                    ▼                                 ▼  │
│                           ┌────────────────┐              ┌────────────┐│
│                           │ Other vision   │              │ General    ││
│                           │ model prompt   │              │ model      ││
│                           └────────────────┘              │ prompt     ││
│                                                           └────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Specialized Chemical Table Processing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Chemical Table-Specific Components                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Specialized Chemical Table Prompt Elements                       │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 1. Column Structure Definition                                   │   │
│  │    "This document contains a chemical data table with columns:   │   │
│  │     - NO (number index)                                          │   │
│  │     - FORMULA (chemical formula like C4H8)                       │   │
│  │     - NAME (chemical name like CIS-2-BUTENE)                     │   │
│  │     - MOLWT (molecular weight like 56.108)                       │   │
│  │     - TFP, TB, TC, PC, VC, ZC, OMEGA, LIQDEN, TDEN, DIPM..."     │   │
│  │                                                                  │   │
│  │ 2. Decimal Precision Requirements                                │   │
│  │    "Extract EVERY digit correctly, especially in decimal values  │   │
│  │     (e.g., 56.108, 134.3, 0.272)"                               │   │
│  │                                                                  │   │
│  │ 3. Formatting Instructions                                       │   │
│  │    "Preserve the exact placement of decimal points"              │   │
│  │    "Maintain proper column alignment with headers"               │   │
│  │    "Do NOT insert formatting markers like 'x6' or 'x1'"          │   │
│  │                                                                  │   │
│  │ 4. Chemical Notation                                             │   │
│  │    "Preserve chemical formulas exactly as shown (e.g., C4H8)"    │   │
│  │    "Preserve chemical names in uppercase (e.g., CIS-2-BUTENE)"   │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Chemical Table Post-Processing Functions                         │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 1. Pattern Detection                                             │   │
│  │    - Detect if content contains chemical table headers           │   │
│  │      (NO, FORMULA, NAME, MOLWT)                                  │   │
│  │                                                                  │   │
│  │ 2. Numerical Format Fixes                                        │   │
│  │    - Fix NxM patterns: (\\d+)x(\\d+) → $1.$2                    │   │
│  │    - Fix missing decimals: (\\s|^)(0)(\\d{3})(\\s|$) → $1$2.$3$4│   │
│  │    - Fix spaced decimals: (\\d+)\\s+(\\d{3})(?=\\s) → $1.$2     │   │
│  │                                                                  │   │
│  │ 3. Chemical Formula Fixes                                        │   │
│  │    - Fix spacing in formulas: C\\s+(\\d)\\s+H\\s+(\\d) → C$1H$2 │   │
│  │                                                                  │   │
│  │ 4. Markdown Cleanup                                              │   │
│  │    - Convert markdown tables to plain text if needed             │   │
│  │    - Clean up excessive whitespace                               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Best Practices for Any LMStudio Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Optimization Strategies for Any LMStudio Model              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. Image Quality Enhancements                                    │   │
│  │    - Increase resolution to 600 DPI                              │   │
│  │    - Convert to grayscale for better OCR                         │   │
│  │    - Enhance contrast (factor 1.5)                               │   │
│  │    - Apply mild sharpening                                       │   │
│  │    - Scale up small images to at least 1000px                    │   │
│  │                                                                  │   │
│  │ 2. Model Parameters                                              │   │
│  │    - Use very low temperature (0.1) for consistent output        │   │
│  │    - Increase max_tokens for complex tables (4096)               │   │
│  │    - Use detailed system messages tuned to the model type        │   │
│  │                                                                  │   │
│  │ 3. Multi-Model Strategy                                          │   │
│  │    - Try specialized OCR models first (MonkeyOCR)                │   │
│  │    - Fall back to vision-language models (InternVL, Qwen-VL)     │   │
│  │    - Use general models only if specialized ones aren't available │   │
│  │                                                                  │   │
│  │ 4. Quality Validation                                            │   │
│  │    - Check for expected patterns (chemical formulas, headers)     │   │
│  │    - Count decimal values and verify format                      │   │
│  │    - Retry with different settings for low-quality results       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Key Implementation Tips                                          │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 1. Always enhance images before OCR                              │   │
│  │ 2. Be specific about table structure in prompts                  │   │
│  │ 3. Provide explicit column names and expected value formats      │   │
│  │ 4. Always apply post-processing to fix common OCR errors         │   │
│  │ 5. Try multiple models when critical accuracy is needed          │   │
│  │ 6. For periodic tables and chemical data, emphasize decimal      │   │
│  │    precision and formula preservation in prompts                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Troubleshooting Common Issues

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Troubleshooting Common OCR Issues                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Numeric Values with 'x6' Patterns                         │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Cause: Model misinterpreting decimal formatting                  │   │
│  │ Solution:                                                        │   │
│  │ 1. Add explicit warning in prompt about 'x6' notation            │   │
│  │ 2. Apply regex fix: (\\d+)x(\\d+) → $1.$2                       │   │
│  │ 3. Try different model (MonkeyOCR often performs better)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Missing Decimal Points                                    │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Cause: Model failing to recognize decimal points in small text   │   │
│  │ Solution:                                                        │   │
│  │ 1. Increase image resolution                                     │   │
│  │ 2. Apply pattern-based fixes (0272 → 0.272)                      │   │
│  │ 3. Include example decimal values in prompt (56.108, 0.272)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Chemical Formulas with Spaces                             │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Cause: Model inserting spaces between characters                 │   │
│  │ Solution:                                                        │   │
│  │ 1. Add formula preservation instructions to prompt               │   │
│  │ 2. Apply regex fix: C\\s+(\\d)\\s+H\\s+(\\d) → C$1H$2           │   │
│  │ 3. Include example formulas in prompt (C4H8)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Inconsistent Results Across Pages                         │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Cause: Varying image quality or content between pages            │   │
│  │ Solution:                                                        │   │
│  │ 1. Add page-specific context to prompts                          │   │
│  │ 2. Apply consistent image enhancement to all pages               │   │
│  │ 3. Consider multi-pass approach with different models            │   │
│  │ 4. Implement quality check and retry logic                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```