# Docling OCR Processing Flow Diagram

## Main Document Processing Flow

```
                              PDF DOCUMENT
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INITIAL ANALYSIS (Docling)                  │
├─────────────────────────────────────────────────────────────────┤
│ • Analyze document structure                                    │
│ • Identify content types (text, tables, images)                 │
│ • Extract basic document properties                             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONTENT TYPE MAPPING                        │
├─────────────────────────────────────────────────────────────────┤
│ • Map content types to appropriate processors                   │
│ • Determine which processor to use for each page/element        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    PARALLEL PROCESSING       │
                  └──────────────────────────────┘
                                 │
             ┌─────────────────────────────────────┐
             │                                     │
  ┌──────────▼───────────┐            ┌────────────▼──────────┐
  │    TEXT CONTENT      │            │     TABLE CONTENT     │
  └──────────┬───────────┘            └────────────┬──────────┘
             │                                     │
    ┌────────▼────────┐                  ┌─────────▼─────────┐
    │     Docling     │                  │      Camelot      │
    │    Processor    │                  │     Processor     │
    └─────────────────┘                  └─────────┬─────────┘
                                                   │
                                         ┌─────────▼─────────┐
                                         │   Table Format    │
                                         │   Extraction      │
                                         └─────────┬─────────┘
                                                   │
                                         ┌─────────▼─────────┐
                                         │   Table to MD     │
                                         │   Conversion      │
                                         └─────────┬─────────┘
                                                   ▼
             │                                     │
  ┌──────────▼───────────┐            ┌────────────▼──────────┐
  │    IMAGE CONTENT     │            │    COMPLEX CONTENT    │
  └──────────┬───────────┘            └────────────┬──────────┘
             │                                     │
             │                                     │
     ┌───────▼──────┐                   ┌──────────▼─────────┐
     │    Is API    │                   │   Smart Processor  │
     │   Available? │                   │   Selection Logic   │
     └───────┬──────┘                   └──────────┬─────────┘
             │                                     │
      ┌──────▼─────┐                              │
      │     YES    │                              │
      └──────┬─────┘                              │
             │                                    │
  ┌──────────▼──────────┐                         │
  │  Gemini Processor   │                         │
  └───────────┬─────────┘                         │
              │                                   │
  ┌───────────▼─────────┐                         │
  │ Gemini Vision API   │                         │
  │ Image Processing    │                         │
  └───────────┬─────────┘                         │
              │                                   │
              │                                   │
              │                                   │
      ┌───────▼───────┐                ┌──────────▼─────────┐
      │      NO       │                │   LMStudio         │
      └───────┬───────┘                │   Processor        │
              │                        └──────────┬─────────┘
              │                                   │
              │                                   │
  ┌───────────▼─────────┐              ┌──────────▼─────────┐
  │ LMStudio Processor  │              │ Extract Pages as   │
  │ (Local Fallback)    │              │ Images             │
  └───────────┬─────────┘              └──────────┬─────────┘
              │                                   │
              │                                   │
              │                        ┌──────────▼─────────┐
              │                        │ Check Connection   │
              │                        │ to LMStudio        │
              │                        └──────────┬─────────┘
              │                                   │
              │                        ┌──────────▼─────────┐
              │                        │ Detect Current     │
              │                        │ LMStudio Model     │
              │                        └──────────┬─────────┘
              │                                   │
              │                        ┌──────────▼─────────┐
              │                        │ Select Optimized   │
              │                        │ Prompt for Model   │
              │                        └──────────┬─────────┘
              │                                   │
              │                        ┌──────────▼─────────┐
              │                        │ Process Image with │
              │                        │ LMStudio API       │
              │                        └──────────┬─────────┘
              │                                   │
              ▼                                   ▼

┌─────────────────────────────────────────────────────────────────┐
│                      RESULT COMBINATION                         │
├─────────────────────────────────────────────────────────────────┤
│ • Combine results from all processors                           │
│ • Prioritize results based on processor strengths               │
│ • Merge content from different sources                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT GENERATION                          │
├─────────────────────────────────────────────────────────────────┤
│ • Generate combined markdown output                             │
│ • Generate JSON output                                          │
│ • Save processor-specific results                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                         FINAL DOCUMENT RESULTS
```

## LMStudio Model Processing Path

```
                       IMAGE FROM PDF
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                 LMStudio Connection Check                 │
├───────────────────────────────────────────────────────────┤
│ • Check if LMStudio API is available at configured URL    │
│ • Verify API is responding                                │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Connection OK?    │
                    └────────┬──────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌────────────────────┐                 ┌─────────────────────┐
│        YES         │                 │         NO          │
└─────────┬──────────┘                 └──────────┬──────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────────┐              ┌─────────────────────┐
│ Detect Current Model    │              │ Fall Back to Basic  │
└─────────────┬───────────┘              │ Text Extraction     │
              │                          └─────────────────────┘
              ▼
┌───────────────────────────────────┐
│ Is a Specific Model Forced?       │
└───────────────┬───────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌────────────────┐    ┌────────────────────┐
│      YES       │    │        NO          │
└───────┬────────┘    └─────────┬──────────┘
        │                       │
        ▼                       ▼
┌────────────────┐    ┌────────────────────┐
│ Use Forced     │    │ Auto-Select Best   │
│ Model          │    │ Available Model    │
└───────┬────────┘    └─────────┬──────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│                 Determine Model Type                      │
├───────────────────────────────────────────────────────────┤
│ • Identify if model is OCR-specific (MonkeyOCR, OCRFlux)  │
│ • Identify if model is vision-capable (InternVL, Qwen)    │
│ • Otherwise classify as general model                     │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                 Select Optimized Prompt                   │
├───────────────────────────────────────────────────────────┤
│ • Choose prompt template based on model type              │
│ • Add document-specific details to prompt                 │
│ • Include page number and context                         │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                 Process Image with API                    │
├───────────────────────────────────────────────────────────┤
│ • Convert image to base64                                 │
│ • Create API payload with prompt and image                │
│ • Send to LMStudio API                                    │
│ • Parse response                                          │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                 Post-process Results                      │
├───────────────────────────────────────────────────────────┤
│ • Apply formatting fixes if needed                        │
│ • Extract content based on model type                     │
│ • Parse JSON for OCRFlux model                            │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                     PROCESSED TEXT CONTENT
```

## Result Combination Flow

```
                    PROCESSOR RESULTS
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                  Collect All Results                     │
├──────────────────────────────────────────────────────────┤
│ • Gather results from all processors                     │
│ • Organize by page number                                │
│ • Track which processor produced each result             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Content Prioritization                  │
├──────────────────────────────────────────────────────────┤
│ • Text: Prefer LMStudio/Gemini for scanned text          │
│ • Text: Prefer Docling for native text                   │
│ • Tables: Prefer Camelot for structured tables           │
│ • Images: Prefer Gemini for image descriptions           │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Merge Page Content                      │
├──────────────────────────────────────────────────────────┤
│ • Combine text from best sources                         │
│ • Insert tables in appropriate locations                 │
│ • Add image descriptions where applicable                │
│ • Preserve document structure                            │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Format Combined Output                  │
├──────────────────────────────────────────────────────────┤
│ • Generate markdown with proper structure                │
│ • Create structured JSON with all content                │
│ • Organize by page number                                │
│ • Include metadata about processors used                 │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Save Results                            │
├──────────────────────────────────────────────────────────┤
│ • Save combined results                                  │
│ • Save processor-specific results                        │
│ • Organize in output directory structure                 │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
                      FINAL OUTPUT FILES
```

## Web Application Flow

```
               USER UPLOADS DOCUMENT
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│               Web Application (Flask)                 │
├───────────────────────────────────────────────────────┤
│ • Receive uploaded file                               │
│ • Create task ID                                      │
│ • Initialize progress tracking (5%)                   │
│ • Start background processing task                    │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────┐
│              Background Processing Task               │
├───────────────────────────────────────────────────────┤
│ • Initialize document processor                       │
│ • Update progress to 10%                              │
│ • Call document_processor.process_document()          │
│ • Update progress throughout processing               │
│ • Handle completion and errors                        │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
                 DOCUMENT PROCESSOR
                (See main flow diagram)
                         │
                         ▼
┌───────────────────────────────────────────────────────┐
│               Progress Tracking Updates               │
├───────────────────────────────────────────────────────┤
│ • Analysis phase: 10-20%                              │
│ • Processor selection: 20-25%                         │
│ • Page processing: 25-90% (scaled by page count)      │
│ • Result combination: 90-95%                          │
│ • Output generation: 95-100%                          │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────┐
│                  Results Display                      │
├───────────────────────────────────────────────────────┤
│ • Show results on web interface                       │
│ • Provide links to output files                       │
│ • Display processing details                          │
│ • Show which processors were used                     │
└───────────────────────────────────────────────────────┘
```