#!/usr/bin/env python3
"""
OCR Model Comparison Test Script
Compare different OCR models (OCRFlux, MonkeyOCR, InternVL3, Gemini) on the same document.
"""

import os
import time

from document_processor import DocumentProcessor
from processors.lmstudio_processor import LMStudioProcessor


def test_model_performance(test_file, model_name, processor_type="lmstudio"):
    """Test a specific model's performance on a document."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_name} ({processor_type})")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        if processor_type == "lmstudio":
            # Test LMStudio model
            processor = DocumentProcessor(enabled_processors=["lmstudio"])

            # Check current model
            lm_processor = LMStudioProcessor()
            current_model = lm_processor.get_model_info()["current_model"]
            print(f"📋 Current LMStudio model: {current_model}")

            if model_name.lower() not in current_model.lower():
                print(f"⚠️  Expected {model_name} but found {current_model}")
                print(f"💡 Please load {model_name} in LMStudio and try again")
                return None

        elif processor_type == "gemini":
            processor = DocumentProcessor(enabled_processors=["gemini"])
        else:
            print(f"❌ Unknown processor type: {processor_type}")
            return None

        # Process single page for comparison
        result = processor.process_document(test_file, page_indices=[1])
        end_time = time.time()

        processing_time = end_time - start_time

        if result and "processor_results" in result:
            processor_result = result["processor_results"].get(processor_type, {})
            if "pages" in processor_result:
                pages = processor_result["pages"]
                total_chars = 0
                total_pages = len(pages)

                print("✅ Processing completed successfully!")
                print(f"⏱️  Processing time: {processing_time:.2f} seconds")
                print(f"📄 Pages processed: {total_pages}")

                for page_num, page_data in pages.items():
                    if "content" in page_data:
                        chars = len(page_data["content"])
                        total_chars += chars
                        print(f"   Page {page_num}: {chars} characters")

                        # Show content preview
                        content_preview = page_data["content"][:200].replace("\n", " ")
                        print(f"   Preview: {content_preview}...")

                chars_per_second = (
                    total_chars / processing_time if processing_time > 0 else 0
                )
                print(f"🚀 Speed: {chars_per_second:.1f} chars/second")

                return {
                    "model": model_name,
                    "processor": processor_type,
                    "processing_time": processing_time,
                    "total_characters": total_chars,
                    "pages_processed": total_pages,
                    "chars_per_second": chars_per_second,
                    "success": True,
                }
            else:
                print("❌ No pages found in results")
                return None
        else:
            print("❌ No results from processor")
            return None

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"❌ Error after {processing_time:.2f} seconds: {e}")
        return None


def main():
    """Main comparison function."""
    print("🔍 OCR Model Performance Comparison")
    print("=" * 60)

    # Find test file
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print(f"❌ Uploads directory not found: {uploads_dir}")
        return

    pdf_files = [f for f in os.listdir(uploads_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDF files found in {uploads_dir}")
        return

    test_file = os.path.join(uploads_dir, pdf_files[0])
    print(f"📄 Test document: {test_file}")

    # Test results storage
    results = []

    # Test current LMStudio model (whatever is loaded)
    print("\n🎯 Testing current LMStudio model...")
    lm_processor = LMStudioProcessor()
    current_model = lm_processor.get_model_info()["current_model"]

    if current_model:
        result = test_model_performance(test_file, current_model, "lmstudio")
        if result:
            results.append(result)
    else:
        print("❌ No model loaded in LMStudio")

    # Optionally test Gemini (uncomment if you want to compare)
    # print(f"\n🌐 Testing Gemini API...")
    # gemini_result = test_model_performance(test_file, "Gemini-1.5-Pro", 'gemini')
    # if gemini_result:
    #     results.append(gemini_result)

    # Show comparison summary
    if results:
        print(f"\n{'='*80}")
        print("📊 PERFORMANCE COMPARISON SUMMARY")
        print(f"{'='*80}")

        print(
            f"{'Model':<25} {'Time (s)':<10} {'Chars':<8} {'Speed (c/s)':<12} {'Status'}"
        )
        print("-" * 80)

        for result in results:
            status = "✅ Success" if result["success"] else "❌ Failed"
            print(
                f"{result['model']:<25} {result['processing_time']:<10.2f} {result['total_characters']:<8} {result['chars_per_second']:<12.1f} {status}"
            )

        # Find best performer
        if len(results) > 1:
            fastest = min(results, key=lambda x: x["processing_time"])
            most_accurate = max(results, key=lambda x: x["total_characters"])

            print("\n🏆 WINNERS:")
            print(
                f"   ⚡ Fastest: {fastest['model']} ({fastest['processing_time']:.2f}s)"
            )
            print(
                f"   📝 Most Content: {most_accurate['model']} ({most_accurate['total_characters']} chars)"
            )

    print("\n💡 RECOMMENDATIONS:")
    print("   1. Test OCRFlux: Load OCRFlux in LMStudio and run this script again")
    print("   2. Compare Models: Switch between MonkeyOCR and OCRFlux to compare")
    print("   3. Test Gemini: Uncomment Gemini test above for cloud comparison")
    print(
        "   4. Choose Best: Use fastest for daily work, most accurate for critical docs"
    )


if __name__ == "__main__":
    main()
