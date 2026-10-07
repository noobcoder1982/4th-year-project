import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
=============================================================================
TruthLens - Input Processing System Test Suite (Phase 1)
=============================================================================
Tests:
1. Direct text extraction and paragraph cleaning
2. PDF file generation and text extraction (pypdf)
3. DOCX file generation and text extraction (python-docx)
4. Plain text / Markdown file extraction (UTF-8)
5. Corrupted / invalid file validation & graceful error reporting
6. Live article URL extraction and boilerplate filtering
7. Invalid URL handling and error reporting
=============================================================================
"""

import os
import tempfile
from input_processor import (
    extract_from_direct_text,
    extract_from_pdf,
    extract_from_docx,
    extract_from_txt,
    extract_from_url,
    ingest_content
)


def run_tests():
    print("=" * 65)
    print("TRUTHLENS INPUT SYSTEM VALIDATION SUITE (PHASE 1)")
    print("=" * 65)

    # ---------------------------------------------------------
    # TEST 1: Direct Text
    # ---------------------------------------------------------
    print("\n[TEST 1] Testing Direct Text Input Processing...")
    sample_text = (
        "Water is composed of hydrogen and oxygen.\n\n"
        "The Pacific Ocean is the largest ocean on Earth."
    )
    res_direct = extract_from_direct_text(sample_text)
    print(f"  Success: {res_direct['success']} | Word Count: {res_direct['word_count']}")
    assert res_direct["success"] is True
    assert res_direct["word_count"] > 10
    assert "hydrogen" in res_direct["text"]
    print("  [OK] PASSED: Direct text cleaned and parsed correctly.")

    # ---------------------------------------------------------
    # TEST 2: PDF Extraction
    # ---------------------------------------------------------
    print("\n[TEST 2] Testing PDF File Parsing...")
    from pypdf import PdfWriter
    writer = PdfWriter()
    page = writer.add_blank_page(width=72 * 8.5, height=72 * 11)
    
    # Create a small valid PDF using reportlab or PyPDF
    # Since reportlab might not be installed, let's create a PDF or test pypdf writing
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, "test_article.pdf")
    
    # We can write a simple valid PDF stream
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 55 >> stream\n"
        b"BT /F1 12 Tf 100 700 Td (Water is composed of hydrogen and oxygen.) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000244 00000 n \n0000000350 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n425\n%%EOF\n"
    )
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    res_pdf = extract_from_pdf(pdf_path)
    print(f"  Success: {res_pdf['success']} | Extracted Text: '{res_pdf['text']}'")
    assert res_pdf["success"] is True
    assert "Water is composed" in res_pdf["text"]
    print("  [OK] PASSED: PDF text layer parsed accurately.")

    # ---------------------------------------------------------
    # TEST 3: DOCX Extraction
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing DOCX File Parsing...")
    import docx
    docx_path = os.path.join(temp_dir, "test_article.docx")
    doc = docx.Document()
    doc.add_heading("Scientific Findings on Earth", level=1)
    doc.add_paragraph("The Sun is a star located at the center of our solar system.")
    doc.add_paragraph("The Great Wall of China was built across historical northern borders.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Fact 1"
    table.rows[0].cells[1].text = "Mount Everest is the highest mountain."
    doc.save(docx_path)

    res_docx = extract_from_docx(docx_path)
    print(f"  Success: {res_docx['success']} | Word Count: {res_docx['word_count']}")
    assert res_docx["success"] is True
    assert "solar system" in res_docx["text"]
    assert "Mount Everest" in res_docx["text"]
    print("  [OK] PASSED: Word (.docx) paragraphs and tables extracted cleanly.")

    # ---------------------------------------------------------
    # TEST 4: Plain Text / Markdown Extraction
    # ---------------------------------------------------------
    print("\n[TEST 4] Testing TXT / Markdown File Parsing...")
    txt_path = os.path.join(temp_dir, "test_article.md")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Climate Study\n\nPhotosynthesis converts solar energy into chemical energy.")

    res_txt = extract_from_txt(txt_path)
    print(f"  Success: {res_txt['success']} | Word Count: {res_txt['word_count']}")
    assert res_txt["success"] is True
    assert "Photosynthesis" in res_txt["text"]
    print("  [OK] PASSED: Plain text / Markdown file parsed correctly.")

    # ---------------------------------------------------------
    # TEST 5: Corrupted / Unsupported File Validation
    # ---------------------------------------------------------
    print("\n[TEST 5] Testing Corrupted / Unsupported File Error Handling...")
    corrupt_pdf_path = os.path.join(temp_dir, "corrupt.pdf")
    with open(corrupt_pdf_path, "wb") as f:
        f.write(b"NOT A REAL PDF FILE GARBAGE DATA 12345")

    res_corrupt = extract_from_pdf(corrupt_pdf_path)
    print(f"  Corrupt File Handled Gracefully: {not res_corrupt['success']}")
    print(f"  User-Friendly Error Message: {res_corrupt['error']}")
    assert res_corrupt["success"] is False
    assert res_corrupt["error"] is not None

    unsupported_path = os.path.join(temp_dir, "image.png")
    with open(unsupported_path, "wb") as f:
        f.write(b"FAKE PNG")
    res_unsupported = ingest_content(file_path=unsupported_path)
    print(f"  Unsupported Ext Handled Gracefully: {not res_unsupported['success']}")
    assert res_unsupported["success"] is False
    print("  [OK] PASSED: Corrupted and unsupported files gracefully rejected with informative errors.")

    # ---------------------------------------------------------
    # TEST 6: Live Web URL Extraction
    # ---------------------------------------------------------
    print("\n[TEST 6] Testing Live Article URL Scraping & Boilerplate Stripping...")
    test_url = "https://en.wikipedia.org/wiki/Apollo_11"
    res_url = extract_from_url(test_url)
    print(f"  Success: {res_url['success']} | Title: '{res_url['title']}' | Word Count: {res_url['word_count']}")
    assert res_url["success"] is True
    assert res_url["word_count"] > 100
    assert "Apollo 11" in res_url["title"] or "Apollo 11" in res_url["text"]
    print("  [OK] PASSED: Live article webpage fetched, boilerplate removed, body extracted.")

    # ---------------------------------------------------------
    # TEST 7: Invalid URL Error Handling
    # ---------------------------------------------------------
    print("\n[TEST 7] Testing Invalid / Non-Existent URL Error Handling...")
    invalid_url = "https://this-domain-does-not-exist-at-all-xyz987654321.com"
    res_inv_url = extract_from_url(invalid_url)
    print(f"  Success: {res_inv_url['success']} | Error: '{res_inv_url['error']}'")
    assert res_inv_url["success"] is False
    assert res_inv_url["error"] is not None
    print("  [OK] PASSED: Non-existent URL failed gracefully without throwing unhandled exceptions.")

    # ---------------------------------------------------------
    # TEST 8: Unified Ingestion Router
    # ---------------------------------------------------------
    print("\n[TEST 8] Testing Unified Ingestion Router (ingest_content)...")
    res_router = ingest_content(file_path=docx_path)
    assert res_router["success"] is True
    assert res_router["source_type"] == "docx"
    print("  [OK] PASSED: Unified router accurately dispatched to docx parser.")

    print("\n" + "=" * 65)
    print("ALL INPUT SYSTEM TESTS PASSED SUCCESSFULLY! (8/8)")
    print("=" * 65)

    # Cleanup temp files
    for p in [pdf_path, docx_path, txt_path, corrupt_pdf_path, unsupported_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


if __name__ == "__main__":
    run_tests()
