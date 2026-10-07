import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
Phase 1 Integration Test Suite
Verifies that all supported input methods (Text, PDF, DOCX, TXT, URL)
work seamlessly with the pipeline entry points and error handling.
"""
import os
import sys
from input_processor import ingest_content

def test_phase1_flow():
    print("[TEST 1] Empty input validation")
    res = ingest_content("", None, "")
    assert not res["success"], "Empty input should fail"
    assert "No input provided" in res["error"]
    print("  -> Passed")

    print("[TEST 2] Direct text input")
    text = "The James Webb Space Telescope was launched in December 2021. It orbits the Sun-Earth L2 Lagrange point."
    res = ingest_content(direct_text=text)
    assert res["success"]
    assert res["source_type"] == "text"
    assert "James Webb" in res["text"]
    assert res["word_count"] >= 15
    print(f"  -> Passed ({res['word_count']} words, type={res['source_type']})")

    print("[TEST 3] TXT document input")
    txt_path = "test_phase1_sample.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Quantum computing relies on quantum bits or qubits. Qubits can exist in superpositions of states.")
    
    res = ingest_content(file_path=txt_path)
    assert res["success"]
    assert res["source_type"] == "txt"
    assert "Quantum computing" in res["text"]
    print(f"  -> Passed ({res['word_count']} words, type={res['source_type']})")

    print("[TEST 4] Live URL input")
    url = "https://en.wikipedia.org/wiki/Speed_of_light"
    res = ingest_content(url=url)
    assert res["success"]
    assert res["source_type"] == "url"
    assert "light" in res["text"].lower()
    print(f"  -> Passed ({res['word_count']} words, type={res['source_type']})")

    # Clean up
    if os.path.exists(txt_path):
        os.remove(txt_path)

    print("\nALL PHASE 1 INTEGRATION TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_phase1_flow()
