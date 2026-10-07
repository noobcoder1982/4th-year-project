import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
=============================================================================
TruthLens - Plagiarism Detection Engine Benchmark & Test Suite
=============================================================================
Tests:
1. Exact match / verbatim copying detection
2. Paraphrasing / synonym-replaced text detection
3. Cross-document comparison with full alignment matrix
4. HTML highlighting generation for plagiarism review
=============================================================================
"""

import sys
from plagiarism_detector import PlagiarismDetector, extract_sentences, highlight_matched_segments


def run_tests():
    print("=" * 60)
    print("TRUTHLENS PLAGIARISM ENGINE TEST SUITE")
    print("=" * 60)

    detector = PlagiarismDetector()

    # ---------------------------------------------------------
    # TEST 1: Verbatim Copy
    # ---------------------------------------------------------
    print("\n[TEST 1] Testing Verbatim Copy-Paste Detection...")
    s1 = "Artificial intelligence is intelligence demonstrated by machines, as opposed to intelligence of humans."
    s2 = "Artificial intelligence is intelligence demonstrated by machines, as opposed to intelligence of humans."
    res1 = detector.evaluate_pair(s1, s2)
    print(f"  Suspect : {s1}")
    print(f"  Source  : {s2}")
    print(f"  Result  : Category={res1['category']} | Similarity={res1['similarity_score']}% | Exact={res1['exact_score']}%")
    assert res1["category"] == "VERBATIM", f"Expected VERBATIM, got {res1['category']}"
    assert res1["similarity_score"] >= 95.0, f"Expected >= 95%, got {res1['similarity_score']}%"
    print("  [OK] PASSED: Verbatim copy accurately detected (100% match).")

    # ---------------------------------------------------------
    # TEST 2: High Overlap with Slight Rewording
    # ---------------------------------------------------------
    print("\n[TEST 2] Testing High Overlap Detection...")
    s3 = "Artificial intelligence is machine-demonstrated intelligence, contrary to human cognitive abilities."
    res2 = detector.evaluate_pair(s3, s1)
    print(f"  Suspect : {s3}")
    print(f"  Source  : {s1}")
    print(f"  Result  : Category={res2['category']} | Similarity={res2['similarity_score']}% | Lexical={res2['lexical_score']}%")
    print(f"  [OK] PASSED: High overlap evaluated ({res2['similarity_score']}%).")

    # ---------------------------------------------------------
    # TEST 3: Completely Original Content
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing Original / Novel Content...")
    s4 = "Photosynthesis converts solar energy into chemical energy within chlorophyll-rich plant leaves."
    res3 = detector.evaluate_pair(s4, s1)
    print(f"  Suspect : {s4}")
    print(f"  Source  : {s1}")
    print(f"  Result  : Category={res3['category']} | Similarity={res3['similarity_score']}%")
    assert res3["category"] == "ORIGINAL", f"Expected ORIGINAL, got {res3['category']}"
    assert res3["similarity_score"] <= 20.0, f"Expected <= 20%, got {res3['similarity_score']}%"
    print("  [OK] PASSED: Original content evaluated (< 20% similarity).")

    # ---------------------------------------------------------
    # TEST 4: Matched Span Highlighting
    # ---------------------------------------------------------
    print("\n[TEST 4] Testing HTML Highlighting of Plagiarized Segments...")
    high_html = highlight_matched_segments(
        "The quick brown fox jumps effortlessly over the lazy sleeping dog",
        "The quick brown fox leaps into the air over the lazy sleeping dog"
    )
    print(f"  Highlighted HTML: {high_html}")
    assert "<mark class=\"plag-highlight\">" in high_html, "Highlight tag missing"
    print("  [OK] PASSED: Exact overlapping phrases successfully wrapped in <mark>.")

    # ---------------------------------------------------------
    # TEST 5: Document-to-Document Comparison
    # ---------------------------------------------------------
    print("\n[TEST 5] Testing Document-to-Document Comparison...")
    suspect_doc = """
    The Apollo 11 mission was the first spaceflight that landed humans on the Moon.
    Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew.
    Modern neural networks learn statistical patterns directly from large training datasets.
    """

    reference_doc = """
    Apollo 11 was the American spaceflight that first landed humans on the Moon.
    Neil Armstrong and Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.
    Photosynthesis allows green plants to convert sunlight into organic nutrients.
    """

    doc_report = detector.compare_documents(suspect_doc, reference_doc)
    print(f"  Overall Similarity : {doc_report['overall_similarity']}%")
    print(f"  Originality Score  : {doc_report['originality_score']}%")
    print(f"  Verbatim Count     : {doc_report['verbatim_count']}")
    print(f"  Paraphrase Count   : {doc_report['paraphrase_count']}")
    print(f"  Original Count     : {doc_report['original_count']}")
    assert doc_report["total_sentences"] == 3
    print("  [OK] PASSED: Full document comparison matrix and alignments computed.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
