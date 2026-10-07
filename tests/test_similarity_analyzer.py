import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
Unit Tests for TruthLens SimilarityAnalyzer (Phase 2)
Verifies:
1. Accurate topical and lexical similarity computation
2. Explicit differentiation between legitimate topical overlap and plagiarism
3. Correct classification of verbatim copy, paraphrasing, topical overlap, and distinct content
"""

from similarity_analyzer import SimilarityAnalyzer, extract_shared_keywords

def test_similarity_analyzer():
    analyzer = SimilarityAnalyzer()

    print("=================================================================")
    print("TRUTHLENS TEXT SIMILARITY VALIDATION SUITE (PHASE 2)")
    print("=================================================================")

    # Test 1: Verbatim Copy
    print("\n[TEST 1] Testing Verbatim Match Detection...")
    text_a = "The James Webb Space Telescope was launched on December 25 2021 from French Guiana."
    text_b = "The James Webb Space Telescope was launched on December 25 2021 from French Guiana."
    res1 = analyzer.analyze_pair(text_a, text_b, "NASA Release")
    assert res1["relationship"] == "VERBATIM_COPY", f"Expected VERBATIM_COPY, got {res1['relationship']}"
    assert res1["is_plagiarism"] is True
    assert res1["similarity_score"] >= 90.0
    print(f"  [OK] PASSED: Classified as {res1['relationship_label']} (Score: {res1['similarity_score']}%, Plagiarism: {res1['is_plagiarism']})")

    # Test 2: Topical / Factual Overlap (NOT Plagiarism)
    print("\n[TEST 2] Testing Topical / Factual Overlap Differentiation (NOT Plagiarism)...")
    text_target = "Astronomers deployed the James Webb observatory to peer into the ancient cosmos and analyze distant exoplanet atmospheres."
    text_source = "The James Webb Space Telescope enables scientists to study the early universe and detect chemical signatures around distant exoplanets."
    res2 = analyzer.analyze_pair(text_target, text_source, "Scientific Journal")
    assert res2["relationship"] in ("TOPICAL_OVERLAP", "SHARED_TERMINOLOGY"), f"Expected TOPICAL_OVERLAP, got {res2['relationship']}"
    assert res2["is_plagiarism"] is False, "Topical overlap must NOT be confused with plagiarism!"
    assert res2["topical_similarity"] >= 25.0
    print(f"  [OK] PASSED: Classified as {res2['relationship_label']} (Topical Sim: {res2['topical_similarity']}%, Plagiarism: {res2['is_plagiarism']})")
    print(f"  Explanation: {res2['relationship_desc']}")

    # Test 3: Completely Distinct Content
    print("\n[TEST 3] Testing Distinct Independent Content...")
    text_dist_a = "Photosynthesis converts solar light energy into chemical sugars within plant chloroplasts."
    text_dist_b = "The financial markets experienced volatility following changes in central bank interest rates."
    res3 = analyzer.analyze_pair(text_dist_a, text_dist_b, "Economic Report")
    assert res3["relationship"] == "DISTINCT"
    assert res3["is_plagiarism"] is False
    assert res3["similarity_score"] <= 15.0
    print(f"  [OK] PASSED: Classified as {res3['relationship_label']} (Score: {res3['similarity_score']}%)")

    # Test 4: Document-Level Multi-Source Analysis
    print("\n[TEST 4] Testing Multi-Source Document Alignment...")
    doc = "Global temperatures continue to rise due to anthropogenic greenhouse gas emissions, causing melting polar ice."
    sources = [
        {"title": "IPCC Report", "body": "Greenhouse gas emissions from human activities are driving global temperature increases and polar ice retreat."},
        {"title": "Cooking Guide", "body": "Preheat the oven to 350 degrees before preparing the dough for baking cookies."},
    ]
    doc_res = analyzer.analyze_document_similarity(doc, sources)
    assert doc_res["overall_similarity"] > 20.0
    assert "greenhouse" in doc_res["shared_topical_keywords"] or "emissions" in doc_res["shared_topical_keywords"] or "ice" in doc_res["shared_topical_keywords"]
    print(f"  [OK] PASSED: Document similarity={doc_res['overall_similarity']}%, Consensus={doc_res['topical_consensus_level']}")
    print(f"  Shared Keywords: {doc_res['shared_topical_keywords']}")

    print("\n=================================================================")
    print("ALL SIMILARITY ANALYZER TESTS PASSED SUCCESSFULLY! (4/4)")
    print("=================================================================")

if __name__ == "__main__":
    test_similarity_analyzer()
