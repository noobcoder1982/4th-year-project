import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
TruthLens Phase 2 Pipeline Integration Test Suite
Verifies:
1. Source credibility scoring, tiers, and backwards compatibility
2. Evidence ranking and score assignment
3. Evidence-grounded claim confidence calculation (without fabrication)
4. Text similarity analysis and its clear distinction from plagiarism
5. Result card and summary generation with Phase 2 badges and metrics
"""

from app import (
    get_source_credibility,
    compute_claim_confidence,
    create_result_card,
    build_summary,
    SourceCredibility
)
from similarity_analyzer import SimilarityAnalyzer

def test_phase2_features():
    print("=================================================================")
    print("TRUTHLENS PHASE 2 PIPELINE INTEGRATION TEST SUITE")
    print("=================================================================")

    # Test 1: Source Credibility Tiers and Scoring
    print("\n[TEST 1] Testing Source Credibility Engine...")
    gov_cred = get_source_credibility("https://www.nasa.gov/mission_pages/station/research")
    assert isinstance(gov_cred, tuple) and len(gov_cred) == 3, "Must unpack as 3-tuple"
    lvl, cls_name, desc = gov_cred
    assert lvl == "HIGH"
    assert gov_cred.score >= 0.90, f"Expected gov score >= 0.90, got {gov_cred.score}"
    assert gov_cred.tier == "TIER_1"

    news_cred = get_source_credibility("https://www.reuters.com/world/europe/article-test")
    assert news_cred.tier == "TIER_2"
    assert news_cred.score >= 0.80

    wiki_cred = get_source_credibility("https://en.wikipedia.org/wiki/Earth")
    assert wiki_cred.tier == "TIER_3"
    assert wiki_cred.score >= 0.60

    penalized_cred = get_source_credibility("https://www.reddit.com/r/science/comments/test")
    assert penalized_cred.tier == "PENALIZED"
    assert penalized_cred.score <= 0.25

    print(f"  [OK] PASSED: NASA={gov_cred.score} ({gov_cred.tier}), Reuters={news_cred.score} ({news_cred.tier}), Wikipedia={wiki_cred.score} ({wiki_cred.tier}), Reddit={penalized_cred.score} ({penalized_cred.tier})")

    # Test 2: Evidence-Grounded Claim Confidence Calculation
    print("\n[TEST 2] Testing Claim Confidence Calculation (Evidence Grounded)...")
    evidence_sample = [
        {
            "title": "NASA Official Report",
            "body": "The James Webb Space Telescope was launched into space on December 25 2021 from Kourou.",
            "url": "https://www.nasa.gov/jwst",
            "credibility_score": 0.95
        },
        {
            "title": "ESA Space Updates",
            "body": "Ariane 5 rocket launched the James Webb Space Telescope in December 2021.",
            "url": "https://www.esa.int/jwst",
            "credibility_score": 0.92
        }
    ]
    claim_true = "The James Webb Space Telescope was launched in December 2021."
    conf_sup, label_sup, reason_sup = compute_claim_confidence(claim_true, "SUPPORTED", evidence_sample)
    assert conf_sup >= 80.0, f"Expected high confidence >= 80, got {conf_sup}"
    assert "High Confidence" in label_sup

    refuted_evidence = [
        {
            "title": "NASA Moon Science",
            "body": "The Moon is a rocky body composed of silicate rock and basalt, not cheese or dairy products.",
            "url": "https://www.nasa.gov/moon",
            "credibility_score": 0.95
        }
    ]
    conf_ref, label_ref, reason_ref = compute_claim_confidence("The Moon is made of green cheese.", "REFUTED", refuted_evidence)
    assert conf_ref >= 75.0, f"Expected refutation confidence >= 75, got {conf_ref}"

    conf_unc, label_unc, reason_unc = compute_claim_confidence("Undiscovered alien signals exist on Mars.", "UNCERTAIN", [])
    assert conf_unc <= 35.0, f"Expected low confidence for uncertain, got {conf_unc}"
    assert "Insufficient" in label_unc or "Inconclusive" in label_unc

    print(f"  [OK] PASSED: Supported Conf={conf_sup}%, Refuted Conf={conf_ref}%, Uncertain Conf={conf_unc}%")

    # Test 3: Text Similarity vs Plagiarism Differentiation
    print("\n[TEST 3] Testing Text Similarity vs Plagiarism Differentiation...")
    sim_analyzer = SimilarityAnalyzer()
    
    # Topical overlap: same topic, independent phrasing
    text_target = "Astronomers observe distant exoplanets to characterize chemical atmospheric fingerprints."
    text_source = "Scientists analyze the atmospheres of exoplanets searching for distinctive chemical signatures."
    sim_res = sim_analyzer.analyze_pair(text_target, text_source, "Scientific Journal")
    assert sim_res["relationship"] in ("TOPICAL_OVERLAP", "SHARED_TERMINOLOGY")
    assert sim_res["is_plagiarism"] is False, "Topical overlap must NOT be flagged as plagiarism"
    print(f"  [OK] PASSED: Topical similarity={sim_res['similarity_score']}%, Plagiarism={sim_res['is_plagiarism']}")
    print(f"  Explanation: {sim_res['relationship_desc']}")

    # Test 4: Result Card Rendering with Phase 2 Features
    print("\n[TEST 4] Testing Result Card HTML Generation with Phase 2 Elements...")
    atomic_claims = [{
        "claim": "Water consists of hydrogen and oxygen atoms.",
        "verdict": "SUPPORTED",
        "confidence": 95.0,
        "confidence_label": "High Confidence",
        "explanation": "Supported by standard chemical definitions.",
        "sources": [
            {
                "title": "USGS Water Science",
                "body": "Water is composed of hydrogen and oxygen (H2O).",
                "url": "https://www.usgs.gov/water",
                "rank": 1,
                "credibility_score": 0.95
            }
        ]
    }]

    card_html = create_result_card(
        sentence="Water consists of hydrogen and oxygen atoms.",
        verdict="SUPPORTED",
        atomic_claims=atomic_claims,
        index=1,
        confidence=95.0,
        similarity_result=sim_res,
        plag_result={"category": "ORIGINAL", "similarity_score": 0.0}
    )

    assert "sim-pill" in card_html, "Must render sim-pill"
    assert "RANKED WEB EVIDENCE" in card_html, "Must render ranked web evidence section"
    assert "#1 RANKED" in card_html, "Must render rank badge"
    assert "95% Conf." in card_html, "Must render confidence in verdict badge"
    print("  [OK] PASSED: Result card rendered with sim-pill, confidence, and ranked evidence badges.")

    # Test 5: Summary Card Rendering with Similarity Strip
    print("\n[TEST 5] Testing Summary Card HTML with Similarity Strip...")
    sim_summary = {
        "overall_similarity": 35.0,
        "topical_overlap_score": 42.0,
        "lexical_overlap_score": 18.0,
        "shared_topical_keywords": ["hydrogen", "oxygen", "water"],
        "topical_consensus_level": "Moderate"
    }
    summary_html = build_summary(
        sentence_results=[{"verdict": "SUPPORTED"}],
        plagiarism_audit={"originality_score": 98.0, "overall_similarity": 2.0},
        similarity_summary=sim_summary
    )
    assert "summary-similarity-strip" in summary_html, "Must render similarity strip"
    assert "Topical Overlap" in summary_html
    print("  [OK] PASSED: Summary card rendered with dual grid and similarity strip.")

    print("\n=================================================================")
    print("ALL PHASE 2 INTEGRATION TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=================================================================")

if __name__ == "__main__":
    test_phase2_features()
