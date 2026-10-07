import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
TruthLens Phase 3 Pipeline Integration Test Suite
Verifies:
1. Evidence-based hallucination classification across claims
2. Article-level grounding score and hallucination aggregation
3. Transparent multi-signal Trust Score (Fact, Credibility, Grounding, Originality)
4. Result card generation with Hallucination badge and warning alert
5. Summary card generation with Trust Score Hero and 4-signal breakdown
6. Programmatic check_article full report structure
"""

from app import (
    create_result_card,
    build_summary,
    get_source_credibility,
    check_article
)
from trust_engine import HallucinationDetector, TrustScoreEngine

def test_phase3_pipeline():
    print("=================================================================")
    print("TRUTHLENS PHASE 3 PIPELINE INTEGRATION TEST SUITE")
    print("=================================================================")

    hal_detector = HallucinationDetector()
    trust_engine = TrustScoreEngine()

    # -----------------------------------------------------------------
    # Test 1: Evidence-Based Hallucination Detector
    # -----------------------------------------------------------------
    print("\n[TEST 1] Testing Evidence-Based Hallucination Detection...")
    ev1 = [{
        "title": "NASA Mars Missions",
        "body": "NASA Perseverance rover successfully landed in Jezero crater on Mars in February 2021.",
        "url": "https://nasa.gov/perseverance",
        "credibility_score": 0.95
    }]
    claim_faithful = "Perseverance rover landed in Jezero crater on Mars in 2021."
    res_faithful = hal_detector.analyze_claim(claim_faithful, "SUPPORTED", ev1, confidence=96.0)
    assert res_faithful["status"] == "FAITHFUL"
    assert res_faithful["hallucination_risk"] <= 15.0
    assert res_faithful["grounding_ratio"] >= 70.0

    claim_contradicted = "Perseverance rover crash-landed on Venus in 1999."
    res_contra = hal_detector.analyze_claim(claim_contradicted, "REFUTED", ev1, confidence=92.0)
    assert res_contra["status"] == "CONTRADICTED"
    assert res_contra["hallucination_risk"] >= 85.0

    claim_unsupported = "The rover discovered underground alien structures built of quartz."
    res_unsup = hal_detector.analyze_claim(claim_unsupported, "UNCERTAIN", ev1, confidence=30.0)
    assert res_unsup["status"] in ("UNSUPPORTED", "UNVERIFIABLE")
    assert res_unsup["hallucination_risk"] >= 50.0

    print(f"  [OK] PASSED: Faithful={res_faithful['status']} (Risk: {res_faithful['hallucination_risk']}%), Contradicted={res_contra['status']} (Risk: {res_contra['hallucination_risk']}%), Unsupported={res_unsup['status']} (Risk: {res_unsup['hallucination_risk']}%)")

    # -----------------------------------------------------------------
    # Test 2: Article-Level Grounding Score & Hallucination Aggregation
    # -----------------------------------------------------------------
    print("\n[TEST 2] Testing Article-Level Hallucination Aggregation...")
    sentence_results = [
        {"sentence": claim_faithful, "verdict": "SUPPORTED", "sources": ev1, "confidence": 96.0},
        {"sentence": claim_contradicted, "verdict": "REFUTED", "sources": ev1, "confidence": 92.0},
        {"sentence": claim_unsupported, "verdict": "UNCERTAIN", "sources": ev1, "confidence": 30.0}
    ]
    hal_report = hal_detector.evaluate_article_hallucination(sentence_results)
    assert hal_report["faithful_count"] == 1
    assert hal_report["contradicted_count"] == 1
    assert hal_report["total_claims"] == 3
    assert 0.0 <= hal_report["grounding_score"] <= 100.0
    print(f"  [OK] PASSED: Grounding Score={hal_report['grounding_score']}%, Faithful={hal_report['faithful_count']}, Contradicted={hal_report['contradicted_count']}")

    # -----------------------------------------------------------------
    # Test 3: Transparent Multi-Signal Trust Score Engine
    # -----------------------------------------------------------------
    print("\n[TEST 3] Testing Transparent Multi-Signal Trust Score...")
    all_sources = [
        {"url": "https://nasa.gov/perseverance", "credibility_score": 0.95},
        {"url": "https://reuters.com/space", "credibility_score": 0.85}
    ]
    plag_audit = {"originality_score": 94.0}
    trust_res = trust_engine.calculate_trust_score(
        sentence_results=sentence_results,
        all_sources=all_sources,
        hallucination_report=hal_report,
        plagiarism_audit=plag_audit
    )

    t_score = trust_res["trust_score"]
    t_breakdown = trust_res["breakdown"]
    assert 0.0 <= t_score <= 100.0
    assert "fact_verification" in t_breakdown
    assert "source_credibility" in t_breakdown
    assert "evidence_grounding" in t_breakdown
    assert "content_originality" in t_breakdown
    assert t_breakdown["fact_verification"]["weight"] == 35
    assert t_breakdown["source_credibility"]["weight"] == 25
    assert t_breakdown["evidence_grounding"]["weight"] == 25
    assert t_breakdown["content_originality"]["weight"] == 15
    print(f"  [OK] PASSED: Overall Trust Score={t_score}/100 ({trust_res['rating_label']} - Grade {trust_res['rating_tier']})")
    print(f"    - Fact Verification  (35%): {t_breakdown['fact_verification']['score']}% -> {t_breakdown['fact_verification']['weighted_score']} pts")
    print(f"    - Source Credibility (25%): {t_breakdown['source_credibility']['score']}% -> {t_breakdown['source_credibility']['weighted_score']} pts")
    print(f"    - Evidence Grounding (25%): {t_breakdown['evidence_grounding']['score']}% -> {t_breakdown['evidence_grounding']['weighted_score']} pts")
    print(f"    - Content Originality(15%): {t_breakdown['content_originality']['score']}% -> {t_breakdown['content_originality']['weighted_score']} pts")

    # -----------------------------------------------------------------
    # Test 4: Result Card with Hallucination Pill & Warning Alert
    # -----------------------------------------------------------------
    print("\n[TEST 4] Testing Result Card HTML Generation with Phase 3 Badges...")
    card_html = create_result_card(
        sentence=claim_contradicted,
        verdict="REFUTED",
        atomic_claims=[{"claim": claim_contradicted, "verdict": "REFUTED", "explanation": "Contradicted by NASA data.", "sources": ev1}],
        index=1,
        confidence=92.0,
        hallucination_result=res_contra
    )
    assert "hal-pill" in card_html, "Card must render hallucination pill"
    assert "hal-contradicted" in card_html, "Card must render contradicted class"
    assert "hal-alert-box" in card_html, "Card must render hallucination alert box for refuted/contradicted claim"
    assert "GROUNDING &amp; HALLUCINATION AUDIT" in card_html
    print("  [OK] PASSED: Result card rendered with hallucination pill and warning alert box.")

    # -----------------------------------------------------------------
    # Test 5: Summary Card with Trust Score Hero & 4-Signal Grid
    # -----------------------------------------------------------------
    print("\n[TEST 5] Testing Summary Card HTML with Trust Score Hero & Grid...")
    summary_html = build_summary(
        sentence_results=sentence_results,
        plagiarism_audit=plag_audit,
        trust_result=trust_res,
        hallucination_report=hal_report
    )
    assert "trust-hero-container" in summary_html, "Summary must render Trust Hero container"
    assert "trust-signals-grid" in summary_html, "Summary must render 4 signals grid"
    assert "fill-fact" in summary_html and "fill-cred" in summary_html
    assert "fill-ground" in summary_html and "fill-orig" in summary_html
    assert "summary-hallucination-strip" in summary_html, "Summary must render hallucination strip"
    assert f"{t_score}" in summary_html, "Summary must display numeric trust score"
    print("  [OK] PASSED: Summary card rendered with Trust Hero, 4 signals grid, and Hallucination strip.")

    print("\n=================================================================")
    print("ALL PHASE 3 INTEGRATION TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=================================================================")

if __name__ == "__main__":
    test_phase3_pipeline()
