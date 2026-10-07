import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
Unit Tests for TruthLens Hallucination Detector & Trust Score Engine (Phase 3)
Verifies:
1. Evidence-based hallucination classification (FAITHFUL, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE)
2. Aggregate article hallucination metrics
3. Transparent multi-signal Trust Score calculation and breakdown
"""

from trust_engine import HallucinationDetector, TrustScoreEngine

def test_trust_engine():
    hal_detector = HallucinationDetector()
    trust_engine = TrustScoreEngine()

    print("=================================================================")
    print("TRUTHLENS HALLUCINATION DETECTOR & TRUST SCORE SUITE (PHASE 3)")
    print("=================================================================")

    # Test 1: Faithful Grounded Claim
    print("\n[TEST 1] Testing Faithful Evidence Grounding...")
    claim1 = "The James Webb Space Telescope orbits the Sun-Earth L2 Lagrange point."
    evidence1 = [{
        "title": "NASA JWST Orbit",
        "body": "The James Webb Space Telescope operates in a halo orbit around the Sun-Earth L2 Lagrange point approximately 1.5 million kilometers from Earth.",
        "credibility_score": 0.95
    }]
    res1 = hal_detector.analyze_claim(claim1, "SUPPORTED", evidence1, confidence=95.0)
    assert res1["status"] == "FAITHFUL"
    assert res1["hallucination_risk"] <= 15.0
    print(f"  [OK] PASSED: Classified as {res1['status_label']} (Risk: {res1['hallucination_risk']}%, Grounding: {res1['grounding_ratio']}%)")

    # Test 2: Contradicted Statement (Direct Hallucination)
    print("\n[TEST 2] Testing Contradicted Statement Detection...")
    claim2 = "Humans possess four lungs inside their chest cavity."
    evidence2 = [{
        "title": "Anatomy Reference",
        "body": "The human respiratory system contains two lungs situated in the thoracic cavity.",
        "credibility_score": 0.94
    }]
    res2 = hal_detector.analyze_claim(claim2, "REFUTED", evidence2, confidence=92.0)
    assert res2["status"] == "CONTRADICTED"
    assert res2["hallucination_risk"] >= 85.0
    print(f"  [OK] PASSED: Classified as {res2['status_label']} (Risk: {res2['hallucination_risk']}%)")

    # Test 3: Unverifiable / Speculative Statement
    print("\n[TEST 3] Testing Unverifiable Claim Handling...")
    claim3 = "Sub-surface crystalline alien pyramids were unearthed beneath Olympus Mons."
    res3 = hal_detector.analyze_claim(claim3, "UNCERTAIN", [], confidence=20.0)
    assert res3["status"] == "UNVERIFIABLE"
    assert res3["hallucination_risk"] >= 60.0
    print(f"  [OK] PASSED: Classified as {res3['status_label']} (Risk: {res3['hallucination_risk']}%)")

    # Test 4: Article-Level Hallucination Aggregation
    print("\n[TEST 4] Testing Article-Level Hallucination Aggregation...")
    sentence_results = [
        {"sentence": claim1, "verdict": "SUPPORTED", "sources": evidence1, "confidence": 95.0},
        {"sentence": claim2, "verdict": "REFUTED", "sources": evidence2, "confidence": 92.0},
        {"sentence": claim3, "verdict": "UNCERTAIN", "sources": [], "confidence": 20.0}
    ]
    hal_report = hal_detector.evaluate_article_hallucination(sentence_results)
    assert hal_report["faithful_count"] == 1
    assert hal_report["contradicted_count"] == 1
    assert hal_report["unverifiable_count"] == 1
    assert hal_report["grounding_score"] > 0
    print(f"  [OK] PASSED: Faithful={hal_report['faithful_count']}, Contradicted={hal_report['contradicted_count']}, Grounding Score={hal_report['grounding_score']}%")

    # Test 5: Transparent Multi-Signal Trust Score Calculation
    print("\n[TEST 5] Testing Transparent Trust Score Engine...")
    all_sources = [
        {"url": "https://nasa.gov/jwst", "credibility_score": 0.95},
        {"url": "https://reuters.com/space", "credibility_score": 0.85}
    ]
    plag_audit = {"originality_score": 92.0}

    trust_result = trust_engine.calculate_trust_score(
        sentence_results=sentence_results,
        all_sources=all_sources,
        hallucination_report=hal_report,
        plagiarism_audit=plag_audit
    )

    score = trust_result["trust_score"]
    breakdown = trust_result["breakdown"]

    assert 0.0 <= score <= 100.0
    assert "fact_verification" in breakdown
    assert "source_credibility" in breakdown
    assert "evidence_grounding" in breakdown
    assert "content_originality" in breakdown

    # Verify weights sum to 100%
    total_weights = sum(b["weight"] for b in breakdown.values())
    assert total_weights == 100

    print(f"  Overall Trust Score : {score}/100 ({trust_result['rating_label']} - Grade {trust_result['rating_tier']})")
    print(f"  Score Breakdown:")
    for key, data in breakdown.items():
        print(f"    - {key.replace('_', ' ').title():22}: Score={data['score']:.1f}/100 | Weight={data['weight']}% | Contrib={data['weighted_score']:.1f} pts")

    print(f"  Assessment Summary  : {trust_result['assessment_summary']}")
    print("\n=================================================================")
    print("ALL TRUST ENGINE TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=================================================================")

if __name__ == "__main__":
    test_trust_engine()
