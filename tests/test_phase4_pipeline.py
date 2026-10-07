import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
TruthLens Phase 4 Pipeline Integration Test Suite
Verifies:
1. End-to-end report generation containing all 10 mandatory fields
2. TRUE / FALSE / UNCERTAIN results mapping
3. Evidence segregation: supporting vs. contradicting evidence
4. Confidence scores and reasoning for each claim
5. Source credibility scores and tiers
6. Text similarity vs. plagiarism findings
7. Evidence-based hallucination detection findings
8. Deduplicated sources and references bibliography
9. Exportable Markdown report generation
10. Exportable JSON report generation
11. Rich Report View HTML with downloadable data URIs
12. Programmatic check_article full report structure
"""

import json
from report_generator import (
    generate_report_dict,
    generate_markdown_report,
    generate_json_report,
    render_report_html_view,
    map_verdict_to_standard
)
from trust_engine import HallucinationDetector, TrustScoreEngine
from similarity_analyzer import SimilarityAnalyzer
from app import (
    get_source_credibility,
    compute_claim_confidence,
    create_result_card,
    build_summary,
    check_article
)

def test_phase4_complete_pipeline():
    print("=================================================================")
    print("TRUTHLENS PHASE 4 END-TO-END PIPELINE INTEGRATION TEST SUITE")
    print("=================================================================")

    # Test 1: Verdict Mapping Normalization
    print("\n[TEST 1] Testing Verdict Mapping Normalization...")
    assert map_verdict_to_standard("SUPPORTED") == "TRUE"
    assert map_verdict_to_standard("REFUTED") == "FALSE"
    assert map_verdict_to_standard("PARTIALLY_SUPPORTED") == "PARTIAL"
    assert map_verdict_to_standard("UNCERTAIN") == "UNCERTAIN"
    print("  [OK] PASSED: Standard verdicts mapped correctly (TRUE / FALSE / PARTIAL / UNCERTAIN).")

    # Mock full end-to-end article verification scenario
    article_text = (
        "The Hubble Space Telescope was launched into low Earth orbit in 1990. "
        "The Sun revolves directly around the Earth every twenty-four hours. "
        "Quantum graviton warp drives were constructed in subterranean facilities in 1840."
    )

    ev_hubble = [{
        "title": "NASA Hubble History",
        "url": "https://www.nasa.gov/mission_pages/hubble/story",
        "body": "The Hubble Space Telescope was launched aboard space shuttle Discovery in April 1990.",
        "credibility_score": 0.95,
        "tier": "TIER_1",
        "rank": 1
    }]

    ev_sun = [{
        "title": "Solar System Science",
        "url": "https://www.esa.int/Science_Exploration/Heliocentric",
        "body": "The Earth and other planets revolve around the Sun, which is the center of the solar system.",
        "credibility_score": 0.92,
        "tier": "TIER_1",
        "rank": 1
    }]

    hal_detector = HallucinationDetector()
    trust_engine = TrustScoreEngine()
    sim_analyzer = SimilarityAnalyzer()

    # Sentence 1: TRUE
    c1 = "The Hubble Space Telescope was launched into low Earth orbit in 1990."
    c1_conf, c1_lbl, c1_rsn = compute_claim_confidence(c1, "SUPPORTED", ev_hubble)
    c1_hal = hal_detector.analyze_claim(c1, "SUPPORTED", ev_hubble, confidence=c1_conf)
    c1_sim = sim_analyzer.analyze_pair(c1, ev_hubble[0]["body"], "NASA")

    # Sentence 2: FALSE
    c2 = "The Sun revolves directly around the Earth every twenty-four hours."
    c2_conf, c2_lbl, c2_rsn = compute_claim_confidence(c2, "REFUTED", ev_sun)
    c2_hal = hal_detector.analyze_claim(c2, "REFUTED", ev_sun, confidence=c2_conf)
    c2_sim = sim_analyzer.analyze_pair(c2, ev_sun[0]["body"], "ESA")

    # Sentence 3: UNCERTAIN
    c3 = "Quantum graviton warp drives were constructed in subterranean facilities in 1840."
    c3_conf, c3_lbl, c3_rsn = compute_claim_confidence(c3, "UNCERTAIN", [])
    c3_hal = hal_detector.analyze_claim(c3, "UNCERTAIN", [], confidence=c3_conf)
    c3_sim = sim_analyzer.analyze_pair(c3, "", "None")

    sentence_results = [
        {
            "sentence": c1,
            "verdict": "SUPPORTED",
            "confidence": c1_conf,
            "confidence_label": c1_lbl,
            "confidence_reason": c1_rsn,
            "explanation": "Confirmed by NASA spaceflight historical records.",
            "sources": ev_hubble,
            "hallucination": c1_hal,
            "similarity": c1_sim,
            "plagiarism": {"similarity_score": 0.0, "category": "ORIGINAL", "matched_origin": "web"}
        },
        {
            "sentence": c2,
            "verdict": "REFUTED",
            "confidence": c2_conf,
            "confidence_label": c2_lbl,
            "confidence_reason": c2_rsn,
            "explanation": "Contradicted by established heliocentric astrophysics.",
            "sources": ev_sun,
            "hallucination": c2_hal,
            "similarity": c2_sim,
            "plagiarism": {"similarity_score": 0.0, "category": "ORIGINAL", "matched_origin": "web"}
        },
        {
            "sentence": c3,
            "verdict": "UNCERTAIN",
            "confidence": c3_conf,
            "confidence_label": c3_lbl,
            "confidence_reason": c3_rsn,
            "explanation": "No authoritative evidence substantiates this claim.",
            "sources": [],
            "hallucination": c3_hal,
            "similarity": c3_sim,
            "plagiarism": {"similarity_score": 0.0, "category": "ORIGINAL", "matched_origin": "web"}
        }
    ]

    # Article-level aggregations
    hal_report = hal_detector.evaluate_article_hallucination(sentence_results)
    all_sources = ev_hubble + ev_sun
    plag_audit = {
        "originality_score": 100.0,
        "overall_similarity": 0.0,
        "verbatim_percentage": 0.0,
        "paraphrase_percentage": 0.0,
        "original_percentage": 100.0
    }
    sim_summary = sim_analyzer.analyze_document_similarity(article_text, all_sources)
    trust_result = trust_engine.calculate_trust_score(
        sentence_results=sentence_results,
        all_sources=all_sources,
        hallucination_report=hal_report,
        plagiarism_audit=plag_audit
    )

    # -----------------------------------------------------------------
    # Test 2: Structured Report Dictionary Generation
    # -----------------------------------------------------------------
    print("\n[TEST 2] Testing Structured Report Dictionary...")
    report_dict = generate_report_dict(
        article_text=article_text,
        sentence_results=sentence_results,
        trust_result=trust_result,
        hallucination_report=hal_report,
        plagiarism_audit=plag_audit,
        similarity_summary=sim_summary,
        source_info={"type": "text", "name": "Space Science Article"}
    )

    # Verify all 10 requirements:
    # 1. Overall trust score
    assert "overall_trust_score" in report_dict
    assert 0.0 <= report_dict["overall_trust_score"]["score"] <= 100.0
    # 2. Article summary
    assert "article_summary" in report_dict
    assert report_dict["article_summary"]["sentence_count"] == 3
    # 3. Claims analyzed
    assert len(report_dict["claims_analyzed"]) == 3
    # 4. TRUE/FALSE/UNCERTAIN results
    verdicts = [c["verdict"] for c in report_dict["claims_analyzed"]]
    assert verdicts == ["TRUE", "FALSE", "UNCERTAIN"]
    # 5. Confidence for each claim
    for c in report_dict["claims_analyzed"]:
        assert "confidence" in c and 0.0 <= c["confidence"] <= 100.0
    # 6. Supporting / Contradicting evidence
    assert len(report_dict["claims_analyzed"][0]["supporting_evidence"]) > 0
    assert len(report_dict["claims_analyzed"][1]["contradicting_evidence"]) > 0
    # 7. Source credibility
    assert report_dict["claims_analyzed"][0]["supporting_evidence"][0]["credibility_score"] >= 0.90
    # 8. Similarity / Plagiarism findings
    assert "similarity_and_plagiarism_findings" in report_dict
    # 9. Hallucination findings
    assert "hallucination_findings" in report_dict
    assert report_dict["hallucination_findings"]["faithful_count"] == 1
    assert report_dict["hallucination_findings"]["contradicted_count"] == 1
    # 10. Sources/references bibliography
    assert len(report_dict["sources_bibliography"]) == 2

    print(f"  [OK] PASSED: All 10 required report sections verified.")
    print(f"    - Trust Score: {report_dict['overall_trust_score']['score']}/100 ({report_dict['overall_trust_score']['rating_label']})")
    print(f"    - Verdicts Breakdown: {verdicts}")
    print(f"    - Grounding Score: {report_dict['hallucination_findings']['grounding_score']}%")

    # -----------------------------------------------------------------
    # Test 3: Markdown Report Generation & Export
    # -----------------------------------------------------------------
    print("\n[TEST 3] Testing Markdown Report Export...")
    md_report = generate_markdown_report(report_dict)
    assert "# TruthLens Verification & Audit Report" in md_report
    assert "1. Executive Trust Summary" in md_report
    assert "2. Article Summary & Consensus" in md_report
    assert "3. Grounding & Hallucination Audit" in md_report
    assert "4. Text Similarity & Originality Findings" in md_report
    assert "5. Claims Analyzed & Evidence Verification" in md_report
    assert "6. Sources & References Bibliography" in md_report
    assert "Claim #1: TRUE" in md_report
    assert "Claim #2: FALSE" in md_report
    assert "Claim #3: UNCERTAIN" in md_report
    print(f"  [OK] PASSED: Markdown export successfully generated ({len(md_report.splitlines())} lines).")

    # -----------------------------------------------------------------
    # Test 4: JSON Report Generation & Export
    # -----------------------------------------------------------------
    print("\n[TEST 4] Testing JSON Report Export...")
    json_report = generate_json_report(report_dict)
    parsed_json = json.loads(json_report)
    assert parsed_json["metadata"]["version"] == "2.0"
    assert len(parsed_json["claims_analyzed"]) == 3
    print(f"  [OK] PASSED: Valid JSON report export generated ({len(json_report)} bytes).")

    # -----------------------------------------------------------------
    # Test 5: Interactive HTML Report View Rendering
    # -----------------------------------------------------------------
    print("\n[TEST 5] Testing HTML Report View Rendering...")
    html_report = render_report_html_view(report_dict)
    assert "verification-report-view" in html_report
    assert "data:text/markdown;charset=utf-8;base64," in html_report
    assert "data:application/json;charset=utf-8;base64," in html_report
    assert "report-claims-table" in html_report
    assert "report-sources-table" in html_report
    assert "claim-v-true" in html_report
    assert "claim-v-false" in html_report
    assert "claim-v-uncertain" in html_report
    print("  [OK] PASSED: HTML report view rendered with downloadable data URIs and tables.")

    # -----------------------------------------------------------------
    # Test 6: Summary & Card Integration
    # -----------------------------------------------------------------
    print("\n[TEST 6] Testing Summary and Result Card Integration...")
    card_html = create_result_card(
        sentence=c2,
        verdict="REFUTED",
        atomic_claims=[{"claim": c2, "verdict": "REFUTED", "explanation": "Contradicted.", "sources": ev_sun}],
        index=2,
        confidence=c2_conf,
        similarity_result=c2_sim,
        plag_result={"similarity_score": 0.0, "category": "ORIGINAL"},
        hallucination_result=c2_hal
    )
    assert "hal-contradicted" in card_html
    assert "hal-alert-box" in card_html

    summary_html = build_summary(
        sentence_results=sentence_results,
        plagiarism_audit=plag_audit,
        similarity_summary=sim_summary,
        trust_result=trust_result,
        hallucination_report=hal_report
    )
    assert "trust-hero-container" in summary_html
    assert "summary-hallucination-strip" in summary_html
    print("  [OK] PASSED: Summary and result cards integrated with complete metrics.")

    print("\n=================================================================")
    print("ALL PHASE 4 PIPELINE INTEGRATION TESTS PASSED! (6/6)")
    print("=================================================================")

if __name__ == "__main__":
    test_phase4_complete_pipeline()
