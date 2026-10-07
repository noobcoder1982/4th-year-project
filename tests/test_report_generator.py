import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
Unit tests for TruthLens Final Verification Report Generator (Phase 4)
Verifies:
1. Full report dictionary generation matching all 10 required fields
2. TRUE / FALSE / UNCERTAIN normalized mapping
3. Supporting and contradicting evidence segregation
4. Deduplicated sources bibliography
5. Exportable Markdown report generation
6. Exportable JSON report generation
7. Interactive Report View HTML rendering
"""

import json
from report_generator import (
    generate_report_dict,
    generate_markdown_report,
    generate_json_report,
    render_report_html_view,
    map_verdict_to_standard
)

def test_report_generation():
    print("=================================================================")
    print("TRUTHLENS REPORT GENERATOR TEST SUITE (PHASE 4)")
    print("=================================================================")

    # Test 1: Verdict Mapping
    print("\n[TEST 1] Testing Verdict Normalization...")
    assert map_verdict_to_standard("SUPPORTED") == "TRUE"
    assert map_verdict_to_standard("REFUTED") == "FALSE"
    assert map_verdict_to_standard("PARTIALLY_SUPPORTED") == "PARTIAL"
    assert map_verdict_to_standard("UNCERTAIN") == "UNCERTAIN"
    print("  [OK] PASSED: Verdicts properly mapped to standard TRUE/FALSE/PARTIAL/UNCERTAIN.")

    # Sample data
    article = "The James Webb Space Telescope operates at Sun-Earth L2. Humans have four lungs."
    ev_nasa = {
        "title": "NASA JWST Facts",
        "url": "https://nasa.gov/jwst",
        "domain": "nasa.gov",
        "body": "JWST orbits the second Lagrange point L2.",
        "credibility_score": 0.95,
        "tier": "TIER_1",
        "rank": 1
    }
    ev_anatomy = {
        "title": "Medical Anatomy Overview",
        "url": "https://nih.gov/anatomy",
        "domain": "nih.gov",
        "body": "The human body contains two lungs in the chest.",
        "credibility_score": 0.96,
        "tier": "TIER_1",
        "rank": 1
    }

    sentence_results = [
        {
            "sentence": "The James Webb Space Telescope operates at Sun-Earth L2.",
            "verdict": "SUPPORTED",
            "confidence": 98.0,
            "confidence_label": "High Confidence",
            "confidence_reason": "Direct confirmation by official NASA mission facts.",
            "explanation": "Verified by NASA documentation.",
            "sources": [ev_nasa],
            "hallucination": {
                "status": "FAITHFUL",
                "status_label": "Grounded & Faithful",
                "hallucination_risk": 4.0,
                "grounding_ratio": 95.0,
                "status_desc": "Substantiated by NASA facts."
            },
            "similarity": {
                "similarity_score": 25.0,
                "relationship": "TOPICAL_OVERLAP",
                "relationship_label": "Topical Overlap",
                "is_plagiarism": False
            },
            "plagiarism": {
                "similarity_score": 0.0,
                "category": "ORIGINAL",
                "matched_origin": "web"
            }
        },
        {
            "sentence": "Humans have four lungs.",
            "verdict": "REFUTED",
            "confidence": 94.0,
            "confidence_label": "High Confidence",
            "confidence_reason": "Direct refutation by medical authorities.",
            "explanation": "Contradicted by established human anatomy.",
            "sources": [ev_anatomy],
            "hallucination": {
                "status": "CONTRADICTED",
                "status_label": "Contradicted Statement",
                "hallucination_risk": 92.0,
                "grounding_ratio": 50.0,
                "status_desc": "Directly contradicts anatomical literature."
            },
            "similarity": {
                "similarity_score": 10.0,
                "relationship": "INDEPENDENT",
                "relationship_label": "Independent Writing",
                "is_plagiarism": False
            },
            "plagiarism": {
                "similarity_score": 0.0,
                "category": "ORIGINAL",
                "matched_origin": "web"
            }
        }
    ]

    trust_res = {
        "trust_score": 62.5,
        "rating_tier": "B",
        "rating_label": "MODERATE / MIXED TRUST",
        "assessment_summary": "Article contains one verified statement and one contradicted anatomical falsehood.",
        "breakdown": {
            "fact_verification": {"score": 38.0, "weight": 35, "weighted_score": 13.3},
            "source_credibility": {"score": 95.5, "weight": 25, "weighted_score": 23.9},
            "evidence_grounding": {"score": 52.0, "weight": 25, "weighted_score": 13.0},
            "content_originality": {"score": 100.0, "weight": 15, "weighted_score": 15.0}
        }
    }

    hal_rep = {
        "overall_hallucination_risk": 48.0,
        "grounding_score": 52.0,
        "faithful_count": 1,
        "contradicted_count": 1,
        "unsupported_count": 0,
        "unverifiable_count": 0,
        "risk_badge": "⚠ MODERATE RISK",
        "summary_desc": "Contains disproven propositions."
    }

    plag_audit = {
        "originality_score": 100.0,
        "overall_similarity": 0.0,
        "verbatim_percentage": 0.0,
        "paraphrase_percentage": 0.0,
        "original_percentage": 100.0
    }

    # Test 2: Structured Report Dictionary Generation
    print("\n[TEST 2] Testing Structured Report Dictionary...")
    report_dict = generate_report_dict(
        article_text=article,
        sentence_results=sentence_results,
        trust_result=trust_res,
        hallucination_report=hal_rep,
        plagiarism_audit=plag_audit,
        source_info={"type": "text", "name": "Direct Input"}
    )

    # Check 10 user requirements
    assert "overall_trust_score" in report_dict, "Req 1: Overall trust score"
    assert "article_summary" in report_dict, "Req 2: Article summary"
    assert "claims_analyzed" in report_dict, "Req 3: Claims analyzed"
    assert report_dict["claims_analyzed"][0]["verdict"] == "TRUE", "Req 4: TRUE/FALSE/UNCERTAIN"
    assert report_dict["claims_analyzed"][1]["verdict"] == "FALSE", "Req 4: TRUE/FALSE/UNCERTAIN"
    assert report_dict["claims_analyzed"][0]["confidence"] == 98.0, "Req 5: Confidence for each claim"
    assert len(report_dict["claims_analyzed"][0]["supporting_evidence"]) > 0, "Req 6: Supporting evidence"
    assert len(report_dict["claims_analyzed"][1]["contradicting_evidence"]) > 0, "Req 6: Contradicting evidence"
    assert report_dict["claims_analyzed"][0]["all_evidence"][0]["credibility_score"] == 0.95, "Req 7: Source credibility"
    assert "similarity_and_plagiarism_findings" in report_dict, "Req 8: Similarity/plagiarism findings"
    assert "hallucination_findings" in report_dict, "Req 9: Hallucination findings"
    assert len(report_dict["sources_bibliography"]) == 2, "Req 10: Deduplicated sources/references"

    print(f"  [OK] PASSED: Report dictionary contains all 10 required sections.")
    print(f"    - Trust Score: {report_dict['overall_trust_score']['score']} (Grade {report_dict['overall_trust_score']['grade']})")
    print(f"    - Claims: {len(report_dict['claims_analyzed'])} verified (1 TRUE, 1 FALSE)")
    print(f"    - References: {len(report_dict['sources_bibliography'])} deduplicated sources")

    # Test 3: Markdown Report Export
    print("\n[TEST 3] Testing Markdown Report Export...")
    md_content = generate_markdown_report(report_dict)
    assert "# TruthLens Verification & Audit Report" in md_content
    assert "Executive Trust Summary" in md_content
    assert "Score Breakdown Pillars" in md_content
    assert "Claim #1: TRUE" in md_content
    assert "Claim #2: FALSE" in md_content
    assert "Sources & References Bibliography" in md_content
    print(f"  [OK] PASSED: Markdown report generated ({len(md_content.splitlines())} lines).")

    # Test 4: JSON Report Export
    print("\n[TEST 4] Testing JSON Report Export...")
    json_str = generate_json_report(report_dict)
    parsed = json.loads(json_str)
    assert parsed["overall_trust_score"]["score"] == 62.5
    print(f"  [OK] PASSED: Valid JSON report generated ({len(json_str)} bytes).")

    # Test 5: Interactive HTML Report View
    print("\n[TEST 5] Testing HTML Report View Rendering...")
    html_view = render_report_html_view(report_dict)
    assert "verification-report-view" in html_view
    assert "data:text/markdown;charset=utf-8;base64," in html_view, "Must contain Markdown download URI"
    assert "data:application/json;charset=utf-8;base64," in html_view, "Must contain JSON download URI"
    assert "report-claims-table" in html_view
    assert "report-sources-table" in html_view
    print(f"  [OK] PASSED: HTML report view rendered with downloadable URIs and tables.")

    print("\n=================================================================")
    print("ALL PHASE 4 REPORT GENERATOR TESTS PASSED! (5/5)")
    print("=================================================================")

if __name__ == "__main__":
    test_report_generation()
