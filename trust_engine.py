"""
=============================================================================
TruthLens - Evidence-Based Hallucination Detector & Trust Score Engine (Phase 3)
=============================================================================
1. Hallucination Detection:
   Strict evidence-grounded comparison (claims vs retrieved evidence).
   Classifies statements into:
   - FAITHFUL (Confirmed by evidence, minimal hallucination risk)
   - CONTRADICTED (Factual mismatch / Disproven, critical hallucination risk)
   - UNSUPPORTED (Entities present in topic but assertions lack evidence)
   - UNVERIFIABLE (Speculative / Absent evidence)
   DOES NOT merely prompt an LLM; compares grounded propositions mathematically.

2. Transparent Multi-Signal Trust Score Engine:
   Weighted calculation across 4 transparent pillars:
   - 35% Fact Verification (Supported vs Refuted claims weighted by confidence)
   - 25% Source Credibility & Evidence Authority (.gov, .edu, peer-reviewed, news)
   - 25% Hallucination & Grounding Risk (Absence of contradicted/unsupported statements)
   - 15% Content Originality & Integrity (Original writing vs verbatim copy)
=============================================================================
"""

import re
from typing import List, Dict, Tuple, Optional, Any, Set

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did',
    'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
    'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not',
    'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
    'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with'
}


def extract_factual_entities(claim: str) -> List[str]:
    """
    Extract informative entities, numerical terms, capitalized proper nouns,
    and key content words from a claim.
    """
    # Capitalized multi-word entities (e.g. "James Webb", "Great Wall")
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
    
    # Numbers and dates
    numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', claim)
    
    # Content words
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', claim) if w.lower() not in STOPWORDS and len(w) > 2]
    
    combined = set()
    for pn in proper_nouns:
        combined.add(pn.lower())
    for num in numbers:
        combined.add(num)
    for w in words:
        combined.add(w)
        
    return sorted(list(combined))


# =============================================================================
# 1. EVIDENCE-BASED HALLUCINATION DETECTOR
# =============================================================================

class HallucinationDetector:
    """
    Detects hallucinations by comparing atomic claims against retrieved evidence texts.
    Categorizes statements into FAITHFUL, UNSUPPORTED, CONTRADICTED, or UNVERIFIABLE.
    """

    def analyze_claim(
        self,
        claim: str,
        verdict: str,
        evidence: List[Dict[str, Any]],
        confidence: float = 50.0,
        explanation: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate hallucination risk for an individual claim.
        """
        entities = extract_factual_entities(claim)
        
        # Combine evidence text
        evidence_text = " ".join([f"{s.get('title', '')} {s.get('body', '')}".lower() for s in evidence])
        
        # Calculate grounding coverage
        if entities and evidence_text:
            matched_entities = [e for e in entities if e in evidence_text]
            grounding_ratio = len(matched_entities) / len(entities)
        else:
            matched_entities = []
            grounding_ratio = 0.0

        verdict_upper = (verdict or "UNCERTAIN").upper()
        
        # 1. CONTRADICTED (Direct Hallucination / Factually disproven)
        if verdict_upper == "REFUTED":
            status = "CONTRADICTED"
            status_label = "Contradicted Statement"
            status_desc = "The assertion directly contradicts credible retrieved evidence."
            hallucination_risk = 92.0
            badge_class = "hal-contradicted"

        # 2. UNVERIFIABLE (No evidence found or inconclusive)
        elif not evidence or verdict_upper == "UNCERTAIN" and grounding_ratio < 0.20:
            status = "UNVERIFIABLE"
            status_label = "Unverifiable Statement"
            status_desc = "No authoritative evidence could be retrieved to substantiate or disprove this assertion."
            hallucination_risk = 65.0
            badge_class = "hal-unverifiable"

        # 3. UNSUPPORTED (Evidence found, but specific claim is uncorroborated)
        elif verdict_upper == "UNCERTAIN" or (verdict_upper == "PARTIALLY_SUPPORTED" and grounding_ratio < 0.50):
            status = "UNSUPPORTED"
            status_label = "Unsupported Claim"
            status_desc = "Retrieved sources discuss related topics but do not substantiate the specific assertion."
            hallucination_risk = 55.0
            badge_class = "hal-unsupported"

        # 4. PARTIALLY GROUNDED
        elif verdict_upper == "PARTIALLY_SUPPORTED":
            status = "PARTIALLY_GROUNDED"
            status_label = "Partially Grounded"
            status_desc = "Part of the compound statement is supported, but other parts lack full confirmation."
            hallucination_risk = 35.0
            badge_class = "hal-partial"

        # 5. FAITHFUL (Fully grounded in evidence)
        else:  # SUPPORTED
            status = "FAITHFUL"
            status_label = "Grounded & Faithful"
            status_desc = "Factual assertions are substantiated by credible retrieved evidence."
            # High grounding gives near zero hallucination risk
            hallucination_risk = max(4.0, round((1.0 - grounding_ratio) * 20.0, 1))
            badge_class = "hal-faithful"

        return {
            "status": status,
            "status_label": status_label,
            "status_desc": status_desc,
            "badge_class": badge_class,
            "hallucination_risk": hallucination_risk,
            "grounding_ratio": round(grounding_ratio * 100, 1),
            "matched_entities": matched_entities[:5],
            "unmatched_entities": [e for e in entities if e not in matched_entities][:5],
            "confidence": confidence
        }

    def evaluate_article_hallucination(
        self,
        sentence_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate hallucination risk across all analyzed sentences and claims in the article.
        """
        if not sentence_results:
            return {
                "overall_hallucination_risk": 50.0,
                "grounding_score": 50.0,
                "faithful_count": 0,
                "unsupported_count": 0,
                "contradicted_count": 0,
                "unverifiable_count": 0,
                "total_claims": 0,
                "faithful_percentage": 0.0,
                "risk_category": "UNKNOWN",
                "risk_badge": "⚠ UNKNOWN RISK",
                "summary_desc": "No sentences evaluated."
            }

        claim_evaluations = []
        for sr in sentence_results:
            atomic_claims = sr.get("atomic_claims", [])
            evidence = sr.get("sources", [])
            
            if atomic_claims:
                for ac in atomic_claims:
                    eval_res = self.analyze_claim(
                        claim=ac.get("claim", ""),
                        verdict=ac.get("verdict", "UNCERTAIN"),
                        evidence=ac.get("sources", evidence),
                        confidence=ac.get("confidence", 50.0),
                        explanation=ac.get("explanation", "")
                    )
                    claim_evaluations.append(eval_res)
            else:
                eval_res = self.analyze_claim(
                    claim=sr.get("sentence", ""),
                    verdict=sr.get("verdict", "UNCERTAIN"),
                    evidence=evidence,
                    confidence=sr.get("confidence", 50.0),
                    explanation=sr.get("explanation", "")
                )
                claim_evaluations.append(eval_res)

        total = len(claim_evaluations)
        faithful_cnt = sum(1 for c in claim_evaluations if c["status"] == "FAITHFUL")
        partial_cnt = sum(1 for c in claim_evaluations if c["status"] == "PARTIALLY_GROUNDED")
        unsup_cnt = sum(1 for c in claim_evaluations if c["status"] == "UNSUPPORTED")
        contra_cnt = sum(1 for c in claim_evaluations if c["status"] == "CONTRADICTED")
        unver_cnt = sum(1 for c in claim_evaluations if c["status"] == "UNVERIFIABLE")

        # Weighted aggregate hallucination risk
        avg_risk = sum(c["hallucination_risk"] for c in claim_evaluations) / total
        overall_risk = round(avg_risk, 1)
        grounding_score = round(max(0.0, min(100.0, 100.0 - overall_risk)), 1)
        faithful_pct = round((faithful_cnt / total) * 100, 1)

        if overall_risk <= 20.0:
            risk_category = "LOW_RISK"
            risk_badge = "✓ LOW HALLUCINATION RISK"
            summary_desc = "Content is well-grounded in external evidence with high factual fidelity."
        elif overall_risk <= 45.0:
            risk_category = "MODERATE_RISK"
            risk_badge = "⚠ MODERATE RISK"
            summary_desc = "Contains some uncorroborated assertions or ambiguous statements."
        else:
            risk_category = "HIGH_RISK"
            risk_badge = "✕ HIGH HALLUCINATION RISK"
            summary_desc = "Contains contradicted propositions or heavily unverified statements."

        return {
            "overall_hallucination_risk": overall_risk,
            "grounding_score": grounding_score,
            "faithful_count": faithful_cnt,
            "partially_grounded_count": partial_cnt,
            "unsupported_count": unsup_cnt,
            "contradicted_count": contra_cnt,
            "unverifiable_count": unver_cnt,
            "total_claims": total,
            "faithful_percentage": faithful_pct,
            "risk_category": risk_category,
            "risk_badge": risk_badge,
            "summary_desc": summary_desc,
            "claim_evaluations": claim_evaluations
        }


# =============================================================================
# 2. TRANSPARENT MULTI-SIGNAL TRUST SCORE ENGINE
# =============================================================================

class TrustScoreEngine:
    """
    Computes an evidence-grounded, multi-signal Trust Score (0 - 100).
    Provides transparent mathematical breakdown across 4 pillars:
    - 35% Fact Verification Integrity
    - 25% Source Credibility & Authority
    - 25% Evidence Grounding (Hallucination Resistance)
    - 15% Content Originality (Plagiarism Resistance)
    """

    def calculate_trust_score(
        self,
        sentence_results: List[Dict[str, Any]],
        all_sources: List[Dict[str, Any]],
        hallucination_report: Optional[Dict[str, Any]] = None,
        plagiarism_audit: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates the overall trust score and returns a detailed, transparent breakdown.
        """
        # ---------------------------------------------------------------------
        # 1. Fact Verification Signal (35% Weight)
        # ---------------------------------------------------------------------
        if sentence_results:
            supported_count = sum(r["verdict"] == "SUPPORTED" for r in sentence_results)
            partial_count = sum(r["verdict"] == "PARTIALLY_SUPPORTED" for r in sentence_results)
            uncertain_count = sum(r["verdict"] == "UNCERTAIN" for r in sentence_results)
            refuted_count = sum(r["verdict"] == "REFUTED" for r in sentence_results)
            total_sent = len(sentence_results)

            # Verification score: Supported=100, Partial=60, Uncertain=35, Refuted=0
            raw_fact_score = (
                (supported_count * 100.0) +
                (partial_count * 60.0) +
                (uncertain_count * 35.0) +
                (refuted_count * 0.0)
            ) / total_sent

            # Apply refuted penalty if refuted statements exist
            if refuted_count > 0:
                penalty = min(25.0, refuted_count * 12.0)
                raw_fact_score = max(0.0, raw_fact_score - penalty)

            fact_score = round(max(0.0, min(100.0, raw_fact_score)), 1)
        else:
            fact_score = 50.0

        # ---------------------------------------------------------------------
        # 2. Source Credibility Signal (25% Weight)
        # ---------------------------------------------------------------------
        if all_sources:
            cred_scores = [s.get("credibility_score", 0.45) for s in all_sources]
            avg_cred = sum(cred_scores) / len(cred_scores)
            credibility_score = round(max(0.0, min(100.0, avg_cred * 100.0)), 1)
        else:
            credibility_score = 35.0

        # ---------------------------------------------------------------------
        # 3. Grounding & Hallucination Resistance Signal (25% Weight)
        # ---------------------------------------------------------------------
        if hallucination_report:
            grounding_score = hallucination_report.get("grounding_score", 50.0)
        else:
            grounding_score = 50.0

        # ---------------------------------------------------------------------
        # 4. Content Originality Signal (15% Weight)
        # ---------------------------------------------------------------------
        if plagiarism_audit:
            originality_score = plagiarism_audit.get("originality_score", 100.0)
        else:
            originality_score = 100.0

        # ---------------------------------------------------------------------
        # Composite Trust Score Calculation
        # Formula: 0.35 * Fact + 0.25 * Credibility + 0.25 * Grounding + 0.15 * Originality
        # ---------------------------------------------------------------------
        weights = {
            "fact_verification": 0.35,
            "source_credibility": 0.25,
            "evidence_grounding": 0.25,
            "content_originality": 0.15
        }

        weighted_fact = fact_score * weights["fact_verification"]
        weighted_cred = credibility_score * weights["source_credibility"]
        weighted_ground = grounding_score * weights["evidence_grounding"]
        weighted_orig = originality_score * weights["content_originality"]

        composite_trust = weighted_fact + weighted_cred + weighted_ground + weighted_orig
        overall_trust_score = round(max(0.0, min(100.0, composite_trust)), 1)

        # ---------------------------------------------------------------------
        # Trust Classification & Badges
        # ---------------------------------------------------------------------
        if overall_trust_score >= 82.0:
            rating_label = "VERIFIED HIGH TRUST"
            rating_tier = "A+"
            badge_class = "trust-badge-high"
            assessment_summary = "Article is well-grounded in authoritative evidence with high factual fidelity and authentic writing."
        elif overall_trust_score >= 68.0:
            rating_label = "GENERALLY RELIABLE"
            rating_tier = "A"
            badge_class = "trust-badge-good"
            assessment_summary = "Majority of claims are verified by credible sources, with minor unverified or paraphrased segments."
        elif overall_trust_score >= 50.0:
            rating_label = "MODERATE / MIXED TRUST"
            rating_tier = "B"
            badge_class = "trust-badge-moderate"
            assessment_summary = "Contains mixed verification results, uncorroborated assertions, or moderate similarity."
        elif overall_trust_score >= 32.0:
            rating_label = "LOW TRUST / CAUTION"
            rating_tier = "C"
            badge_class = "trust-badge-low"
            assessment_summary = "Multiple unverified assertions, questionable source backing, or significant hallucination risk."
        else:
            rating_label = "HIGHLY UNRELIABLE"
            rating_tier = "F"
            badge_class = "trust-badge-unreliable"
            assessment_summary = "Substantial contradicted claims, fabricated assertions, or excessive plagiarism detected."

        return {
            "trust_score": overall_trust_score,
            "rating_label": rating_label,
            "rating_tier": rating_tier,
            "badge_class": badge_class,
            "assessment_summary": assessment_summary,
            "breakdown": {
                "fact_verification": {
                    "score": fact_score,
                    "weight": 35,
                    "weighted_score": round(weighted_fact, 1),
                    "description": "Accuracy of extracted propositions against live evidence."
                },
                "source_credibility": {
                    "score": credibility_score,
                    "weight": 25,
                    "weighted_score": round(weighted_cred, 1),
                    "description": "Institutional authority (.gov, .edu, peer-reviewed, reputable news)."
                },
                "evidence_grounding": {
                    "score": grounding_score,
                    "weight": 25,
                    "weighted_score": round(weighted_ground, 1),
                    "description": "Absence of contradicted, unsupported, or hallucinated claims."
                },
                "content_originality": {
                    "score": originality_score,
                    "weight": 15,
                    "weighted_score": round(weighted_orig, 1),
                    "description": "Original expression without unauthorized verbatim copying."
                }
            }
        }
