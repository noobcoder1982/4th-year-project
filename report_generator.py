"""
=============================================================================
TruthLens - Final Verification Report Generator (Phase 4)
=============================================================================
Generates comprehensive verification reports containing:
1. Overall Trust Score & transparent 4-pillar breakdown
2. Article Summary (word count, sentence count, consensus, metadata)
3. Claims Analyzed with normalized TRUE / FALSE / UNCERTAIN verdicts
4. Evidence-Grounded Confidence for each claim
5. Supporting and Contradicting Evidence segregation
6. Source Credibility metrics and authoritative tiers
7. Similarity vs. Plagiarism findings (distinct contextual alignment)
8. Hallucination findings (faithful, contradicted, unsupported, unverifiable)
9. Deduplicated Sources and References bibliography
10. Export formats: Downloadable JSON, Markdown, and rich interactive HTML
=============================================================================
"""

import json
import base64
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional

def map_verdict_to_standard(raw_verdict: str) -> str:
    """
    Map internal verdicts (SUPPORTED, REFUTED, PARTIALLY_SUPPORTED, UNCERTAIN)
    to standard user-facing verdicts (TRUE, FALSE, PARTIAL, UNCERTAIN).
    """
    v = (raw_verdict or "UNCERTAIN").upper().strip()
    if v == "SUPPORTED":
        return "TRUE"
    elif v == "REFUTED":
        return "FALSE"
    elif v == "PARTIALLY_SUPPORTED":
        return "PARTIAL"
    elif v == "UNCERTAIN":
        return "UNCERTAIN"
    return v


def generate_report_dict(
    article_text: str,
    sentence_results: List[Dict[str, Any]],
    trust_result: Optional[Dict[str, Any]] = None,
    hallucination_report: Optional[Dict[str, Any]] = None,
    plagiarism_audit: Optional[Dict[str, Any]] = None,
    similarity_summary: Optional[Dict[str, Any]] = None,
    source_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compile all verification, credibility, similarity, plagiarism, and hallucination
    findings into a complete, structured verification report dictionary.
    """
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Metadata
    s_info = source_info or {}
    metadata = {
        "report_id": f"TL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": now_iso,
        "engine": "TruthLens Fact Verification & Originality Suite",
        "version": "2.0",
        "source_type": s_info.get("type", "text"),
        "source_name": s_info.get("name", "Direct Input"),
        "source_title": s_info.get("title", "")
    }

    # 2. Article Summary
    words = article_text.split() if article_text else []
    total_words = len(words)
    total_sentences = len(sentence_results)

    supported_count = sum(1 for r in sentence_results if r.get("verdict") == "SUPPORTED")
    refuted_count = sum(1 for r in sentence_results if r.get("verdict") == "REFUTED")
    partial_count = sum(1 for r in sentence_results if r.get("verdict") == "PARTIALLY_SUPPORTED")
    uncertain_count = sum(1 for r in sentence_results if r.get("verdict") == "UNCERTAIN")

    if refuted_count >= 1:
        if refuted_count >= supported_count:
            consensus = "LIKELY MISLEADING"
        else:
            consensus = "MIXED / CONFLICTING"
    elif partial_count > 0:
        consensus = "PARTIALLY ACCURATE"
    elif uncertain_count > total_sentences / 2:
        consensus = "INSUFFICIENT EVIDENCE"
    else:
        consensus = "HIGH FACTUAL INTEGRITY"

    article_summary = {
        "word_count": total_words,
        "sentence_count": total_sentences,
        "verification_consensus": consensus,
        "verdict_distribution": {
            "true": supported_count,
            "false": refuted_count,
            "partial": partial_count,
            "uncertain": uncertain_count
        }
    }

    # 3. Overall Trust Score
    t_res = trust_result or {}
    overall_trust = {
        "score": t_res.get("trust_score", 50.0),
        "grade": t_res.get("rating_tier", "B"),
        "rating_label": t_res.get("rating_label", "MODERATE TRUST"),
        "assessment_summary": t_res.get("assessment_summary", ""),
        "signal_breakdown": t_res.get("breakdown", {})
    }

    # 4. Claims Analyzed
    claims_analyzed = []
    sources_catalog = {}

    for idx, sr in enumerate(sentence_results, 1):
        raw_v = sr.get("verdict", "UNCERTAIN")
        std_v = map_verdict_to_standard(raw_v)
        sent_text = sr.get("sentence", "")
        conf_val = sr.get("confidence", 50.0)
        
        # Check atomic claims
        atomic_list = sr.get("atomic_claims", [])
        atomic_claims_formatted = []
        for at_idx, at in enumerate(atomic_list, 1):
            at_raw_v = at.get("verdict", "UNCERTAIN")
            atomic_claims_formatted.append({
                "sub_id": f"{idx}.{at_idx}",
                "claim": at.get("claim", ""),
                "verdict": map_verdict_to_standard(at_raw_v),
                "confidence": at.get("confidence", 50.0),
                "confidence_label": at.get("confidence_label", "Moderate"),
                "explanation": at.get("explanation", "")
            })

        # Sources & Evidence segregation
        evidence_items = sr.get("sources", [])
        supporting_evidence = []
        contradicting_evidence = []
        evidence_formatted = []

        for ev in evidence_items:
            ev_url = ev.get("url", "")
            ev_title = ev.get("title", "")
            ev_body = ev.get("body", "")
            ev_domain = ev.get("domain", "")
            if not ev_domain and ev_url:
                try:
                    import urllib.parse
                    ev_domain = urllib.parse.urlparse(ev_url).netloc
                except Exception:
                    ev_domain = "web"
            ev_cred = ev.get("credibility_score", 0.5)
            ev_tier = ev.get("tier", "TIER_3")
            ev_rank = ev.get("rank", 1)

            ev_entry = {
                "rank": ev_rank,
                "title": ev_title,
                "url": ev_url,
                "domain": ev_domain,
                "snippet": ev_body,
                "credibility_score": ev_cred,
                "credibility_tier": ev_tier
            }
            evidence_formatted.append(ev_entry)

            # Categorize into supporting vs contradicting
            if raw_v == "SUPPORTED":
                supporting_evidence.append(ev_entry)
            elif raw_v == "REFUTED":
                contradicting_evidence.append(ev_entry)
            else:
                supporting_evidence.append(ev_entry)

            # Add to bibliography catalog
            if ev_url and ev_url not in sources_catalog:
                sources_catalog[ev_url] = {
                    "url": ev_url,
                    "title": ev_title,
                    "domain": ev_domain,
                    "credibility_score": ev_cred,
                    "credibility_tier": ev_tier,
                    "referenced_in_claims": [idx]
                }
            elif ev_url and idx not in sources_catalog[ev_url]["referenced_in_claims"]:
                sources_catalog[ev_url]["referenced_in_claims"].append(idx)

        # Hallucination info
        hal_info = sr.get("hallucination") or {}
        hal_status = hal_info.get("status", "FAITHFUL")
        hal_risk = hal_info.get("hallucination_risk", 0.0)
        ground_ratio = hal_info.get("grounding_ratio", 0.0)

        # Similarity & Plagiarism info
        sim_info = sr.get("similarity") or {}
        plag_info = sr.get("plagiarism") or {}

        claims_analyzed.append({
            "claim_id": idx,
            "sentence": sent_text,
            "verdict": std_v,
            "verdict_raw": raw_v,
            "confidence": conf_val,
            "confidence_label": sr.get("confidence_label", "Moderate Confidence"),
            "confidence_reason": sr.get("confidence_reason", ""),
            "explanation": sr.get("explanation", ""),
            "atomic_claims": atomic_claims_formatted,
            "hallucination": {
                "status": hal_status,
                "label": hal_info.get("status_label", hal_status),
                "risk_score": hal_risk,
                "grounding_ratio": ground_ratio,
                "description": hal_info.get("status_desc", "")
            },
            "similarity": {
                "score": sim_info.get("similarity_score", 0.0),
                "topical_overlap": sim_info.get("topical_similarity", 0.0),
                "lexical_overlap": sim_info.get("lexical_overlap", 0.0),
                "relationship": sim_info.get("relationship", "INDEPENDENT"),
                "relationship_label": sim_info.get("relationship_label", "Independent Writing"),
                "is_plagiarism": sim_info.get("is_plagiarism", False)
            },
            "plagiarism": {
                "score": plag_info.get("similarity_score", 0.0),
                "category": plag_info.get("category", "ORIGINAL"),
                "matched_origin": plag_info.get("matched_origin", "web")
            },
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "all_evidence": evidence_formatted,
            "context_meta": sr.get("context_meta", {})
        })

    # 5. Hallucination Findings Summary
    h_rep = hallucination_report or {}
    hallucination_findings = {
        "overall_hallucination_risk": h_rep.get("overall_hallucination_risk", 0.0),
        "grounding_score": h_rep.get("grounding_score", 50.0),
        "risk_category": h_rep.get("risk_category", "LOW_RISK"),
        "risk_badge": h_rep.get("risk_badge", "LOW RISK"),
        "faithful_count": h_rep.get("faithful_count", 0),
        "partially_grounded_count": h_rep.get("partially_grounded_count", 0),
        "unsupported_count": h_rep.get("unsupported_count", 0),
        "contradicted_count": h_rep.get("contradicted_count", 0),
        "unverifiable_count": h_rep.get("unverifiable_count", 0),
        "summary": h_rep.get("summary_desc", "")
    }

    # 6. Similarity & Plagiarism Findings Summary
    p_aud = plagiarism_audit or {}
    s_sum = similarity_summary or {}
    similarity_plagiarism_findings = {
        "originality_score": p_aud.get("originality_score", 100.0),
        "overall_plagiarism_similarity": p_aud.get("overall_similarity", 0.0),
        "verbatim_percentage": p_aud.get("verbatim_percentage", 0.0),
        "paraphrase_percentage": p_aud.get("paraphrase_percentage", 0.0),
        "original_percentage": p_aud.get("original_percentage", 100.0),
        "topical_overlap_score": s_sum.get("topical_overlap_score", 0.0),
        "lexical_overlap_score": s_sum.get("lexical_overlap_score", 0.0),
        "shared_topical_keywords": s_sum.get("shared_topical_keywords", []),
        "topical_consensus_level": s_sum.get("topical_consensus_level", "Moderate")
    }

    # 7. Deduplicated Bibliography of Sources
    sources_bibliography = list(sources_catalog.values())
    sources_bibliography.sort(key=lambda s: s["credibility_score"], reverse=True)

    return {
        "metadata": metadata,
        "overall_trust_score": overall_trust,
        "article_summary": article_summary,
        "claims_analyzed": claims_analyzed,
        "hallucination_findings": hallucination_findings,
        "similarity_and_plagiarism_findings": similarity_plagiarism_findings,
        "sources_bibliography": sources_bibliography
    }


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """
    Format complete verification report into a publication-grade Markdown document.
    """
    meta = report_data.get("metadata", {})
    trust = report_data.get("overall_trust_score", {})
    summary = report_data.get("article_summary", {})
    claims = report_data.get("claims_analyzed", [])
    hal = report_data.get("hallucination_findings", {})
    sim_plag = report_data.get("similarity_and_plagiarism_findings", {})
    sources = report_data.get("sources_bibliography", [])
    breakdown = trust.get("signal_breakdown", {})

    lines = []
    lines.append(f"# TruthLens Verification & Audit Report")
    lines.append(f"**Report ID:** `{meta.get('report_id')}` | **Date:** {meta.get('generated_at')}")
    lines.append(f"**Source Origin:** {meta.get('source_type', '').upper()} ({meta.get('source_name')})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Executive Trust Score
    lines.append("## 1. Executive Trust Summary")
    lines.append(f"**Overall Trust Score:** **{trust.get('score', 0)} / 100** (Grade: **{trust.get('grade')}** &bull; **{trust.get('rating_label')}**)")
    lines.append(f"> {trust.get('assessment_summary')}")
    lines.append("")
    lines.append("### Score Breakdown Pillars")
    lines.append("| Signal Pillar | Score | Weight | Contribution | Metric Details |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    
    for key, name in [
        ("fact_verification", "Fact Verification"),
        ("source_credibility", "Source Credibility"),
        ("evidence_grounding", "Evidence Grounding"),
        ("content_originality", "Content Originality")
    ]:
        b = breakdown.get(key, {})
        lines.append(f"| **{name}** | {b.get('score', 0)}% | {b.get('weight', 0)}% | {b.get('weighted_score', 0)} pts | {b.get('desc', '')} |")
    lines.append("")

    # 2. Article Statistics & Consensus
    lines.append("## 2. Article Summary & Consensus")
    v_dist = summary.get("verdict_distribution", {})
    lines.append(f"- **Total Sentences Evaluated:** {summary.get('sentence_count')}")
    lines.append(f"- **Total Words:** {summary.get('word_count')}")
    lines.append(f"- **Overall Factual Consensus:** **{summary.get('verification_consensus')}**")
    lines.append(f"- **Verdict Counts:** TRUE: **{v_dist.get('true', 0)}** | FALSE: **{v_dist.get('false', 0)}** | PARTIAL: **{v_dist.get('partial', 0)}** | UNCERTAIN: **{v_dist.get('uncertain', 0)}**")
    lines.append("")

    # 3. Grounding & Hallucination Audit
    lines.append("## 3. Grounding & Hallucination Audit")
    lines.append(f"- **Overall Hallucination Risk:** **{hal.get('overall_hallucination_risk')}%** ({hal.get('risk_badge')})")
    lines.append(f"- **Evidence Grounding Score:** **{hal.get('grounding_score')}%**")
    lines.append(f"- **Faithful Claims:** {hal.get('faithful_count')} | **Contradicted Claims:** {hal.get('contradicted_count')} | **Unsupported:** {hal.get('unsupported_count')} | **Unverifiable:** {hal.get('unverifiable_count')}")
    lines.append(f"> {hal.get('summary')}")
    lines.append("")

    # 4. Text Similarity vs. Plagiarism Findings
    lines.append("## 4. Text Similarity & Originality Findings")
    lines.append(f"- **Content Originality Score:** **{sim_plag.get('originality_score')}%** (Unique Content)")
    lines.append(f"- **Verbatim Copy:** {sim_plag.get('verbatim_percentage')}% | **Paraphrased:** {sim_plag.get('paraphrase_percentage')}% | **Original Writing:** {sim_plag.get('original_percentage')}%")
    lines.append(f"- **Topical Text Overlap:** {sim_plag.get('topical_overlap_score')}% (Lexical Overlap: {sim_plag.get('lexical_overlap_score')}%)")
    kws = ", ".join(sim_plag.get("shared_topical_keywords", [])[:8])
    if kws:
        lines.append(f"- **Key Shared Topical Terminology:** `{kws}`")
    lines.append("")

    # 5. Claim-by-Claim Verification
    lines.append("## 5. Claims Analyzed & Evidence Verification")
    for cl in claims:
        lines.append(f"### Claim #{cl['claim_id']}: {cl['verdict']}")
        lines.append(f"> “{cl['sentence']}”")
        lines.append("")
        lines.append(f"- **Verdict:** **{cl['verdict']}** (Confidence: **{cl['confidence']}%** - *{cl['confidence_label']}*)")
        if cl.get("confidence_reason"):
            lines.append(f"- **Confidence Basis:** {cl['confidence_reason']}")
        lines.append(f"- **Grounding Status:** **{cl['hallucination']['status']}** (Hallucination Risk: {cl['hallucination']['risk_score']}%, Grounding Coverage: {cl['hallucination']['grounding_ratio']}%)")
        lines.append(f"- **Similarity / Originality:** {cl['similarity']['relationship_label']} ({cl['similarity']['score']}%) &bull; Plagiarism: {cl['plagiarism']['category']}")
        if cl.get("explanation"):
            lines.append(f"- **AI Fact Analysis:** {cl['explanation']}")
        lines.append("")

        # Supporting & Contradicting Evidence
        sup_ev = cl.get("supporting_evidence", [])
        contra_ev = cl.get("contradicting_evidence", [])

        if cl['verdict'] == "FALSE" and contra_ev:
            lines.append("#### Contradicting Web Evidence:")
            for ev in contra_ev[:3]:
                lines.append(f"- **[{ev['domain']}]** {ev['title']} (Credibility: {ev['credibility_score']})")
                lines.append(f"  > “{ev['snippet'][:250]}...”")
                if ev.get('url'):
                    lines.append(f"  Source: {ev['url']}")
            lines.append("")
        elif sup_ev:
            lines.append("#### Retrieved Supporting Evidence:")
            for ev in sup_ev[:3]:
                lines.append(f"- **[{ev['domain']}]** {ev['title']} (Credibility: {ev['credibility_score']})")
                lines.append(f"  > “{ev['snippet'][:250]}...”")
                if ev.get('url'):
                    lines.append(f"  Source: {ev['url']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # 6. Deduplicated References
    lines.append("## 6. Sources & References Bibliography")
    if sources:
        lines.append("| Domain | Source Title & URL | Credibility Score | Tier | Cited In Claims |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for s in sources:
            cited_str = ", ".join([f"#{c}" for c in s.get("referenced_in_claims", [])])
            lines.append(f"| **{s.get('domain')}** | [{s.get('title')}]({s.get('url')}) | **{s.get('credibility_score')}** | `{s.get('credibility_tier')}` | {cited_str} |")
    else:
        lines.append("No external references cited.")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated automatically by TruthLens AI Article Verification Suite.*")
    return "\n".join(lines)


def generate_json_report(report_data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format complete verification report into a JSON string.
    """
    return json.dumps(report_data, indent=indent, ensure_ascii=False)


def render_report_html_view(report_data: Dict[str, Any]) -> str:
    """
    Render a stunning, interactive, print-ready Report View in HTML
    with downloadable JSON and Markdown data URIs.
    """
    meta = report_data.get("metadata", {})
    trust = report_data.get("overall_trust_score", {})
    summary = report_data.get("article_summary", {})
    claims = report_data.get("claims_analyzed", [])
    hal = report_data.get("hallucination_findings", {})
    sim_plag = report_data.get("similarity_and_plagiarism_findings", {})
    sources = report_data.get("sources_bibliography", [])
    breakdown = trust.get("signal_breakdown", {})

    # Generate data URIs for immediate download
    md_text = generate_markdown_report(report_data)
    json_text = generate_json_report(report_data)
    
    md_b64 = base64.b64encode(md_text.encode("utf-8")).decode("utf-8")
    json_b64 = base64.b64encode(json_text.encode("utf-8")).decode("utf-8")

    md_download_href = f"data:text/markdown;charset=utf-8;base64,{md_b64}"
    json_download_href = f"data:application/json;charset=utf-8;base64,{json_b64}"

    v_dist = summary.get("verdict_distribution", {})

    # Claims table rows
    claims_rows_html = ""
    for c in claims:
        v_class = "claim-v-true" if c["verdict"] == "TRUE" else "claim-v-false" if c["verdict"] == "FALSE" else "claim-v-partial" if c["verdict"] == "PARTIAL" else "claim-v-uncertain"
        h_class = "hal-badge-faithful" if c["hallucination"]["status"] == "FAITHFUL" else "hal-badge-contra" if c["hallucination"]["status"] == "CONTRADICTED" else "hal-badge-unsup"
        
        ctx = c.get("context_meta", {})
        memory_html = ""
        if ctx and ctx.get("has_anaphora") and ctx.get("resolved_antecedent"):
            memory_html = f'<div class="r-claim-memory-tag" style="display:inline-flex; align-items:center; gap:5px; margin-top:6px; font-size:12px; font-weight:600; padding:3px 10px; border-radius:6px; background:rgba(99, 102, 241, 0.12); color:#4338ca; border:1px solid rgba(99, 102, 241, 0.25);"><span>🧠</span> <span>Discourse Link: <em>"{ctx.get("trigger")}"</em> &rarr; <strong>{ctx.get("resolved_antecedent")}</strong></span></div>'

        claims_rows_html += f"""
        <tr class="report-claim-row">
            <td class="r-col-id">#{c['claim_id']}</td>
            <td class="r-col-text">
                <div class="r-claim-sentence">“{c['sentence']}”</div>
                {memory_html}
                {f'<div class="r-claim-explanation">AI Analysis: {c["explanation"]}</div>' if c.get("explanation") else ''}
            </td>
            <td class="r-col-verdict">
                <span class="r-v-badge {v_class}">{c['verdict']}</span>
                <div class="r-conf-sub">{round(c['confidence'])}% Conf.</div>
            </td>
            <td class="r-col-hal">
                <span class="r-hal-badge {h_class}">{c['hallucination']['label']}</span>
                <div class="r-risk-sub">{c['hallucination']['risk_score']}% Risk</div>
            </td>
            <td class="r-col-sim">
                <div class="r-sim-label">{c['similarity']['relationship_label']}</div>
                <div class="r-sim-sub">{c['similarity']['score']}% match</div>
            </td>
        </tr>
        """

    # Sources table rows
    sources_rows_html = ""
    for s in sources[:8]:
        cited_str = ", ".join([f"#{c}" for c in s.get("referenced_in_claims", [])])
        sources_rows_html += f"""
        <tr>
            <td><strong>{s.get('domain')}</strong></td>
            <td><a href="{s.get('url')}" target="_blank" class="r-src-link">{s.get('title')} ↗</a></td>
            <td><span class="r-cred-score">{s.get('credibility_score')}</span></td>
            <td><span class="r-tier-pill">{s.get('credibility_tier')}</span></td>
            <td><span class="r-cited-pill">{cited_str}</span></td>
        </tr>
        """

    return f"""
<div class="verification-report-view">
    <!-- Report Action & Export Header -->
    <div class="report-action-header">
        <div class="report-header-left">
            <div class="report-badge-pill">COMPREHENSIVE VERIFICATION DOSSIER</div>
            <div class="report-id-text">ID: {meta.get('report_id')} &bull; {meta.get('generated_at')}</div>
        </div>
        <div class="report-export-buttons">
            <a href="{md_download_href}" download="TruthLens_Report_{meta.get('report_id')}.md" class="export-btn export-md-btn">
                ⬇ Export Markdown (.md)
            </a>
            <a href="{json_download_href}" download="TruthLens_Report_{meta.get('report_id')}.json" class="export-btn export-json-btn">
                ⬇ Export JSON (.json)
            </a>
        </div>
    </div>

    <!-- Executive Overview Card -->
    <div class="report-executive-card">
        <div class="report-exec-left">
            <div class="report-score-circle {trust.get('signal_breakdown', {}).get('badge_class', 'trust-badge-good')}">
                <div class="r-score-val">{trust.get('score')}</div>
                <div class="r-score-label">TRUST SCORE</div>
            </div>
            <div class="report-exec-info">
                <div class="r-grade-row">
                    <span class="r-grade-badge">GRADE {trust.get('grade')}</span>
                    <span class="r-consensus-badge">{summary.get('verification_consensus')}</span>
                </div>
                <div class="r-exec-desc">{trust.get('assessment_summary')}</div>
                <div class="r-stat-pills-row">
                    <span class="r-stat-pill"><strong>{summary.get('sentence_count')}</strong> Sentences</span>
                    <span class="r-stat-pill"><strong>{summary.get('word_count')}</strong> Words</span>
                    <span class="r-stat-pill true-pill"><strong>{v_dist.get('true', 0)}</strong> True</span>
                    <span class="r-stat-pill false-pill"><strong>{v_dist.get('false', 0)}</strong> False</span>
                    <span class="r-stat-pill partial-pill"><strong>{v_dist.get('partial', 0)}</strong> Partial</span>
                    <span class="r-stat-pill uncertain-pill"><strong>{v_dist.get('uncertain', 0)}</strong> Uncertain</span>
                </div>
            </div>
        </div>
    </div>

    <!-- 4 Pillars Breakdown -->
    <div class="report-section-title">TRANSPARENT TRUST BREAKDOWN</div>
    <div class="report-pillars-grid">
        <div class="report-pillar-card">
            <div class="r-pillar-head">
                <span>Fact Verification</span>
                <strong>{breakdown.get('fact_verification', {}).get('score', 0)}%</strong>
            </div>
            <div class="r-pillar-bar"><div class="r-pillar-fill fill-fact" style="width:{breakdown.get('fact_verification', {}).get('score', 0)}%;"></div></div>
            <div class="r-pillar-sub">Weight 35% &bull; {breakdown.get('fact_verification', {}).get('weighted_score', 0)} contribution pts</div>
        </div>

        <div class="report-pillar-card">
            <div class="r-pillar-head">
                <span>Source Credibility</span>
                <strong>{breakdown.get('source_credibility', {}).get('score', 0)}%</strong>
            </div>
            <div class="r-pillar-bar"><div class="r-pillar-fill fill-cred" style="width:{breakdown.get('source_credibility', {}).get('score', 0)}%;"></div></div>
            <div class="r-pillar-sub">Weight 25% &bull; {breakdown.get('source_credibility', {}).get('weighted_score', 0)} contribution pts</div>
        </div>

        <div class="report-pillar-card">
            <div class="r-pillar-head">
                <span>Evidence Grounding</span>
                <strong>{breakdown.get('evidence_grounding', {}).get('score', 0)}%</strong>
            </div>
            <div class="r-pillar-bar"><div class="r-pillar-fill fill-ground" style="width:{breakdown.get('evidence_grounding', {}).get('score', 0)}%;"></div></div>
            <div class="r-pillar-sub">Weight 25% &bull; {breakdown.get('evidence_grounding', {}).get('weighted_score', 0)} contribution pts</div>
        </div>

        <div class="report-pillar-card">
            <div class="r-pillar-head">
                <span>Content Originality</span>
                <strong>{breakdown.get('content_originality', {}).get('score', 0)}%</strong>
            </div>
            <div class="r-pillar-bar"><div class="r-pillar-fill fill-orig" style="width:{breakdown.get('content_originality', {}).get('score', 0)}%;"></div></div>
            <div class="r-pillar-sub">Weight 15% &bull; {breakdown.get('content_originality', {}).get('weighted_score', 0)} contribution pts</div>
        </div>
    </div>

    <!-- Claims Analysis Table -->
    <div class="report-section-title">CLAIM-BY-CLAIM VERIFICATION AUDIT</div>
    <div class="report-table-container">
        <table class="report-claims-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Claim Statement &amp; Analysis</th>
                    <th>Verdict &amp; Conf.</th>
                    <th>Hallucination</th>
                    <th>Similarity</th>
                </tr>
            </thead>
            <tbody>
                {claims_rows_html}
            </tbody>
        </table>
    </div>

    <!-- Deduplicated Source Bibliography -->
    <div class="report-section-title">RETRIEVED SOURCES &amp; CREDIBILITY BIBLIOGRAPHY</div>
    <div class="report-table-container">
        <table class="report-sources-table">
            <thead>
                <tr>
                    <th>Domain</th>
                    <th>Source Document Title</th>
                    <th>Credibility</th>
                    <th>Tier</th>
                    <th>Claims Cited</th>
                </tr>
            </thead>
            <tbody>
                {sources_rows_html}
            </tbody>
        </table>
    </div>
</div>
"""
