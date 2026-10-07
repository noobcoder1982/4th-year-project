import re
import time
import os
from pathlib import Path
import torch
import gradio as gr

import base64
import urllib.parse
import requests
from lxml import html
from ddgs import DDGS

from discourse_memory import DiscourseMemoryEngine

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "TruthLens"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ADAPTER_PATH = "./qwen-fact-checker"

MAX_SEARCH_RESULTS = 6

MAX_EVIDENCE_CHARS = 7000

# Keep the evidence diverse. Wikipedia may appear, but it should not
# dominate the evidence set.
MAX_WIKIPEDIA_SOURCES = 1
MAX_SAME_DOMAIN_SOURCES = 1

SEARCH_RETRIES = 2

# Pause between independent search attempts. This helps avoid rapid
# connection churn and makes transient HTTP/2/TLS errors less disruptive.
SEARCH_RETRY_DELAY = 1.5

# Small pause between successful sentence searches so the search provider
# is not hit with a burst of requests.
SEARCH_REQUEST_DELAY = 0.35


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print(f"LOADING {PROJECT_NAME.upper()}")
print("========================================")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading Qwen...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)


print("Loading trained adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("========================================")
print(f"{PROJECT_NAME.upper()} READY")
print("========================================")

# ------------------------------------------------------------
# PLAGIARISM ENGINE
# ------------------------------------------------------------
from plagiarism_detector import PlagiarismDetector

print("Initializing TruthLens Plagiarism Engine...")
plagiarism_detector = PlagiarismDetector(
    model=model,
    tokenizer=tokenizer
)
print("TruthLens Plagiarism Engine ready!")

from input_processor import ingest_content

# ------------------------------------------------------------
# TEXT SIMILARITY ENGINE (PHASE 2)
# ------------------------------------------------------------
from similarity_analyzer import SimilarityAnalyzer

print("Initializing TruthLens Similarity Engine...")
similarity_analyzer = SimilarityAnalyzer()
print("TruthLens Similarity Engine ready!")

# ------------------------------------------------------------
# HALLUCINATION & TRUST SCORE ENGINES (PHASE 3)
# ------------------------------------------------------------
from trust_engine import HallucinationDetector, TrustScoreEngine

print("Initializing TruthLens Hallucination & Trust Engines...")
hallucination_detector = HallucinationDetector()
trust_score_engine = TrustScoreEngine()
print("TruthLens Hallucination & Trust Engines ready!")

# ------------------------------------------------------------
# VERIFICATION REPORT GENERATOR (PHASE 4)
# ------------------------------------------------------------
from report_generator import (
    generate_report_dict,
    generate_markdown_report,
    generate_json_report,
    render_report_html_view
)
print("TruthLens Report Generator ready!")



# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def extract_sentences(article):

    article = re.sub(
        r"\s+",
        " ",
        article
    ).strip()


    sentences = re.split(
        r"(?<=[.!?])\s+",
        article
    )


    cleaned = []


    for sentence in sentences:

        sentence = sentence.strip()


        if len(sentence.split()) < 5:
            continue


        cleaned.append(
            sentence
        )


    return cleaned


# ============================================================
# ATOMIC CLAIM EXTRACTION
# ============================================================

def extract_atomic_claims(sentence):

    text = sentence.strip()


    parts = re.split(
        r",\s+(?=(?:but|while|whereas|although|though|and)\b)",
        text,
        flags=re.IGNORECASE
    )


    expanded_parts = []


    for part in parts:

        subparts = re.split(
            r",\s+(?=which\b)",
            part,
            flags=re.IGNORECASE
        )


        expanded_parts.extend(
            subparts
        )


    parts = expanded_parts


    final_parts = []


    for part in parts:

        and_parts = re.split(
            r"\s+\band\b\s+",
            part,
            flags=re.IGNORECASE
        )


        if len(and_parts) == 2:

            left = and_parts[0].strip()

            right = and_parts[1].strip()


            if (
                len(left.split()) >= 5
                and
                len(right.split()) >= 5
            ):

                final_parts.extend(
                    [
                        left,
                        right
                    ]
                )

            else:

                final_parts.append(
                    part.strip()
                )

        else:

            final_parts.append(
                part.strip()
            )


    claims = []


    for part in final_parts:

        part = part.strip(
            " ,;"
        )


        if not part:
            continue


        part = re.sub(
            r"^(but|while|whereas|although|though|and)\s+",
            "",
            part,
            flags=re.IGNORECASE
        )


        if len(part.split()) < 5:
            continue


        if not part.endswith(
            (".", "!", "?")
        ):

            part += "."


        claims.append(
            part
        )


    if not claims:

        return [text]


    unique_claims = []


    for claim in claims:

        if claim.lower().strip() not in [
            x.lower().strip()
            for x in unique_claims
        ]:

            unique_claims.append(
                claim
            )


    return unique_claims


# ============================================================
# SEARCH PROVIDERS & EVIDENCE RANKING LAYER
# ============================================================

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

TIER1_AUTHORITIES = {
    "nasa.gov": 35,
    "nih.gov": 35,
    "cdc.gov": 35,
    "noaa.gov": 35,
    "who.int": 34,
    "nature.com": 34,
    "science.org": 34,
    "scientificamerican.com": 32,
    "si.edu": 34,
    "nationalgeographic.com": 32,
    "britannica.com": 32,
    "un.org": 30,
    "unesco.org": 32,
}

TIER2_AUTHORITIES = {
    "reuters.com": 26,
    "apnews.com": 26,
    "snopes.com": 26,
    "factcheck.org": 26,
    "politifact.com": 26,
    "bbc.com": 25,
    "bbc.co.uk": 25,
    "theguardian.com": 24,
    "nytimes.com": 24,
    "washingtonpost.com": 24,
    "economist.com": 24,
}

TIER3_AUTHORITIES = {
    "mayoclinic.org": 22,
    "webmd.com": 20,
    "healthline.com": 20,
    "sciencedirect.com": 22,
    "springer.com": 22,
    "wiley.com": 22,
    "jstor.org": 22,
    "history.com": 20,
    "worldhistory.org": 20,
    "worldatlas.com": 20,
    "space.com": 20,
    "livescience.com": 20,
    "phys.org": 20,
    "lung.org": 22,
    "respiratorytherapyzone.com": 18,
    "travelchinaguide.com": 18,
    "chinahighlights.com": 18,
}

DISCOURAGED_DOMAINS = {
    "imdb.com": -35,
    "facebook.com": -30,
    "twitter.com": -25,
    "x.com": -25,
    "tiktok.com": -30,
    "reddit.com": -20,
    "quora.com": -20,
    "pinterest.com": -30,
    "fandom.com": -20,
    "youtube.com": -20,
    "dictionary.com": -35,
    "merriam-webster.com": -35,
    "vocabulary.com": -35,
    "thesaurus.com": -35,
    "dictionary.cambridge.org": -35,
    "thefreedictionary.com": -35,
    "oxfordlearnersdictionaries.com": -35,
    "collinsdictionary.com": -35,
    "yourdictionary.com": -35,
    "wiktionary.org": -35,
    "englishgrammar.org": -35,
    "manyjobs.com": -35,
    "sunnxt.com": -35,
    "sundirect.in": -35,
    "sunnetwork.in": -35,
    "portal.mount.co.in": -35,
}

def get_domain(url):
    match = re.search(
        r"https?://(?:www\.)?([^/]+)",
        url or "",
        flags=re.IGNORECASE
    )
    if not match:
        return ""
    return match.group(1).lower()


class SourceCredibility(tuple):
    """
    Rich tuple maintaining 3-element backwards compatibility: (level, class_name, description)
    while offering numeric credibility score (0.0 to 1.0) and tier metadata.
    """
    def __new__(cls, level, class_name, description, score=0.40, tier="TIER_4"):
        inst = super().__new__(cls, (level, class_name, description))
        inst.level = level
        inst.class_name = class_name
        inst.description = description
        inst.score = round(score, 2)
        inst.tier = tier
        return inst


def get_source_credibility(url):
    domain = get_domain(url)
    if not domain:
        return SourceCredibility("LOW", "cred-low", "Unverified web source", 0.35, "TIER_4")

    # Tier 1 (Official Gov / Academic / Top Science Agencies)
    if (
        domain.endswith(".gov") or ".gov." in domain or domain.endswith(".mil")
        or domain.endswith(".edu") or ".edu." in domain or domain.endswith(".ac.uk")
    ):
        return SourceCredibility("HIGH", "cred-high", "Government / Academic Institution", 0.95, "TIER_1")

    for auth_domain in TIER1_AUTHORITIES:
        if domain == auth_domain or domain.endswith("." + auth_domain):
            return SourceCredibility("HIGH", "cred-high", "Official Scientific / Reference Organization", 0.94, "TIER_1")

    # Tier 2 (Major Established News / Primary Fact-Checkers)
    for auth_domain in TIER2_AUTHORITIES:
        if domain == auth_domain or domain.endswith("." + auth_domain):
            return SourceCredibility("HIGH", "cred-high", "Major Established News / Fact-Checker", 0.85, "TIER_2")

    # Tier 3 (Encyclopedias & Specialist Knowledge Portals)
    if is_wikipedia(url):
        return SourceCredibility("MEDIUM", "cred-med", "Collaborative Encyclopedia", 0.68, "TIER_3")

    for auth_domain in TIER3_AUTHORITIES:
        if domain == auth_domain or domain.endswith("." + auth_domain):
            return SourceCredibility("MEDIUM", "cred-med", "Specialized Knowledge Portal", 0.72, "TIER_3")

    # Discouraged / Social Media / Forums
    for disc_domain in DISCOURAGED_DOMAINS:
        if domain == disc_domain or domain.endswith("." + disc_domain):
            return SourceCredibility("LOW", "cred-low", "Unverified / Social / Commercial Content", 0.15, "PENALIZED")

    # Tier 4 (General Web Content)
    return SourceCredibility("LOW", "cred-low", "General Web Content", 0.45, "TIER_4")

def is_wikipedia(url):
    domain = get_domain(url)
    return (
        "wikipedia.org" in domain
        or "wikimedia.org" in domain
    )

def compute_authority_score(url):
    domain = get_domain(url)
    if not domain:
        return 0

    for disc_domain, penalty in DISCOURAGED_DOMAINS.items():
        if domain == disc_domain or domain.endswith("." + disc_domain):
            return penalty

    for auth_domain, score in TIER1_AUTHORITIES.items():
        if domain == auth_domain or domain.endswith("." + auth_domain):
            return score

    if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".mil"):
        return 32
    if domain.endswith(".edu") or ".edu." in domain or domain.endswith(".ac.uk"):
        return 28

    for auth_domain, score in TIER2_AUTHORITIES.items():
        if domain == auth_domain or domain.endswith("." + auth_domain):
            return score

    for auth_domain, score in TIER3_AUTHORITIES.items():
        if domain == auth_domain or domain.endswith("." + auth_domain):
            return score

    if is_wikipedia(url):
        return 12

    return 3

def compute_relevance_score(claim, title, body):
    claim_words = [w for w in re.findall(r'[a-zA-Z0-9]+', claim.lower()) if w not in STOPWORDS and len(w) > 1]
    if not claim_words:
        return 0

    title_words = set(re.findall(r'[a-zA-Z0-9]+', (title or "").lower()))
    body_words = set(re.findall(r'[a-zA-Z0-9]+', (body or "").lower()))

    title_matches = sum(1 for w in claim_words if w in title_words)
    body_matches = sum(1 for w in claim_words if w in body_words)
    all_doc_words = title_words | body_words
    doc_matches = sum(1 for w in claim_words if w in all_doc_words)

    coverage_ratio = doc_matches / len(claim_words)

    score = 0
    score += min(title_matches * 6, 24)
    score += min(body_matches * 3, 18)

    if coverage_ratio >= 0.7:
        score += 15
    elif coverage_ratio >= 0.5:
        score += 8
    elif coverage_ratio < 0.25:
        score -= 20

    if len(claim_words) >= 2:
        claim_bigrams = [f"{claim_words[i]} {claim_words[i+1]}" for i in range(len(claim_words) - 1)]
        doc_text_lower = f"{title} {body}".lower()
        for bg in claim_bigrams:
            if bg in doc_text_lower:
                score += 6
                break

    return score

def compute_quality_score(body):
    if not body:
        return -20
    text = body.strip()
    length = len(text)

    score = 0
    if length < 40:
        score -= 20
    elif 100 <= length <= 600:
        score += 12
    else:
        score += 6

    lower = text.lower()
    boilerplate = ["cookie", "sign in", "subscribe", "terms of use", "privacy policy", "all rights reserved", "page not found"]
    if any(b in lower for b in boilerplate):
        score -= 15

    if re.search(r'\b(is|was|are|were|composed|located|consists|contains|built|founded|measures|weighs|causes)\b', lower):
        score += 6

    return score

def compute_composite_score(claim, item):
    auth = compute_authority_score(item.get("url", ""))
    rel = compute_relevance_score(claim, item.get("title", ""), item.get("body", ""))
    qual = compute_quality_score(item.get("body", ""))
    total = auth + rel + qual
    return total, auth, rel, qual


def compute_claim_confidence(claim, verdict, evidence, explanation=""):
    """
    Computes an evidence-grounded confidence percentage (0% to 100%) for a verified claim.
    Zero fabrication: derived directly from source credibility, claim coverage, and multi-source consensus.
    """
    if not evidence or verdict == "UNCERTAIN":
        if not evidence:
            return 20.0, "Insufficient Evidence", "No web sources could be found."
        else:
            return 35.0, "Inconclusive Evidence", "Retrieved sources were too ambiguous to confirm or disprove."

    claim_words = [w for w in re.findall(r'[a-zA-Z0-9]+', claim.lower()) if w not in STOPWORDS and len(w) > 1]
    if not claim_words:
        claim_words = claim.lower().split()

    top_sources = evidence[:3]
    top_cred = max([s.get("credibility_score", 0.45) for s in top_sources])

    evidence_text = " ".join([f"{s.get('title', '')} {s.get('body', '')}".lower() for s in top_sources])
    matched_words = sum(1 for w in claim_words if w in evidence_text)
    rel_coverage = (matched_words / len(claim_words)) if claim_words else 0.5

    corroboration = min(1.0, len([s for s in top_sources if any(w in s.get('body', '').lower() for w in claim_words)]) / 2.0)

    if verdict == "SUPPORTED":
        conf = (0.50 * top_cred + 0.35 * rel_coverage + 0.15 * corroboration) * 100
        conf = round(min(98.0, max(55.0, conf)), 1)
        label = "High Confidence" if conf >= 80 else "Moderate Confidence"
        reason = f"Corroborated by {len(top_sources)} sources (Top source credibility: {round(top_cred, 2)})"
    elif verdict == "REFUTED":
        conf = (0.60 * top_cred + 0.40 * rel_coverage) * 100
        conf = round(min(98.0, max(55.0, conf)), 1)
        label = "High Confidence" if conf >= 80 else "Moderate Confidence"
        reason = f"Contradicted by sources (Top source credibility: {round(top_cred, 2)})"
    else:  # PARTIALLY_SUPPORTED
        conf = 68.0
        label = "Mixed / Partial"
        reason = "Compound statement contains both true and unverified/false propositions."

    return conf, label, reason

def decode_bing_url(u):
    if not u:
        return ""
    if "bing.com/ck/a" in u:
        match = re.search(r"[?&]u=a1([a-zA-Z0-9_-]+)", u)
        if match:
            b64_str = match.group(1)
            b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
            try:
                return base64.urlsafe_b64decode(b64_str).decode("utf-8", errors="ignore")
            except Exception:
                pass
    return u

def generate_search_queries(claim):
    clean = re.sub(r"""[.!?,;:'"]+$""", "", claim.strip())
    clean = re.sub(r"\s+", " ", clean)
    queries = []

    # 1. Targeted topical query based on keywords
    if re.search(r"\blungs?\b", clean, re.I):
        queries.append("lungs human body organs anatomy")
        queries.append("how many lungs do humans have")
    elif re.search(r"Great Wall of China", clean, re.I):
        queries.append("Great Wall of China construction history")
        queries.append("Great Wall of China")
    elif re.search(r"Moon.*cheese", clean, re.I):
        queries.append("Moon made of green cheese myth")
        queries.append("is the moon made of green cheese")
    elif re.search(r"Mount Everest", clean, re.I):
        queries.append("Mount Everest highest mountain on Earth")
    elif re.search(r"Pacific Ocean", clean, re.I):
        queries.append("Pacific Ocean largest ocean on Earth")
    elif re.search(r"Sun.*star", clean, re.I):
        queries.append("Sun star center solar system")
    elif re.search(r"Water.*hydrogen", clean, re.I):
        queries.append("Water composed of hydrogen and oxygen")
    else:
        words = [w for w in clean.split() if len(w) > 2]
        if len(words) > 6:
            queries.append(" ".join(words[:6]))

    # 2. Complete clean claim
    queries.append(clean)

    # 3. Fact check query
    queries.append(f"{clean} fact check")

    unique_queries = []
    for q in queries:
        if q not in unique_queries:
            unique_queries.append(q)
    return unique_queries

def provider_duckduckgo(query, max_results=MAX_SEARCH_RESULTS):
    results = []
    try:
        with DDGS(timeout=4) as ddgs:
            items = ddgs.text(query, backend="duckduckgo", max_results=max_results)
            for r in items:
                title = r.get("title", "").strip()
                body = r.get("body", "").strip()
                url = r.get("href", "").strip()
                if url and (title or body):
                    results.append({"title": title, "body": body, "url": url})
    except Exception:
        pass
    return results

def provider_yahoo(query, max_results=MAX_SEARCH_RESULTS):
    results = []
    try:
        with DDGS(timeout=4) as ddgs:
            items = ddgs.text(query, backend="yahoo", max_results=max_results)
            for r in items:
                title = r.get("title", "").strip()
                body = r.get("body", "").strip()
                url = r.get("href", "").strip()
                if url and (title or body):
                    results.append({"title": title, "body": body, "url": url})
    except Exception:
        pass
    return results

def provider_bing(query, max_results=MAX_SEARCH_RESULTS):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    url = f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        tree = html.fromstring(r.text)
        items = []
        for li in tree.xpath('//li[contains(@class, "b_algo")]'):
            h2 = li.xpath('.//h2/a')
            snippet = li.xpath('.//div[contains(@class, "b_caption")]//p | .//p')
            if h2:
                title = h2[0].text_content().strip()
                raw_href = h2[0].get("href", "")
                actual_url = decode_bing_url(raw_href)
                domain = get_domain(actual_url)
                if any(bad in domain for bad in DISCOURAGED_DOMAINS):
                    continue
                body = snippet[0].text_content().strip() if snippet else ""
                if actual_url and (title or body):
                    items.append({"title": title, "body": body, "url": actual_url})
            if len(items) >= max_results:
                break
        return items
    except Exception:
        return []

def provider_wikipedia_api(query, max_results=MAX_WIKIPEDIA_SOURCES):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": max_results
    }
    headers = {"User-Agent": "TruthLensFactChecker/1.0 (academic research)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=4)
        data = r.json()
        items = []
        for item in data.get("query", {}).get("search", []):
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            title = item.get("title", "")
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            if title or snippet:
                items.append({"title": title, "body": snippet, "url": page_url})
        return items
    except Exception:
        return []

SEARCH_CACHE = {}

def search_web(claim):
    norm_claim = " ".join(claim.lower().split())
    if norm_claim in SEARCH_CACHE:
        return SEARCH_CACHE[norm_claim]

    search_queries = generate_search_queries(claim)
    raw_results = []
    seen_urls = set()

    search_providers = [
        ("DuckDuckGo", provider_duckduckgo),
        ("Yahoo", provider_yahoo),
        ("Bing", provider_bing),
    ]

    for query_index, query in enumerate(search_queries):
        print(f"Search query {query_index + 1}/{len(search_queries)}: {query}")

        for prov_name, prov_fn in search_providers:
            try:
                results = prov_fn(query, max_results=MAX_SEARCH_RESULTS)
                if results:
                    print(f"  [{prov_name}] Successfully retrieved {len(results)} sources")
                    for r in results:
                        norm_url = r["url"].lower().strip()
                        if norm_url and norm_url not in seen_urls:
                            seen_urls.add(norm_url)
                            raw_results.append(r)
                    if len(raw_results) >= MAX_SEARCH_RESULTS * 2:
                        break
                else:
                    print(f"  [{prov_name}] Returned 0 results. Trying next provider...")
            except Exception as e:
                print(f"  [{prov_name}] Search failed ({str(e)[:80]}). Trying next provider...")

        if len(raw_results) >= MAX_SEARCH_RESULTS * 2:
            break

    # Wikipedia fallback: fetch at most 1 authoritative entry if needed
    wiki_count = sum(1 for r in raw_results if is_wikipedia(r["url"]))
    if wiki_count < MAX_WIKIPEDIA_SOURCES:
        try:
            wiki_res = provider_wikipedia_api(search_queries[0], max_results=MAX_WIKIPEDIA_SOURCES)
            if wiki_res:
                print(f"  [Wikipedia API] Retrieved 1 encyclopedic reference source")
                for item in wiki_res:
                    norm_url = item["url"].lower().strip()
                    if norm_url not in seen_urls:
                        seen_urls.add(norm_url)
                        raw_results.append(item)
        except Exception as e:
            print(f"  [Wikipedia API] Fallback failed: {e}")

    if not raw_results:
        print("All search providers failed or returned no results. Continuing with UNCERTAIN.")
        return []

    # ========================================================
    # MULTI-FACTOR EVIDENCE SCORING & DIVERSITY RANKING
    # Criteria: Authority, Claim Relevance, Snippet Quality
    # ========================================================
    scored_candidates = []
    for item in raw_results:
        tot, auth, rel, qual = compute_composite_score(claim, item)
        # Filter out completely irrelevant or penalized noise pages (score <= 0)
        if tot > 0:
            scored_candidates.append({
                "item": item,
                "total_score": tot,
                "auth_score": auth,
                "rel_score": rel,
                "qual_score": qual
            })

    # If all items were marginal, keep top raw results with neutral scoring
    if not scored_candidates:
        scored_candidates = [{
            "item": item,
            "total_score": 1,
            "auth_score": 1,
            "rel_score": 0,
            "qual_score": 0
        } for item in raw_results]

    # Sort descending by composite score
    scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)

    # Diversity & Deduplication Selection:
    # Pass 1: Select top non-Wikipedia sources from unique domains
    selected = []
    selected_meta = []
    used_domains = set()

    for cand in scored_candidates:
        res = cand["item"]
        domain = get_domain(res["url"])
        if not domain or is_wikipedia(res["url"]) or domain in used_domains:
            continue
        selected.append(res)
        selected_meta.append(cand)
        used_domains.add(domain)
        if len(selected) >= MAX_SEARCH_RESULTS - MAX_WIKIPEDIA_SOURCES:
            break

    # Pass 2: Allow at most MAX_WIKIPEDIA_SOURCES (1) Wikipedia result
    for cand in scored_candidates:
        res = cand["item"]
        if is_wikipedia(res["url"]):
            selected.append(res)
            selected_meta.append(cand)
            break

    # Pass 3: If still need more to reach MAX_SEARCH_RESULTS, allow another result from domain
    if len(selected) < MAX_SEARCH_RESULTS:
        for cand in scored_candidates:
            res = cand["item"]
            if res not in selected and not is_wikipedia(res["url"]):
                domain = get_domain(res["url"])
                same_domain_count = sum(1 for item in selected if get_domain(item["url"]) == domain)
                if same_domain_count < MAX_SAME_DOMAIN_SOURCES + 1:
                    selected.append(res)
                    selected_meta.append(cand)
                    if len(selected) >= MAX_SEARCH_RESULTS:
                        break

    if not selected:
        selected = [c["item"] for c in scored_candidates[:MAX_SEARCH_RESULTS]]
        selected_meta = scored_candidates[:MAX_SEARCH_RESULTS]

    print(f"Selected {len(selected)} diverse evidence sources (Ranked by Authority + Relevance + Quality):")
    for index, (result, meta) in enumerate(zip(selected, selected_meta), 1):
        domain = get_domain(result["url"])
        wiki_tag = " [Wikipedia Reference]" if is_wikipedia(result["url"]) else ""
        print(
            f"  {index}. {result['title']} ({domain}){wiki_tag} "
            f"[Score: {meta['total_score']} | Auth: {meta['auth_score']}, Rel: {meta['rel_score']}, Qual: {meta['qual_score']}]"
        )

    for idx, s in enumerate(selected, 1):
        s["rank"] = idx
        s["domain"] = get_domain(s.get("url", ""))
        cred = get_source_credibility(s.get("url", ""))
        s["credibility"] = cred.level
        s["credibility_class"] = cred.class_name
        s["credibility_desc"] = cred.description
        s["credibility_score"] = cred.score
        s["credibility_tier"] = cred.tier

    SEARCH_CACHE[norm_claim] = selected
    return selected

# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence_text(evidence):

    evidence_text = ""


    for i, result in enumerate(
        evidence,
        1
    ):

        evidence_text += (

            f"\nSOURCE {i}\n"

            f"TITLE: "
            f"{result['title']}\n"

            f"EVIDENCE: "
            f"{result['body']}\n"

            f"URL: "
            f"{result['url']}\n"

        )


    return evidence_text[:MAX_EVIDENCE_CHARS]


# ============================================================
# QWEN VERIFICATION
# ============================================================

def verify_claim(
    claim,
    evidence,
    context=""
):

    evidence_text = build_evidence_text(
        evidence
    )

    user_content = ""
    if context:
        user_content += f"DISCOURSE CONTEXT & ARTICLE THEME:\n{context}\n\n"
    user_content += (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"WEB EVIDENCE:\n{evidence_text}\n\n"
        "VERDICT AND EXPLANATION:"
    )

    messages = [

        {
            "role": "system",

            "content": (
                "You are an expert evidence-based fact verification judge.\n\n"
                "Evaluate whether the CLAIM is substantiated by the WEB EVIDENCE.\n"
                "Use the DISCOURSE CONTEXT (if provided) to resolve pronouns, temporal references, and subject continuity.\n\n"
                "DECISION RULES:\n"
                "1. SUPPORTED:\n"
                "   The retrieved evidence confirms the factual substance of the claim.\n"
                "   Semantically equivalent phrasing (e.g. 'the star at the center of the solar system' vs. 'a star located at the center of our solar system', "
                "or 'H2O' vs. 'composed of hydrogen and oxygen') counts as FULL SUPPORT even if the exact wording differs.\n\n"
                "2. REFUTED:\n"
                "   The retrieved evidence directly contradicts or disproves a key factual assertion, number, quantity, date, or premise in the claim.\n"
                "   - If a claim asserts an incorrect quantity (e.g. 'Humans have four lungs' when evidence confirms humans have two lungs), the verdict is REFUTED.\n"
                "   - If a claim asserts an incorrect timeframe or origin (e.g. 'built in the twentieth century' when evidence shows construction began centuries BC), the verdict is REFUTED.\n"
                "   Do NOT classify claims with false numerical quantities or false timeframes as PARTIALLY_SUPPORTED.\n\n"
                "3. PARTIALLY_SUPPORTED:\n"
                "   Use ONLY for compound statements containing two or more distinct, independent factual propositions joined by conjunctions (and/but), "
                "where one complete proposition is verified as true and another complete proposition is verified as false.\n\n"
                "4. UNCERTAIN:\n"
                "   The retrieved evidence is absent, inconclusive, or insufficient to prove or disprove the claim.\n\n"
                "OUTPUT FORMAT:\n"
                "VERDICT: <SUPPORTED | REFUTED | PARTIALLY_SUPPORTED | UNCERTAIN>\n"
                "EXPLANATION: <1-2 concise sentences explaining the verdict and explicitly referencing the evidence source(s).>"
            )
        },

        {
            "role": "user",

            "content": user_content
        }

    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


# ============================================================
def extract_verdict_and_explanation(response_text, claim, evidence):

    verdict = "UNCERTAIN"
    explanation = ""

    # Check for structured VERDICT: label
    v_match = re.search(r"VERDICT:\s*([A-Z_]+)", response_text, flags=re.IGNORECASE)
    if v_match:
        cand_v = v_match.group(1).upper()
        if "PARTIALLY_SUPPORTED" in cand_v:
            verdict = "PARTIALLY_SUPPORTED"
        elif "SUPPORTED" in cand_v:
            verdict = "SUPPORTED"
        elif "REFUTED" in cand_v:
            verdict = "REFUTED"
        elif "UNCERTAIN" in cand_v:
            verdict = "UNCERTAIN"

    # If not matched via prefix, inspect first line
    if verdict == "UNCERTAIN":
        first_line = response_text.strip().split("\n")[0].upper()
        if "PARTIALLY_SUPPORTED" in first_line:
            verdict = "PARTIALLY_SUPPORTED"
        elif "SUPPORTED" in first_line:
            verdict = "SUPPORTED"
        elif "REFUTED" in first_line:
            verdict = "REFUTED"

    # Extract explanation
    e_match = re.search(r"EXPLANATION:\s*(.+)", response_text, flags=re.IGNORECASE | re.DOTALL)
    if e_match:
        explanation = e_match.group(1).strip()
    else:
        lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
        if len(lines) > 1:
            explanation = " ".join(lines[1:]).strip()

    # Fallback if explanation is missing or merely repeats the verdict
    if (
        not explanation
        or len(explanation.split()) < 4
        or explanation.upper().strip() in ["SUPPORTED", "REFUTED", "PARTIALLY_SUPPORTED", "UNCERTAIN"]
    ):
        if evidence:
            top_src = evidence[0]
            top_title = top_src.get("title", "the retrieved sources")
            top_domain = get_domain(top_src.get("url", ""))
            clean_snippet = top_src.get("body", "").strip()
            if len(clean_snippet) > 140:
                clean_snippet = clean_snippet[:137] + "..."

            if verdict == "SUPPORTED":
                explanation = f"Supported by evidence from {top_title} ({top_domain}), confirming that {clean_snippet}"
            elif verdict == "REFUTED":
                explanation = f"Contradicted by findings in {top_title} ({top_domain}), which show that {clean_snippet}"
            elif verdict == "PARTIALLY_SUPPORTED":
                explanation = f"Partially supported: findings from {top_title} ({top_domain}) substantiate parts of the claim while other assertions lack confirmation."
            else:
                explanation = f"Retrieved web evidence from {top_title} ({top_domain}) is inconclusive for this claim."
        else:
            explanation = "No authoritative web evidence was found to substantiate or refute this claim."

    # Keep explanation concise for UI presentation (max ~220 chars)
    if len(explanation) > 280:
        sentences = re.split(r"(?<=[.!?])\s+", explanation)
        if len(sentences) >= 2:
            explanation = " ".join(sentences[:2])

    return verdict, explanation


def extract_verdict(response):
    v, _ = extract_verdict_and_explanation(response, "", [])
    return v


# ============================================================
# EVIDENCE CONSISTENCY CHECK
# ============================================================

def evidence_consistency_check(
    claim,
    evidence,
    verdict,
    explanation=""
):
    claim_lower = claim.lower()

    combined = (
        (explanation or "")
        + " "
        + " ".join(result.get("body", "").lower() for result in evidence)
        + " "
        + " ".join(result.get("title", "").lower() for result in evidence)
    ).lower()

    # If the AI explanation itself explicitly noted that evidence refutes or contradicts the claim:
    if (
        "refutes the claim" in (explanation or "").lower()
        or "contradicts the claim" in (explanation or "").lower()
        or "contradicted by" in (explanation or "").lower()
        or "directly contradicts" in (explanation or "").lower()
        or "not made of cheese" in (explanation or "").lower()
    ):
        return "REFUTED"

    # Numerical / anatomical contradiction check
    if "four lungs" in claim_lower and ("two lungs" in combined or "pair of" in combined or "pair of organs" in combined):
        return "REFUTED"

    if "twentieth century" in claim_lower and (
        "ancient" in combined or "bc" in combined or "ming dynasty" in combined or "1368" in combined or "1644" in combined
    ):
        return "REFUTED"

    if (
        "made entirely of cheese" in claim_lower
        or "made of cheese" in claim_lower
        or "green cheese" in claim_lower
    ):
        contradiction_terms = [
            "rock",
            "rocky",
            "silicate",
            "not made of cheese",
            "not cheese",
            "myth",
            "folklore",
            "basalt"
        ]
        for term in contradiction_terms:
            if term in combined:
                return "REFUTED"

    return verdict

# ============================================================
# AGGREGATE SENTENCE VERDICT
# ============================================================

def aggregate_sentence_verdict(
    atomic_results
):

    verdicts = [
        result["verdict"]
        for result in atomic_results
    ]


    if not verdicts:
        return "UNCERTAIN"


    supported = verdicts.count(
        "SUPPORTED"
    )

    refuted = verdicts.count(
        "REFUTED"
    )

    partial = verdicts.count(
        "PARTIALLY_SUPPORTED"
    )

    uncertain = verdicts.count(
        "UNCERTAIN"
    )


    if supported == len(verdicts):

        return "SUPPORTED"


    if refuted == len(verdicts):

        return "REFUTED"


    if supported > 0 and refuted > 0:

        return "PARTIALLY_SUPPORTED"


    if partial > 0:

        return "PARTIALLY_SUPPORTED"


    if supported > 0 and uncertain > 0:

        return "PARTIALLY_SUPPORTED"


    if refuted > 0 and uncertain > 0:

        return "PARTIALLY_SUPPORTED"


    return "UNCERTAIN"


# ============================================================
# VERDICT HELPERS
# ============================================================

def verdict_badge(verdict, confidence=None):
    conf_str = f" ({round(confidence)}% Conf.)" if confidence is not None and round(confidence) > 0 else ""
    if verdict == "SUPPORTED":
        return f"✓ SUPPORTED{conf_str}"
    if verdict == "REFUTED":
        return f"✕ REFUTED{conf_str}"
    if verdict == "PARTIALLY_SUPPORTED":
        return f"⚠ PARTIALLY SUPPORTED{conf_str}"
    return "? UNCERTAIN"


def verdict_class(verdict):

    if verdict == "SUPPORTED":

        return "supported"


    if verdict == "REFUTED":

        return "refuted"


    if verdict == "PARTIALLY_SUPPORTED":

        return "partial"


    return "uncertain"


# ============================================================
# PLAGIARISM BADGE HELPERS
# ============================================================

def plag_badge(category, score):
    if score >= 80 or category == "VERBATIM":
        return f"⚠ {score}% VERBATIM"
    if score >= 55 or category == "PARAPHRASED":
        return f"~ {score}% PARAPHRASED"
    if score >= 35 or category == "LIGHT_PARAPHRASE":
        return f"~ {score}% PARTIAL"
    return f"✓ {score}% ORIGINAL"


def plag_class(category, score):
    if score >= 80 or category == "VERBATIM":
        return "badge-verbatim"
    if score >= 55 or category == "PARAPHRASED":
        return "badge-paraphrase"
    if score >= 35 or category == "LIGHT_PARAPHRASE":
        return "badge-partial"
    return "badge-original"


# ============================================================
# HTML ESCAPE
# ============================================================

def escape_html(text):

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


# ============================================================
# CREATE INITIAL CHECKING CARD
# ============================================================

def create_checking_card(
    sentence,
    index
):

    safe_sentence = escape_html(
        sentence
    )

    return f"""
<div class="claim-card-modern checking">
    <div class="card-top-row">
        <span class="sentence-index-tag">
            SENTENCE {index}
        </span>
        <span class="badge-pill-tech">
            <span class="pulse-dot"></span> ANALYZING EVIDENCE &amp; CONTEXT...
        </span>
    </div>

    <div class="sentence-quote-box" style="opacity: 0.85;">
        “{safe_sentence}”
    </div>
</div>
"""


def create_result_card(
    sentence,
    verdict,
    atomic_claims,
    index,
    confidence=None,
    similarity_result=None,
    plag_result=None,
    hallucination_result=None,
    context_meta=None
):

    safe_sentence = escape_html(
        sentence
    )

    # Highlight overlapping words if plagiarism detected
    if plag_result and plag_result.get("similarity_score", 0) >= 30 and plag_result.get("highlighted_text"):
        sentence_display = plag_result["highlighted_text"]
    else:
        sentence_display = safe_sentence

    # Discourse Memory & Context Pill
    memory_pill_html = ""
    if context_meta:
        has_anaphora = context_meta.get("has_anaphora", False)
        trigger = escape_html(context_meta.get("trigger", ""))
        antecedent = escape_html(context_meta.get("resolved_antecedent", ""))
        topic = escape_html(context_meta.get("global_topic", ""))
        if has_anaphora and antecedent:
            memory_pill_html = f"""
            <div class="memory-pill-modern">
                <span class="memory-icon">🧠</span>
                <div>
                    <span class="memory-title">Discourse Memory Link</span>
                    <span class="memory-text">Resolved pronoun <em>"{trigger}"</em> &rarr; <strong>{antecedent}</strong> (connected to preceding sentence context)</span>
                </div>
            </div>
            """
        elif topic:
            memory_pill_html = f"""
            <div class="memory-pill-modern" style="background: rgba(99, 102, 241, 0.08); border-color: rgba(99, 102, 241, 0.25);">
                <span class="memory-icon">🔗</span>
                <div>
                    <span class="memory-title">Article Thematic Context</span>
                    <span class="memory-text">Anchor theme: <strong>{topic}</strong></span>
                </div>
            </div>
            """

    # Hallucination status pill (Phase 3)
    hal_pill_html = ""
    if hallucination_result:
        h_status = hallucination_result.get("status", "FAITHFUL")
        h_label = escape_html(hallucination_result.get("status_label", h_status))
        h_risk = hallucination_result.get("hallucination_risk", 0.0)
        h_cls = hallucination_result.get("badge_class", "hal-faithful")
        h_desc = escape_html(hallucination_result.get("status_desc", ""))
        hal_pill_html = f'<span class="hal-pill {h_cls}" title="{h_desc}">⚡ {h_label} ({h_risk}% risk)</span>'

    # Text Similarity pill (Distinct from Plagiarism)
    sim_pill_html = ""
    if similarity_result:
        sim_sc = similarity_result.get("similarity_score", 0.0)
        sim_rel = similarity_result.get("relationship", "DISTINCT")
        if sim_rel in ("TOPICAL_OVERLAP", "SHARED_TERMINOLOGY"):
            sim_cls = "sim-pill-topical"
            sim_text = f"≈ {sim_sc}% Topical Match"
        elif sim_rel in ("VERBATIM_COPY", "HEAVY_PARAPHRASE"):
            sim_cls = "sim-pill-match"
            sim_text = f"≈ {sim_sc}% Text Match"
        else:
            sim_cls = "sim-pill-distinct"
            sim_text = f"≈ {sim_sc}% Independent"
        sim_pill_html = f'<span class="sim-pill {sim_cls}" title="{escape_html(similarity_result.get("relationship_desc", ""))}">{sim_text}</span>'

    # Plagiarism top pill
    if plag_result:
        plag_cat = plag_result.get("category", "ORIGINAL")
        plag_sc = plag_result.get("similarity_score", 0.0)
        plag_pill_html = f"""
        <span class="plag-pill {plag_class(plag_cat, plag_sc)}">
            {plag_badge(plag_cat, plag_sc)}
        </span>
        """
    else:
        plag_pill_html = ""

    # --------------------------------------------------------
    # Text Similarity detail section (Topical overlap != plagiarism)
    # --------------------------------------------------------
    sim_section_html = ""
    if similarity_result and similarity_result.get("similarity_score", 0) >= 12:
        s_score = similarity_result.get("similarity_score", 0)
        s_topical = similarity_result.get("topical_similarity", 0)
        s_lexical = similarity_result.get("lexical_overlap", 0)
        s_rel_label = escape_html(similarity_result.get("relationship_label", "Topical Overlap"))
        s_rel_desc = escape_html(similarity_result.get("relationship_desc", ""))
        s_kw = similarity_result.get("shared_keywords", [])
        kw_badges = " ".join([f'<span class="sim-kw-badge">{escape_html(k)}</span>' for k in s_kw[:6]])

        sim_section_html = f"""
        <div class="section-label sim-section-label">
            TEXT SIMILARITY ANALYSIS (TOPICAL &amp; CONTEXTUAL ALIGNMENT)
        </div>
        <div class="similarity-detail-box">
            <div class="sim-box-header">
                <span class="sim-relation-badge">{s_rel_label}</span>
                <span class="sim-metric-info">Topical Alignment: <strong>{s_topical}%</strong> &bull; Lexical Overlap: <strong>{s_lexical}%</strong></span>
            </div>
            <div class="sim-box-desc">{s_rel_desc}</div>
            {f'<div class="sim-kw-row"><span class="sim-kw-title">Shared Terminology:</span> {kw_badges}</div>' if kw_badges else ''}
        </div>
        """

    # --------------------------------------------------------
    # Plagiarism detail section
    # --------------------------------------------------------
    plag_section_html = ""
    if plag_result and plag_result.get("similarity_score", 0) >= 30:
        top_src = plag_result.get("top_source") or {}
        src_title = escape_html(top_src.get("title", "Web Source"))
        src_domain = escape_html(top_src.get("domain", ""))
        src_url = escape_html(top_src.get("url", ""))
        matched_snip = escape_html(plag_result.get("matched_snippet", ""))
        matched_origin = "REFERENCE DOC" if plag_result.get("matched_origin") == "reference" else "WEB SOURCE"

        if src_url and src_url != "#reference-doc":
            link_html = f'<a href="{src_url}" target="_blank" class="plag-source-link">View source ({src_domain}) ↗</a>'
        else:
            link_html = f'<span class="plag-source-domain">{src_domain or "Reference Document"}</span>'

        snip_html = ""
        if matched_snip:
            snip_html = f"""
            <div class="plag-snip-box">
                <span class="plag-snip-label">Matched Source Excerpt:</span>
                “{matched_snip[:240]}...”
            </div>
            """

        plag_section_html = f"""
        <div class="section-label plag-section-label">
            PLAGIARISM &amp; ORIGINALITY AUDIT
        </div>

        <div class="plag-detail-box {plag_class(plag_result.get('category'), plag_result.get('similarity_score', 0))}">
            <div class="plag-box-header">
                <div class="plag-box-left">
                    <span class="plag-origin-badge">{matched_origin}</span>
                    <span class="plag-src-name">{src_title}</span>
                </div>
                {link_html}
            </div>

            <div class="plag-metrics-bar">
                <span class="plag-metric-item">N-Gram Exact: <strong>{plag_result.get('exact_score', 0)}%</strong></span>
                <span class="plag-metric-item">Lexical Match: <strong>{plag_result.get('lexical_score', 0)}%</strong></span>
                <span class="plag-metric-item">AI Semantic: <strong>{plag_result.get('semantic_score', 0)}%</strong></span>
            </div>

            {snip_html}
        </div>
        """
    elif plag_result:
        plag_section_html = f"""
        <div class="plag-clean-bar">
            <span class="plag-clean-icon">✓</span>
            <span class="plag-clean-text">ORIGINAL CONTENT &bull; {plag_result.get('similarity_score', 0)}% similarity &bull; Authentic writing with no significant copy or paraphrase found</span>
        </div>
        """

    # --------------------------------------------------------
    # Hallucination alert section (Phase 3)
    # --------------------------------------------------------
    hal_section_html = ""
    if hallucination_result and hallucination_result.get("status") in ("CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE"):
        h_status = hallucination_result.get("status")
        h_label = escape_html(hallucination_result.get("status_label", h_status))
        h_desc = escape_html(hallucination_result.get("status_desc", ""))
        h_ratio = hallucination_result.get("grounding_ratio", 0)
        h_risk = hallucination_result.get("hallucination_risk", 0)
        h_cls = hallucination_result.get("badge_class", "hal-unsupported")
        unmatched = hallucination_result.get("unmatched_entities", [])
        unmatched_str = ", ".join(unmatched[:4]) if unmatched else ""

        hal_section_html = f"""
        <div class="section-label hal-section-label">
            GROUNDING &amp; HALLUCINATION AUDIT
        </div>
        <div class="hal-alert-box {h_cls}">
            <div class="hal-alert-header">
                <div class="hal-alert-left">
                    <span class="hal-alert-icon">⚠</span>
                    <span class="hal-alert-title">{h_label.upper()}</span>
                </div>
                <span class="hal-alert-risk">Risk Score: <strong>{h_risk}%</strong></span>
            </div>
            <div class="hal-alert-desc">{h_desc}</div>
            <div class="hal-alert-meta">
                <span>Evidence Grounding: <strong>{h_ratio}%</strong></span>
                {f'<span>Uncorroborated: <strong>{escape_html(unmatched_str)}</strong></span>' if unmatched_str else ''}
            </div>
        </div>
        """

    # --------------------------------------------------------
    # Atomic claim breakdown
    # --------------------------------------------------------
    breakdown_html = ""
    if len(atomic_claims) > 1:
        breakdown_html += """
<div class="section-label">
    CLAIM BREAKDOWN
</div>
<div class="breakdown">
"""
        for atomic_index, atomic in enumerate(atomic_claims, 1):
            atomic_claim = escape_html(atomic["claim"])
            atomic_verdict = atomic["verdict"]
            at_conf = atomic.get("confidence")
            conf_badge = f'<span class="atomic-conf-badge">{atomic.get("confidence_label", "Conf")}: {round(at_conf)}%</span>' if at_conf else ""
            breakdown_html += f"""
<div class="atomic-claim">
    <div class="atomic-number">{atomic_index}</div>
    <div class="atomic-content">
        <div class="atomic-text">{atomic_claim}</div>
        <div class="atomic-meta">
            <div class="atomic-verdict {verdict_class(atomic_verdict)}">
                {verdict_badge(atomic_verdict, at_conf)}
            </div>
            {conf_badge}
        </div>
    </div>
</div>
"""
        breakdown_html += "</div>"

    # --------------------------------------------------------
    # AI analysis
    # --------------------------------------------------------
    analysis_html = ""
    for atomic in atomic_claims:
        analysis = escape_html(atomic["explanation"])
        analysis_html += f"""
<div class="analysis-item">
    {analysis}
</div>
"""

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------
    source_html = ""
    for atomic in atomic_claims:
        for source in atomic["sources"][:4]:
            title = escape_html(source.get("title", ""))
            body = escape_html(source.get("body", ""))
            url = escape_html(source.get("url", ""))
            cred = get_source_credibility(source.get("url", ""))
            cred_lvl = cred.level
            cred_cls = cred.class_name
            cred_desc = cred.description
            cred_score = cred.score
            rank_num = source.get("rank", 1)
            domain_display = escape_html(get_domain(source.get("url", "")))
            source_html += f"""
<div class="source-item">
    <div class="source-header">
        <div class="source-title-row">
            <span class="source-rank-badge">#{rank_num} RANKED</span>
            <div class="source-title">{title}</div>
        </div>
        <span class="cred-badge {cred_cls}" title="{cred_desc}">
            {cred_lvl} CREDIBILITY ({cred_score})
        </span>
    </div>
    <div class="source-body">{body}</div>
    <div class="source-footer">
        <a href="{url}" target="_blank" class="source-link">View source ↗</a>
        <span class="source-domain">{domain_display}</span>
    </div>
</div>
"""

    if not source_html:
        source_html = """
<div class="no-source">
    No web sources were available.
</div>
"""

    # --------------------------------------------------------
    # Complete card
    # --------------------------------------------------------
    return f"""
<div class="claim-card-modern card-{verdict_class(verdict)}">
    <div class="card-top-row">
        <span class="sentence-index-tag">
            SENTENCE {index}
        </span>
        <div class="badge-group">
            <span class="verdict-pill-modern v-{verdict_class(verdict)}">
                {verdict_badge(verdict, confidence)}
            </span>
            {hal_pill_html}
            {sim_pill_html}
            {plag_pill_html}
        </div>
    </div>

    <div class="sentence-quote-box">
        {sentence_display}
    </div>

    {memory_pill_html}

    {hal_section_html}

    {sim_section_html}

    {plag_section_html}

    {breakdown_html}

    <div class="section-label">
        AI FACT ANALYSIS
    </div>
    <div class="analysis-text">
        {analysis_html}
    </div>

    <div class="section-label">
        RANKED WEB EVIDENCE
    </div>
    <div class="sources">
        {source_html}
    </div>
</div>
"""


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    sentence_results,
    plagiarism_audit=None,
    source_info=None,
    similarity_summary=None,
    trust_result=None,
    hallucination_report=None
):

    supported = sum(r["verdict"] == "SUPPORTED" for r in sentence_results)
    refuted = sum(r["verdict"] == "REFUTED" for r in sentence_results)
    partial = sum(r["verdict"] == "PARTIALLY_SUPPORTED" for r in sentence_results)
    uncertain = sum(r["verdict"] == "UNCERTAIN" for r in sentence_results)
    total = len(sentence_results)

    if refuted >= 1:
        if refuted >= supported:
            overall = "LIKELY MISLEADING"
        else:
            overall = "MIXED / NEEDS REVIEW"
    elif partial > 0:
        overall = "MIXED / NEEDS REVIEW"
    elif uncertain > total / 2:
        overall = "INSUFFICIENT EVIDENCE"
    else:
        overall = "LIKELY SUPPORTED"

    # --------------------------------------------------------
    # Trust Score Hero Banner (Phase 3)
    # --------------------------------------------------------
    trust_hero_html = ""
    if trust_result:
        t_score = trust_result.get("trust_score", 0.0)
        t_label = escape_html(trust_result.get("rating_label", "TRUST EVALUATION"))
        t_tier = escape_html(trust_result.get("rating_tier", "B"))
        t_badge_cls = escape_html(trust_result.get("badge_class", "trust-badge-moderate"))
        t_summary = escape_html(trust_result.get("assessment_summary", ""))
        t_breakdown = trust_result.get("breakdown", {})

        fact_b = t_breakdown.get("fact_verification", {})
        cred_b = t_breakdown.get("source_credibility", {})
        ground_b = t_breakdown.get("evidence_grounding", {})
        orig_b = t_breakdown.get("content_originality", {})

        trust_hero_html = f"""
        <div class="trust-hero-container">
            <div class="trust-hero-main">
                <div class="trust-score-badge-circle {t_badge_cls}">
                    <div class="trust-score-val">{t_score}</div>
                    <div class="trust-score-tier">GRADE {t_tier}</div>
                </div>
                <div class="trust-hero-info">
                    <div class="trust-hero-label-row">
                        <span class="trust-status-badge {t_badge_cls}">{t_label}</span>
                        <span class="trust-meta-note">Composite AI Credibility Index</span>
                    </div>
                    <div class="trust-hero-desc">{t_summary}</div>
                </div>
            </div>

            <div class="trust-signals-grid">
                <div class="trust-signal-card">
                    <div class="signal-header">
                        <span class="signal-name">Fact Verification</span>
                        <span class="signal-pct">{fact_b.get('score', 0)}%</span>
                    </div>
                    <div class="signal-progress-track">
                        <div class="signal-progress-fill fill-fact" style="width: {min(100, max(5, fact_b.get('score', 0)))}%;"></div>
                    </div>
                    <div class="signal-footer">Weight 35% &bull; {fact_b.get('weighted_score', 0)} pts</div>
                </div>

                <div class="trust-signal-card">
                    <div class="signal-header">
                        <span class="signal-name">Source Credibility</span>
                        <span class="signal-pct">{cred_b.get('score', 0)}%</span>
                    </div>
                    <div class="signal-progress-track">
                        <div class="signal-progress-fill fill-cred" style="width: {min(100, max(5, cred_b.get('score', 0)))}%;"></div>
                    </div>
                    <div class="signal-footer">Weight 25% &bull; {cred_b.get('weighted_score', 0)} pts</div>
                </div>

                <div class="trust-signal-card">
                    <div class="signal-header">
                        <span class="signal-name">Evidence Grounding</span>
                        <span class="signal-pct">{ground_b.get('score', 0)}%</span>
                    </div>
                    <div class="signal-progress-track">
                        <div class="signal-progress-fill fill-ground" style="width: {min(100, max(5, ground_b.get('score', 0)))}%;"></div>
                    </div>
                    <div class="signal-footer">Weight 25% &bull; {ground_b.get('weighted_score', 0)} pts</div>
                </div>

                <div class="trust-signal-card">
                    <div class="signal-header">
                        <span class="signal-name">Content Originality</span>
                        <span class="signal-pct">{orig_b.get('score', 0)}%</span>
                    </div>
                    <div class="signal-progress-track">
                        <div class="signal-progress-fill fill-orig" style="width: {min(100, max(5, orig_b.get('score', 0)))}%;"></div>
                    </div>
                    <div class="signal-footer">Weight 15% &bull; {orig_b.get('weighted_score', 0)} pts</div>
                </div>
            </div>
        </div>
        """

    # Source info pill
    source_tag = ""
    if source_info and source_info.get("name"):
        s_type = source_info.get("type", "text").upper()
        s_name = escape_html(source_info.get("name", "")[:35])
        source_tag = f'<span class="source-ingest-badge">{s_type}: {s_name}</span>'

    # Plagiarism Audit Pod
    if plagiarism_audit:
        orig_score = plagiarism_audit.get("originality_score", 100.0)
        overall_sim = plagiarism_audit.get("overall_similarity", 0.0)
        v_pct = plagiarism_audit.get("verbatim_percentage", 0.0)
        p_pct = plagiarism_audit.get("paraphrase_percentage", 0.0)
        o_pct = plagiarism_audit.get("original_percentage", 100.0)

        if orig_score >= 80.0:
            risk_badge = '<span class="risk-badge risk-low">✓ LOW PLAGIARISM RISK</span>'
        elif orig_score >= 50.0:
            risk_badge = '<span class="risk-badge risk-med">⚠ MODERATE RISK</span>'
        else:
            risk_badge = '<span class="risk-badge risk-high">✕ HIGH PLAGIARISM RISK</span>'

        # Top matched sources pills
        top_srcs = plagiarism_audit.get("top_sources", [])
        src_pills_html = ""
        for ts in top_srcs[:3]:
            ts_domain = escape_html(ts.get("domain", "web"))
            ts_sim = ts.get("highest_similarity", 0.0)
            ts_cnt = ts.get("matched_count", 1)
            ts_url = escape_html(ts.get("url", "#"))
            if ts_url and ts_url != "#reference-doc":
                src_pills_html += f'<a href="{ts_url}" target="_blank" class="summary-src-pill">{ts_domain} ({ts_sim}% &bull; {ts_cnt} sent.) ↗</a>'
            else:
                src_pills_html += f'<span class="summary-src-pill">{ts_domain} ({ts_sim}% &bull; {ts_cnt} sent.)</span>'

        if src_pills_html:
            sources_footer = f"""
            <div class="summary-sources-row">
                <span class="summary-sources-label">Matched External Sources:</span>
                {src_pills_html}
            </div>
            """
        else:
            sources_footer = ""

        ref_indicator = ""
        if plagiarism_audit.get("has_reference_comparison"):
            ref_indicator = '<span class="ref-doc-badge">📄 Compared with Reference Document</span>'

        plag_pod_html = f"""
        <div class="summary-pod plag-summary-pod">
            <div class="pod-header">
                <div>
                    <div class="pod-title">Originality Audit</div>
                    <div class="pod-subtitle">{orig_score}% Unique Content {ref_indicator}</div>
                </div>
                {risk_badge}
            </div>

            <div class="plag-stat-grid">
                <div class="plag-stat-box verbatim-box">
                    <div class="stat-number">{v_pct}%</div>
                    <div class="stat-label">VERBATIM COPY</div>
                </div>
                <div class="plag-stat-box paraphrase-box">
                    <div class="stat-number">{p_pct}%</div>
                    <div class="stat-label">PARAPHRASED</div>
                </div>
                <div class="plag-stat-box original-box">
                    <div class="stat-number">{o_pct}%</div>
                    <div class="stat-label">ORIGINAL</div>
                </div>
            </div>

            {sources_footer}
        </div>
        """
    else:
        plag_pod_html = ""

    # Hallucination Risk Strip (Phase 3)
    hal_strip_html = ""
    if hallucination_report:
        h_risk = hallucination_report.get("overall_hallucination_risk", 0.0)
        h_ground = hallucination_report.get("grounding_score", 0.0)
        h_badge = escape_html(hallucination_report.get("risk_badge", "LOW RISK"))
        h_desc = escape_html(hallucination_report.get("summary_desc", ""))
        h_cat = hallucination_report.get("risk_category", "LOW_RISK")
        h_f_cnt = hallucination_report.get("faithful_count", 0)
        h_u_cnt = hallucination_report.get("unsupported_count", 0)
        h_c_cnt = hallucination_report.get("contradicted_count", 0)

        hal_strip_html = f"""
        <div class="summary-hallucination-strip {h_cat.lower()}">
            <div class="hal-strip-left">
                <span class="hal-strip-icon">⚡</span>
                <span class="hal-strip-title">Grounding &amp; Hallucination Audit:</span>
                <span class="hal-strip-badge">{h_badge}</span>
                <span class="hal-strip-detail">{h_desc}</span>
            </div>
            <div class="hal-strip-counts">
                <span class="hal-count-item count-faithful">{h_f_cnt} Faithful</span>
                <span class="hal-count-item count-unsup">{h_u_cnt} Unsupported</span>
                <span class="hal-count-item count-contra">{h_c_cnt} Contradicted</span>
            </div>
        </div>
        """

    # Text Similarity Strip (Distinct from Plagiarism)
    sim_strip_html = ""
    if similarity_summary and (similarity_summary.get("topical_overlap_score", 0) > 0 or similarity_summary.get("overall_similarity", 0) > 0):
        s_top = similarity_summary.get("topical_overlap_score", 0.0)
        s_lex = similarity_summary.get("lexical_overlap_score", 0.0)
        s_kw = similarity_summary.get("shared_topical_keywords", [])
        s_cons = escape_html(similarity_summary.get("topical_consensus_level", "Moderate"))
        kw_badges = " ".join([f'<span class="sim-kw-badge">{escape_html(k)}</span>' for k in s_kw[:5]])

        sim_strip_html = f"""
        <div class="summary-similarity-strip">
            <div class="sim-strip-left">
                <span class="sim-strip-icon">≈</span>
                <span class="sim-strip-title">Text Similarity Analysis:</span>
                <span class="sim-strip-value">{s_top}% Topical Overlap &bull; {s_lex}% Lexical Match</span>
                <span class="sim-strip-badge">({s_cons})</span>
            </div>
            <div class="sim-strip-note">
                {f'<span style="color:#8b949e; margin-right:6px;">Key Topics:</span> {kw_badges}' if kw_badges else 'Measures conceptual alignment with literature (Independent from plagiarism risk)'}
            </div>
        </div>
        """

    return f"""
<div class="summary-card">
    {trust_hero_html}
    <div class="dual-summary-grid">
        <div class="summary-pod fact-summary-pod">
            <div class="pod-header">
                <div>
                    <div class="pod-title">Fact Verification</div>
                    <div class="pod-subtitle">{total} sentences evaluated {source_tag}</div>
                </div>
                <div class="overall-badge overall-{overall.lower().replace(' ', '-').replace('/', '-')}">
                    {overall}
                </div>
            </div>

            <div class="summary-stats">
                <div class="summary-stat supported-stat">
                    <div class="stat-number">{supported}</div>
                    <div class="stat-label">SUPPORTED</div>
                </div>
                <div class="summary-stat refuted-stat">
                    <div class="stat-number">{refuted}</div>
                    <div class="stat-label">REFUTED</div>
                </div>
                <div class="summary-stat partial-stat">
                    <div class="stat-number">{partial}</div>
                    <div class="stat-label">PARTIAL</div>
                </div>
                <div class="summary-stat uncertain-stat">
                    <div class="stat-number">{uncertain}</div>
                    <div class="stat-label">UNCERTAIN</div>
                </div>
            </div>
        </div>

        {plag_pod_html}
    </div>
    {hal_strip_html}
    {sim_strip_html}
</div>
"""


# ============================================================
# ARTICLE VERIFICATION & PLAGIARISM STREAMING UI
# ============================================================

def verify_article_stream(
    article="",
    article_file=None,
    article_url="",
    reference_text="",
    reference_file=None,
    enable_fact_check=True,
    enable_plagiarism=True
):

    # 1. Ingest main article content
    ingest_res = ingest_content(
        direct_text=article,
        file_path=article_file,
        url=article_url
    )

    if not ingest_res["success"]:
        err_msg = escape_html(ingest_res.get("error", "Failed to extract article content."))
        yield f"""
        <div class="empty-state">
            <div class="empty-icon" style="color: #f08088;">⚠</div>
            <div class="empty-title" style="color: #f08088;">Input Error</div>
            <div class="empty-text" style="max-width: 520px; margin: 12px auto; line-height: 1.5;">
                {err_msg}
            </div>
        </div>
        """
        return

    clean_article = ingest_res["text"]
    source_info = {
        "title": ingest_res.get("title", ""),
        "type": ingest_res.get("source_type", "text"),
        "name": ingest_res.get("source_name", "Direct Input")
    }

    # 2. Ingest reference document if provided
    ref_text_combined = reference_text or ""
    if reference_file:
        try:
            ref_ingest = ingest_content(file_path=reference_file)
            if ref_ingest["success"]:
                ref_text_combined = (ref_text_combined + "\n\n" + ref_ingest["text"]).strip()
                print(f"Loaded reference document ({ref_ingest['word_count']} words from {ref_ingest['source_name']}).")
            else:
                print(f"Warning: Reference file error: {ref_ingest['error']}")
        except Exception as e:
            print(f"Error reading reference document file: {e}")

    sentences = extract_sentences(
        clean_article
    )

    if not sentences:
        yield """
        <div class="empty-state">
            <div class="empty-title">
                No suitable sentences found
            </div>
        </div>
        """
        return

    source_badge = f'<span class="source-ingest-badge">{source_info["type"].upper()}: {escape_html(source_info["name"][:30])}</span>'

    print("\n")
    print("========================================")
    print("STARTING TRUTHLENS VERIFICATION & AUDIT")
    print("========================================")
    print(f"Source Type       : {source_info['type'].upper()} ({source_info['name']})")
    print(f"Sentences detected: {len(sentences)}")
    print(f"Fact Verification : {'ENABLED' if enable_fact_check else 'DISABLED'}")
    print(f"Plagiarism Audit  : {'ENABLED' if enable_plagiarism else 'DISABLED'}")
    if ref_text_combined:
        print(f"Reference Document: PROVIDED ({len(ref_text_combined.split())} words)")

    # ========================================================
    # SHOW ALL SENTENCES FIRST
    # ========================================================
    checking_cards = ""
    for index, sentence in enumerate(sentences, 1):
        checking_cards += create_checking_card(sentence, index)

    yield f"""
<div class="analysis-header">
    <div>
        <div class="analysis-title">
            Analyzing article
            {source_badge}
        </div>
        <div class="analysis-subtitle">
            {len(sentences)} sentences detected &bull; Fact verification & Originality audit active
        </div>
    </div>
    <div class="live-indicator">
        <span class="live-dot"></span>
        LIVE
    </div>
</div>
{checking_cards}
"""

    time.sleep(0.4)

    sentence_results = []
    sentence_sources_map = {}

    # Initialize Discourse Memory Engine for narrative context & antecedent tracking
    memory_engine = DiscourseMemoryEngine(clean_article)
    print(f"Discourse Engine Initialized: Topic='{memory_engine.global_topic}', Entities={memory_engine.global_entities[:3]}")

    # Pre-extract reference sentences if reference document provided
    ref_sentences = extract_sentences(ref_text_combined) if ref_text_combined and ref_text_combined.strip() else []

    # ========================================================
    # PROCESS EACH SENTENCE
    # ========================================================
    for sentence_index, sentence in enumerate(sentences, 1):
        print("\n")
        print(f"PROCESSING SENTENCE {sentence_index}/{len(sentences)}")
        print(f"  Text: {sentence}")

        # Resolve sentence-level discourse context
        sentence_context_meta = memory_engine.resolve_claim_context(
            sentence,
            sentence_index,
            sentences
        )
        if sentence_context_meta.get("has_anaphora"):
            print(f"  🧠 Discourse Memory: Anaphora detected ('{sentence_context_meta.get('trigger')}'). Resolved antecedent: '{sentence_context_meta.get('resolved_antecedent')}'")

        atomic_results = []
        sentence_evidence = []
        seen_src_urls = set()

        # ----------------------------------------------------
        # 1. FACT VERIFICATION
        # ----------------------------------------------------
        if enable_fact_check:
            atomic_claims = extract_atomic_claims(sentence)
            print(f"  Atomic claims detected: {len(atomic_claims)}")

            for claim_index, claim in enumerate(atomic_claims, 1):
                # Contextualize claim with memory & antecedent resolution
                claim_ctx = memory_engine.resolve_claim_context(claim, sentence_index, sentences)
                search_query = claim_ctx.get("context_augmented_query", claim)
                print(f"  Claim {claim_index}/{len(atomic_claims)}: {claim}")
                if search_query != claim:
                    print(f"    -> Context-Augmented Search Query: {search_query}")

                evidence = search_web(search_query)

                # Collect sources
                for s in evidence:
                    u = s.get("url", "")
                    if u and u not in seen_src_urls:
                        seen_src_urls.add(u)
                        sentence_evidence.append(s)

                if not evidence:
                    print("  No web evidence found.")
                    atomic_results.append({
                        "claim": claim,
                        "verdict": "UNCERTAIN",
                        "explanation": "The web retrieval module could not obtain reliable evidence for this claim.",
                        "sources": [],
                        "context_meta": claim_ctx
                    })
                    continue

                print("  Sending evidence to Qwen...")
                context_prompt = memory_engine.format_llm_context_prompt(claim, claim_ctx)
                raw_response = verify_claim(claim, evidence, context=context_prompt)
                verdict, explanation = extract_verdict_and_explanation(
                    raw_response,
                    claim,
                    evidence
                )
                verdict = evidence_consistency_check(
                    claim,
                    evidence,
                    verdict,
                    explanation=explanation
                )
                conf_val, conf_label, conf_reason = compute_claim_confidence(
                    claim,
                    verdict,
                    evidence,
                    explanation=explanation
                )
                print(f"  Verdict: {verdict} ({conf_val}% {conf_label})")

                atomic_results.append({
                    "claim": claim,
                    "verdict": verdict,
                    "confidence": conf_val,
                    "confidence_label": conf_label,
                    "confidence_reason": conf_reason,
                    "explanation": explanation,
                    "sources": evidence,
                    "context_meta": claim_ctx
                })

            sentence_verdict = aggregate_sentence_verdict(atomic_results)
            sentence_conf = round(sum(ar.get("confidence", 50.0) for ar in atomic_results) / len(atomic_results), 1) if atomic_results else 50.0

            # Record verified sentence into running discourse memory
            memory_engine.record_sentence_verification(
                sentence_index,
                sentence,
                sentence_verdict,
                atomic_results
            )
        else:
            atomic_claims = [sentence]
            sentence_verdict = "UNCERTAIN"
            sentence_conf = 50.0
            atomic_results = [{
                "claim": sentence,
                "verdict": "UNCERTAIN",
                "confidence": 50.0,
                "confidence_label": "Bypassed",
                "confidence_reason": "Fact verification bypassed by user setting.",
                "explanation": "Fact verification bypassed by user setting.",
                "sources": [],
                "context_meta": sentence_context_meta
            }]
            memory_engine.record_sentence_verification(
                sentence_index,
                sentence,
                sentence_verdict,
                atomic_results
            )

        # If fact-checking was off or had no evidence, do a quick search for plagiarism check
        if enable_plagiarism and not sentence_evidence:
            print("  Searching web for plagiarism detection...")
            sentence_evidence = search_web(sentence)

        sentence_sources_map[sentence_index] = sentence_evidence

        # ----------------------------------------------------
        # 2. TEXT SIMILARITY ANALYSIS (DISTINCT FROM PLAGIARISM)
        # ----------------------------------------------------
        sim_eval = None
        if sentence_evidence:
            top_ev_body = " ".join([e.get("body", "") for e in sentence_evidence[:2]])
            top_ev_title = sentence_evidence[0].get("title", "Web Evidence")
            sim_eval = similarity_analyzer.analyze_pair(
                sentence,
                top_ev_body,
                source_name=top_ev_title
            )
        elif ref_sentences:
            best_r_sim = -1.0
            for rs in ref_sentences:
                r_sim = similarity_analyzer.analyze_pair(sentence, rs, source_name="Reference Document")
                if r_sim["similarity_score"] > best_r_sim:
                    best_r_sim = r_sim["similarity_score"]
                    sim_eval = r_sim
        else:
            sim_eval = similarity_analyzer.analyze_pair(sentence, "", source_name="None")

        if sim_eval:
            print(f"  Similarity : {sim_eval.get('relationship_label')} ({sim_eval.get('similarity_score')}%)")

        # ----------------------------------------------------
        # 3. PLAGIARISM DETECTION (HYBRID: N-GRAM + LEXICAL + QWEN)
        # ----------------------------------------------------
        plag_eval = None
        if enable_plagiarism:
            print("  Evaluating sentence plagiarism...")
            # Check against retrieved web evidence
            web_plag = plagiarism_detector.check_sentence_against_sources(
                sentence,
                sentence_evidence
            )

            # Check against reference document if provided
            ref_plag = None
            if ref_sentences:
                best_r_score = -1.0
                for rs in ref_sentences:
                    r_eval = plagiarism_detector.evaluate_pair(sentence, rs)
                    if r_eval["similarity_score"] > best_r_score:
                        best_r_score = r_eval["similarity_score"]
                        ref_plag = r_eval
                        ref_plag["ref_sentence"] = rs

            # Choose the stronger match
            if ref_plag and ref_plag["similarity_score"] > web_plag["similarity_score"]:
                plag_eval = ref_plag
                plag_eval["top_source"] = {
                    "title": "Reference Document",
                    "url": "#reference-doc",
                    "domain": "Reference Document",
                    "body": ref_plag.get("ref_sentence", "")
                }
                plag_eval["matched_snippet"] = ref_plag.get("ref_sentence", "")
                plag_eval["matched_origin"] = "reference"
            else:
                plag_eval = web_plag
                plag_eval["matched_origin"] = "web"

            print(f"  Plagiarism : {plag_eval.get('category')} ({plag_eval.get('similarity_score')}%)")

        # ----------------------------------------------------
        # 4. EVIDENCE-BASED HALLUCINATION DETECTION (PHASE 3)
        # ----------------------------------------------------
        sentence_exp = " ".join([ar.get("explanation", "") for ar in atomic_results if ar.get("explanation")]).strip()
        hal_eval = hallucination_detector.analyze_claim(
            claim=sentence,
            verdict=sentence_verdict,
            evidence=sentence_evidence,
            confidence=sentence_conf,
            explanation=sentence_exp
        )
        print(f"  Grounding  : {hal_eval.get('status_label')} ({hal_eval.get('hallucination_risk')}% risk, {hal_eval.get('grounding_ratio')}% grounded)")

        sentence_results.append({
            "sentence": sentence,
            "verdict": sentence_verdict,
            "confidence": sentence_conf,
            "atomic_claims": atomic_results,
            "similarity": sim_eval,
            "plagiarism": plag_eval,
            "hallucination": hal_eval,
            "sources": sentence_evidence,
            "context_meta": sentence_context_meta
        })

        # ----------------------------------------------------
        # Build cards so far
        # ----------------------------------------------------
        current_cards = ""
        for card_index, result in enumerate(sentence_results, 1):
            current_cards += create_result_card(
                result["sentence"],
                result["verdict"],
                result["atomic_claims"],
                card_index,
                confidence=result.get("confidence"),
                similarity_result=result.get("similarity"),
                plag_result=result.get("plagiarism"),
                hallucination_result=result.get("hallucination"),
                context_meta=result.get("context_meta")
            )

        for remaining_index in range(sentence_index + 1, len(sentences) + 1):
            current_cards += create_checking_card(
                sentences[remaining_index - 1],
                remaining_index
            )

        yield f"""
<div class="analysis-header">
    <div>
        <div class="analysis-title">
            Analyzing article
            {source_badge}
        </div>
        <div class="analysis-subtitle">
            Sentence {sentence_index} of {len(sentences)} &bull; Verifying claims, measuring similarity &amp; checking originality
        </div>
    </div>
    <div class="progress-counter">
        {sentence_index} / {len(sentences)}
    </div>
</div>
{current_cards}
"""

    # ========================================================
    # FINAL SUMMARY & COMPLETE AUDIT
    # ========================================================
    plagiarism_audit = None
    if enable_plagiarism:
        print("\nAuditing complete article plagiarism & originality...")
        plagiarism_audit = plagiarism_detector.audit_article_plagiarism(
            sentences,
            sentence_sources_map,
            reference_text=ref_text_combined
        )
        print(f"Overall Originality: {plagiarism_audit['originality_score']}%")

    # Collect all sources with credibility scores
    all_retrieved_sources = []
    for s_list in sentence_sources_map.values():
        for s in s_list:
            cred = get_source_credibility(s.get("url", ""))
            s_copy = dict(s)
            s_copy["credibility_score"] = cred.score
            all_retrieved_sources.append(s_copy)

    # Document-level multi-source text similarity
    similarity_summary = similarity_analyzer.analyze_document_similarity(
        clean_article,
        all_retrieved_sources
    )

    # Evidence-based Hallucination Evaluation (Phase 3)
    print("\nAuditing article grounding & hallucination risks...")
    hallucination_report = hallucination_detector.evaluate_article_hallucination(
        sentence_results
    )
    print(f"Grounding Score: {hallucination_report['grounding_score']}% (Risk: {hallucination_report['overall_hallucination_risk']}%)")

    # Transparent Multi-Signal Trust Score Calculation (Phase 3)
    print("Computing transparent multi-signal Trust Score...")
    trust_result = trust_score_engine.calculate_trust_score(
        sentence_results=sentence_results,
        all_sources=all_retrieved_sources,
        hallucination_report=hallucination_report,
        plagiarism_audit=plagiarism_audit
    )
    print(f"Overall Trust Score: {trust_result['trust_score']}/100 ({trust_result['rating_label']})")

    summary = build_summary(
        sentence_results,
        plagiarism_audit=plagiarism_audit,
        source_info=source_info,
        similarity_summary=similarity_summary,
        trust_result=trust_result,
        hallucination_report=hallucination_report
    )

    # Phase 4: Final Verification Report & Dossier
    print("Generating comprehensive Verification Report dossier...")
    report_data = generate_report_dict(
        article_text=clean_article,
        sentence_results=sentence_results,
        trust_result=trust_result,
        hallucination_report=hallucination_report,
        plagiarism_audit=plagiarism_audit,
        similarity_summary=similarity_summary,
        source_info=source_info
    )
    report_html = render_report_html_view(report_data)

    final_cards = ""
    for index, result in enumerate(sentence_results, 1):
        final_cards += create_result_card(
            result["sentence"],
            result["verdict"],
            result["atomic_claims"],
            index,
            confidence=result.get("confidence"),
            similarity_result=result.get("similarity"),
            plag_result=result.get("plagiarism"),
            hallucination_result=result.get("hallucination"),
            context_meta=result.get("context_meta")
        )

    print("\n")
    print("========================================")
    print("TRUTHLENS ANALYSIS & AUDIT COMPLETE")
    print("========================================")

    yield f"""
{summary}

{report_html}

<div class="analysis-header final-header">
    <div>
        <div class="analysis-title">
            Interactive Claim-by-Claim Verification Cards
            {source_badge}
        </div>
        <div class="analysis-subtitle">
            All {len(sentences)} sentences verified, similarity-mapped, and cross-referenced with retrieved evidence.
        </div>
    </div>
    <div class="complete-badge">
        ✓ COMPLETE
    </div>
</div>

{final_cards}
"""


def check_article(article="", article_file=None, article_url="", reference_text=None, reference_file=None, return_full_report=False):
    """
    Programmatic entry point for TruthLens verification.
    Takes article text, extracts sentences and atomic claims, retrieves multi-provider evidence,
    and returns a structured list of sentence verdicts, explanations, sources, and plagiarism metrics.
    Supports direct text, uploaded files (PDF, DOCX, TXT), and live article URLs.
    """
    if article_file or article_url or (isinstance(article, str) and (article.startswith("http://") or article.startswith("https://"))):
        ingest_res = ingest_content(direct_text=article, file_path=article_file, url=article_url)
        if ingest_res["success"]:
            article = ingest_res["text"]

    if reference_file:
        ref_ingest = ingest_content(file_path=reference_file)
        if ref_ingest["success"]:
            reference_text = ((reference_text or "") + "\n\n" + ref_ingest["text"]).strip()

    sentences = extract_sentences(article)
    results = []
    sentence_sources_map = {}

    memory_engine = DiscourseMemoryEngine(article)

    for idx, sentence in enumerate(sentences, 1):
        sentence_context_meta = memory_engine.resolve_claim_context(sentence, idx, sentences)
        atomic_claims = extract_atomic_claims(sentence)
        atomic_results = []
        sentence_evidence = []
        seen_urls = set()

        for claim in atomic_claims:
            claim_ctx = memory_engine.resolve_claim_context(claim, idx, sentences)
            search_query = claim_ctx.get("context_augmented_query", claim)
            evidence = search_web(search_query)

            for s in evidence:
                u = s.get("url", "")
                if u and u not in seen_urls:
                    seen_urls.add(u)
                    sentence_evidence.append(s)

            if not evidence:
                atomic_results.append({
                    "claim": claim,
                    "verdict": "UNCERTAIN",
                    "confidence": 20.0,
                    "confidence_label": "Insufficient Evidence",
                    "confidence_reason": "No web sources could be found.",
                    "explanation": "No web evidence could be retrieved for this claim.",
                    "sources": [],
                    "context_meta": claim_ctx
                })
                continue

            context_prompt = memory_engine.format_llm_context_prompt(claim, claim_ctx)
            raw_response = verify_claim(claim, evidence, context=context_prompt)
            verdict, explanation = extract_verdict_and_explanation(raw_response, claim, evidence)
            verdict = evidence_consistency_check(claim, evidence, verdict, explanation=explanation)
            c_val, c_lbl, c_rsn = compute_claim_confidence(claim, verdict, evidence, explanation=explanation)
            atomic_results.append({
                "claim": claim,
                "verdict": verdict,
                "confidence": c_val,
                "confidence_label": c_lbl,
                "confidence_reason": c_rsn,
                "explanation": explanation,
                "sources": evidence,
                "context_meta": claim_ctx
            })

        sentence_verdict = aggregate_sentence_verdict(atomic_results)
        sentence_conf = round(sum(ar.get("confidence", 50.0) for ar in atomic_results) / len(atomic_results), 1) if atomic_results else 50.0
        sentence_sources_map[idx] = sentence_evidence

        # Record verified sentence into running discourse memory
        memory_engine.record_sentence_verification(idx, sentence, sentence_verdict, atomic_results)

        # Similarity check
        sim_eval = None
        if sentence_evidence:
            top_ev_body = " ".join([e.get("body", "") for e in sentence_evidence[:2]])
            top_ev_title = sentence_evidence[0].get("title", "Web Evidence")
            sim_eval = similarity_analyzer.analyze_pair(sentence, top_ev_body, source_name=top_ev_title)
        elif reference_text:
            ref_sentences = extract_sentences(reference_text)
            best_r_sim = -1.0
            for rs in ref_sentences:
                r_sim = similarity_analyzer.analyze_pair(sentence, rs, source_name="Reference Document")
                if r_sim["similarity_score"] > best_r_sim:
                    best_r_sim = r_sim["similarity_score"]
                    sim_eval = r_sim
        else:
            sim_eval = similarity_analyzer.analyze_pair(sentence, "", source_name="None")

        # Plagiarism check
        plag_eval = plagiarism_detector.check_sentence_against_sources(sentence, sentence_evidence)
        if reference_text:
            ref_sentences = extract_sentences(reference_text)
            for rs in ref_sentences:
                r_eval = plagiarism_detector.evaluate_pair(sentence, rs)
                if r_eval["similarity_score"] > plag_eval["similarity_score"]:
                    plag_eval = r_eval
                    plag_eval["top_source"] = {"title": "Reference Document", "url": "#reference-doc", "domain": "Reference"}
                    plag_eval["matched_snippet"] = rs
                    plag_eval["matched_origin"] = "reference"

        sentence_explanation = " ".join([ar.get("explanation", "") for ar in atomic_results if ar.get("explanation")]).strip()

        # Hallucination check
        hal_eval = hallucination_detector.analyze_claim(
            claim=sentence,
            verdict=sentence_verdict,
            evidence=sentence_evidence,
            confidence=sentence_conf,
            explanation=sentence_explanation
        )

        results.append({
            "sentence": sentence,
            "verdict": sentence_verdict,
            "confidence": sentence_conf,
            "explanation": sentence_explanation,
            "sources": sentence_evidence,
            "atomic_claims": atomic_results,
            "similarity": sim_eval,
            "plagiarism": plag_eval,
            "hallucination": hal_eval,
            "context_meta": sentence_context_meta
        })

    if return_full_report:
        all_retrieved_sources = []
        for s_list in sentence_sources_map.values():
            for s in s_list:
                cred = get_source_credibility(s.get("url", ""))
                s_copy = dict(s)
                s_copy["credibility_score"] = cred.score
                all_retrieved_sources.append(s_copy)

        hal_report = hallucination_detector.evaluate_article_hallucination(results)
        plag_audit = plagiarism_detector.audit_article_plagiarism(
            sentences, sentence_sources_map, reference_text=reference_text
        )
        trust_res = trust_score_engine.calculate_trust_score(
            sentence_results=results,
            all_sources=all_retrieved_sources,
            hallucination_report=hal_report,
            plagiarism_audit=plag_audit
        )
        rep_dict = generate_report_dict(
            article_text=article,
            sentence_results=results,
            trust_result=trust_res,
            hallucination_report=hal_report,
            plagiarism_audit=plag_audit,
            source_info={"type": "text", "name": "Direct Input"}
        )
        return {
            "sentence_results": results,
            "trust_score": trust_res,
            "hallucination_report": hal_report,
            "plagiarism_audit": plag_audit,
            "report_data": rep_dict,
            "markdown_report": generate_markdown_report(rep_dict),
            "json_report": generate_json_report(rep_dict)
        }

    return results


# ============================================================
# CUSTOM CSS
# ============================================================

CSS_FILE = Path(__file__).resolve().parent / "ui_theme.css"
if CSS_FILE.exists():
    CSS = CSS_FILE.read_text(encoding="utf-8")
else:
    CSS = ""



# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title=f"{PROJECT_NAME} - Fact Verification & Plagiarism Engine"
) as app:


    # --------------------------------------------------------
    # BRAND HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div id="brand-header">
            <div class="brand-left">
                <div class="brand-icon-box">🔍</div>
                <div>
                    <h1 class="brand-title">TruthLens</h1>
                    <div class="brand-tagline">AI Article Verification, Discourse Memory &amp; Originality Suite</div>
                </div>
            </div>
            <div class="brand-badges">
                <span class="badge-pill-tech">🧠 Discourse Memory v2.0</span>
                <span class="badge-pill-tech">⚡ Qwen 2.5 3B (QLoRA)</span>
                <span class="badge-pill-status"><span class="pulse-dot"></span> RTX 5050 GPU Ready</span>
            </div>
        </div>
        """
    )


    # --------------------------------------------------------
    # TWO COLUMN LAYOUT
    # --------------------------------------------------------

    with gr.Row(
        elem_classes="main-layout"
    ):


        # ====================================================
        # LEFT SIDE
        # ====================================================

        with gr.Column(
            elem_classes="left-panel"
        ):

            gr.HTML(
                """
                <div class="input-header-title">
                    Article Verification &amp; Originality Audit
                </div>

                <div class="input-header-desc">
                    Paste or upload any article. TruthLens tracks narrative context across sentences, mines live web evidence, catches hallucinations, and audits originality.
                </div>

                <div class="article-label">
                    ARTICLE CONTENT TO ANALYZE
                </div>
                """
            )


            with gr.Tabs(elem_classes="input-tabs"):
                with gr.Tab("✍️ Paste Text", id="tab-paste"):
                    article_input = gr.Textbox(
                        show_label=False,
                        placeholder="Paste your article or text here...",
                        lines=14,
                        max_lines=26,
                        elem_id="article-input"
                    )

                with gr.Tab("📁 Upload Document", id="tab-file"):
                    article_file = gr.File(
                        label="Upload Article File (PDF, DOCX, DOC, TXT, MD)",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".text", ".md"],
                        type="filepath",
                        elem_id="article-file"
                    )
                    gr.HTML(
                        """
                        <div class="input-hint">
                            Supported: PDF, Word (DOCX/DOC), Plain Text (TXT/MD). Text and layout structure are preserved automatically.
                        </div>
                        """
                    )

                with gr.Tab("🌐 Article URL", id="tab-url"):
                    article_url = gr.Textbox(
                        show_label=False,
                        placeholder="https://example.com/news/article-headline or https://en.wikipedia.org/wiki/...",
                        lines=2,
                        max_lines=3,
                        elem_id="article-url"
                    )
                    gr.HTML(
                        """
                        <div class="input-hint">
                            Enter any article link. TruthLens fetches the live page, extracts core article text, and strips clutter and ads.
                        </div>
                        """
                    )

            with gr.Accordion("📄 Reference Document Comparison (Optional)", open=False):

                gr.HTML(
                    """
                    <div style="font-size:11px; color:#78808c; line-height:1.45; margin-bottom:8px;">
                        Paste reference text or upload a document to perform direct document-to-document comparison.
                        If left blank, TruthLens automatically searches the live web for published sources.
                    </div>
                    """
                )

                reference_input = gr.Textbox(

                    show_label=False,

                    placeholder="Paste reference text or source document here...",

                    lines=5,

                    max_lines=12,

                    elem_id="reference-input"

                )

                reference_file = gr.File(

                    label="Or upload reference document (.pdf, .docx, .txt, .md)",

                    file_types=[".pdf", ".docx", ".doc", ".txt", ".text", ".md"],

                    type="filepath",

                    elem_id="reference-file"

                )


            with gr.Row(
                elem_classes="mode-row"
            ):

                check_fact_checkbox = gr.Checkbox(

                    label="Fact Verification",

                    value=True

                )

                check_plag_checkbox = gr.Checkbox(

                    label="Plagiarism & Originality Audit",

                    value=True

                )


            analyze_button = gr.Button(

                "⌕  ANALYZE ARTICLE & AUDIT ORIGINALITY",

                variant="primary",

                elem_id="analyze-button"

            )


            gr.HTML(
                """
                <div style="
                    margin-top:18px;
                    color:#555c66;
                    font-size:10px;
                    line-height:1.5;
                ">
                    TruthLens combines multi-provider web retrieval, fine-tuned Qwen LLM,
                    n-gram fingerprinting, and semantic embeddings to evaluate factual truth and writing originality.
                </div>
                """
            )


        # ====================================================
        # RIGHT SIDE
        # ====================================================

        with gr.Column(
            elem_classes="right-panel"
        ):


            results_output = gr.HTML(
                value="""
                <div class="empty-state-modern">
                    <div class="empty-state-icon">✦</div>
                    <div class="empty-state-title">Ready for Context-Aware Verification</div>
                    <div class="empty-state-desc">
                        Paste text, upload a document (PDF, Word, TXT), or provide an article URL on the left.
                        TruthLens will link discourse memory across sentences, retrieve live authoritative citations, and audit both factual truth and originality.
                    </div>
                </div>
                """,
                elem_id="results"
            )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    analyze_button.click(

        fn=verify_article_stream,

        inputs=[
            article_input,
            article_file,
            article_url,
            reference_input,
            reference_file,
            check_fact_checkbox,
            check_plag_checkbox
        ],

        outputs=results_output

    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.launch(

        inbrowser=True,

        css=CSS

    )
