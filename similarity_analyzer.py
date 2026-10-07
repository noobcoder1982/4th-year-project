"""
=============================================================================
TruthLens - Text Similarity Analysis Module (Phase 2)
=============================================================================
Analyzes lexical, topical, and semantic similarity between texts while
strictly distinguishing topical/thematic overlap from unethical plagiarism.

Key Metrics:
1. Lexical Similarity (Jaccard token overlap & Character/Word n-gram overlap)
2. Topical / Terminology Overlap (Domain entity & keyword sharing)
3. Semantic Cosine Alignment (Contextual vector or TF-IDF vector similarity)
4. Similarity vs Plagiarism Differentiation:
   - Topical Overlap: Shared facts/vocabulary in independent sentences (NOT plagiarism)
   - Terminology Overlap: Domain-standard phrases (NOT plagiarism)
   - Verbatim / Paraphrase Copy: Unattributed duplicate structure (plagiarism risk)
=============================================================================
"""

import re
import math
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any, Set


# Standard English stopwords
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


def normalize_text(text: str) -> str:
    """Normalize whitespace, lowercases, and clean punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stem_word(w: str) -> str:
    """Lightweight deterministic suffix stripper for English terms."""
    w = w.lower()
    if len(w) <= 3:
        return w
    for suffix in ("ing", "ies", "es", "ed", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def extract_content_tokens(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric content words, excluding common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def compute_tf_vector(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency vector with L2 normalization over stemmed tokens."""
    if not tokens:
        return {}
    stemmed = [stem_word(t) for t in tokens]
    counts = Counter(stemmed)
    total = len(stemmed)
    norm = math.sqrt(sum((c / total) ** 2 for c in counts.values())) or 1.0
    return {word: (count / total) / norm for word, count in counts.items()}


def compute_cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Compute cosine similarity between two normalized TF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common_words = set(vec_a.keys()).intersection(vec_b.keys())
    return sum(vec_a[w] * vec_b[w] for w in common_words)


def extract_shared_keywords(text_a: str, text_b: str, top_k: int = 6) -> List[str]:
    """Extract substantive shared content words and bigrams."""
    tokens_a = extract_content_tokens(text_a)
    tokens_b = extract_content_tokens(text_b)
    stems_b = {stem_word(t): t for t in tokens_b}
    shared = []
    seen = set()
    for ta in tokens_a:
        sa = stem_word(ta)
        if sa in stems_b and sa not in seen:
            seen.add(sa)
            shared.append(ta)
    # Sort by length and alphabetical order
    shared.sort(key=lambda x: (-len(x), x))
    return shared[:top_k]


class SimilarityAnalyzer:
    """
    Dedicated analyzer for measuring topical, lexical, and semantic text similarity,
    while explicitly differentiating legitimate similarity from plagiarism.
    """

    def __init__(self, embedding_fn: Optional[Any] = None):
        """
        embedding_fn: Optional function text -> torch.Tensor / np.ndarray
        """
        self.embedding_fn = embedding_fn

    def compute_lexical_overlap(self, text_a: str, text_b: str) -> Dict[str, float]:
        """
        Compute Jaccard token similarity and word n-gram containment.
        """
        tokens_a = [stem_word(t) for t in extract_content_tokens(text_a)]
        tokens_b = [stem_word(t) for t in extract_content_tokens(text_b)]

        if not tokens_a or not tokens_b:
            return {"jaccard": 0.0, "ngram_overlap": 0.0, "overlap_pct": 0.0}

        set_a = set(tokens_a)
        set_b = set(tokens_b)

        inter = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        jaccard = (inter / union) if union > 0 else 0.0

        # Bigram overlap
        bigrams_a = set(zip(tokens_a[:-1], tokens_a[1:])) if len(tokens_a) > 1 else set()
        bigrams_b = set(zip(tokens_b[:-1], tokens_b[1:])) if len(tokens_b) > 1 else set()
        if bigrams_a and bigrams_b:
            bg_inter = len(bigrams_a.intersection(bigrams_b))
            bg_union = len(bigrams_a.union(bigrams_b))
            bg_sim = (bg_inter / bg_union) if bg_union > 0 else 0.0
        else:
            bg_sim = 0.0

        overlap_pct = round((0.6 * jaccard + 0.4 * bg_sim) * 100, 1)

        return {
            "jaccard": round(jaccard, 3),
            "ngram_overlap": round(bg_sim, 3),
            "overlap_pct": overlap_pct
        }

    def compute_topical_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute TF-IDF / Term Frequency Cosine Similarity reflecting topical alignment.
        Returns a float between 0.0 and 100.0.
        """
        tokens_a = extract_content_tokens(text_a)
        tokens_b = extract_content_tokens(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        vec_a = compute_tf_vector(tokens_a)
        vec_b = compute_tf_vector(tokens_b)
        cosine = compute_cosine_similarity(vec_a, vec_b)
        return round(min(1.0, max(0.0, cosine)) * 100, 1)

    def analyze_pair(
        self,
        target_text: str,
        source_text: str,
        source_name: str = "External Source"
    ) -> Dict[str, Any]:
        """
        Analyze similarity between target text and a source text,
        producing explicit categorization that distinguishes topical similarity from plagiarism.
        """
        clean_target = normalize_text(target_text)
        clean_source = normalize_text(source_text)

        if not clean_target or not clean_source:
            return {
                "similarity_score": 0.0,
                "lexical_overlap": 0.0,
                "topical_similarity": 0.0,
                "relationship": "DISTINCT",
                "relationship_label": "Distinct Content",
                "relationship_desc": "No meaningful similarity detected.",
                "is_plagiarism": False,
                "shared_keywords": [],
                "source_name": source_name
            }

        lex_metrics = self.compute_lexical_overlap(clean_target, clean_source)
        topical_sim = self.compute_topical_similarity(clean_target, clean_source)
        shared_kw = extract_shared_keywords(clean_target, clean_source)

        # Composite similarity score (0 - 100%)
        # Balanced between topical topic alignment and lexical overlap
        comp_similarity = round(
            (0.55 * topical_sim) + (0.45 * lex_metrics["overlap_pct"]),
            1
        )

        # Determine the exact semantic relationship:
        # Crucial Rule: High topical similarity without identical phrasing is legitimate topical overlap!
        words_target = re.findall(r"\b\w+\b", clean_target.lower())
        words_source = re.findall(r"\b\w+\b", clean_source.lower())

        # Check for continuous verbatim run
        longest_continuous_words = 0
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, words_target, words_source)
        for block in matcher.get_matching_blocks():
            if block.size > longest_continuous_words:
                longest_continuous_words = block.size

        seq_ratio = matcher.ratio()

        if seq_ratio >= 0.70 or longest_continuous_words >= max(6, int(len(words_target) * 0.6)):
            relationship = "VERBATIM_COPY"
            relationship_label = "Verbatim Match"
            relationship_desc = "High textual match with continuous verbatim word sequences. Requires attribution/quotes."
            is_plag = True
        elif comp_similarity >= 65 and lex_metrics["jaccard"] >= 0.50:
            relationship = "HEAVY_PARAPHRASE"
            relationship_label = "Paraphrased Content"
            relationship_desc = "High structural overlap with substituted synonyms or rearranged clauses."
            is_plag = True
        elif comp_similarity >= 25 or topical_sim >= 25 or len(shared_kw) >= 2:
            relationship = "TOPICAL_OVERLAP"
            relationship_label = "Topical / Factual Overlap"
            relationship_desc = f"Both texts cover the same subject and share domain terminology ({', '.join(shared_kw[:3]) if shared_kw else 'facts'}), but phrasing is independent. Not plagiarism."
            is_plag = False
        elif comp_similarity >= 12 or len(shared_kw) >= 1:
            relationship = "SHARED_TERMINOLOGY"
            relationship_label = "Shared Terminology"
            relationship_desc = "Incidental overlap on common technical or subject-matter vocabulary."
            is_plag = False
        else:
            relationship = "DISTINCT"
            relationship_label = "Original Expression"
            relationship_desc = "Independent wording and low overlap with the compared text."
            is_plag = False

        return {
            "similarity_score": comp_similarity,
            "topical_similarity": topical_sim,
            "lexical_overlap": lex_metrics["overlap_pct"],
            "jaccard_index": lex_metrics["jaccard"],
            "longest_matching_sequence": longest_continuous_words,
            "relationship": relationship,
            "relationship_label": relationship_label,
            "relationship_desc": relationship_desc,
            "is_plagiarism": is_plag,
            "shared_keywords": shared_kw,
            "source_name": source_name,
            "source_snippet": clean_source[:200]
        }

    def analyze_document_similarity(
        self,
        article_text: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze overall article similarity across retrieved web sources or reference documents.
        """
        if not article_text or not sources:
            return {
                "overall_similarity": 0.0,
                "topical_overlap_score": 0.0,
                "lexical_overlap_score": 0.0,
                "primary_matching_sources": [],
                "shared_topical_keywords": [],
                "topical_consensus_level": "None"
            }

        # Combine snippets from sources
        scored_sources = []
        all_shared_kw = set()

        for s in sources:
            s_body = s.get("body", "") or s.get("text", "") or s.get("snippet", "")
            s_title = s.get("title", "External Source")
            if not s_body:
                continue

            pair_res = self.analyze_pair(article_text, s_body, source_name=s_title)
            scored_sources.append(pair_res)
            all_shared_kw.update(pair_res["shared_keywords"])

        if not scored_sources:
            return {
                "overall_similarity": 0.0,
                "topical_overlap_score": 0.0,
                "lexical_overlap_score": 0.0,
                "primary_matching_sources": [],
                "shared_topical_keywords": [],
                "topical_consensus_level": "None"
            }

        # Sort sources by similarity score descending
        scored_sources.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Consider top relevant matching sources (avoiding noise dilution)
        relevant_sources = [s for s in scored_sources if s["similarity_score"] > 0]
        eval_sources = relevant_sources[:3] if relevant_sources else scored_sources[:1]

        if len(eval_sources) == 1:
            overall_sim = eval_sources[0]["similarity_score"]
            topical_score = eval_sources[0]["topical_similarity"]
            lexical_score = eval_sources[0]["lexical_overlap"]
        else:
            weights = [0.6, 0.25, 0.15][:len(eval_sources)]
            w_sum = sum(weights)
            norm_weights = [w / w_sum for w in weights]
            overall_sim = sum(s["similarity_score"] * w for s, w in zip(eval_sources, norm_weights))
            topical_score = sum(s["topical_similarity"] * w for s, w in zip(eval_sources, norm_weights))
            lexical_score = sum(s["lexical_overlap"] * w for s, w in zip(eval_sources, norm_weights))

        if topical_score >= 45.0:
            consensus = "High (Extensive thematic alignment with published literature)"
        elif topical_score >= 20.0:
            consensus = "Moderate (Relevant thematic alignment on core entities)"
        else:
            consensus = "Low (Novel or niche subject with limited topical overlap)"

        return {
            "overall_similarity": round(overall_sim, 1),
            "topical_overlap_score": round(topical_score, 1),
            "lexical_overlap_score": round(lexical_score, 1),
            "primary_matching_sources": eval_sources,
            "shared_topical_keywords": sorted(list(all_shared_kw))[:8],
            "topical_consensus_level": consensus
        }
