"""
=============================================================================
TruthLens - Advanced Hybrid Plagiarism Detection Engine
=============================================================================
Combines:
1. Exact N-Gram Shingling & Winnowing (Verbatim Copy Detection)
2. Lexical & Syntactic Similarity (SequenceMatcher Gestalt Pattern + Jaccard)
3. Semantic & Paraphrase Similarity (Qwen Contextual Embedding Cosine Similarity)
4. Dual Mode:
   - Web-Based Plagiarism Search (DuckDuckGo search + snippet alignment)
   - Document-to-Document Comparison (Side-by-side reference comparison)
=============================================================================
"""

import re
import difflib
import unicodedata
from typing import List, Dict, Tuple, Optional, Any
import torch
import numpy as np


# Common English Stopwords for normalized shingling
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


# =============================================================================
# TEXT NORMALIZATION & PREPROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Normalize whitespace and unicode characters."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str, keep_case: bool = False) -> List[str]:
    """Tokenize text into lowercase alphanumeric word tokens."""
    if not keep_case:
        text = text.lower()
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text)


def extract_sentences(text: str) -> List[str]:
    """Extract distinct sentences of sufficient length from an article."""
    text = clean_text(text)
    if not text:
        return []

    # Split on sentence terminals while keeping abbreviations relatively intact
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        words = s.split()
        if len(words) >= 4:
            sentences.append(s)
    return sentences if sentences else [text]


def get_word_ngrams(tokens: List[str], n: int = 4) -> set:
    """Return a set of word n-grams (shingles)."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def get_char_ngrams(text: str, n: int = 5) -> set:
    """Return character n-grams for typo-tolerant matching."""
    text_clean = re.sub(r"\s+", " ", text.lower().strip())
    if len(text_clean) < n:
        return {text_clean} if text_clean else set()
    return {text_clean[i:i + n] for i in range(len(text_clean) - n + 1)}


# =============================================================================
# LEXICAL & STRING-MATCHING METRICS
# =============================================================================

def compute_ngram_containment(suspect_text: str, source_text: str, n: int = 4) -> float:
    """
    Compute word n-gram containment:
    Containment = |Ngrams(suspect) ∩ Ngrams(source)| / |Ngrams(suspect)|
    """
    s_tokens = tokenize_words(suspect_text)
    src_tokens = tokenize_words(source_text)

    if not s_tokens:
        return 0.0

    s_ngrams = get_word_ngrams(s_tokens, n=n)
    src_ngrams = get_word_ngrams(src_tokens, n=n)

    if not s_ngrams:
        return 0.0

    intersection = s_ngrams.intersection(src_ngrams)
    return len(intersection) / len(s_ngrams)


def compute_jaccard_similarity(text_a: str, text_b: str, remove_stopwords: bool = True) -> float:
    """Compute Jaccard similarity over word tokens."""
    tokens_a = set(tokenize_words(text_a))
    tokens_b = set(tokenize_words(text_b))

    if remove_stopwords:
        tokens_a = {t for t in tokens_a if t not in STOPWORDS}
        tokens_b = {t for t in tokens_b if t not in STOPWORDS}

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


def compute_sequence_matcher_ratio(text_a: str, text_b: str) -> Tuple[float, int]:
    """
    Compute Gestalt pattern matching ratio using difflib.SequenceMatcher.
    Returns (ratio, length of longest common matching block in words).
    """
    tokens_a = tokenize_words(text_a)
    tokens_b = tokenize_words(text_b)

    if not tokens_a or not tokens_b:
        return 0.0, 0

    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    ratio = matcher.ratio()

    # Find longest continuous match
    longest_block = 0
    for block in matcher.get_matching_blocks():
        if block.size > longest_block:
            longest_block = block.size

    return ratio, longest_block


def highlight_matched_segments(suspect_text: str, source_text: str) -> str:
    """
    Highlight overlapping word sequences in suspect_text using HTML <mark>.
    Finds contiguous matching blocks of 3+ words.
    """
    s_words = suspect_text.split()
    src_words = source_text.split()

    s_lower = [re.sub(r'[^\w]', '', w.lower()) for w in s_words]
    src_lower = [re.sub(r'[^\w]', '', w.lower()) for w in src_words]

    matcher = difflib.SequenceMatcher(None, s_lower, src_lower)
    matching_blocks = [b for b in matcher.get_matching_blocks() if b.size >= 2]

    # Mark suspect words that belong to matching blocks
    matched_indices = set()
    for block in matching_blocks:
        for i in range(block.a, block.a + block.size):
            matched_indices.add(i)

    # Build highlighted text
    result_words = []
    in_mark = False

    for idx, word in enumerate(s_words):
        is_matched = idx in matched_indices
        if is_matched and not in_mark:
            result_words.append('<mark class="plag-highlight">')
            in_mark = True
        elif not is_matched and in_mark:
            result_words.append('</mark>')
            in_mark = False
        result_words.append(word)

    if in_mark:
        result_words.append('</mark>')

    return " ".join(result_words)


# =============================================================================
# HYBRID PLAGIARISM DETECTOR CLASS
# =============================================================================

class PlagiarismDetector:
    """
    State-of-the-art hybrid plagiarism detector combining N-gram fingerprinting,
    Gestalt sequence matching, and deep contextual LLM embeddings (Qwen).
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_cache: Dict[str, torch.Tensor] = {}

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Extract mean-pooled sentence embedding from Qwen LLM.
        Caches embeddings for efficiency.
        """
        if self.model is None or self.tokenizer is None:
            return None

        clean = clean_text(text)
        if not clean:
            return None

        if clean in self.embedding_cache:
            return self.embedding_cache[clean]

        try:
            device = self.model.device
            inputs = self.tokenizer(
                clean,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                # Mean pool over attention mask
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                sum_hidden = torch.sum(hidden * mask, 1)
                counts = torch.clamp(mask.sum(1), min=1e-9)
                embedding = sum_hidden / counts
                norm_emb = torch.nn.functional.normalize(embedding, p=2, dim=1)

            # Store on CPU to conserve GPU VRAM
            emb_cpu = norm_emb.cpu()
            self.embedding_cache[clean] = emb_cpu
            return emb_cpu
        except Exception as e:
            # Fallback if VRAM or generation exception occurs
            return None

    def compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between sentence embeddings."""
        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)

        if emb_a is not None and emb_b is not None:
            sim = torch.mm(emb_a, emb_b.T).item()
            return max(0.0, min(1.0, float(sim)))

        # Fallback pseudo-semantic similarity using character 4-gram Jaccard
        char_a = get_char_ngrams(text_a, n=4)
        char_b = get_char_ngrams(text_b, n=4)
        if not char_a or not char_b:
            return 0.0
        return len(char_a.intersection(char_b)) / len(char_a.union(char_b))

    def evaluate_pair(self, suspect_text: str, source_text: str) -> Dict[str, Any]:
        """
        Evaluate plagiarism between a suspect sentence and a candidate source text.
        Returns detailed scores, classification, and highlighted text.
        """
        clean_suspect = clean_text(suspect_text)
        clean_source = clean_text(source_text)

        if not clean_suspect or not clean_source:
            return {
                "similarity_score": 0.0,
                "exact_score": 0.0,
                "lexical_score": 0.0,
                "semantic_score": 0.0,
                "category": "ORIGINAL",
                "label": "Original",
                "badge_class": "badge-original",
                "longest_match_words": 0,
                "highlighted_text": clean_suspect,
                "source_text": clean_source
            }

        # 1. Exact N-Gram Containment
        ngram_3 = compute_ngram_containment(clean_suspect, clean_source, n=3)
        ngram_4 = compute_ngram_containment(clean_suspect, clean_source, n=4)
        ngram_5 = compute_ngram_containment(clean_suspect, clean_source, n=5)
        exact_score = (0.2 * ngram_3) + (0.5 * ngram_4) + (0.3 * ngram_5)

        # 2. SequenceMatcher and Longest Continuous Block
        seq_ratio, longest_block = compute_sequence_matcher_ratio(clean_suspect, clean_source)
        jaccard_score = compute_jaccard_similarity(clean_suspect, clean_source)
        lexical_score = (0.6 * seq_ratio) + (0.4 * jaccard_score)

        # 3. Semantic / Embedding Cosine Similarity
        semantic_score = self.compute_semantic_similarity(clean_suspect, clean_source)

        # 4. Multi-Layer Classification
        suspect_len = len(tokenize_words(clean_suspect))

        # Direct / Verbatim Match
        # High n-gram overlap OR long continuous verbatim segment
        is_verbatim = (
            exact_score >= 0.60
            or seq_ratio >= 0.75
            or (suspect_len >= 5 and longest_block >= min(7, suspect_len - 1))
        )

        # Heavy Paraphrasing: High semantic similarity with moderate lexical overlap
        is_paraphrase = (
            not is_verbatim
            and (
                (semantic_score >= 0.84 and (lexical_score >= 0.28 or exact_score >= 0.20))
                or (semantic_score >= 0.90 and lexical_score >= 0.20)
                or (seq_ratio >= 0.50 and semantic_score >= 0.75)
            )
        )

        # Light Paraphrase / Moderate Similarity
        is_light = (
            not is_verbatim
            and not is_paraphrase
            and (
                (semantic_score >= 0.74 and lexical_score >= 0.25)
                or (lexical_score >= 0.40)
                or (exact_score >= 0.30)
            )
        )

        # Compute composite calibrated score (0 - 100%)
        if is_verbatim:
            category = "VERBATIM"
            label = "Verbatim Match"
            badge_class = "badge-verbatim"
            # High score 80-100%
            raw = max(exact_score, seq_ratio)
            score = 80.0 + (raw * 20.0)
        elif is_paraphrase:
            category = "PARAPHRASED"
            label = "Paraphrased"
            badge_class = "badge-paraphrase"
            # Score 60-85%
            score = 55.0 + (0.4 * lexical_score + 0.6 * semantic_score) * 35.0
        elif is_light:
            category = "LIGHT_PARAPHRASE"
            label = "Partial Similarity"
            badge_class = "badge-partial"
            # Score 35-59%
            score = 35.0 + (0.5 * lexical_score + 0.5 * semantic_score) * 24.0
        else:
            category = "ORIGINAL"
            label = "Original Content"
            badge_class = "badge-original"
            # Score 0-34%
            score = (0.4 * lexical_score + 0.6 * max(0.0, semantic_score - 0.4)) * 34.0

        score = max(0.0, min(100.0, round(score, 1)))

        # Highlight matched segments
        highlighted = highlight_matched_segments(clean_suspect, clean_source)

        return {
            "similarity_score": score,
            "exact_score": round(exact_score * 100, 1),
            "lexical_score": round(lexical_score * 100, 1),
            "semantic_score": round(semantic_score * 100, 1),
            "category": category,
            "label": label,
            "badge_class": badge_class,
            "longest_match_words": longest_block,
            "highlighted_text": highlighted,
            "source_text": clean_source
        }

    # =========================================================================
    # DOCUMENT-TO-DOCUMENT COMPARISON MODE
    # =========================================================================

    def compare_documents(self, suspect_doc: str, reference_doc: str) -> Dict[str, Any]:
        """
        Compare suspect document against a reference document side-by-side.
        Returns document similarity %, sentence alignments, and match breakdowns.
        """
        suspect_sentences = extract_sentences(suspect_doc)
        ref_sentences = extract_sentences(reference_doc)

        if not suspect_sentences or not ref_sentences:
            return {
                "overall_similarity": 0.0,
                "originality_score": 100.0,
                "verbatim_percentage": 0.0,
                "paraphrase_percentage": 0.0,
                "original_percentage": 100.0,
                "sentence_matches": [],
                "verbatim_count": 0,
                "paraphrase_count": 0,
                "original_count": len(suspect_sentences),
                "total_sentences": len(suspect_sentences)
            }

        sentence_matches = []
        verbatim_count = 0
        paraphrase_count = 0
        original_count = 0

        # For each suspect sentence, find the closest reference sentence
        for s_idx, s_sent in enumerate(suspect_sentences, 1):
            best_match = None
            best_score = -1.0

            for r_idx, r_sent in enumerate(ref_sentences, 1):
                eval_res = self.evaluate_pair(s_sent, r_sent)
                if eval_res["similarity_score"] > best_score:
                    best_score = eval_res["similarity_score"]
                    best_match = eval_res
                    best_match["ref_sentence"] = r_sent
                    best_match["ref_index"] = r_idx

            if best_match is None:
                best_match = {
                    "similarity_score": 0.0,
                    "category": "ORIGINAL",
                    "label": "Original Content",
                    "badge_class": "badge-original",
                    "ref_sentence": "",
                    "ref_index": 0,
                    "highlighted_text": s_sent,
                    "source_text": ""
                }

            best_match["suspect_sentence"] = s_sent
            best_match["suspect_index"] = s_idx

            if best_match["category"] == "VERBATIM":
                verbatim_count += 1
            elif best_match["category"] in ("PARAPHRASED", "LIGHT_PARAPHRASE"):
                paraphrase_count += 1
            else:
                original_count += 1

            sentence_matches.append(best_match)

        total_s = len(suspect_sentences)
        avg_sim = sum(m["similarity_score"] for m in sentence_matches) / total_s
        verbatim_pct = round((verbatim_count / total_s) * 100, 1)
        paraphrase_pct = round((paraphrase_count / total_s) * 100, 1)
        original_pct = round((original_count / total_s) * 100, 1)
        overall_sim = round(avg_sim, 1)
        orig_score = max(0.0, round(100.0 - overall_sim, 1))

        return {
            "overall_similarity": overall_sim,
            "originality_score": orig_score,
            "verbatim_percentage": verbatim_pct,
            "paraphrase_percentage": paraphrase_pct,
            "original_percentage": original_pct,
            "sentence_matches": sentence_matches,
            "verbatim_count": verbatim_count,
            "paraphrase_count": paraphrase_count,
            "original_count": original_count,
            "total_sentences": total_s
        }

    # =========================================================================
    # WEB-BASED PLAGIARISM CHECK
    # =========================================================================

    def check_sentence_against_sources(
        self,
        sentence: str,
        sources: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Check a sentence against a list of retrieved web source dicts (title, body, url, domain).
        Finds the highest-matching web source and evaluates plagiarism.
        """
        if not sources:
            return {
                "similarity_score": 0.0,
                "category": "ORIGINAL",
                "label": "Original Content",
                "badge_class": "badge-original",
                "top_source": None,
                "highlighted_text": sentence,
                "matched_snippet": ""
            }

        best_score = -1.0
        best_eval = None
        best_source = None
        best_snippet = ""

        for src in sources:
            body = src.get("body", "")
            title = src.get("title", "")
            combined_candidate = f"{title}. {body}"

            # Evaluate sentence against full candidate snippet
            eval_res = self.evaluate_pair(sentence, combined_candidate)

            # Also check sentence-by-sentence inside candidate body for tighter match
            body_sentences = extract_sentences(body)
            for bs in body_sentences:
                sub_eval = self.evaluate_pair(sentence, bs)
                if sub_eval["similarity_score"] > eval_res["similarity_score"]:
                    eval_res = sub_eval
                    combined_candidate = bs

            if eval_res["similarity_score"] > best_score:
                best_score = eval_res["similarity_score"]
                best_eval = eval_res
                best_source = src
                best_snippet = combined_candidate

        if best_eval is None or best_score < 35.0:
            return {
                "similarity_score": round(max(0.0, best_score if best_score > 0 else 0.0), 1),
                "category": "ORIGINAL",
                "label": "Original Content",
                "badge_class": "badge-original",
                "top_source": best_source if best_score >= 25.0 else None,
                "highlighted_text": sentence,
                "matched_snippet": best_snippet if best_score >= 25.0 else ""
            }

        best_eval["top_source"] = best_source
        best_eval["matched_snippet"] = best_snippet
        return best_eval

    def audit_article_plagiarism(
        self,
        sentences: List[str],
        sentence_sources_map: Dict[int, List[Dict[str, str]]],
        reference_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete plagiarism audit for an article.
        Combines Web Sources for each sentence with an optional Reference Document.
        """
        if not sentences:
            return {
                "overall_similarity": 0.0,
                "originality_score": 100.0,
                "verbatim_percentage": 0.0,
                "paraphrase_percentage": 0.0,
                "original_percentage": 100.0,
                "sentence_audits": [],
                "top_sources": [],
                "has_reference_comparison": False
            }

        # If a reference document is provided, prepare reference sentences
        ref_sentences = extract_sentences(reference_text) if reference_text and reference_text.strip() else []

        sentence_audits = []
        verbatim_count = 0
        paraphrase_count = 0
        original_count = 0

        source_frequency: Dict[str, Dict[str, Any]] = {}

        for idx, sentence in enumerate(sentences, 1):
            web_sources = sentence_sources_map.get(idx, [])

            # Check against web sources
            web_res = self.check_sentence_against_sources(sentence, web_sources)

            # Check against reference document (if provided)
            ref_match = None
            if ref_sentences:
                best_r_score = -1.0
                for r_sent in ref_sentences:
                    r_eval = self.evaluate_pair(sentence, r_sent)
                    if r_eval["similarity_score"] > best_r_score:
                        best_r_score = r_eval["similarity_score"]
                        ref_match = r_eval
                        ref_match["ref_sentence"] = r_sent

            # Pick the higher match between Web Source and Reference Document
            final_audit = web_res
            matched_origin = "web"

            if ref_match and ref_match["similarity_score"] > web_res["similarity_score"]:
                final_audit = ref_match
                final_audit["top_source"] = {
                    "title": "Reference Document",
                    "url": "#reference-doc",
                    "domain": "Uploaded Reference",
                    "body": ref_match.get("ref_sentence", "")
                }
                final_audit["matched_snippet"] = ref_match.get("ref_sentence", "")
                matched_origin = "reference"

            final_audit["sentence"] = sentence
            final_audit["index"] = idx
            final_audit["matched_origin"] = matched_origin

            if final_audit["category"] == "VERBATIM":
                verbatim_count += 1
            elif final_audit["category"] in ("PARAPHRASED", "LIGHT_PARAPHRASE"):
                paraphrase_count += 1
            else:
                original_count += 1

            # Track top sources
            src = final_audit.get("top_source")
            if src and final_audit["category"] != "ORIGINAL":
                url = src.get("url", "")
                if url:
                    if url not in source_frequency:
                        source_frequency[url] = {
                            "title": src.get("title", url),
                            "url": url,
                            "domain": src.get("domain", ""),
                            "matched_count": 0,
                            "highest_similarity": 0.0
                        }
                    source_frequency[url]["matched_count"] += 1
                    source_frequency[url]["highest_similarity"] = max(
                        source_frequency[url]["highest_similarity"],
                        final_audit["similarity_score"]
                    )

            sentence_audits.append(final_audit)

        total_s = len(sentences)
        avg_similarity = sum(a["similarity_score"] for a in sentence_audits) / total_s
        verbatim_pct = round((verbatim_count / total_s) * 100, 1)
        paraphrase_pct = round((paraphrase_count / total_s) * 100, 1)
        original_pct = round((original_count / total_s) * 100, 1)
        overall_sim = round(avg_similarity, 1)
        orig_score = max(0.0, round(100.0 - overall_sim, 1))

        # Sort top sources by matched count then similarity
        top_sources_list = sorted(
            source_frequency.values(),
            key=lambda x: (x["matched_count"], x["highest_similarity"]),
            reverse=True
        )[:5]

        return {
            "overall_similarity": overall_sim,
            "originality_score": orig_score,
            "verbatim_percentage": verbatim_pct,
            "paraphrase_percentage": paraphrase_pct,
            "original_percentage": original_pct,
            "sentence_audits": sentence_audits,
            "verbatim_count": verbatim_count,
            "paraphrase_count": paraphrase_count,
            "original_count": original_count,
            "total_sentences": total_s,
            "top_sources": top_sources_list,
            "has_reference_comparison": bool(ref_sentences)
        }
