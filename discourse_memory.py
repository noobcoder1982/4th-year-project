"""
=============================================================================
TruthLens - Discourse Memory & Contextual Coreference Engine
=============================================================================
Solves sentence-isolation issues during verification:
1. Extracts global article topic, primary entities, and subject matter.
2. Maintains a running discourse history of processed sentences and claims.
3. Performs anaphoric coreference resolution:
   Resolves ambiguous pronouns ('It', 'He', 'She', 'They', 'This mission', etc.)
   to their real-world antecedents from preceding sentences.
4. Generates context-augmented search queries so search engines retrieve
   accurate evidence instead of failing on isolated pronouns.
5. Provides rich narrative context to Qwen so the model understands the
   relationship between preceding and subsequent statements.
6. Tracks cross-claim consistency and narrative contradictions.
=============================================================================
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set

# Pronouns and anaphoric trigger phrases that indicate dependency on prior context
PRONOUN_TRIGGERS = {
    'it', 'its', 'itself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'these', 'that', 'those'
}

ANAPHORIC_NOUN_PHRASES = [
    r'\bthe\s+(?:mission|project|agency|company|organization|study|research|experiment|telescope|spacecraft|discovery|theory|author|scientist|president|minister|system|model|algorithm|tool)\b',
    r'\bthis\s+(?:mission|project|agency|company|organization|study|research|experiment|telescope|spacecraft|discovery|theory|author|scientist|president|minister|system|model|algorithm|tool)\b',
    r'\bsuch\s+(?:findings|results|actions|claims|events|phenomena)\b',
    r'\bthese\s+(?:findings|results|scientists|researchers|events)\b',
    r'\blater\s+that\s+year\b',
    r'\bsubsequently\b',
    r'\bas\s+a\s+result\b',
    r'\bdue\s+to\s+this\b'
]

COMMON_STOPWORDS = {
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


class DiscourseMemoryEngine:
    """
    Maintains discourse context and memory across the entire article.
    Prevents verification blind-spots caused by evaluating sentences in isolation.
    """

    def __init__(self, article_text: str = ""):
        self.raw_article = article_text.strip() if article_text else ""
        self.global_topic = ""
        self.global_entities: List[str] = []
        self.discourse_history: List[Dict[str, Any]] = []
        self.active_antecedent: str = ""
        
        if self.raw_article:
            self.initialize_article_context(self.raw_article)

    def initialize_article_context(self, article_text: str) -> None:
        """
        Analyze article to extract dominant named entities, headline/topic, and key theme.
        """
        self.raw_article = article_text.strip()
        self.discourse_history = []
        
        # 1. Extract candidate proper noun entities across the text
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', self.raw_article)
        entity_counts: Dict[str, int] = {}
        for pn in proper_nouns:
            pn_clean = pn.strip()
            # Ignore single capitalized sentence-starting stopwords (e.g. "The", "However")
            if pn_clean.lower() in COMMON_STOPWORDS and len(pn_clean.split()) == 1:
                continue
            if len(pn_clean) < 3:
                continue
            entity_counts[pn_clean] = entity_counts.get(pn_clean, 0) + 1

        # Sort entities by frequency
        sorted_entities = sorted(entity_counts.items(), key=lambda x: -x[1])
        self.global_entities = [e[0] for e in sorted_entities[:8]]

        # 2. Determine global topic / title
        # First paragraph or first sentence usually defines the core topic
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', self.raw_article) if s.strip()]
        if sentences:
            first_sent = sentences[0]
            # If headline exists (short line before period), or top entities
            if self.global_entities:
                top_2 = self.global_entities[:2]
                self.global_topic = " & ".join(top_2)
            else:
                self.global_topic = " ".join(first_sent.split()[:7])
        else:
            self.global_topic = "General Subject"

        self.active_antecedent = self.global_entities[0] if self.global_entities else self.global_topic

    def detect_anaphora(self, sentence: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if a sentence begins with or relies on an anaphoric pronoun or vague noun phrase.
        Returns: (has_anaphora, trigger_phrase)
        """
        clean = sentence.strip()
        words = clean.split()
        if not words:
            return False, None

        first_word = re.sub(r'[^a-zA-Z]', '', words[0]).lower()
        if first_word in PRONOUN_TRIGGERS:
            return True, words[0]

        # Check two-word starts like "This mission", "The agency", "These findings"
        if len(words) >= 2:
            two_words = f"{words[0]} {words[1]}".lower()
            if any(re.match(pattern, two_words, re.IGNORECASE) for pattern in ANAPHORIC_NOUN_PHRASES):
                return True, two_words

        # Check full sentence for strong anaphoric noun phrases
        for pattern in ANAPHORIC_NOUN_PHRASES:
            m = re.search(pattern, clean, re.IGNORECASE)
            if m:
                return True, m.group(0)

        return False, None

    def resolve_claim_context(
        self,
        claim: str,
        sentence_index: int,
        all_sentences: List[str]
    ) -> Dict[str, Any]:
        """
        Contextualize a single claim with prior discourse memory and antecedent resolution.
        Returns context resolution metadata for search and verification.
        """
        has_anaphora, trigger = self.detect_anaphora(claim)
        
        # Immediate preceding sentence (if any)
        prev_sentence = all_sentences[sentence_index - 2] if sentence_index > 1 and len(all_sentences) >= sentence_index - 1 else ""
        
        # Resolve active subject antecedent:
        # If previous sentence exists, extract its primary subject/proper noun
        resolved_antecedent = self.active_antecedent
        if prev_sentence:
            prev_entities = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', prev_sentence)
            valid_prev = [p for p in prev_entities if p.lower() not in COMMON_STOPWORDS and len(p) > 2]
            if valid_prev:
                resolved_antecedent = valid_prev[0]
                self.active_antecedent = resolved_antecedent

        # Formulate contextual search query
        context_query = claim
        if has_anaphora and resolved_antecedent:
            # If starts with pronoun, replace or prefix with resolved entity
            words = claim.split()
            if words and re.sub(r'[^a-zA-Z]', '', words[0]).lower() in PRONOUN_TRIGGERS:
                context_query = f"{resolved_antecedent} {' '.join(words[1:])}"
            else:
                context_query = f"{resolved_antecedent} {claim}"
        elif self.global_topic and not any(e.lower() in claim.lower() for e in self.global_entities[:3]):
            # If claim is very generic, augment with main topic
            context_query = f"{self.global_topic}: {claim}"

        # Context summary for LLM prompt
        prior_claims_summary = []
        for item in self.discourse_history[-3:]:  # Last 3 claims in memory
            prior_claims_summary.append(
                f"- Sentence {item['index']}: \"{item['text'][:80]}...\" -> Verdict: {item.get('verdict', 'PENDING')}"
            )

        return {
            "has_anaphora": has_anaphora,
            "trigger": trigger,
            "resolved_antecedent": resolved_antecedent,
            "context_augmented_query": context_query,
            "global_topic": self.global_topic,
            "prev_sentence": prev_sentence,
            "prior_claims_summary": "\n".join(prior_claims_summary) if prior_claims_summary else "Initial opening statement."
        }

    def record_sentence_verification(
        self,
        sentence_index: int,
        sentence_text: str,
        verdict: str,
        atomic_results: List[Dict[str, Any]]
    ) -> None:
        """
        Record a verified sentence into running discourse memory.
        """
        # Update active antecedent if new strong proper noun introduced
        found_entities = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', sentence_text)
        valid_entities = [e for e in found_entities if e.lower() not in COMMON_STOPWORDS and len(e) > 2]
        if valid_entities:
            self.active_antecedent = valid_entities[0]

        self.discourse_history.append({
            "index": sentence_index,
            "text": sentence_text,
            "verdict": verdict,
            "atomic_results": atomic_results,
            "active_antecedent": self.active_antecedent
        })

    def format_llm_context_prompt(
        self,
        claim: str,
        context_meta: Dict[str, Any]
    ) -> str:
        """
        Formats a clean, non-intrusive discourse context block for the Qwen verification prompt.
        """
        topic = context_meta.get("global_topic", "")
        prev = context_meta.get("prev_sentence", "")
        antecedent = context_meta.get("resolved_antecedent", "")
        priors = context_meta.get("prior_claims_summary", "")

        context_lines = []
        if topic:
            context_lines.append(f"Article Theme / Primary Subject: {topic}")
        if antecedent and context_meta.get("has_anaphora"):
            context_lines.append(f"Subject / Antecedent Resolution: Pronouns in claim refer to '{antecedent}'")
        if prev:
            context_lines.append(f"Immediate Preceding Sentence: \"{prev}\"")
        if priors and priors != "Initial opening statement.":
            context_lines.append(f"Discourse Verification Memory:\n{priors}")

        if not context_lines:
            return ""

        return "\n".join(context_lines)
