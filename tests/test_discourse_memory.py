import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from discourse_memory import DiscourseMemoryEngine

def run_tests():
    print("=================================================================")
    print("TRUTHLENS DISCOURSE MEMORY & CONTEXT RESOLUTION TEST SUITE")
    print("=================================================================")

    article = (
        "NASA successfully launched the James Webb Space Telescope in December 2021. "
        "It reached the Sun-Earth L2 Lagrange point approximately 30 days later. "
        "The telescope began transmitting unprecedented deep space infrared imagery to Earth. "
        "However, critics claimed that the observatory had collided with a major asteroid."
    )

    engine = DiscourseMemoryEngine(article)

    # TEST 1: Initialization & Topic Extraction
    print("\n[TEST 1] Testing Global Topic & Entity Extraction...")
    assert len(engine.global_entities) >= 1, "Failed to extract entities"
    assert "James Webb" in " ".join(engine.global_entities) or "NASA" in " ".join(engine.global_entities)
    print(f"  [OK] Extracted Entities: {engine.global_entities}")
    print(f"  [OK] Global Topic: {engine.global_topic}")

    sentences = [
        "NASA successfully launched the James Webb Space Telescope in December 2021.",
        "It reached the Sun-Earth L2 Lagrange point approximately 30 days later.",
        "The telescope began transmitting unprecedented deep space infrared imagery to Earth.",
        "However, critics claimed that the observatory had collided with a major asteroid."
    ]

    # TEST 2: Pronoun Antecedent Resolution on Sentence 2 ("It reached...")
    print("\n[TEST 2] Testing Anaphora Detection & Coreference Resolution...")
    meta_s2 = engine.resolve_claim_context(
        claim="It reached the Sun-Earth L2 Lagrange point approximately 30 days later.",
        sentence_index=2,
        all_sentences=sentences
    )
    assert meta_s2["has_anaphora"] is True, "Failed to detect pronoun 'It'"
    assert "James Webb" in meta_s2["resolved_antecedent"] or "NASA" in meta_s2["resolved_antecedent"]
    print(f"  [OK] Detected Trigger: '{meta_s2['trigger']}'")
    print(f"  [OK] Resolved Antecedent: '{meta_s2['resolved_antecedent']}'")
    print(f"  [OK] Context Augmented Query: '{meta_s2['context_augmented_query']}'")
    assert "James Webb" in meta_s2["context_augmented_query"] or "NASA" in meta_s2["context_augmented_query"]

    # TEST 3: Discourse Memory Recording
    print("\n[TEST 3] Testing Discourse Memory Recording...")
    engine.record_sentence_verification(
        sentence_index=1,
        sentence_text=sentences[0],
        verdict="SUPPORTED",
        atomic_results=[{"claim": sentences[0], "verdict": "SUPPORTED"}]
    )
    engine.record_sentence_verification(
        sentence_index=2,
        sentence_text=sentences[1],
        verdict="SUPPORTED",
        atomic_results=[{"claim": sentences[1], "verdict": "SUPPORTED"}]
    )
    assert len(engine.discourse_history) == 2
    print(f"  [OK] Recorded {len(engine.discourse_history)} verified items into memory.")

    # TEST 4: Context Prompt for Sentence 3
    print("\n[TEST 4] Testing LLM Context Prompt Formatting...")
    meta_s3 = engine.resolve_claim_context(
        claim="However, critics claimed that the observatory had collided with a major asteroid.",
        sentence_index=4,
        all_sentences=sentences
    )
    prompt_context = engine.format_llm_context_prompt(
        claim="However, critics claimed that the observatory had collided with a major asteroid.",
        context_meta=meta_s3
    )
    assert "Sentence 1" in prompt_context or "Sentence 2" in prompt_context
    assert "Discourse Verification Memory" in prompt_context
    print("  [OK] Generated LLM Context Block:")
    print("  " + "\n  ".join(prompt_context.split("\n")))

    print("\n=================================================================")
    print("ALL DISCOURSE MEMORY TESTS PASSED SUCCESSFULLY! (4/4)")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
