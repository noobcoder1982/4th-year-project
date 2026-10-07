import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
TruthLens Demo Benchmark Test Script
Tests the 5-claim benchmark and a mixed-claim test using app.py verification pipeline.
"""
import sys
import time

print("Loading TruthLens pipeline...")
from app import check_article

def run_benchmark():
    benchmark_article = (
        "Water is composed of hydrogen and oxygen.\n"
        "The Sun is a star located at the center of our solar system.\n"
        "Humans have four lungs that work together to supply oxygen to the bloodstream.\n"
        "The Pacific Ocean is the largest ocean on Earth.\n"
        "The Great Wall of China was built entirely in the twentieth century."
    )

    print("\n" + "=" * 60)
    print("RUNNING 5-CLAIM BENCHMARK ARTICLE")
    print("=" * 60)
    print(benchmark_article)
    print("-" * 60)

    start_time = time.time()
    results = check_article(benchmark_article)
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    expected = [
        ("Water", "SUPPORTED"),
        ("Sun", "SUPPORTED"),
        ("lungs", "REFUTED"),
        ("Pacific", "SUPPORTED"),
        ("Great Wall", "REFUTED")
    ]

    all_matched = True
    for i, res in enumerate(results):
        sentence = res["sentence"]
        verdict = res["verdict"]
        sources = res.get("sources", [])
        explanation = res.get("explanation", "")
        
        exp_topic, exp_verdict = expected[i]
        matched = (verdict == exp_verdict)
        if not matched:
            all_matched = False
        
        status_sym = "[MATCH]" if matched else "[MISMATCH]"
        print(f"\nSentence {i+1}: {sentence}")
        print(f"  Result:   {status_sym} Verdict: {verdict} (Expected: {exp_verdict})")
        print(f"  AI Explanation: {explanation[:120]}...")
        print(f"  Retrieved sources ({len(sources)}):")
        for s in sources[:3]:
            cred = s.get('credibility', 'MEDIUM')
            print(f"    - [{cred}] {s.get('domain', '')} : {s.get('title', '')[:50]}")

    print("\n" + "-" * 60)
    print(f"Total processing time: {total_time:.1f}s")
    print(f"All 5 claims match expected behavior: {all_matched}")
    print("=" * 60)

    # Mixed-claim test
    print("\n" + "=" * 60)
    print("RUNNING MIXED-CLAIM ARTICLE TEST")
    print("=" * 60)
    mixed_article = (
        "Mount Everest is the highest mountain on Earth above sea level.\n"
        "The Moon is primarily made of green cheese and dairy products."
    )
    print(mixed_article)
    print("-" * 60)

    start_time = time.time()
    mixed_results = check_article(mixed_article)
    mixed_time = time.time() - start_time

    for i, res in enumerate(mixed_results):
        print(f"\nSentence {i+1}: {res['sentence']}")
        print(f"  Verdict: {res['verdict']}")
        print(f"  AI Explanation: {res.get('explanation', '')[:120]}...")
        for s in res.get('sources', [])[:2]:
            print(f"    - [{s.get('credibility')}] {s.get('domain')}: {s.get('title')[:45]}")

    print("\n" + "=" * 60)
    print("DEMO BENCHMARK COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
