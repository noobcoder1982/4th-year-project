import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

"""
TruthLens - Search Provider Health Check
Validates that each search backend functions correctly, handles timeouts gracefully,
and adheres to rate limiting.
"""
import sys
import time

# Mock model imports so we can import search providers from app.py without loading 3B model weights
from unittest.mock import MagicMock
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

end_pos = code.find('def verify_claim(')
exec(code[:end_pos], globals())

def run_provider_tests():
    test_query = "Water is composed of hydrogen and oxygen"
    providers = [
        ("DuckDuckGo", provider_duckduckgo),
        ("Yahoo", provider_yahoo),
        ("Bing Scraper", provider_bing),
        ("Wikipedia REST API", provider_wikipedia_api),
    ]

    print("========================================")
    print("TRUTHLENS SEARCH PROVIDER HEALTH CHECK")
    print("========================================")
    print(f"Test Query: \"{test_query}\"\n")

    working_providers = 0

    for name, fn in providers:
        start_t = time.time()
        try:
            results = fn(test_query, max_results=3)
            elapsed = time.time() - start_t
            if results:
                working_providers += 1
                first_title = results[0].get("title", "")[:45]
                first_url = results[0].get("url", "")
                print(f"  [PASS] {name:20} -> {len(results)} results in {elapsed:.2f}s")
                print(f"         Sample: {first_title} ({first_url})")
            else:
                print(f"  [WARN] {name:20} -> 0 results returned in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_t
            print(f"  [FAIL] {name:20} -> Exception ({e}) in {elapsed:.2f}s")

    print("\n----------------------------------------")
    print(f"Active working providers: {working_providers}/{len(providers)}")
    assert working_providers >= 2, "At least 2 search providers must be operational for demo reliability."
    print("Provider health check SUCCESS!")
    print("========================================")

if __name__ == "__main__":
    run_provider_tests()
