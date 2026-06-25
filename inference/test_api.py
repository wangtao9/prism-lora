"""API integration tests for prism-lora multi-adapter vLLM service.

Usage: python -m inference.test_api [--base-url URL]

Tests:
  1. Server connection (list models)
  2. Judge detects UPDATE (conflict case)
  3. Judge detects KEEP (no conflict case)
  4. Poet generates poetry (output length > 20)
"""

import asyncio
import sys

# Ensure project root is on path
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.client import PrismClient


async def run_tests(base_url: str) -> bool:
    client = PrismClient(base_url=base_url)
    passed = 0
    failed = 0

    # Test 1: Server connection
    try:
        models = await client.list_models()
        assert len(models) >= 3, f"Expected 3+ models, got {models}"
        print(f"  [PASS] Server connection: {models}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Server connection: {e}")
        failed += 1

    # Test 2: Judge detects UPDATE (conflict)
    try:
        result = await client.judge("张三喜欢吃苹果", "张三不喜欢吃苹果")
        assert "UPDATE" in result.upper(), f"Expected UPDATE in output, got: {result[:100]}"
        print(f"  [PASS] Judge conflict detection (UPDATE)")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] Judge conflict detection: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] Judge conflict detection: {e}")
        failed += 1

    # Test 3: Judge detects KEEP (no conflict)
    try:
        result = await client.judge("张三喜欢吃苹果", "张三喜欢吃香蕉")
        assert "KEEP" in result.upper(), f"Expected KEEP in output, got: {result[:100]}"
        print(f"  [PASS] Judge no-conflict detection (KEEP)")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] Judge no-conflict detection: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] Judge no-conflict detection: {e}")
        failed += 1

    # Test 4: Poet generates poetry
    try:
        result = await client.poet("写一首关于秋天的七言绝句")
        assert len(result) > 20, f"Expected >20 chars, got {len(result)}: {result[:80]}"
        print(f"  [PASS] Poet generation (len={len(result)})")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] Poet generation: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] Poet generation: {e}")
        failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    success = asyncio.run(run_tests(args.base_url))
    sys.exit(0 if success else 1)