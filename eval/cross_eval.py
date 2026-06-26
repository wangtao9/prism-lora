"""Cross-domain evaluation: test each adapter on the OTHER domain's task.

This proves specialization:
- Judge LoRA should improve on judge task but NOT on poet task
- Poet LoRA should improve on poet task but NOT on judge task
- Only if each LoRA improves ONLY in its domain do we have true specialization

Uses first 100 examples from each test set for speed.

Usage:
  python -m eval.cross_eval [--base-url URL] [--output-dir DIR]
"""

import argparse
import asyncio
import json
import os
import sys
import time

from openai import AsyncOpenAI
from sklearn.metrics import accuracy_score, f1_score

from configs.config import BASE_MODEL, JUDGE_ADAPTER, POET_ADAPTER, VLLM_BASE_URL

# Import from sibling modules
from eval.judge_eval import (
    JUDGE_SYSTEM_PROMPT,
    MODE_MAP,
    parse_judge_response,
    query_vllm_async,
    load_test_data,
    TASK_PARAMS,
)
from eval.poet_eval import (
    POET_SYSTEM_PROMPT,
    FORM_SPEC,
    detect_expected_form,
    detect_topic,
    evaluate_form_compliance,
    evaluate_rhyme_compliance,
    evaluate_topic_relevance,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Cross-eval: judge-on-poet task
# ---------------------------------------------------------------------------
async def cross_eval_judge_on_poet(
    model_name: str,
    poet_test_data: list,
    base_url: str,
) -> dict:
    """Evaluate a model on the poet task (used for cross-eval)."""
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    results = []

    async def _query_one(example):
        convs = example["conversations"]
        instruction = convs[1]["value"]
        expected_form = detect_expected_form(instruction)
        topic = detect_topic(instruction)

        try:
            model_output = await query_vllm_async(
                client, model_name, POET_SYSTEM_PROMPT, instruction,
                max_tokens=TASK_PARAMS["poet"]["max_tokens"],
                temperature=TASK_PARAMS["poet"]["temperature"],
            )
        except Exception:
            model_output = ""

        form_score = evaluate_form_compliance(model_output, expected_form) if expected_form else 0
        rhyme_score = evaluate_rhyme_compliance(model_output, expected_form) if expected_form else 0
        topic_score = evaluate_topic_relevance(model_output, topic) if topic else 0.5

        return {
            "form_compliance": form_score,
            "rhyme_compliance": rhyme_score,
            "topic_relevance": topic_score,
        }

    results = await asyncio.gather(*[_query_one(ex) for ex in poet_test_data])

    n = len(results)
    return {
        "model": model_name,
        "avg_form_compliance": round(sum(r["form_compliance"] for r in results) / n, 4) if n else 0,
        "avg_rhyme_compliance": round(sum(r["rhyme_compliance"] for r in results) / n, 4) if n else 0,
        "avg_topic_relevance": round(sum(r["topic_relevance"] for r in results) / n, 4) if n else 0,
    }


# ---------------------------------------------------------------------------
# Cross-eval: poet-on-judge task
# ---------------------------------------------------------------------------
async def cross_eval_poet_on_judge(
    model_name: str,
    judge_test_data: list,
    base_url: str,
) -> dict:
    """Evaluate a model on the judge task (used for cross-eval)."""
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    async def _query_one(example):
        convs = example["conversations"]
        human_msg = convs[1]["value"]
        gpt_msg = convs[2]["value"]
        true_decision = parse_judge_response(gpt_msg)
        if true_decision == "UNKNOWN":
            return None

        try:
            model_output = await query_vllm_async(
                client, model_name, JUDGE_SYSTEM_PROMPT, human_msg,
                max_tokens=TASK_PARAMS["judge"]["max_tokens"],
                temperature=TASK_PARAMS["judge"]["temperature"],
            )
            pred_decision = parse_judge_response(model_output)
        except Exception:
            pred_decision = "UNKNOWN"

        return (true_decision, pred_decision)

    raw_results = await asyncio.gather(*[_query_one(ex) for ex in judge_test_data])

    # Filter None (UNKNOWN ground truth) and UNKNOWN predictions
    pairs = [(t, p) for r in raw_results if r is not None for t, p in [r] if p != "UNKNOWN"]
    labels = [t for t, _ in pairs]
    predictions = [p for _, p in pairs]

    if not predictions:
        return {"model": model_name, "accuracy": 0, "f1_update": 0, "error": "No valid predictions"}

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, pos_label="UPDATE", average="binary", zero_division=0)

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1_update": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Run cross-eval: 6 combinations (3 models x 2 tasks)
# ---------------------------------------------------------------------------
async def run_cross_eval(base_url: str, output_dir: str) -> None:
    """Run comprehensive cross-domain evaluation with 100-example subsets."""
    os.makedirs(output_dir, exist_ok=True)

    # Load test data (use first 100 examples for speed)
    judge_test_data = load_test_data("judge")[:100]
    poet_test_data = load_test_data("poet")[:100]

    print(f"\n=== Cross-Domain Evaluation ===")
    print(f"  Judge subset: {len(judge_test_data)} examples")
    print(f"  Poet subset: {len(poet_test_data)} examples")

    all_results = {}

    # [1/6] Base model on judge task
    print(f"\n[1/6] Base model -> Judge task...")
    t0 = time.time()
    base_judge = await cross_eval_poet_on_judge(BASE_MODEL, judge_test_data, base_url)
    all_results["base_judge"] = base_judge
    print(f"  Accuracy: {base_judge.get('accuracy', 'N/A')}, F1: {base_judge.get('f1_update', 'N/A')} ({time.time()-t0:.1f}s)")

    # [2/6] Judge LoRA on judge task
    print(f"\n[2/6] Judge LoRA -> Judge task...")
    t0 = time.time()
    judge_judge = await cross_eval_poet_on_judge(JUDGE_ADAPTER, judge_test_data, base_url)
    all_results["judge_judge"] = judge_judge
    print(f"  Accuracy: {judge_judge.get('accuracy', 'N/A')}, F1: {judge_judge.get('f1_update', 'N/A')} ({time.time()-t0:.1f}s)")

    # [3/6] Poet LoRA on judge task
    print(f"\n[3/6] Poet LoRA -> Judge task...")
    t0 = time.time()
    poet_judge = await cross_eval_poet_on_judge(POET_ADAPTER, judge_test_data, base_url)
    all_results["poet_judge"] = poet_judge
    print(f"  Accuracy: {poet_judge.get('accuracy', 'N/A')}, F1: {poet_judge.get('f1_update', 'N/A')} ({time.time()-t0:.1f}s)")

    # [4/6] Base model on poet task
    print(f"\n[4/6] Base model -> Poet task...")
    t0 = time.time()
    base_poet = await cross_eval_judge_on_poet(BASE_MODEL, poet_test_data, base_url)
    all_results["base_poet"] = base_poet
    print(f"  Form: {base_poet.get('avg_form_compliance', 'N/A')}, Rhyme: {base_poet.get('avg_rhyme_compliance', 'N/A')} ({time.time()-t0:.1f}s)")

    # [5/6] Poet LoRA on poet task
    print(f"\n[5/6] Poet LoRA -> Poet task...")
    t0 = time.time()
    poet_poet = await cross_eval_judge_on_poet(POET_ADAPTER, poet_test_data, base_url)
    all_results["poet_poet"] = poet_poet
    print(f"  Form: {poet_poet.get('avg_form_compliance', 'N/A')}, Rhyme: {poet_poet.get('avg_rhyme_compliance', 'N/A')} ({time.time()-t0:.1f}s)")

    # [6/6] Judge LoRA on poet task
    print(f"\n[6/6] Judge LoRA -> Poet task...")
    t0 = time.time()
    judge_poet = await cross_eval_judge_on_poet(JUDGE_ADAPTER, poet_test_data, base_url)
    all_results["judge_poet"] = judge_poet
    print(f"  Form: {judge_poet.get('avg_form_compliance', 'N/A')}, Rhyme: {judge_poet.get('avg_rhyme_compliance', 'N/A')} ({time.time()-t0:.1f}s)")

    # ----- Save results -----
    with open(os.path.join(output_dir, "cross_eval.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_dir}/cross_eval.json")

    # ----- Specialization matrix -----
    print(f"\n{'='*70}")
    print(f"{'Cross-Domain Specialization Matrix':^70}")
    print(f"{'='*70}")

    print(f"\n{'Model':<20} {'Conflict(Acc)':<16} {'Conflict(F1)':<16} {'Poet(Form)':<16} {'Poet(Rhyme)':<16}")
    print(f"{'-'*70}")

    models = [
        ("Base Model", "base_judge", "base_poet"),
        ("Judge LoRA", "judge_judge", "judge_poet"),
        ("Poet LoRA", "poet_judge", "poet_poet"),
    ]

    for display_name, judge_key, poet_key in models:
        j = all_results.get(judge_key, {})
        p = all_results.get(poet_key, {})
        j_acc = j.get("accuracy", "N/A")
        j_f1 = j.get("f1_update", "N/A")
        p_form = p.get("avg_form_compliance", "N/A")
        p_rhyme = p.get("avg_rhyme_compliance", "N/A")

        j_acc_str = f"{j_acc:.4f}" if isinstance(j_acc, (int, float)) else str(j_acc)
        j_f1_str = f"{j_f1:.4f}" if isinstance(j_f1, (int, float)) else str(j_f1)
        p_form_str = f"{p_form:.4f}" if isinstance(p_form, (int, float)) else str(p_form)
        p_rhyme_str = f"{p_rhyme:.4f}" if isinstance(p_rhyme, (int, float)) else str(p_rhyme)

        print(f"{display_name:<20} {j_acc_str:<16} {j_f1_str:<16} {p_form_str:<16} {p_rhyme_str:<16}")

    # ----- 4-condition specialization verdict -----
    print(f"\n{'='*70}")
    base_j_acc = all_results.get("base_judge", {}).get("accuracy", 0)
    judge_j_acc = all_results.get("judge_judge", {}).get("accuracy", 0)
    poet_j_acc = all_results.get("poet_judge", {}).get("accuracy", 0)
    base_p_form = all_results.get("base_poet", {}).get("avg_form_compliance", 0)
    poet_p_form = all_results.get("poet_poet", {}).get("avg_form_compliance", 0)
    judge_p_form = all_results.get("judge_poet", {}).get("avg_form_compliance", 0)

    # Condition 1: Judge LoRA improves on judge task
    cond1 = judge_j_acc > base_j_acc
    # Condition 2: Poet LoRA does NOT improve on judge task (abs(delta) < 0.05)
    cond2 = abs(poet_j_acc - base_j_acc) < 0.05
    # Condition 3: Poet LoRA improves on poet task
    cond3 = poet_p_form > base_p_form
    # Condition 4: Judge LoRA does NOT improve on poet task (abs(delta) < 0.05)
    cond4 = abs(judge_p_form - base_p_form) < 0.05

    print(f"Specialization Analysis (4 Conditions):")
    print(f"  Condition 1: Judge LoRA improves on judge task (delta > 0): "
          f"{'PASS' if cond1 else 'FAIL'} "
          f"(base={base_j_acc:.4f} -> judge={judge_j_acc:.4f}, delta={judge_j_acc - base_j_acc:+.4f})")
    print(f"  Condition 2: Poet LoRA does NOT improve on judge task (|delta| < 0.05): "
          f"{'PASS' if cond2 else 'FAIL'} "
          f"(base={base_j_acc:.4f} -> poet={poet_j_acc:.4f}, delta={poet_j_acc - base_j_acc:+.4f})")
    print(f"  Condition 3: Poet LoRA improves on poet task (delta > 0): "
          f"{'PASS' if cond3 else 'FAIL'} "
          f"(base={base_p_form:.4f} -> poet={poet_p_form:.4f}, delta={poet_p_form - base_p_form:+.4f})")
    print(f"  Condition 4: Judge LoRA does NOT improve on poet task (|delta| < 0.05): "
          f"{'PASS' if cond4 else 'FAIL'} "
          f"(base={base_p_form:.4f} -> judge={judge_p_form:.4f}, delta={judge_p_form - base_p_form:+.4f})")

    all_pass = cond1 and cond2 and cond3 and cond4
    verdict = "TRUE SPECIALIZATION" if all_pass else "NOT PROVEN"
    print(f"\n  Overall: {verdict}")
    if all_pass:
        print(f"  -> LoRA adapters achieved domain-specific enhancement without interfering with other domains.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation (3 models x 2 tasks, 100-example subsets)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for result JSONs (default: <project>/results)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )

    asyncio.run(run_cross_eval(args.base_url, output_dir))


if __name__ == "__main__":
    main()