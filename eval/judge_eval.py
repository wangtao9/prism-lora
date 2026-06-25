"""Judge adapter evaluation: memory conflict detection (UPDATE vs KEEP).

Evaluates the base model and Judge LoRA adapter on binary classification.
Computes accuracy, precision, recall, and F1 (pos_label="UPDATE") using sklearn.

Includes error analysis: prints Top 5 wrong predictions.

Usage:
  python -m eval.judge_eval [--base-url URL] [--output-dir DIR]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

from openai import AsyncOpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Model / prompt constants
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ADAPTER = "judge"

MODE_MAP = {
    "base": BASE_MODEL,
    "judge": JUDGE_ADAPTER,
    "poet": "poet",
}

JUDGE_SYSTEM_PROMPT = (
    "你是一个记忆冲突检测专家。给定旧记忆和新事实，你需要判断它们是否"
    "在同一维度上存在冲突。如果冲突则输出UPDATE并用新事实替换旧记忆，"
    "如果不冲突则输出KEEP让旧记忆保持不变。"
)

BASE_SYSTEM_PROMPT = "You are a helpful AI assistant."

SYSTEM_PROMPT_MAP = {
    "judge": JUDGE_SYSTEM_PROMPT,
    "poet": (
        "你是一位精通古诗词的创作大师，擅长根据要求创作符合格律和意境的古典诗词。"
        "你的创作严格遵守古典诗词的体裁规范，包括字数、行数和押韵。"
    ),
    "base": BASE_SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# Temperature / token presets
# ---------------------------------------------------------------------------
TASK_PARAMS = {
    "judge": {"temperature": 0.0, "max_tokens": 256},
    "poet": {"temperature": 0.7, "max_tokens": 256},
}

# ---------------------------------------------------------------------------
# Helper: load test data
# ---------------------------------------------------------------------------
def load_test_data(task: str) -> list:
    """Load test data from data/{task}/test.json."""
    path = os.path.join(DATA_DIR, task, "test.json")
    if not os.path.exists(path):
        print(f"ERROR: Test data file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: async query vLLM
# ---------------------------------------------------------------------------
async def query_vllm_async(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """Perform a single async vLLM OpenAI API query and return the response text."""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Parse judge decision from model output (3-strategy parser)
# ---------------------------------------------------------------------------
def parse_judge_response(text: str) -> str:
    """Parse the decision (UPDATE/KEEP) from model output.

    Strategy:
        1. Try direct JSON parse of the full text.
        2. Try regex extraction of a JSON block inside the text.
        3. Try direct keyword matching for UPDATE/KEEP.
        4. Return "UNKNOWN" if all fail.
    """
    # Strategy 1: direct JSON parse
    try:
        obj = json.loads(text)
        if "decision" in obj:
            decision = obj["decision"]
            if decision.upper() in ("UPDATE", "KEEP"):
                return decision.upper()
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: regex extraction of JSON block
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if "decision" in obj:
                decision = obj["decision"]
                if decision.upper() in ("UPDATE", "KEEP"):
                    return decision.upper()
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: direct keyword matching
    text_upper = text.upper()
    if "UPDATE" in text_upper:
        return "UPDATE"
    if "KEEP" in text_upper:
        return "KEEP"

    # Strategy 4: fallback
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Async Judge evaluation
# ---------------------------------------------------------------------------
async def evaluate_judge_model(
    client: AsyncOpenAI,
    model_name: str,
    test_data: list,
    base_url: str,
) -> dict:
    """Evaluate judge task: compute Accuracy, Precision, Recall, F1.

    Uses precision_recall_fscore_support from sklearn (binary, pos_label="UPDATE").

    Conversations layout:
      [0] system, [1] human (user input), [2] gpt (ground truth with UPDATE/KEEP)

    Returns dict with: accuracy, precision, recall, f1, per_class_report, total, valid, details
    """
    params = TASK_PARAMS["judge"]
    y_true = []
    y_pred = []
    details = []
    invalid = 0

    # Build list of (user_input, ground_truth) pairs
    tasks_items = []
    for item in test_data:
        conversations = item["conversations"]
        user_input = conversations[1]["value"]
        ground_truth_raw = conversations[2]["value"]

        gt_decision = parse_judge_response(ground_truth_raw)
        if gt_decision == "UNKNOWN":
            continue

        y_true.append(gt_decision)
        tasks_items.append((user_input, gt_decision))

    # Concurrent queries via asyncio.gather
    async def _query_one(user_input: str, gt_decision: str):
        try:
            model_output = await query_vllm_async(
                client, model_name, JUDGE_SYSTEM_PROMPT, user_input,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )
            pred_decision = parse_judge_response(model_output)
        except Exception as e:
            model_output = ""
            pred_decision = "UNKNOWN"
            return {
                "pred": pred_decision,
                "detail": {
                    "input": user_input[:80],
                    "ground_truth": gt_decision,
                    "prediction": pred_decision,
                    "error": str(e),
                },
            }

        return {
            "pred": pred_decision,
            "detail": {
                "input": user_input[:80],
                "ground_truth": gt_decision,
                "prediction": pred_decision,
                "model_output": model_output[:200],
            },
        }

    results = await asyncio.gather(*[_query_one(ui, gt) for ui, gt in tasks_items])

    for r in results:
        y_pred.append(r["pred"])
        details.append(r["detail"])
        if r["pred"] == "UNKNOWN":
            invalid += 1

    total = len(y_true)
    valid = total - invalid

    y_true_valid = [y_true[i] for i in range(total) if y_pred[i] != "UNKNOWN"]
    y_pred_valid = [y_pred[i] for i in range(total) if y_pred[i] != "UNKNOWN"]

    if len(y_true_valid) == 0:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        per_class_report = {}
    else:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_valid, y_pred_valid,
            average="binary",
            pos_label="UPDATE",
        )
        report = classification_report(
            y_true_valid, y_pred_valid,
            labels=["UPDATE", "KEEP"],
            pos_label="UPDATE",
            output_dict=True,
            zero_division=0,
        )
        per_class_report = report

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_class_report": per_class_report,
        "total": total,
        "valid": valid,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Run comparison: base vs judge LoRA
# ---------------------------------------------------------------------------
async def run_judge_comparison(base_url: str, output_dir: str) -> None:
    """Evaluate base model vs judge adapter, save JSONs, print comparison + error analysis."""
    os.makedirs(output_dir, exist_ok=True)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    # Load test data
    test_data = load_test_data("judge")
    print(f"\n=== Judge Task: {len(test_data)} test samples ===")

    # Evaluate base model
    print(f"  Evaluating base model ({BASE_MODEL}) ...")
    t0 = time.time()
    base_results = await evaluate_judge_model(client, BASE_MODEL, test_data, base_url)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Evaluate judge LoRA
    print(f"  Evaluating Judge LoRA ({JUDGE_ADAPTER}) ...")
    t0 = time.time()
    judge_results = await evaluate_judge_model(client, JUDGE_ADAPTER, test_data, base_url)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Save results (summary only, no details)
    base_summary = {k: v for k, v in base_results.items() if k != "details"}
    judge_summary = {k: v for k, v in judge_results.items() if k != "details"}

    with open(os.path.join(output_dir, "judge_base.json"), "w", encoding="utf-8") as f:
        json.dump(base_summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "judge_lora.json"), "w", encoding="utf-8") as f:
        json.dump(judge_summary, f, indent=2, ensure_ascii=False)

    # ----- Error analysis: Top 5 wrong predictions -----
    print(f"\n{'='*60}")
    print(f"Error Analysis (Judge LoRA)")
    print(f"{'='*60}")

    judge_details = judge_results.get("details", [])
    wrong_predictions = [
        d for d in judge_details
        if d.get("prediction") != d.get("ground_truth") and d.get("prediction") != "UNKNOWN"
    ]

    print(f"\nJudge LoRA wrong predictions ({len(wrong_predictions)} cases):")
    for i, d in enumerate(wrong_predictions[:5]):
        print(f"  [{i+1}] True: {d['ground_truth']}, Pred: {d['prediction']}")
        print(f"      Prompt: {d['input']}...")
        print()

    # ----- Comparison table -----
    print(f"{'='*60}")
    print(f"{'Memory Conflict Detection: Base vs Judge LoRA':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Base Model':<15} {'Judge LoRA':<15} {'Delta':<10}")
    print(f"{'-'*60}")

    for metric_key, display in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]:
        base_val = base_results.get(metric_key, 0)
        judge_val = judge_results.get(metric_key, 0)
        delta = judge_val - base_val
        print(f"{display:<20} {base_val:<15.4f} {judge_val:<15.4f} {delta:<+10.4f}")

    valid = judge_results.get("valid", 0)
    total = judge_results.get("total", 0)
    print(f"{'Valid/Total':<20} {base_results.get('valid',0)}/{base_results.get('total',0):<13} {valid}/{total:<13}")

    print(f"\nResults saved to {output_dir}/judge_base.json, {output_dir}/judge_lora.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate judge adapter (memory conflict detection)",
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

    asyncio.run(run_judge_comparison(args.base_url, output_dir))


if __name__ == "__main__":
    main()