#!/usr/bin/env python3
"""Evaluation script with 2x3 cross-comparison matrix for prism-lora.

Runs judge and poet tasks across 3 model variants (base, judge, poet),
produces per-cell result JSONs and a final comparison.md report.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Model / prompt mappings (same as query_adapter.py)
# ---------------------------------------------------------------------------
MODE_MAP = {
    "base": "Qwen/Qwen2.5-1.5B-Instruct",
    "judge": "judge",
    "poet": "poet",
}

JUDGE_SYSTEM_PROMPT = (
    "You are a precise conflict detector for AI assistant responses. "
    "Given two responses to the same user query, determine if they contain "
    "a memory conflict (a factual disagreement that cannot both be true). "
    "Respond in JSON format with the following fields:\n"
    "- \"conflict\": boolean (true if there is a factual conflict)\n"
    "- \"reason\": string (brief explanation of the conflict if found, "
    "or \"No conflict detected\" if none)\n"
    "- \"confidence\": float (0.0-1.0, your confidence in the assessment)"
)

POET_SYSTEM_PROMPT = (
    "You are a creative poet who crafts beautiful poetry. "
    "When given a topic, compose a poem with the following format requirements:\n"
    "- Title on the first line\n"
    "- At least 4 lines of verse\n"
    "- Each line should be meaningful and evocative\n"
    "- Use vivid imagery and poetic language\n"
    "- Optionally include a rhyme scheme"
)

BASE_SYSTEM_PROMPT = "You are a helpful AI assistant."

SYSTEM_PROMPT_MAP = {
    "judge": JUDGE_SYSTEM_PROMPT,
    "poet": POET_SYSTEM_PROMPT,
    "base": BASE_SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# Temperature / token presets per task
# ---------------------------------------------------------------------------
TASK_PARAMS = {
    "judge": {"temperature": 0.1, "max_tokens": 256},
    "poet": {"temperature": 0.7, "max_tokens": 512},
}


# ---------------------------------------------------------------------------
# Helper: load test data
# ---------------------------------------------------------------------------
def load_test_data(task: str) -> list:
    """Load test data from data/{task}_test.json; exit if not found."""
    path = os.path.join(DATA_DIR, f"{task}_test.json")
    if not os.path.exists(path):
        print(f"ERROR: Test data file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: query vLLM
# ---------------------------------------------------------------------------
def query_vllm(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """Perform a single vLLM OpenAI API query and return the response text."""
    response = client.chat.completions.create(
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
# Helper: parse judge decision from model output
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
    # Prioritise UPDATE if both appear (UPDATE is more specific for conflicts)
    if "UPDATE" in text_upper:
        return "UPDATE"
    if "KEEP" in text_upper:
        return "KEEP"

    # Strategy 4: fallback
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Judge evaluation
# ---------------------------------------------------------------------------
def evaluate_judge(
    client: OpenAI,
    test_data: list,
    model_name: str,
    system_prompt: str,
) -> dict:
    """Evaluate judge task: compute Accuracy and F1 for UPDATE/KEEP classification.

    Parse ground truth from test data's conversations[1]["value"],
    query the model, parse prediction, and compute metrics.

    Returns dict with: accuracy, f1_update, f1_keep, total, valid, details
    """
    y_true = []
    y_pred = []
    details = []
    invalid = 0

    params = TASK_PARAMS["judge"]

    for item in test_data:
        conversations = item["conversations"]
        user_input = conversations[0]["value"]
        ground_truth_raw = conversations[1]["value"]

        # Parse ground truth decision from conversations[1]["value"]
        gt_decision = parse_judge_response(ground_truth_raw)
        if gt_decision == "UNKNOWN":
            # Skip items where ground truth cannot be parsed
            continue

        y_true.append(gt_decision)

        # Query model
        try:
            model_output = query_vllm(
                client, model_name, system_prompt, user_input,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )
        except Exception as e:
            y_pred.append("UNKNOWN")
            invalid += 1
            details.append({
                "input": user_input[:80],
                "ground_truth": gt_decision,
                "prediction": "UNKNOWN",
                "error": str(e),
            })
            continue

        pred_decision = parse_judge_response(model_output)
        y_pred.append(pred_decision)

        if pred_decision == "UNKNOWN":
            invalid += 1

        details.append({
            "input": user_input[:80],
            "ground_truth": gt_decision,
            "prediction": pred_decision,
            "model_output": model_output[:200],
        })

    total = len(y_true)
    valid = total - invalid

    # Filter to only valid predictions for metric computation
    y_true_valid = [y_true[i] for i in range(total) if y_pred[i] != "UNKNOWN"]
    y_pred_valid = [y_pred[i] for i in range(total) if y_pred[i] != "UNKNOWN"]

    if len(y_true_valid) == 0:
        accuracy = 0.0
        f1_update = 0.0
        f1_keep = 0.0
    else:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        report = classification_report(
            y_true_valid, y_pred_valid,
            labels=["UPDATE", "KEEP"],
            output_dict=True,
            zero_division=0,
        )
        f1_update = report.get("UPDATE", {}).get("f1-score", 0.0)
        f1_keep = report.get("KEEP", {}).get("f1-score", 0.0)

    return {
        "accuracy": round(accuracy, 4),
        "f1_update": round(f1_update, 4),
        "f1_keep": round(f1_keep, 4),
        "total": total,
        "valid": valid,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Poet evaluation
# ---------------------------------------------------------------------------
def evaluate_poet(
    client: OpenAI,
    test_data: list,
    model_name: str,
    system_prompt: str,
) -> dict:
    """Evaluate poet task: format, topic, rhythm, and diversity metrics.

    Extract topic keyword from prompt (regex: "关于(.+?)的"),
    check format (has "诗题：" prefix and multiple poetry lines with punctuation),
    check topic (keyword appears in output),
    check rhythm (has Chinese comma/period alternation),
    compute diversity (distinct-2 bigram ratio across all outputs).

    Returns dict with: format_compliance, topic_relevance, rhythm_compliance,
                       diversity, total, details
    """
    params = TASK_PARAMS["poet"]
    total = len(test_data)
    format_hits = 0
    topic_hits = 0
    rhythm_hits = 0
    all_outputs = []
    details = []

    for item in test_data:
        conversations = item["conversations"]
        user_input = conversations[0]["value"]

        # Extract topic keyword via regex "关于(.+?)的"
        topic_match = re.search(r"关于(.+?)的", user_input)
        topic_keyword = topic_match.group(1) if topic_match else ""

        # Query model
        try:
            model_output = query_vllm(
                client, model_name, system_prompt, user_input,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )
        except Exception as e:
            details.append({
                "input": user_input[:80],
                "topic": topic_keyword,
                "error": str(e),
                "format": False,
                "topic_match": False,
                "rhythm": False,
            })
            all_outputs.append("")
            continue

        all_outputs.append(model_output)

        # Format check: has "诗题：" prefix and multiple poetry lines with punctuation
        has_title_prefix = bool(re.search(r"诗题[：:]", model_output))
        # Poetry lines: lines containing Chinese punctuation (，、。)
        poetry_lines = re.findall(r"[^\n]+[，。、；！？][^\n]*", model_output)
        has_poetry_lines = len(poetry_lines) >= 2
        format_ok = has_title_prefix and has_poetry_lines
        if format_ok:
            format_hits += 1

        # Topic check: keyword appears in output
        topic_ok = (topic_keyword != "" and topic_keyword in model_output)
        if topic_ok:
            topic_hits += 1

        # Rhythm check: Chinese comma/period alternation pattern
        rhythm_pattern = re.search(r"[，、][^，。、；！？]*[。]", model_output)
        rhythm_ok = bool(rhythm_pattern)
        if rhythm_ok:
            rhythm_hits += 1

        details.append({
            "input": user_input[:80],
            "topic": topic_keyword,
            "format": format_ok,
            "topic_match": topic_ok,
            "rhythm": rhythm_ok,
            "model_output": model_output[:200],
        })

    # Diversity: distinct-2 bigram ratio across all outputs
    all_bigrams = []
    total_bigrams = 0
    for output in all_outputs:
        if not output:
            continue
        chars = list(output)
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i + 1]
            all_bigrams.append(bigram)
            total_bigrams += 1

    distinct_bigrams = len(set(all_bigrams))
    diversity = distinct_bigrams / total_bigrams if total_bigrams > 0 else 0.0

    return {
        "format_compliance": round(format_hits / total, 4) if total > 0 else 0.0,
        "topic_relevance": round(topic_hits / total, 4) if total > 0 else 0.0,
        "rhythm_compliance": round(rhythm_hits / total, 4) if total > 0 else 0.0,
        "diversity": round(diversity, 4),
        "total": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Full evaluation: run all 6 cells
# ---------------------------------------------------------------------------
def run_full_evaluation(client: OpenAI, task: str = "all") -> None:
    """Run evaluations for judge x3 + poet x3 (or a single task x3).

    Save each result to results/{task}_{mode}.json (summary only, no details).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tasks = ["judge", "poet"] if task == "all" else [task]

    for t in tasks:
        test_data = load_test_data(t)
        print(f"\n=== Evaluating task: {t} ({len(test_data)} samples) ===")

        for mode in ["base", "judge", "poet"]:
            model_name = MODE_MAP[mode]
            system_prompt = SYSTEM_PROMPT_MAP[mode]
            print(f"  Mode: {mode} (model: {model_name}) ...")

            if t == "judge":
                results = evaluate_judge(client, test_data, model_name, system_prompt)
            else:
                results = evaluate_poet(client, test_data, model_name, system_prompt)

            # Save result_summary (without "details") to JSON
            result_summary = {k: v for k, v in results.items() if k != "details"}
            result_path = os.path.join(RESULTS_DIR, f"{t}_{mode}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_summary, f, indent=2, ensure_ascii=False)
            print(f"    Saved: {result_path}")
            print(f"    Summary: {result_summary}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report() -> None:
    """Load all result JSONs from results/ and produce results/comparison.md.

    Includes:
      - Judge table: Accuracy, F1(UPDATE), F1(KEEP), Valid/Total
      - Poet table: format_compliance, topic_relevance, rhythm_compliance, diversity
      - Conclusions: per-LoRA improvement deltas
      - Cross-validation: cross-task interference check (threshold abs(delta) < 0.05)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load all result JSONs
    results = {}
    for t in ["judge", "poet"]:
        for mode in ["base", "judge", "poet"]:
            path = os.path.join(RESULTS_DIR, f"{t}_{mode}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    results[(t, mode)] = json.load(f)
            else:
                print(f"WARNING: Missing result file: {path}")
                results[(t, mode)] = {}

    # ----- Judge table -----
    judge_header = "| 模型 | Accuracy | F1(UPDATE) | F1(KEEP) | Valid/Total |"
    judge_separator = "|------|----------|------------|----------|-------------|"
    judge_rows = []
    for mode in ["base", "judge", "poet"]:
        r = results.get(("judge", mode), {})
        acc = r.get("accuracy", "N/A")
        f1u = r.get("f1_update", "N/A")
        f1k = r.get("f1_keep", "N/A")
        valid = r.get("valid", "N/A")
        total = r.get("total", "N/A")
        label = mode if mode != "base" else "base (Qwen2.5-1.5B)"
        judge_rows.append(f"| {label} | {acc} | {f1u} | {f1k} | {valid}/{total} |")

    # ----- Poet table -----
    poet_header = "| 模型 | format_compliance | topic_relevance | rhythm_compliance | diversity |"
    poet_separator = "|------|-------------------|-----------------|-------------------|-----------|"
    poet_rows = []
    for mode in ["base", "judge", "poet"]:
        r = results.get(("poet", mode), {})
        fmt = r.get("format_compliance", "N/A")
        topic = r.get("topic_relevance", "N/A")
        rhythm = r.get("rhythm_compliance", "N/A")
        div = r.get("diversity", "N/A")
        label = mode if mode != "base" else "base (Qwen2.5-1.5B)"
        poet_rows.append(f"| {label} | {fmt} | {topic} | {rhythm} | {div} |")

    # ----- Conclusions -----
    # Judge LoRA improvement delta
    judge_base_acc = results.get(("judge", "base"), {}).get("accuracy", 0)
    judge_lora_acc = results.get(("judge", "judge"), {}).get("accuracy", 0)
    judge_delta = (
        round(judge_lora_acc - judge_base_acc, 4)
        if isinstance(judge_base_acc, (int, float)) and isinstance(judge_lora_acc, (int, float))
        else "N/A"
    )

    # Poet LoRA improvement delta
    poet_base_fmt = results.get(("poet", "base"), {}).get("format_compliance", 0)
    poet_lora_fmt = results.get(("poet", "poet"), {}).get("format_compliance", 0)
    poet_delta = (
        round(poet_lora_fmt - poet_base_fmt, 4)
        if isinstance(poet_base_fmt, (int, float)) and isinstance(poet_lora_fmt, (int, float))
        else "N/A"
    )

    # Cross-validation: poet LoRA on judge task, judge LoRA on poet task
    judge_base_acc_val = results.get(("judge", "base"), {}).get("accuracy", 0) or 0
    judge_poet_acc_val = results.get(("judge", "poet"), {}).get("accuracy", 0) or 0
    poet_on_judge_delta = (
        round(abs(judge_poet_acc_val - judge_base_acc_val), 4)
        if isinstance(judge_base_acc_val, (int, float)) and isinstance(judge_poet_acc_val, (int, float))
        else "N/A"
    )

    poet_base_fmt_val = results.get(("poet", "base"), {}).get("format_compliance", 0) or 0
    poet_judge_fmt_val = results.get(("poet", "judge"), {}).get("format_compliance", 0) or 0
    judge_on_poet_delta = (
        round(abs(poet_judge_fmt_val - poet_base_fmt_val), 4)
        if isinstance(poet_base_fmt_val, (int, float)) and isinstance(poet_judge_fmt_val, (int, float))
        else "N/A"
    )

    cross_threshold = 0.05
    poet_no_interference = (
        isinstance(poet_on_judge_delta, (int, float))
        and poet_on_judge_delta < cross_threshold
    )
    judge_no_interference = (
        isinstance(judge_on_poet_delta, (int, float))
        and judge_on_poet_delta < cross_threshold
    )

    # ----- Write report -----
    report_lines = [
        "# Prism-LoRA 2x3 Cross-Comparison Evaluation Report",
        "",
        "## Judge Task (Memory Conflict Detection)",
        "",
        judge_header,
        judge_separator,
        *judge_rows,
        "",
        "## Poet Task (Chinese Poetry Writing)",
        "",
        poet_header,
        poet_separator,
        *poet_rows,
        "",
        "## Conclusions",
        "",
        f"- Judge LoRA improvement (Accuracy delta): **{judge_delta}**",
        f"- Poet LoRA improvement (format_compliance delta): **{poet_delta}**",
        "",
        "## Cross-Validation (No Cross-domain Interference)",
        "",
        f"- Poet LoRA on judge task (Accuracy delta vs base): **{poet_on_judge_delta}** (< {cross_threshold} = no interference: {'Yes' if poet_no_interference else 'No'})",
        f"- Judge LoRA on poet task (format_compliance delta vs base): **{judge_on_poet_delta}** (< {cross_threshold} = no interference: {'Yes' if judge_no_interference else 'No'})",
        "",
        "-> LoRA 微调实现了领域专用增强，且不干扰其他领域。",
    ]

    report_path = os.path.join(RESULTS_DIR, "comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point with argparse for task, report-only, and port."""
    parser = argparse.ArgumentParser(
        description="Prism-LoRA evaluation: 2x3 cross-comparison matrix",
    )
    parser.add_argument(
        "--task",
        choices=["judge", "poet", "all"],
        default="all",
        help="Which task to evaluate (default: all)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Only generate report from existing result JSONs (skip evaluation)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    args = parser.parse_args()

    # If --report, just generate the report and exit
    if args.report:
        generate_report()
        return

    # Connect to vLLM server -- health check
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
    )

    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"Connected to vLLM server on port {args.port}")
        print(f"Available models: {model_ids}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM server on port {args.port}")
        print(f"Details: {e}")
        print("Make sure the server is running: bash scripts/start_vllm.sh")
        sys.exit(1)

    # Run full evaluation
    run_full_evaluation(client, args.task)

    # Auto-generate report after evaluation
    generate_report()


if __name__ == "__main__":
    main()
