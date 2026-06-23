#!/usr/bin/env python3
"""Evaluation script with 2x3 cross-comparison matrix for prism-lora.

Runs judge and poet tasks across 3 model variants (base, judge, poet),
produces per-cell result JSONs, a final comparison.md report, and
matplotlib visualization plots.

Improvements over previous version:
- 平水韵 (Pingshui Yun) rhyme detection for poet evaluation
- Async inference via AsyncOpenAI + asyncio.gather
- Formal 4-condition specialization verdict
- Form-specific breakdown (五言绝句, 七言绝句, 五言律诗, 七言律诗)
- matplotlib visualization (bar charts + cross-domain heatmap)
- Updated data loading (data/{task}/test.json, 3-round conversations)
- Judge labels use UPDATE/KEEP with pos_label="UPDATE" for F1
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter

from openai import AsyncOpenAI
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Model / prompt mappings
# ---------------------------------------------------------------------------
MODE_MAP = {
    "base": "Qwen/Qwen2.5-1.5B-Instruct",
    "judge": "judge",
    "poet": "poet",
}

JUDGE_SYSTEM_PROMPT = (
    "你是一个记忆冲突检测专家。给定旧记忆和新事实，你需要判断它们是否"
    "在同一维度上存在冲突。如果冲突则输出UPDATE并用新事实替换旧记忆，"
    "如果不冲突则输出KEEP让旧记忆保持不变。"
)

POET_SYSTEM_PROMPT = (
    "你是一位精通古诗词的创作大师，擅长根据要求创作符合格律和意境的古典诗词。"
    "你的创作严格遵守古典诗词的体裁规范，包括字数、行数和押韵。"
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
    "judge": {"temperature": 0.0, "max_tokens": 256},
    "poet": {"temperature": 0.7, "max_tokens": 256},
}

# ---------------------------------------------------------------------------
# Poetry form specifications
# ---------------------------------------------------------------------------
FORM_SPEC = {
    "五言绝句": {"lines": 4, "chars_per_line": 5},
    "七言绝句": {"lines": 4, "chars_per_line": 7},
    "五言律诗": {"lines": 8, "chars_per_line": 5},
    "七言律诗": {"lines": 8, "chars_per_line": 7},
}

# ---------------------------------------------------------------------------
# 平水韵 (Pingshui Yun) rhyme groups
# ---------------------------------------------------------------------------
RHYME_GROUPS = {
    "东": ["东", "同", "中", "风", "空", "公", "功", "红", "虹", "钟", "松", "冬", "终", "宗", "宫", "穷", "虫"],
    "江": ["江", "窗", "邦", "缸", "双", "霜", "庞", "腔", "港"],
    "支": ["支", "枝", "奇", "宜", "期", "词", "诗", "时", "知", "微", "飞", "归", "悲", "姿", "丝", "池", "迟", "辞", "师"],
    "微": ["微", "归", "飞", "悲", "衣", "非", "稀", "威", "辉", "挥", "违", "围", "碑", "吹", "随", "堆"],
    "鱼": ["鱼", "书", "居", "虚", "渠", "疏", "梳", "初", "车", "如", "庐", "猪", "除", "厨", "驱"],
    "虞": ["虞", "无", "图", "湖", "扶", "夫", "驱", "殊", "珠", "枯", "姑", "徒", "途", "炉"],
    "齐": ["齐", "溪", "西", "鸡", "归", "泥", "提", "迷", "低", "啼", "池", "离"],
    "佳": ["佳", "街", "鞋", "牌", "柴", "崖", "涯", "排", "乖"],
    "灰": ["灰", "回", "梅", "杯", "催", "哀", "开", "台", "才", "来", "栽", "培", "胎"],
    "真": ["真", "人", "春", "新", "身", "尘", "神", "亲", "邻", "因", "银", "巾", "民", "晨", "宾", "臣"],
    "文": ["文", "云", "分", "群", "军", "君", "门", "闻", "村", "春", "痕", "魂", "温", "盆", "奔"],
    "元": ["元", "言", "轩", "烦", "园", "泉", "源", "传", "缘", "烟", "弦", "年", "前", "连"],
    "寒": ["寒", "难", "看", "山", "关", "欢", "端", "酸", "宽", "安", "兰", "滩", "残", "弹", "丹", "干"],
    "删": ["删", "山", "关", "还", "湾", "环", "颜", "闲", "间", "寒", "看", "难"],
    "先": ["先", "年", "天", "千", "山", "烟", "边", "仙", "传", "缘", "泉", "连", "弦", "前"],
    "萧": ["萧", "桥", "遥", "条", "销", "宵", "潮", "骄", "飘", "苗", "雕", "娇"],
    "肴": ["肴", "交", "高", "茅", "抛", "包", "梢", "嘲", "刀", "烧", "毛"],
    "豪": ["豪", "高", "劳", "涛", "曹", "袍", "桥", "骚", "槽", "刀", "毛", "遭"],
    "歌": ["歌", "多", "河", "过", "波", "陀", "罗", "和", "戈", "何", "阿", "坡", "哥"],
    "麻": ["麻", "花", "家", "霞", "华", "沙", "茶", "鸦", "夸", "瓜", "车", "牙", "差", "纱"],
    "阳": ["阳", "伤", "光", "长", "香", "乡", "堂", "凉", "霜", "方", "亡", "芒", "狂", "忙", "荒", "行", "量"],
    "庚": ["庚", "明", "行", "惊", "城", "生", "声", "轻", "平", "兵", "名", "晴", "营", "更", "迎", "登"],
    "青": ["青", "轻", "经", "星", "明", "亭", "庭", "平", "屏", "城", "冰", "清"],
    "尤": ["尤", "流", "秋", "游", "愁", "楼", "头", "舟", "州", "求", "牛", "收", "休", "留", "谋", "丘"],
    "侵": ["侵", "心", "深", "林", "音", "阴", "今", "金", "吟", "琴", "临", "尘", "春", "人"],
    "覃": ["覃", "南", "参", "甘", "三", "男", "蓝", "堪", "含", "贪", "担", "谈", "寒"],
    "盐": ["盐", "添", "廉", "帘", "严", "占", "甜", "签", "店", "炎", "潜", "淹"],
    "咸": ["咸", "凡", "衫", "监", "岩", "严", "含", "函", "甘"],
}

# Build reverse lookup: char -> rhyme group
CHAR_TO_RHYME = {}
for group_name, chars in RHYME_GROUPS.items():
    for ch in chars:
        CHAR_TO_RHYME[ch] = group_name


# ---------------------------------------------------------------------------
# Helper: load test data (subdirectory layout)
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
    if "UPDATE" in text_upper:
        return "UPDATE"
    if "KEEP" in text_upper:
        return "KEEP"

    # Strategy 4: fallback
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Poetry evaluation helpers
# ---------------------------------------------------------------------------
def extract_poem_lines(text: str) -> list:
    """Extract the pure poetry lines from model output."""
    lines = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        cleaned = re.sub('[，。！？；：、""''（）《》\\s]', '', stripped)
        if len(cleaned) >= 3:
            lines.append(cleaned)
    return lines


def detect_expected_form(instruction: str) -> str | None:
    """Detect which poetry form the instruction asks for."""
    for form in FORM_SPEC:
        if form in instruction:
            return form
    return None


def evaluate_form_compliance(output: str, expected_form: str) -> float:
    """Check if the poetry follows the specified form (line count + chars per line).
    Returns 0-1 score."""
    spec = FORM_SPEC.get(expected_form)
    if not spec:
        return 0.0

    lines = extract_poem_lines(output)
    expected_lines = spec["lines"]
    expected_chars = spec["chars_per_line"]

    if len(lines) == 0:
        return 0.0

    line_ratio = min(len(lines), expected_lines) / expected_lines
    chars_match = sum(1 for line in lines if len(line) == expected_chars)
    chars_ratio = chars_match / expected_lines if expected_lines > 0 else 0

    return round((line_ratio * 0.3 + chars_ratio * 0.7), 4)


def evaluate_rhyme_compliance(output: str, expected_form: str) -> float:
    """Check if the poetry follows 平水韵 rhyme patterns.

    For 绝句: 偶数句末字 (lines 2, 4) should share the same rhyme group.
    For 律诗: 偶数句末字 (lines 2, 4, 6, 8) should share the same rhyme group.

    Returns 0-1 score = proportion of rhyme chars matching the most common group.
    """
    lines = extract_poem_lines(output)
    spec = FORM_SPEC.get(expected_form)
    if not spec or len(lines) < 2:
        return 0.0

    last_chars = [line[-1] for line in lines if len(line) >= 2]

    # Determine which lines should rhyme (偶数句末字)
    rhyme_line_indices = []
    if spec["lines"] == 4:  # 绝句
        rhyme_line_indices = [1, 3]
    elif spec["lines"] == 8:  # 律诗
        rhyme_line_indices = [1, 3, 5, 7]

    rhyme_chars = []
    for idx in rhyme_line_indices:
        if idx < len(last_chars):
            rhyme_chars.append(last_chars[idx])

    if len(rhyme_chars) < 2:
        return 0.0

    groups = [CHAR_TO_RHYME.get(ch, None) for ch in rhyme_chars]
    non_null_groups = [g for g in groups if g is not None]

    if len(non_null_groups) < 2:
        return 0.0

    group_counts = Counter(non_null_groups)
    most_common = group_counts.most_common(1)[0]

    score = most_common[1] / len(rhyme_chars)
    return round(score, 4)


TOPIC_KEYWORDS = [
    "春雨", "秋月", "山水", "离别", "思乡", "登高", "夜思",
    "梅花", "荷花", "柳树", "春风", "秋霜", "雪景", "落日",
    "江水", "渔舟", "田园", "边塞", "月夜", "寒夜",
    "晨曦", "暮色", "远行", "怀古",
]

TOPIC_EXPANSIONS = {
    "春雨": ["雨", "春", "润", "花", "绿"],
    "秋月": ["月", "秋", "光", "霜", "寒"],
    "山水": ["山", "水", "溪", "峰", "崖", "泉"],
    "离别": ["别", "离", "送", "远", "归"],
    "思乡": ["乡", "思", "故", "远", "归", "望"],
    "登高": ["高", "登", "山", "望", "远"],
    "夜思": ["夜", "思", "月", "星", "灯"],
    "梅花": ["梅", "花", "寒", "香", "雪"],
    "荷花": ["荷", "花", "莲", "池", "水"],
    "柳树": ["柳", "树", "絮", "春", "绿"],
    "春风": ["春风", "花", "绿", "暖"],
    "雪景": ["雪", "白", "寒", "冬", "冰"],
    "落日": ["日", "落", "暮", "夕", "晖"],
    "江水": ["江", "水", "舟", "波", "流"],
    "渔舟": ["渔", "舟", "水", "江", "网"],
    "田园": ["田", "园", "村", "农", "稻"],
    "边塞": ["边", "塞", "军", "战", "关"],
    "月夜": ["月", "夜", "光", "影", "星"],
}


def detect_topic(instruction: str) -> str:
    """Extract topic keyword from instruction via known topic list."""
    for t in TOPIC_KEYWORDS:
        if t in instruction:
            return t
    return ""


def evaluate_topic_relevance(output: str, topic: str) -> float:
    """Compute topic relevance score based on keyword overlap and thematic expansion.
    Returns 0-1 score."""
    if not topic:
        return 0.5

    topic_chars = set(topic)
    poem_chars = set(output)
    overlap = topic_chars & poem_chars

    if len(topic_chars) == 0:
        return 0.5

    base_score = len(overlap) / len(topic_chars)

    expanded = TOPIC_EXPANSIONS.get(topic, list(topic_chars))
    expanded_chars = set(expanded)
    expanded_overlap = expanded_chars & poem_chars
    expanded_score = len(expanded_overlap) / len(expanded_chars) if expanded_chars else 0

    final_score = max(base_score, 0.5 * base_score + 0.5 * expanded_score)
    return round(min(final_score, 1.0), 4)


# ---------------------------------------------------------------------------
# Async Judge evaluation
# ---------------------------------------------------------------------------
async def evaluate_judge_async(
    client: AsyncOpenAI,
    test_data: list,
    model_name: str,
    system_prompt: str,
) -> dict:
    """Evaluate judge task asynchronously: compute Accuracy and F1 for UPDATE/KEEP.

    Conversations layout:
      [0] system, [1] human (user input), [2] gpt (ground truth with UPDATE/KEEP)

    Returns dict with: accuracy, f1_update, f1_keep, total, valid, details
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
                client, model_name, system_prompt, user_input,
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
        f1_update = 0.0
        f1_keep = 0.0
    else:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        report = classification_report(
            y_true_valid, y_pred_valid,
            labels=["UPDATE", "KEEP"],
            pos_label="UPDATE",
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
# Async Poet evaluation with 平水韵 rhyme + form breakdown
# ---------------------------------------------------------------------------
async def evaluate_poet_async(
    client: AsyncOpenAI,
    test_data: list,
    model_name: str,
    system_prompt: str,
) -> dict:
    """Evaluate poet task asynchronously: format, rhyme (平水韵), topic, diversity,
    plus per-form breakdown.

    Conversations layout:
      [0] system, [1] human (instruction), [2] gpt (reference poem)

    Returns dict with:
      format_compliance, rhyme_compliance, topic_relevance, diversity,
      per_form_metrics, total, details
    """
    params = TASK_PARAMS["poet"]
    total = len(test_data)

    # Build list of items with pre-extracted metadata
    items_meta = []
    for item in test_data:
        conversations = item["conversations"]
        instruction = conversations[1]["value"]
        expected_form = detect_expected_form(instruction)
        topic = detect_topic(instruction)
        items_meta.append((instruction, expected_form, topic))

    # Concurrent queries
    async def _query_one(instruction: str):
        try:
            model_output = await query_vllm_async(
                client, model_name, system_prompt, instruction,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )
        except Exception:
            model_output = ""
        return model_output

    all_outputs = await asyncio.gather(*[_query_one(m[0]) for m in items_meta])

    # Compute metrics per item
    format_hits = 0
    rhyme_hits = 0
    topic_hits = 0
    per_form_data = {form: {"format": [], "rhyme": []} for form in FORM_SPEC}
    details = []

    for i, (output, (instruction, expected_form, topic)) in enumerate(zip(all_outputs, items_meta)):
        # Form compliance
        form_ok = False
        form_score = 0.0
        if expected_form:
            form_score = evaluate_form_compliance(output, expected_form)
            form_ok = form_score >= 0.5
        if form_ok:
            format_hits += 1

        # Rhyme compliance (平水韵)
        rhyme_ok = False
        rhyme_score = 0.0
        if expected_form:
            rhyme_score = evaluate_rhyme_compliance(output, expected_form)
            rhyme_ok = rhyme_score >= 0.5
        if rhyme_ok:
            rhyme_hits += 1

        # Topic relevance
        topic_ok = False
        topic_score = evaluate_topic_relevance(output, topic)
        if topic and topic_score >= 0.3:
            topic_ok = True
        if topic_ok:
            topic_hits += 1

        # Per-form accumulation
        if expected_form and expected_form in per_form_data:
            per_form_data[expected_form]["format"].append(form_score)
            per_form_data[expected_form]["rhyme"].append(rhyme_score)

        details.append({
            "input": instruction[:80],
            "expected_form": expected_form,
            "topic": topic,
            "format": form_ok,
            "rhyme": rhyme_ok,
            "topic_match": topic_ok,
            "model_output": output[:200] if output else "",
        })

    # Diversity: distinct-2 bigram ratio across all outputs
    all_bigrams = []
    total_bigrams = 0
    for output in all_outputs:
        if not output:
            continue
        chars = list(output)
        for j in range(len(chars) - 1):
            bigram = chars[j] + chars[j + 1]
            all_bigrams.append(bigram)
            total_bigrams += 1

    distinct_bigrams = len(set(all_bigrams))
    diversity = distinct_bigrams / total_bigrams if total_bigrams > 0 else 0.0

    # Per-form averages
    per_form_metrics = {}
    for form, scores in per_form_data.items():
        if scores["format"]:
            fmt_avg = round(sum(scores["format"]) / len(scores["format"]), 4)
            rhyme_avg = round(sum(scores["rhyme"]) / len(scores["rhyme"]), 4)
            per_form_metrics[form] = {
                "count": len(scores["format"]),
                "avg_format_compliance": fmt_avg,
                "avg_rhyme_compliance": rhyme_avg,
            }

    return {
        "format_compliance": round(format_hits / total, 4) if total > 0 else 0.0,
        "rhyme_compliance": round(rhyme_hits / total, 4) if total > 0 else 0.0,
        "topic_relevance": round(topic_hits / total, 4) if total > 0 else 0.0,
        "diversity": round(diversity, 4),
        "per_form_metrics": per_form_metrics,
        "total": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Full async evaluation: run all 6 cells
# ---------------------------------------------------------------------------
async def run_full_evaluation_async(base_url: str, task: str = "all") -> None:
    """Run evaluations for judge x3 + poet x3 (or a single task x3) asynchronously.

    Save each result to results/{task}_{mode}.json (summary only, no details).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    tasks_list = ["judge", "poet"] if task == "all" else [task]

    for t in tasks_list:
        test_data = load_test_data(t)
        print(f"\n=== Evaluating task: {t} ({len(test_data)} samples) ===")

        for mode in ["base", "judge", "poet"]:
            model_name = MODE_MAP[mode]
            system_prompt = SYSTEM_PROMPT_MAP[mode]
            print(f"  Mode: {mode} (model: {model_name}) ...")
            t0 = time.time()

            if t == "judge":
                results = await evaluate_judge_async(client, test_data, model_name, system_prompt)
            else:
                results = await evaluate_poet_async(client, test_data, model_name, system_prompt)

            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

            result_summary = {k: v for k, v in results.items() if k != "details"}
            result_path = os.path.join(RESULTS_DIR, f"{t}_{mode}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_summary, f, indent=2, ensure_ascii=False)
            print(f"    Saved: {result_path}")
            print(f"    Summary: {result_summary}")


# ---------------------------------------------------------------------------
# Report generation with specialization verdict
# ---------------------------------------------------------------------------
def generate_report() -> None:
    """Load all result JSONs from results/ and produce results/comparison.md.

    Includes:
      - Judge table: Accuracy, F1(UPDATE), F1(KEEP), Valid/Total
      - Poet table: format_compliance, rhyme_compliance, topic_relevance, diversity
      - Per-form breakdown table
      - Conclusions: per-LoRA improvement deltas
      - 4-condition specialization verdict
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
    poet_header = "| 模型 | format_compliance | rhyme_compliance | topic_relevance | diversity |"
    poet_separator = "|------|-------------------|------------------|-----------------|-----------|"
    poet_rows = []
    for mode in ["base", "judge", "poet"]:
        r = results.get(("poet", mode), {})
        fmt = r.get("format_compliance", "N/A")
        rhyme = r.get("rhyme_compliance", "N/A")
        topic = r.get("topic_relevance", "N/A")
        div = r.get("diversity", "N/A")
        label = mode if mode != "base" else "base (Qwen2.5-1.5B)"
        poet_rows.append(f"| {label} | {fmt} | {rhyme} | {topic} | {div} |")

    # ----- Per-form breakdown -----
    per_form_lines = []
    for mode in ["base", "judge", "poet"]:
        r = results.get(("poet", mode), {})
        per_form = r.get("per_form_metrics", {})
        label = mode if mode != "base" else "base"
        for form in FORM_SPEC:
            fm = per_form.get(form, {})
            if fm:
                per_form_lines.append(
                    f"  - {label} / {form}: format={fm.get('avg_format_compliance', 'N/A')}, "
                    f"rhyme={fm.get('avg_rhyme_compliance', 'N/A')} (n={fm.get('count', 'N/A')})"
                )

    # ----- Conclusions: deltas -----
    judge_base_acc = results.get(("judge", "base"), {}).get("accuracy", 0) or 0
    judge_lora_acc = results.get(("judge", "judge"), {}).get("accuracy", 0) or 0
    judge_delta = round(judge_lora_acc - judge_base_acc, 4)

    poet_base_fmt = results.get(("poet", "base"), {}).get("format_compliance", 0) or 0
    poet_lora_fmt = results.get(("poet", "poet"), {}).get("format_compliance", 0) or 0
    poet_delta_fmt = round(poet_lora_fmt - poet_base_fmt, 4)

    poet_base_rhyme = results.get(("poet", "base"), {}).get("rhyme_compliance", 0) or 0
    poet_lora_rhyme = results.get(("poet", "poet"), {}).get("rhyme_compliance", 0) or 0
    poet_delta_rhyme = round(poet_lora_rhyme - poet_base_rhyme, 4)

    # ----- 4-condition specialization verdict -----
    # Condition 1: Judge LoRA improves on judge task (delta > 0)
    cond1 = judge_delta > 0
    # Condition 2: Poet LoRA does NOT improve on judge task (abs(delta) < 0.05)
    poet_on_judge_acc = results.get(("judge", "poet"), {}).get("accuracy", 0) or 0
    poet_judge_delta = round(poet_on_judge_acc - judge_base_acc, 4)
    cond2 = abs(poet_judge_delta) < 0.05
    # Condition 3: Poet LoRA improves on poet task (delta > 0)
    cond3 = poet_delta_fmt > 0
    # Condition 4: Judge LoRA does NOT improve on poet task (abs(delta) < 0.05)
    judge_on_poet_fmt = results.get(("poet", "judge"), {}).get("format_compliance", 0) or 0
    judge_poet_delta = round(judge_on_poet_fmt - poet_base_fmt, 4)
    cond4 = abs(judge_poet_delta) < 0.05

    all_pass = cond1 and cond2 and cond3 and cond4
    verdict = "✓ TRUE SPECIALIZATION" if all_pass else "✗ NOT PROVEN"

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
        "## Per-Form Breakdown",
        "",
        *per_form_lines,
        "",
        "## Conclusions",
        "",
        f"- Judge LoRA improvement (Accuracy delta): **{judge_delta:+.4f}**",
        f"- Poet LoRA improvement (format_compliance delta): **{poet_delta_fmt:+.4f}**",
        f"- Poet LoRA improvement (rhyme_compliance delta): **{poet_delta_rhyme:+.4f}**",
        "",
        "## Specialization Verdict (4 Conditions)",
        "",
        f"- Condition 1: Judge LoRA improves on judge task (delta > 0): **{'PASS' if cond1 else 'FAIL'}** (Δ = {judge_delta:+.4f})",
        f"- Condition 2: Poet LoRA does NOT improve on judge task (|delta| < 0.05): **{'PASS' if cond2 else 'FAIL'}** (Δ = {poet_judge_delta:+.4f})",
        f"- Condition 3: Poet LoRA improves on poet task (delta > 0): **{'PASS' if cond3 else 'FAIL'}** (Δ = {poet_delta_fmt:+.4f})",
        f"- Condition 4: Judge LoRA does NOT improve on poet task (|delta| < 0.05): **{'PASS' if cond4 else 'FAIL'}** (Δ = {judge_poet_delta:+.4f})",
        "",
        f"### Result: **{verdict}**",
        "",
    ]

    report_path = os.path.join(RESULTS_DIR, "comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved to: {report_path}")


# ---------------------------------------------------------------------------
# Visualization (matplotlib)
# ---------------------------------------------------------------------------
def generate_plots() -> None:
    """Generate 3 evaluation result plots in results/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not installed, skipping visualization.")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    BASE_COLOR = '#3498db'
    LORA_COLOR = '#e74c3c'
    POET_LORA_COLOR = '#2ecc71'

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Judge comparison bar chart ---
    judge_base = None
    judge_lora = None
    for mode in ["base", "judge"]:
        path = os.path.join(RESULTS_DIR, f"judge_{mode}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if mode == "base":
                    judge_base = data
                else:
                    judge_lora = data

    if judge_base and judge_lora:
        metrics = ["accuracy", "f1_update", "f1_keep"]
        labels = ["Accuracy", "F1(UPDATE)", "F1(KEEP)"]
        base_vals = [judge_base.get(m, 0) for m in metrics]
        lora_vals = [judge_lora.get(m, 0) for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35

        bars_base = ax.bar(x - width / 2, base_vals, width,
                           label="Base Model", color=BASE_COLOR, edgecolor='white', linewidth=1)
        bars_lora = ax.bar(x + width / 2, lora_vals, width,
                           label="Judge LoRA", color=LORA_COLOR, edgecolor='white', linewidth=1)

        for bar in bars_base:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        for bar in bars_lora:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        for i, (b, l) in enumerate(zip(base_vals, lora_vals)):
            delta = l - b
            ax.annotate(f'Delta={delta:+.3f}',
                        xy=(x[i], max(b, l) + 0.05),
                        ha='center', va='bottom', fontsize=9, color='red')

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Judge Task: Base Model vs Judge LoRA", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "judge_comparison.png"), dpi=150)
        print(f"  Saved: {RESULTS_DIR}/judge_comparison.png")
        plt.close()

    # --- 2. Poet comparison bar chart ---
    poet_base = None
    poet_lora = None
    for mode in ["base", "poet"]:
        path = os.path.join(RESULTS_DIR, f"poet_{mode}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if mode == "base":
                    poet_base = data
                else:
                    poet_lora = data

    if poet_base and poet_lora:
        metrics = ["format_compliance", "rhyme_compliance", "topic_relevance"]
        labels = ["Format Compliance", "Rhyme Compliance", "Topic Relevance"]
        base_vals = [poet_base.get(m, 0) for m in metrics]
        lora_vals = [poet_lora.get(m, 0) for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35

        bars_base = ax.bar(x - width / 2, base_vals, width,
                           label="Base Model", color=BASE_COLOR, edgecolor='white', linewidth=1)
        bars_lora = ax.bar(x + width / 2, lora_vals, width,
                           label="Poet LoRA", color=POET_LORA_COLOR, edgecolor='white', linewidth=1)

        for bar in bars_base:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        for bar in bars_lora:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        for i, (b, l) in enumerate(zip(base_vals, lora_vals)):
            delta = l - b
            ax.annotate(f'Delta={delta:+.3f}',
                        xy=(x[i], max(b, l) + 0.05),
                        ha='center', va='bottom', fontsize=9, color='green')

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Poet Task: Base Model vs Poet LoRA", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "poet_comparison.png"), dpi=150)
        print(f"  Saved: {RESULTS_DIR}/poet_comparison.png")
        plt.close()

    # --- 3. Cross-domain heatmap ---
    # Build 3-model x 2-task matrix from results JSONs
    matrix_data = []
    models_labels = ["Base Model", "Judge LoRA", "Poet LoRA"]
    tasks_labels = ["Judge (Accuracy)", "Poet (Format)"]
    mode_keys_judge = ["base", "judge", "poet"]
    mode_keys_poet = ["base", "poet", "judge"]  # row order matches display

    judge_scores = []
    for mode in mode_keys_judge:
        path = os.path.join(RESULTS_DIR, f"judge_{mode}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                judge_scores.append(json.load(f).get("accuracy", 0))
        else:
            judge_scores.append(0)

    poet_scores = []
    for mode in mode_keys_poet:
        path = os.path.join(RESULTS_DIR, f"poet_{mode}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                poet_scores.append(json.load(f).get("format_compliance", 0))
        else:
            poet_scores.append(0)

    matrix = np.array([judge_scores, poet_scores]).T

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(tasks_labels)))
    ax.set_yticks(np.arange(len(models_labels)))
    ax.set_xticklabels(tasks_labels, fontsize=11)
    ax.set_yticklabels(models_labels, fontsize=11)

    for i in range(len(models_labels)):
        for j in range(len(tasks_labels)):
            val = matrix[i, j]
            text_color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=text_color, fontsize=12, fontweight='bold')

    # Highlight specialization cells with colored borders
    # Judge LoRA -> Judge Task (row 1, col 0)
    ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='red', linewidth=3))
    # Poet LoRA -> Poet Task (row 2, col 1)
    ax.add_patch(plt.Rectangle((1, 2), 1, 1, fill=False, edgecolor='green', linewidth=3))

    ax.set_title("Cross-Domain Evaluation: Model x Task", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_domain_heatmap.png"), dpi=150)
    print(f"  Saved: {RESULTS_DIR}/cross_domain_heatmap.png")
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def async_main(args) -> None:
    """Async entry point: run evaluation then generate report + plots."""
    base_url = f"http://localhost:{args.port}/v1"

    # Health check
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    try:
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"Connected to vLLM server on port {args.port}")
        print(f"Available models: {model_ids}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM server on port {args.port}")
        print(f"Details: {e}")
        print("Make sure the server is running: bash scripts/start_vllm.sh")
        sys.exit(1)

    # Run full evaluation
    await run_full_evaluation_async(base_url, args.task)

    # Auto-generate report after evaluation
    generate_report()

    # Auto-generate plots
    generate_plots()


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
        help="Only generate report + plots from existing result JSONs (skip evaluation)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    args = parser.parse_args()

    # If --report, just generate report + plots and exit
    if args.report:
        generate_report()
        generate_plots()
        return

    # Otherwise run async evaluation pipeline
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
