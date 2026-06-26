"""Poet adapter evaluation: Chinese poetry generation with form/rhyme/topic metrics.

Evaluates the base model and Poet LoRA adapter on poetry generation.
Uses weighted form compliance (0.3*line_count + 0.7*chars_per_line),
平水韵 (Pingshui Yun) rhyme detection, and expanded topic relevance.

Usage:
  python -m eval.poet_eval [--base-url URL] [--output-dir DIR]
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

from configs.config import (
    BASE_MODEL,
    POET_ADAPTER,
    POET_SYSTEM_PROMPT,
    VLLM_BASE_URL,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

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
# Topic keywords and expansions
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = [
    "春雨", "秋月", "山水", "离别", "思乡", "登高", "夜思",
    "梅花", "荷花", "柳树", "春风", "秋霜", "雪景", "落日",
    "江水", "渔舟", "田园", "边塞", "月夜", "寒夜",
    "晨曦", "暮色", "远行", "怀古",
]

TOPIC_EXPANSIONS = {
    "春雨": ["春", "雨", "润", "花", "风", "绿"],
    "秋月": ["月", "秋", "光", "霜", "寒"],
    "山水": ["山", "水", "溪", "峰", "崖", "泉"],
    "离别": ["别", "离", "远", "行", "归", "送", "泪", "愁", "孤"],
    "思乡": ["乡", "思", "故", "远", "归", "望", "梦", "忆", "念"],
    "登高": ["高", "登", "山", "望", "远", "云", "险", "巅"],
    "夜思": ["夜", "思", "月", "星", "灯", "影", "眠", "梦"],
    "梅花": ["梅", "花", "寒", "香", "雪", "枝", "傲", "暗"],
    "荷花": ["荷", "花", "莲", "池", "水", "叶", "清", "香"],
    "柳树": ["柳", "树", "絮", "春", "绿", "枝", "垂", "丝"],
    "春风": ["春", "风", "花", "绿", "暖", "吹", "柔"],
    "秋霜": ["秋", "霜", "冷", "寒", "白", "叶", "露"],
    "雪景": ["雪", "白", "寒", "冬", "冰", "霜", "梅", "银"],
    "落日": ["日", "落", "暮", "夕", "晖", "霞", "山", "远"],
    "江水": ["江", "水", "舟", "波", "流", "岸", "潮", "渔"],
    "渔舟": ["渔", "舟", "水", "江", "网", "客", "帆", "晚"],
    "田园": ["田", "园", "村", "农", "稻", "桑", "溪", "耕"],
    "边塞": ["边", "塞", "军", "战", "关", "旗", "马", "月"],
    "月夜": ["月", "夜", "光", "影", "星", "寒", "桂", "明"],
    "寒夜": ["寒", "夜", "冷", "风", "雪", "月", "灯", "孤"],
    "晨曦": ["晨", "曦", "光", "日", "露", "鸟", "晓", "曙"],
    "暮色": ["暮", "色", "夕", "晚", "霞", "归", "暗", "余"],
    "远行": ["远", "行", "路", "山", "水", "程", "风", "客"],
    "怀古": ["古", "怀", "史", "遗迹", "兴", "亡", "叹", "旧"],
}

# ---------------------------------------------------------------------------
# Temperature / token presets
# ---------------------------------------------------------------------------
TASK_PARAMS = {
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
# Poetry extraction
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


# ---------------------------------------------------------------------------
# Form detection
# ---------------------------------------------------------------------------
def detect_expected_form(instruction: str) -> str | None:
    """Detect which poetry form the instruction asks for."""
    for form in FORM_SPEC:
        if form in instruction:
            return form
    return None


# ---------------------------------------------------------------------------
# Form compliance (weighted: 0.3 * line_count + 0.7 * chars_per_line)
# ---------------------------------------------------------------------------
def evaluate_form_compliance(output: str, expected_form: str) -> float:
    """Check if the poetry follows the specified form.

    Weighted scoring: 0.3 * line_count_ratio + 0.7 * chars_per_line_ratio.
    Returns 0-1 score.
    """
    spec = FORM_SPEC.get(expected_form)
    if not spec:
        return 0.0

    lines = extract_poem_lines(output)
    expected_lines = spec["lines"]
    expected_chars = spec["chars_per_line"]

    if len(lines) == 0:
        return 0.0

    # Line count score: how close is the line count to expected?
    line_count_ratio = min(len(lines), expected_lines) / expected_lines

    # Chars per line score: proportion of expected lines with correct char count
    chars_match = sum(1 for line in lines if len(line) == expected_chars)
    chars_per_line_ratio = chars_match / expected_lines if expected_lines > 0 else 0

    # Weighted combination
    return round(line_count_ratio * 0.3 + chars_per_line_ratio * 0.7, 4)


# ---------------------------------------------------------------------------
# Rhyme compliance (平水韵)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------
def detect_topic(instruction: str) -> str:
    """Extract topic keyword from instruction via known topic list."""
    for t in TOPIC_KEYWORDS:
        if t in instruction:
            return t
    return ""


# ---------------------------------------------------------------------------
# Topic relevance (keyword overlap + TOPIC_EXPANSIONS lookup)
# ---------------------------------------------------------------------------
def evaluate_topic_relevance(output: str, topic: str) -> float:
    """Compute topic relevance score based on keyword overlap and thematic expansion.

    Uses TOPIC_EXPANSIONS lookup for expanded topic character sets.
    final_score = max(base_score, 0.5 * base_score + 0.5 * expanded_score)

    Returns 0-1 score.
    """
    if not topic:
        return 0.5

    topic_chars = set(topic)
    poem_chars = set(output)
    overlap = topic_chars & poem_chars

    if len(topic_chars) == 0:
        return 0.5

    # Direct keyword overlap
    base_score = len(overlap) / len(topic_chars)

    # Expanded topic lookup (thematic related chars)
    expanded = TOPIC_EXPANSIONS.get(topic, list(topic_chars))
    expanded_chars = set(expanded)
    expanded_overlap = expanded_chars & poem_chars
    expanded_score = len(expanded_overlap) / len(expanded_chars) if expanded_chars else 0

    final_score = max(base_score, 0.5 * base_score + 0.5 * expanded_score)
    return round(min(final_score, 1.0), 4)


# ---------------------------------------------------------------------------
# Async Poet evaluation with per-form breakdown
# ---------------------------------------------------------------------------
async def evaluate_poet_model(
    client: AsyncOpenAI,
    model_name: str,
    test_data: list,
    base_url: str,
) -> dict:
    """Evaluate poet task: format, rhyme (平水韵), topic, diversity, per-form breakdown.

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
                client, model_name, POET_SYSTEM_PROMPT, instruction,
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
# Run comparison: base vs poet LoRA
# ---------------------------------------------------------------------------
async def run_poet_comparison(base_url: str, output_dir: str) -> None:
    """Evaluate base model vs poet adapter, save JSONs, print comparison + per-form breakdown + samples."""
    os.makedirs(output_dir, exist_ok=True)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    # Load test data
    test_data = load_test_data("poet")
    print(f"\n=== Poet Task: {len(test_data)} test samples ===")

    # Evaluate base model
    print(f"  Evaluating base model ({BASE_MODEL}) ...")
    t0 = time.time()
    base_results = await evaluate_poet_model(client, BASE_MODEL, test_data, base_url)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Evaluate poet LoRA
    print(f"  Evaluating Poet LoRA ({POET_ADAPTER}) ...")
    t0 = time.time()
    poet_results = await evaluate_poet_model(client, POET_ADAPTER, test_data, base_url)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # Save results (summary only, no details)
    base_summary = {k: v for k, v in base_results.items() if k != "details"}
    poet_summary = {k: v for k, v in poet_results.items() if k != "details"}

    with open(os.path.join(output_dir, "poet_base.json"), "w", encoding="utf-8") as f:
        json.dump(base_summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "poet_lora.json"), "w", encoding="utf-8") as f:
        json.dump(poet_summary, f, indent=2, ensure_ascii=False)

    # ----- Comparison table -----
    print(f"\n{'='*60}")
    print(f"{'Poet Task: Base Model vs Poet LoRA':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Base Model':<15} {'Poet LoRA':<15} {'Delta':<10}")
    print(f"{'-'*65}")

    for metric_key, display in [
        ("format_compliance", "Format Compliance"),
        ("rhyme_compliance", "Rhyme Compliance"),
        ("topic_relevance", "Topic Relevance"),
        ("diversity", "Diversity"),
    ]:
        base_val = base_results.get(metric_key, 0)
        poet_val = poet_results.get(metric_key, 0)
        delta = poet_val - base_val
        print(f"{display:<25} {base_val:<15.4f} {poet_val:<15.4f} {delta:<+10.4f}")

    # ----- Per-form breakdown -----
    print(f"\nPer-form breakdown:")
    for form in FORM_SPEC:
        base_form = base_results.get("per_form_metrics", {}).get(form, {})
        poet_form = poet_results.get("per_form_metrics", {}).get(form, {})
        if base_form or poet_form:
            b_fmt = base_form.get("avg_format_compliance", 0)
            p_fmt = poet_form.get("avg_format_compliance", 0)
            b_rhyme = base_form.get("avg_rhyme_compliance", 0)
            p_rhyme = poet_form.get("avg_rhyme_compliance", 0)
            b_n = base_form.get("count", 0)
            p_n = poet_form.get("count", 0)
            print(f"  {form}:")
            print(f"    Format: {b_fmt:.4f} -> {p_fmt:.4f} (n={b_n}/{p_n})")
            print(f"    Rhyme:  {b_rhyme:.4f} -> {p_rhyme:.4f}")

    # ----- Sample outputs -----
    print(f"\n{'='*60}")
    print("Sample outputs comparison:")
    print(f"{'='*60}")

    base_details = base_results.get("details", [])
    poet_details = poet_results.get("details", [])
    for i in range(min(3, len(base_details))):
        bd = base_details[i]
        pd = poet_details[i]
        print(f"\nInstruction: {bd['input']}...")
        print(f"  Base:  form={bd['format']}, rhyme={bd['rhyme']}, topic={bd['topic_match']}")
        print(f"         {bd['model_output'][:100]}...")
        print(f"  Poet:  form={pd['format']}, rhyme={pd['rhyme']}, topic={pd['topic_match']}")
        print(f"         {pd['model_output'][:100]}...")

    print(f"\nResults saved to {output_dir}/poet_base.json, {output_dir}/poet_lora.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate poet adapter (Chinese poetry generation)",
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

    asyncio.run(run_poet_comparison(args.base_url, output_dir))


if __name__ == "__main__":
    main()