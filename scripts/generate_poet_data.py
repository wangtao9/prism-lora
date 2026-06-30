#!/usr/bin/env python3
"""
Poet LoRA 数据生成脚本：使用 OpenAI 兼容 API 生成原创古诗。

支持任意 OpenAI 兼容的 LLM 提供商（Claude、GLM、DeepSeek 等），
通过 --base-url / --model / --api-key 配置。

替代 prepare_data.py 中直接使用 POEMS_DB 名篇原文的做法。
每个 (form, topic, poet_style) 组合生成一首原创诗，
然后与指令模板配对组装成训练数据。

特性：
  - 信号量限流，支持并发控制
  - JSONL 缓存，支持断点续传
  - 四阶段验证（形式/押韵/原创性/非空）
  - 自动重试（最多 3 次）

用法：
  # GLM
  LLM_API_KEY=xxx python scripts/generate_poet_data.py \\
      --base-url https://open.bigmodel.cn/api/paas/v4 \\
      --model glm-4-flash

  # DeepSeek
  LLM_API_KEY=xxx python scripts/generate_poet_data.py \\
      --base-url https://api.deepseek.com/v1 \\
      --model deepseek-chat

  # Claude (via OpenAI compat)
  LLM_API_KEY=xxx python scripts/generate_poet_data.py \\
      --base-url https://api.anthropic.com/v1 \\
      --model claude-haiku-4-5-20251001
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "poet", "raw_poetry")
CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.jsonl")

sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# 诗歌配置（与 prepare_data.py / poet_eval.py 共享）
# ---------------------------------------------------------------------------
POETRY_FORMS = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]

FORM_SPEC = {
    "五言绝句": {"lines": 4, "chars_per_line": 5},
    "七言绝句": {"lines": 4, "chars_per_line": 7},
    "五言律诗": {"lines": 8, "chars_per_line": 5},
    "七言律诗": {"lines": 8, "chars_per_line": 7},
}

TOPICS = [
    "春雨", "秋月", "山水", "离别", "思乡", "登高", "夜思",
    "梅花", "荷花", "柳树", "春风", "秋霜", "雪景", "落日",
    "渔舟", "田园", "边塞", "故人", "归途",
    "月夜", "寒夜", "晨曦", "暮色", "远行", "怀古",
    "听雨", "咏竹", "望远", "松涛", "清泉", "孤舟",
    "长河", "烽火", "古道", "深林", "寒山", "碧水",
    "暮雨", "晓风", "残阳", "孤雁", "绿野", "烟波",
]

POETS_TO_IMITATE = [
    "李白", "杜甫", "王维", "苏轼",
]

INSTRUCTION_TEMPLATES = [
    "写一首{form}，以{topic}为题。",
    "请创作一首{form}，描写{topic}的意境。",
    "以{topic}为主题，创作一首{form}。要求意境深远。",
    "模仿{poet}的风格，写一首关于{topic}的{form}。",
    "请以{poet}的笔触，描绘{topic}，体裁为{form}。",
    "创作{form}一首，咏{topic}。",
    "用{form}的形式，表达对{topic}的感受。",
    "以{topic}入诗，作{form}一首。",
]

POET_SYSTEM_PROMPT = (
    "你是一位精通古诗词的创作大师，擅长根据要求创作符合格律和意境的古典诗词。"
    "你的创作严格遵守古典诗词的体裁规范，包括字数、行数和押韵。"
)

# ---------------------------------------------------------------------------
# POEMS_DB（仅作风格参考，不再用于训练数据）
# ---------------------------------------------------------------------------

# 从 prepare_data.py 导入 POEMS_DB
from scripts.prepare_data import POEMS_DB


def get_poet_style_examples(poet: str, form: str) -> str:
    """从 POEMS_DB 中取指定诗人在指定体裁下的 1-2 首诗，作为风格参考。"""
    poems = POEMS_DB.get(form, [])
    matching = [p for p in poems if p["author"] == poet]
    if not matching:
        # 退而求其次：该诗人的任意体裁
        for f, plist in POEMS_DB.items():
            matching = [p for p in plist if p["author"] == poet]
            if matching:
                break
    if matching:
        example = matching[0]
        return f"《{example['title']}》({example['author']})\n{example['text']}"
    return ""


# ---------------------------------------------------------------------------
# 押韵验证（复用 eval/poet_eval.py 的 RHYME_GROUPS）
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

CHAR_TO_RHYME = {}
for _grp, _chars in RHYME_GROUPS.items():
    for _ch in _chars:
        CHAR_TO_RHYME[_ch] = _grp


def _strip_punctuation(text: str) -> str:
    """去除中文标点和空格。"""
    return re.sub(r'[，。！？；：、""''（）《》\s]', '', text)


def _split_poem_lines(text: str) -> list[str]:
    """将诗歌文本按行拆分，兼容两种 LLM 输出格式：
    1. 每行一句（标准）：  床前明月光，\n疑是地上霜。
    2. 两句一行（常见）：  床前明月光，疑是地上霜。\n举头望明月，低头思故乡。
    当某行字数是预期每行字数的 2 倍时，按逗号/句号拆成两行。
    """
    raw_lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            raw_lines.append(line)

    # 尝试检测每行字数，判断是否需要拆分
    # 先按纯文字（去标点）计算每行字数
    char_counts = []
    for line in raw_lines:
        cleaned = _strip_punctuation(line)
        if cleaned:
            char_counts.append(len(cleaned))

    if not char_counts:
        return []

    # 如果大多数行字数接近 2x 预期，说明两句写在一行
    # 用中位数检测：如果中位数 > 6，大概率需要拆
    char_counts.sort()
    median_chars = char_counts[len(char_counts) // 2]

    need_split = median_chars > 6  # 超过6字大概率是两句合一

    if not need_split:
        # 格式1：直接返回去标点后的行
        result = []
        for line in raw_lines:
            cleaned = _strip_punctuation(line)
            if cleaned:
                result.append(cleaned)
        return result

    # 格式2：按中文逗号和句号拆分
    result = []
    for line in raw_lines:
        # 按中文逗号、句号拆成子句
        sub_lines = re.split(r'[，。！？；]', line)
        for sub in sub_lines:
            cleaned = _strip_punctuation(sub)
            if cleaned:
                result.append(cleaned)
    return result


def check_form_compliance(text: str, expected_form: str) -> tuple[bool, float]:
    """检查诗歌形式合规性。返回 (合规, 得分)。"""
    spec = FORM_SPEC[expected_form]
    expected_lines = spec["lines"]
    expected_chars = spec["chars_per_line"]

    lines = _split_poem_lines(text)

    if not lines:
        return False, 0.0

    # 行数检查
    line_score = 1.0 if len(lines) == expected_lines else 0.0

    # 字数检查：至少 80% 的行字数正确
    correct_lines = sum(1 for l in lines if len(l) == expected_chars)
    char_ratio = correct_lines / len(lines) if lines else 0
    char_score = 1.0 if char_ratio >= 0.8 else char_ratio / 0.8

    score = 0.3 * line_score + 0.7 * char_score
    passed = score >= 0.7 and len(lines) == expected_lines
    return passed, round(score, 3)


def check_rhyme_compliance(text: str) -> tuple[bool, float]:
    """检查押韵合规性。取偶数句末字判断是否同韵。返回 (合规, 得分)。"""
    lines = _split_poem_lines(text)

    if len(lines) < 2:
        return False, 0.0

    # 取偶数句（0-indexed: 1, 3, 5, 7）的末字
    rhyme_chars = []
    for i in range(1, len(lines), 2):
        last_char = _strip_punctuation(lines[i])[-1:]
        if last_char:
            rhyme_chars.append(last_char)

    if not rhyme_chars:
        return False, 0.0

    # 检查韵字是否属于同一韵部
    rhyme_groups_found = set()
    for ch in rhyme_chars:
        if ch in CHAR_TO_RHYME:
            rhyme_groups_found.add(CHAR_TO_RHYME[ch])

    if not rhyme_groups_found:
        return False, 0.0

    # 所有的韵字都在同一韵部 → 满分
    if len(rhyme_groups_found) == 1:
        return True, 1.0

    # 大部分同韵 → 部分得分
    main_group = max(rhyme_groups_found, key=lambda g: sum(1 for ch in rhyme_chars if CHAR_TO_RHYME.get(ch) == g))
    match_ratio = sum(1 for ch in rhyme_chars if CHAR_TO_RHYME.get(ch) == main_group) / len(rhyme_chars)
    return match_ratio >= 0.5, round(match_ratio, 3)


def check_originality(text: str, threshold: float = 0.8) -> tuple[bool, float]:
    """检查原创性：与 POEMS_DB 不应行级大面积重叠。返回 (原创, 重叠率)。"""
    # 提取生成诗的行
    gen_lines = set()
    for line in _split_poem_lines(text):
        if len(line) >= 3:
            gen_lines.add(line)

    if not gen_lines:
        return False, 1.0

    # 提取 POEMS_DB 所有行
    db_lines = set()
    for form, poems in POEMS_DB.items():
        for poem in poems:
            for line in poem["text"].split("\n"):
                cleaned = _strip_punctuation(line.strip())
                if cleaned and len(cleaned) >= 3:
                    db_lines.add(cleaned)

    # 计算重叠率
    overlap_count = sum(1 for line in gen_lines if line in db_lines)
    overlap_ratio = overlap_count / len(gen_lines)

    return overlap_ratio < threshold, round(overlap_ratio, 3)


def validate_poem(text: str, expected_form: str) -> tuple[bool, dict]:
    """四阶段验证。返回 (通过, 详情dict)。"""
    if not text or not text.strip():
        return False, {"error": "empty", "stage": "nonnull"}

    form_ok, form_score = check_form_compliance(text, expected_form)
    if not form_ok:
        return False, {"error": "form", "stage": "form", "form_score": form_score}

    rhyme_ok, rhyme_score = check_rhyme_compliance(text)
    if not rhyme_ok:
        return False, {"error": "rhyme", "stage": "rhyme", "rhyme_score": rhyme_score}

    orig_ok, overlap_ratio = check_originality(text)
    if not orig_ok:
        return False, {"error": "originality", "stage": "originality", "overlap_ratio": overlap_ratio}

    return True, {
        "form_score": form_score,
        "rhyme_score": rhyme_score,
        "overlap_ratio": overlap_ratio,
    }


# ---------------------------------------------------------------------------
# LLM API 集成（OpenAI 兼容接口）
# ---------------------------------------------------------------------------

def build_llm_prompt(form: str, topic: str, poet_style: str | None = None) -> str:
    """构造 LLM prompt。"""
    spec = FORM_SPEC[form]
    lines = spec["lines"]
    chars = spec["chars_per_line"]

    parts = [
        f"请创作一首原创古诗，要求如下：",
        f"",
        f"体裁：{form}（{lines}行，每行{chars}个字）",
        f"主题：{topic}",
    ]

    if poet_style:
        style_ref = get_poet_style_examples(poet_style, form)
        parts.append(f"风格：模仿{poet_style}的风格特点")
        if style_ref:
            parts.append(f"")
            parts.append(f"参考{poet_style}的风格特点（仅供参考风格，不要照搬）：")
            parts.append(style_ref)

    parts.extend([
        f"",
        f"重要：请创作一首全新的、原创的诗，不要直接引用任何已有诗作。",
        f"只输出诗的正文，每行用换行分隔，行内用中文逗号和句号分隔。",
        f"不要输出标题、作者或任何额外说明。",
        f"",
        f"输出格式：",
        f"第一行",
        f"第二行",
        f"...",
        f"第{lines}行",
    ])

    return "\n".join(parts)


def build_generation_requests() -> list[dict]:
    """构建去重的生成请求列表。"""
    requests = []
    seen = set()

    for form in POETRY_FORMS:
        for topic in TOPICS:
            # 无诗人风格版本
            key = (form, topic, None)
            if key not in seen:
                seen.add(key)
                requests.append({
                    "form": form,
                    "topic": topic,
                    "poet_style": None,
                    "prompt": build_llm_prompt(form, topic),
                })

            # 有诗人风格版本
            for poet in POETS_TO_IMITATE:
                key = (form, topic, poet)
                if key not in seen:
                    seen.add(key)
                    requests.append({
                        "form": form,
                        "topic": topic,
                        "poet_style": poet,
                        "prompt": build_llm_prompt(form, topic, poet),
                    })

    return requests


async def generate_one_poem(
    client,
    model: str,
    request: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict | None:
    """单次 LLM API 调用，带信号量限流和重试。"""
    from openai import AsyncOpenAI

    async with semaphore:
        for attempt in range(max_retries):
            try:
                prompt = request["prompt"]
                # 重试时追加提示
                if attempt > 0:
                    prompt += "\n\n请确保诗的格式严格按照上述要求，每行字数必须准确。"

                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": POET_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"enable_thinking": False},
                )

                poem_text = response.choices[0].message.content.strip()

                # 兼容 thinking 模型：如果返回内容被 <thinking> 包裹，提取正文
                if not poem_text and hasattr(response.choices[0].message, 'reasoning_content'):
                    reasoning = response.choices[0].message.reasoning_content
                    if reasoning:
                        poem_text = reasoning.strip()

                # 再次兜底：从 thinking 标签中提取
                if not poem_text:
                    full_response = str(response.choices[0].message)
                    import re as _re
                    think_match = _re.search(r'</think>(.*)', full_response, _re.DOTALL)
                    if think_match:
                        poem_text = think_match.group(1).strip()

                # 验证
                valid, details = validate_poem(poem_text, request["form"])

                if valid:
                    return {
                        "form": request["form"],
                        "topic": request["topic"],
                        "poet_style": request["poet_style"],
                        "poem_text": poem_text,
                        "validation": details,
                        "attempts": attempt + 1,
                    }
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)  # 重试间隔
                        continue
                    else:
                        return {
                            "form": request["form"],
                            "topic": request["topic"],
                            "poet_style": request["poet_style"],
                            "poem_text": poem_text,
                            "validation": details,
                            "attempts": attempt + 1,
                            "failed": True,
                        }

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    return {
                        "form": request["form"],
                        "topic": request["topic"],
                        "poet_style": request["poet_style"],
                        "poem_text": "",
                        "validation": {"error": str(e), "stage": "api"},
                        "attempts": attempt + 1,
                        "failed": True,
                    }

    return None


async def generate_all_poems(
    requests: list[dict],
    base_url: str,
    api_key: str,
    model: str,
    max_concurrent: int = 5,
) -> list[dict]:
    """批量生成诗歌，带缓存和进度。"""
    from openai import AsyncOpenAI

    os.makedirs(CACHE_DIR, exist_ok=True)

    # 加载缓存
    cached = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        key = (entry["form"], entry["topic"], entry.get("poet_style"))
                        cached[key] = entry
                    except json.JSONDecodeError:
                        pass
        print(f"  Loaded {len(cached)} cached poems from {CACHE_FILE}")

    # 过滤已缓存的请求
    to_generate = []
    for req in requests:
        key = (req["form"], req["topic"], req.get("poet_style"))
        if key not in cached:
            to_generate.append(req)

    print(f"  Total requests: {len(requests)}, Cached: {len(cached)}, To generate: {len(to_generate)}")

    if not to_generate:
        print("  All requests already cached. Nothing to generate.")
        return list(cached.values())

    # 创建 API 客户端
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    semaphore = asyncio.Semaphore(max_concurrent)

    # 分批生成（每批 max_concurrent 个并发）
    results = list(cached.values())
    failed_count = 0
    done_count = 0
    total = len(to_generate)

    print(f"  Generating {total} poems (concurrency={max_concurrent})...")

    # 使用 asyncio.Semaphore 控制并发，逐个完成即时打印进度
    t0 = time.time()

    async def _run_with_progress(idx: int, req: dict):
        nonlocal done_count, failed_count
        result = await generate_one_poem(client, model, req, semaphore)
        done_count += 1

        if result is None or isinstance(result, Exception):
            failed_count += 1
            elapsed = time.time() - t0
            speed = done_count / elapsed if elapsed > 0 else 0
            print(f"  [{done_count}/{total}] FAIL (speed: {speed:.1f}/s)")
            return None

        # 写入缓存
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result.get("failed"):
            failed_count += 1
            details = result.get("validation", {})
            elapsed = time.time() - t0
            speed = done_count / elapsed if elapsed > 0 else 0
            print(f"  [{done_count}/{total}] {result['form']}/{result['topic']} — "
                  f"FAIL: {details.get('error', 'unknown')} (speed: {speed:.1f}/s)")
        else:
            elapsed = time.time() - t0
            speed = done_count / elapsed if elapsed > 0 else 0
            eta = (total - done_count) / speed if speed > 0 else 0
            print(f"  [{done_count}/{total}] {result['form']}/{result['topic']} — "
                  f"OK (speed: {speed:.1f}/s, ETA: {eta:.0f}s)")

        return result

    tasks = [_run_with_progress(i, req) for i, req in enumerate(to_generate)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in batch_results:
        if result is not None and not isinstance(result, Exception):
            results.append(result)

    elapsed = time.time() - t0
    successful = len(results) - failed_count
    print(f"  Generation complete: {successful} valid, {failed_count} failed, {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# 训练数据组装
# ---------------------------------------------------------------------------

def assemble_training_data(generated_poems: list[dict]) -> None:
    """将生成的诗歌与指令模板配对，输出 sharegpt JSON。"""
    # 只保留验证通过的诗歌
    valid_poems = [p for p in generated_poems if not p.get("failed") and p.get("poem_text")]

    print(f"\n=== Assembling Training Data ===")
    print(f"  Valid poems: {len(valid_poems)}")

    if not valid_poems:
        print("  ERROR: No valid poems to assemble. Exiting.")
        return

    # 按 form 分组
    by_form = {}
    for p in valid_poems:
        form = p["form"]
        if form not in by_form:
            by_form[form] = []
        by_form[form].append(p)

    print(f"  Per form: {', '.join(f'{k}={len(v)}' for k, v in by_form.items())}")

    # 将诗歌分配到 train/val/test（同首诗不出现在多个 split）
    import random
    random.seed(42)

    train_poems = []
    val_poems = []
    test_poems = []

    for form, poems in by_form.items():
        random.shuffle(poems)
        n = len(poems)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.12))
        # rest goes to test

        train_poems.extend(poems[:n_train])
        val_poems.extend(poems[n_train:n_train + n_val])
        test_poems.extend(poems[n_train + n_val:])

    print(f"  Split: train={len(train_poems)}, val={len(val_poems)}, test={len(test_poems)}")

    # 对每首诗生成多个指令变体
    def build_records(poems: list[dict], variants_per_poem: int = 5) -> list[dict]:
        records = []
        for poem in poems:
            form = poem["form"]
            topic = poem["topic"]
            poet = poem.get("poet_style")
            poem_text = poem["poem_text"]

            # 收集适用的模板
            applicable_templates = []
            for tmpl in INSTRUCTION_TEMPLATES:
                if "{poet}" in tmpl:
                    if poet:
                        applicable_templates.append(tmpl.format(form=form, topic=topic, poet=poet))
                else:
                    applicable_templates.append(tmpl.format(form=form, topic=topic))

            # 选择 variants_per_poem 个变体
            if len(applicable_templates) >= variants_per_poem:
                selected = random.sample(applicable_templates, variants_per_poem)
            else:
                selected = applicable_templates * (variants_per_poem // len(applicable_templates) + 1)
                selected = selected[:variants_per_poem]

            for instruction in selected:
                records.append({
                    "conversations": [
                        {"from": "system", "value": POET_SYSTEM_PROMPT},
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": poem_text},
                    ]
                })

        return records

    train_records = build_records(train_poems, variants_per_poem=3)
    val_records = build_records(val_poems, variants_per_poem=3)
    test_records = build_records(test_poems, variants_per_poem=3)

    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    # 写出 JSON
    poet_dir = os.path.join(DATA_DIR, "poet")
    os.makedirs(poet_dir, exist_ok=True)

    for name, data, path in [
        ("train", train_records, os.path.join(poet_dir, "train.json")),
        ("val", val_records, os.path.join(poet_dir, "val.json")),
        ("test", test_records, os.path.join(poet_dir, "test.json")),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  poet/{name}.json: {len(data)} samples")

    # 打印样例
    if train_records:
        sample = train_records[0]
        print(f"\n  --- Sample (first record) ---")
        for turn in sample["conversations"]:
            role = turn["from"]
            val_preview = turn["value"][:100] + "..." if len(turn["value"]) > 100 else turn["value"]
            print(f"    {role}: {val_preview}")

    print(f"\n  Done! Poet training data saved to {poet_dir}/")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

async def async_main(base_url: str, model: str, api_key: str, max_concurrent: int) -> None:
    print("=== Poet LoRA Data Generation (LLM API) ===")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Cache: {CACHE_FILE}")
    print()

    # 构建生成请求
    requests = build_generation_requests()
    print(f"  Generation requests: {len(requests)}")
    print()

    # 批量生成
    results = await generate_all_poems(requests, base_url, api_key, model, max_concurrent)

    # 组装训练数据
    assemble_training_data(results)

    # 更新 dataset_info.json
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    dataset_info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

    dataset_info["poet_train"] = {"file_name": "poet/train.json", "formatting": "sharegpt"}
    dataset_info["poet_val"] = {"file_name": "poet/val.json", "formatting": "sharegpt"}

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"\n  dataset_info.json updated at {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate original poetry data using OpenAI-compatible LLM API",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        help="OpenAI-compatible API base URL (default: LLM_BASE_URL env or GLM endpoint)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "glm-4-flash"),
        help="Model name (default: LLM_MODEL env or glm-4-flash)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLM_API_KEY", ""),
        help="API key (default: LLM_API_KEY env)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: LLM_API_KEY is not set.")
        print("Set it via --api-key or environment variable:")
        print("  export LLM_API_KEY=your-api-key")
        print("  # or")
        print("  LLM_API_KEY=xxx python scripts/generate_poet_data.py")
        sys.exit(1)

    asyncio.run(async_main(args.base_url, args.model, args.api_key, args.concurrent))


if __name__ == "__main__":
    main()