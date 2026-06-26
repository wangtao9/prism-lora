"""Centralized configuration for prism-lora.

Single source of truth: configs/config.yaml
Python code and shell scripts all read from this file.
Environment variables override YAML values.

Usage:
    from configs.config import BASE_MODEL, render_yaml

    # Read a config value
    print(BASE_MODEL)

    # Render a training YAML template (e.g. judge_lora.yaml) to a
    # temporary file with ${...} placeholders resolved, then pass
    # the temp file to llamafactory-cli.
    rendered = render_yaml("configs/judge_lora.yaml")
    subprocess.run(["llamafactory-cli", "train", rendered])
"""

import os
import re
import tempfile

import yaml

# ---------------------------------------------------------------------------
# Locate config.yaml
# ---------------------------------------------------------------------------
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.yaml")


def _load_config() -> dict:
    """Load config.yaml, with environment variable overrides."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment variable overrides take precedence
    if "PRISM_BASE_MODEL" in os.environ:
        cfg["base_model"] = os.environ["PRISM_BASE_MODEL"]
    if "PRISM_VLLM_PORT" in os.environ:
        cfg["vllm_port"] = int(os.environ["PRISM_VLLM_PORT"])

    return cfg


_CFG = _load_config()

# ---------------------------------------------------------------------------
# Public constants — imported by other modules
# ---------------------------------------------------------------------------
BASE_MODEL: str = _CFG["base_model"]
HF_MODEL_ID: str = _CFG["hf_model_id"]

JUDGE_ADAPTER: str = _CFG["judge_adapter"]
POET_ADAPTER: str = _CFG["poet_adapter"]

VLLM_PORT: int = int(_CFG["vllm_port"])
VLLM_BASE_URL: str = f"http://localhost:{VLLM_PORT}/v1"
VLLM_GPU_UTIL: float = float(_CFG["vllm_gpu_util"])
VLLM_MAX_MODEL_LEN: int = int(_CFG["vllm_max_model_len"])

MODE_MAP = {
    "judge": JUDGE_ADAPTER,
    "poet": POET_ADAPTER,
    "base": BASE_MODEL,
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "你是一个记忆冲突检测专家。给定旧记忆和新事实，你需要判断它们是否"
    "在同一维度上存在冲突。如果冲突则输出UPDATE并用新事实替换旧记忆，"
    "如果不冲突则输出KEEP让两条记忆共存。请以JSON格式输出："
    "{\"decision\": \"UPDATE/KEEP\", \"reason\": \"...\", "
    "\"updated_memory\": \"...\"}"
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
# YAML template renderer
# ---------------------------------------------------------------------------
# Template variables available in training YAML files:
#   ${base_model}     — model path
#   ${template}       — LLaMAFactory template name
#   ${flash_attn}     — flash attention type
#   Plus any key from config.yaml


def _build_vars() -> dict:
    """Build the variable dict for ${...} substitution from config.yaml."""
    return dict(_CFG)


def render_yaml(template_path: str) -> str:
    """Render a YAML template by replacing ${var} placeholders.

    Replaces ``${key}`` with the corresponding value from config.yaml.
    Writes the resolved YAML to a **temporary file** (auto-deleted when
    the process exits) and returns the temp file path.

    The source template is never modified — llamafactory-cli receives
    the rendered copy.

    Args:
        template_path: Path to the YAML template (e.g. configs/judge_lora.yaml).

    Returns:
        Path to the rendered temporary YAML file.
    """
    variables = _build_vars()

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise ValueError(
                f"Unknown variable ${{{key}}} in {template_path}. "
                f"Available: {sorted(variables.keys())}"
            )
        return str(variables[key])

    resolved = re.sub(r"\$\{(\w+)\}", _replace, content)

    # Write to a temp file that auto-cleans on process exit
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="prism_lora_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(resolved)

    return tmp_path