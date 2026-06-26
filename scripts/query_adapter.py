#!/usr/bin/env python3
"""Interactive and single-query script for dynamic LoRA adapter switching.

Switches between judge, poet, and base model modes via the vLLM OpenAI API,
demonstrating multi-adapter LoRA serving with dynamic routing.
"""

import argparse
import sys

from openai import OpenAI

from configs.config import (
    BASE_MODEL,
    MODE_MAP,
    JUDGE_SYSTEM_PROMPT,
    POET_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_MAP,
    VLLM_BASE_URL,
)


def query_once(client: OpenAI, mode: str, user_input: str, max_tokens: int = 256) -> str:
    """Perform a single inference query with the specified adapter mode.

    Args:
        client: OpenAI client connected to vLLM server.
        mode: One of 'judge', 'poet', or 'base'.
        user_input: The user prompt to send.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response text.
    """
    model = MODE_MAP[mode]
    system_prompt = SYSTEM_PROMPT_MAP[mode]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_tokens=max_tokens,
        temperature=0.1 if mode == "judge" else 0.7,
    )

    return response.choices[0].message.content


def interactive_mode(client: OpenAI, max_tokens: int = 256) -> None:
    """Run an interactive REPL with dynamic adapter switching.

    Commands:
        /judge  - Switch to judge adapter
        /poet   - Switch to poet adapter
        /base   - Switch to base model (no adapter)
        /quit   - Exit the REPL
    """
    current_mode = "base"
    print(f"Interactive mode started. Current mode: {current_mode}")
    print("Commands: /judge, /poet, /base, /quit")
    print()

    while True:
        try:
            user_input = input(f"[{current_mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            command = user_input.lower()
            if command == "/quit":
                print("Goodbye!")
                break
            elif command in ("/judge", "/poet", "/base"):
                current_mode = command.lstrip("/")
                print(f"Switched to {current_mode} mode (model: {MODE_MAP[current_mode]})")
            else:
                print(f"Unknown command: {user_input}")
                print("Available commands: /judge, /poet, /base, /quit")
            continue

        # Regular query
        try:
            result = query_once(client, current_mode, user_input, max_tokens)
            print(f"\n{result}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def demo_mode(client: OpenAI, max_tokens: int = 256) -> None:
    """Run a demo showing all three adapter modes with sample queries."""
    print("=" * 60)
    print("Prism-LoRA 多适配器推理演示")
    print("=" * 60)
    print()

    judge_query = (
        "旧记忆：张三喜欢吃苹果\n"
        "新事实：张三不喜欢吃苹果\n"
        "请判断新事实与旧记忆的关系，并决定处理策略。"
    )
    poet_query = "请写一首关于秋天的七言绝句，风格要求：意境深远。"
    base_query = "你好，请介绍一下你自己。"

    for mode, query, label in [
        ("judge", judge_query, "Judge 模式 — 记忆冲突检测"),
        ("poet", poet_query, "Poet 模式 — 古诗创作"),
        ("base", base_query, "基座模型 — 通用对话"),
    ]:
        print(f"--- {label} ---")
        print(f"Model: {MODE_MAP[mode]}")
        print(f"Query: {query}")
        try:
            result = query_once(client, mode, query, max_tokens)
            print(f"Response:\n{result}")
        except Exception as e:
            print(f"Error: {e}")
        print()

    print("=" * 60)
    print("演示完成。使用 --interactive 进入交互式切换模式。")
    print("=" * 60)


def main() -> None:
    """Entry point with argparse for mode, input, interactive, port, and max-tokens."""
    parser = argparse.ArgumentParser(
        description="Query vLLM with dynamic LoRA adapter switching",
    )
    parser.add_argument(
        "--mode",
        choices=["judge", "poet", "base"],
        default=None,
        help="Adapter mode to use (judge, poet, or base)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single query input text",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive REPL with /judge, /poet, /base, /quit commands",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VLLM_BASE_URL.split(":")[-1].rstrip("/v1"),
        help=f"vLLM server port (default: {VLLM_BASE_URL.split(":")[-1].rstrip('/v1')})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens in response (default: 256)",
    )

    args = parser.parse_args()

    # Create OpenAI client pointing to vLLM server
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",  # vLLM does not require an API key
    )

    # Health check — verify the server is reachable
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

    # Dispatch based on arguments
    if args.interactive:
        interactive_mode(client, args.max_tokens)
    elif args.mode and args.input:
        result = query_once(client, args.mode, args.input, args.max_tokens)
        print(f"[{args.mode}] Response:\n{result}")
    else:
        # No specific input or interactive flag — run demo
        demo_mode(client, args.max_tokens)


if __name__ == "__main__":
    main()
