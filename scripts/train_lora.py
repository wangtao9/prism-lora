#!/usr/bin/env python3
"""
prism-lora LoRA 训练脚本
通过 LLaMAFactory CLI 执行 Judge / Poet LoRA 微调

用法：
  python scripts/train_lora.py --task judge
  python scripts/train_lora.py --task poet

数据集注册使用 data/dataset_info.json + configs/ 中 dataset_dir: data 的相对路径方式，
与 LLaMAFactory 标准用法一致，无需自定义 llamafactory_data 目录。
"""

import argparse
import os
import subprocess
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 任务 → YAML 配置映射
TASK_CONFIG = {
    "judge": os.path.join(PROJECT_ROOT, "configs", "judge_lora.yaml"),
    "poet": os.path.join(PROJECT_ROOT, "configs", "poet_lora.yaml"),
}


def check_dataset_info(task_name: str):
    """验证 dataset_info.json 中是否有对应的数据集注册"""
    info_path = os.path.join(PROJECT_ROOT, "data", "dataset_info.json")

    if not os.path.isfile(info_path):
        print(f"Error: data/dataset_info.json not found at {info_path}")
        print("Please run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    import json
    with open(info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    key = f"{task_name}_train"
    if key not in dataset_info:
        print(f"Error: dataset '{key}' not registered in dataset_info.json")
        print(f"Available datasets: {list(dataset_info.keys())}")
        print("Please run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    # 验证数据文件实际存在
    rel_path = dataset_info[key]["file_name"]
    abs_path = os.path.join(PROJECT_ROOT, "data", rel_path)
    if not os.path.isfile(abs_path):
        print(f"Error: training data file not found: {abs_path}")
        print("Please run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    print(f"Dataset '{key}' registered: {rel_path} → verified at {abs_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter via LLaMAFactory")
    parser.add_argument(
        "--task", required=True, choices=["judge", "poet"],
        help="Which adapter to train: judge or poet",
    )
    args = parser.parse_args()

    task = args.task
    config_file = TASK_CONFIG[task]

    if not os.path.isfile(config_file):
        print(f"Error: config file not found: {config_file}")
        sys.exit(1)

    # 验证数据集注册（LLaMAFactory 通过 dataset_dir: data + dataset_info.json 自动处理）
    check_dataset_info(task)

    # 构建命令（LLaMAFactory 会自动从 YAML 中的 dataset_dir 找 dataset_info.json）
    cmd = ["llamafactory-cli", "train", config_file]
    print(f"Running: {cmd}")
    print(f"  Config: {config_file}")
    print(f"  Dataset lookup: data/dataset_info.json (relative paths, dataset_dir=data)")

    # 执行训练（LLaMAFactory 用 YAML 中 dataset_dir=data 自动定位）
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"Training completed successfully (exit code {result.returncode})")
    except subprocess.CalledProcessError as e:
        print(f"Error: LLaMAFactory training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: llamafactory-cli not found. Please install LLaMAFactory:")
        print("  pip install llamafactory")
        sys.exit(1)


if __name__ == "__main__":
    main()
