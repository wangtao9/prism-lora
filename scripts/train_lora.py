#!/usr/bin/env python3
"""
prism-lora LoRA 训练脚本
通过 LLaMAFactory CLI 执行 Judge / Poet LoRA 微调
"""

import argparse
import json
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


def register_dataset(task_name: str) -> str:
    """
    将训练数据注册到 LLaMAFactory 本地数据集索引中。
    返回 dataset_info.json 的路径。
    """
    # 数据文件路径
    train_file = os.path.join(PROJECT_ROOT, "data", f"{task_name}_train.json")
    if not os.path.isfile(train_file):
        print(f"Error: training data file not found: {train_file}")
        sys.exit(1)

    # 创建 LLaMAFactory 数据目录和索引文件
    lf_data_dir = os.path.join(PROJECT_ROOT, "llamafactory_data")
    os.makedirs(lf_data_dir, exist_ok=True)

    info_path = os.path.join(lf_data_dir, "dataset_info.json")

    # 加载已有的索引（如果存在）
    dataset_info = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

    # 注册数据集
    dataset_info[f"{task_name}_train"] = {
        "file_name": os.path.abspath(train_file),
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
        },
    }

    # 写回索引文件
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Dataset '{task_name}_train' registered in {info_path}")
    return lf_data_dir


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter via LLaMAFactory")
    parser.add_argument(
        "--task",
        required=True,
        choices=["judge", "poet"],
        help="Which adapter to train: judge or poet",
    )
    args = parser.parse_args()

    task = args.task
    config_file = TASK_CONFIG[task]

    if not os.path.isfile(config_file):
        print(f"Error: config file not found: {config_file}")
        sys.exit(1)

    # 注册数据集
    lf_data_dir = register_dataset(task)

    # 设置环境变量，让 LLaMAFactory 使用我们的本地数据目录
    env = os.environ.copy()
    env["LLAMAFACTORY_DATASET_DIR"] = os.path.abspath(lf_data_dir)

    # 构建命令
    cmd = ["llamafactory-cli", "train", config_file]
    print(f"Running: {cmd}")
    print(f"  LLAMAFACTORY_DATASET_DIR={env['LLAMAFACTORY_DATASET_DIR']}")

    # 执行训练
    try:
        result = subprocess.run(cmd, env=env, check=True)
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
