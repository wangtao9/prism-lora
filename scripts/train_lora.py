#!/usr/bin/env python3
"""
prism-lora LoRA 训练脚本
通过 LLaMAFactory CLI 执行 Judge / Poet LoRA 微调

用法：
  python scripts/train_lora.py --task judge
  python scripts/train_lora.py --task poet

训练 YAML 使用 ${base_model}、${template}、${flash_attn} 等占位符，
运行时由 render_yaml() 从 configs/config.yaml 读取值并渲染到临时文件，
源模板不会被修改。
"""

import argparse
import os
import subprocess
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 从统一配置导入
sys.path.insert(0, PROJECT_ROOT)
from configs.config import BASE_MODEL, render_yaml

# 任务 → YAML 模板映射
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
    template_file = TASK_CONFIG[task]

    if not os.path.isfile(template_file):
        print(f"Error: config template not found: {template_file}")
        sys.exit(1)

    # 验证数据集注册
    check_dataset_info(task)

    # 渲染 YAML 模板：将 ${base_model}、${template}、${flash_attn} 等占位符
    # 替换为 configs/config.yaml 中的实际值，输出到临时文件
    rendered_config = render_yaml(template_file)
    print(f"  Template: {template_file}")
    print(f"  Rendered: {rendered_config}")
    print(f"  Model:    {BASE_MODEL}")

    # 构建命令（LLaMAFactory 使用渲染后的临时 YAML）
    cmd = ["llamafactory-cli", "train", rendered_config]
    print(f"Running: {' '.join(cmd)}")

    # 执行训练
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
    finally:
        # 清理临时文件
        if os.path.exists(rendered_config):
            os.remove(rendered_config)
            print(f"  Cleaned up: {rendered_config}")


if __name__ == "__main__":
    main()