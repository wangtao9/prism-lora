# prism-lora

LoRA + vLLM 多适配器推理实验：记忆冲突检测（Judge）× 古诗写作（Poet）

## 项目简介

微调两个 LoRA 适配器于 Qwen2.5-1.5B-Instruct 基座模型：
- **Judge LoRA**: 记忆冲突检测与更新——判断两条事实是否在同一维度有冲突（UPDATE 或 KEEP）
- **Poet LoRA**: 古诗写作——根据主题和风格要求创作古诗并赏析

使用 vLLM `--enable-lora` 加载多个适配器，通过 OpenAI-compatible API 的 `model` 参数动态切换角色，并通过交叉评测验证领域专用增强效果。

## 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU (16GB+ VRAM)
- CUDA 11.8+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 一站式运行

```bash
bash run_all.sh
```

这会自动完成：数据合成 → LoRA 训练 → vLLM 启动 → 评测 → 生成对比报告。

### 分步运行

```bash
# 1. 合成数据
python scripts/prepare_data.py

# 2. 训练 LoRA（需要 GPU）
python scripts/train_lora.py --task judge
python scripts/train_lora.py --task poet

# 3. 启动 vLLM 服务
bash scripts/start_vllm.sh

# 4. 评测（vLLM 服务运行中）
python scripts/evaluate.py

# 5. 生成报告
python scripts/evaluate.py --report
```

### 交互式推理

```bash
# 交互式模式，可动态切换 Judge/Poet/Base
python scripts/query_adapter.py --interactive

# 单次查询
python scripts/query_adapter.py --mode judge --input "旧记忆：张三喜欢吃苹果\n新事实：张三不喜欢吃苹果"
python scripts/query_adapter.py --mode poet --input "写一首关于秋天的七言绝句"
```

## 评测结果

查看 `results/comparison.md` 获取完整对比报告。

实验设计为 2×3 交叉矩阵：

| 任务 | 基座模型 | Judge LoRA | Poet LoRA |
|------|---------|------------|-----------|
| Judge | baseline | ✓ 预期提升 | ✗ 交叉验证 |
| Poet | baseline | ✗ 交叉验证 | ✓ 预期提升 |

## 项目结构

```
prism-lora/
├── configs/          # LLaMAFactory 训练配置
├── data/             # 合成的训练/测试数据
├── adapters/         # LoRA adapter 输出
├── scripts/          # 工具脚本
├── results/          # 评测结果 + 对比报告
├── run_all.sh        # 一站式运行
└── requirements.txt  # 依赖声明
```

## 技术栈

- 基座模型: Qwen2.5-1.5B-Instruct
- 训练框架: LLaMAFactory
- 推理引擎: vLLM (多适配器模式)
- 评测: scikit-learn + 自定义指标
