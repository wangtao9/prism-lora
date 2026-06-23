# LoRA + vLLM 多适配器推理设计文档

**项目名**: prism-lora  
**日期**: 2026-06-23  
**基座模型**: Qwen2.5-1.5B-Instruct  
**训练框架**: LLaMAFactory  
**硬件**: 单卡 NVIDIA 16GB+

## 1. 项目目标

微调两个 LoRA 适配器，使用 vLLM 多适配器推理实现动态角色切换，并通过实验验证 LoRA 微调确实在各自领域产生了专用增强。

- **Judge LoRA**: 记忆冲突检测与更新——判断两条事实是否在同一维度有冲突，决定 KEEP（共存）还是 UPDATE（替换）
- **Poet LoRA**: 古诗写作——根据主题和风格要求创作古诗并赏析

## 2. 项目结构

```
prism-lora1/
├── README.md
├── run_all.sh                # 一站式启动脚本（串联所有环节）
├── configs/
│   ├── judge_lora.yaml       # LLaMAFactory 训练配置（Judge）
│   ├── poet_lora.yaml        # LLaMAFactory 训练配置（Poet）
│   └── vllm_config.json      # vLLM 多适配器启动配置
├── data/
│   ├── judge_train.json      # 记忆冲突训练数据 (~800条)
│   ├── judge_test.json       # 记忆冲突评测数据 (~200条)
│   ├── poet_train.json       # 古诗训练数据 (~500条)
│   └── poet_test.json        # 古诗评测数据 (~100条)
├── scripts/
│   ├── prepare_data.py       # 数据合成脚本
│   ├── train_lora.py         # 调用 LLaMAFactory 训练
│   ├── start_vllm.sh         # 启动 vLLM 多适配器服务
│   ├── query_adapter.py      # 动态切换适配器推理
│   ├── evaluate.py           # 评测对比脚本
│   └── cleanup.sh            # 清理中间文件
├── adapters/                 # LoRA adapter 输出目录
│   ├── judge/
│   └── poet/
├── results/                  # 评测结果输出
│   ├── judge_baseline.json
│   ├── judge_lora.json
│   ├── poet_baseline.json
│   ├── poet_lora.json
│   └── comparison.md         # 最终对比报告
```

## 3. 数据设计

### 3.1 Judge（记忆冲突检测与更新）

**核心任务**: 给模型两条事实（旧记忆 + 新事实），判断是否在同一维度有冲突。

**输入格式**:
```
旧记忆：张三喜欢吃苹果
新事实：张三不喜欢吃苹果
请判断新事实与旧记忆的关系，并决定处理策略。
```

**输出格式（结构化JSON）**:
```json
{
  "decision": "UPDATE",
  "reason": "两者描述同一维度（张三对苹果的喜好），但值相反，存在冲突",
  "updated_memory": "张三不喜欢吃苹果"
}
```

**三类数据分布**:

| 类型 | 示例 | decision | 占比 |
|------|------|----------|------|
| 同维度值冲突 | 旧:"张三喜欢吃苹果" 新:"张三不喜欢吃苹果" | UPDATE | ~40% |
| 同维度数值更新 | 旧:"北京人口2000万" 新:"北京人口2200万" | UPDATE | ~10% |
| 不同维度共存 | 旧:"张三喜欢吃苹果" 新:"张三喜欢吃香蕉" | KEEP | ~30% |
| 不同领域共存 | 旧:"张三30岁" 新:"张三在清华工作" | KEEP | ~20% |

- 训练集 ~800 条，测试集 ~200 条，正负比例（UPDATE:KEEP）约 1:1
- 数据由 `scripts/prepare_data.py` 通过模板+知识库自动合成

### 3.2 Poet（古诗写作）

**输入格式**:
```
请写一首关于秋天的七言绝句，风格要求：意境深远。
```

**输出格式**:
```
诗题：秋思
空山新雨后，天气晚来秋。
明月松间照，清泉石上流。

赏析：此诗以秋山雨后之景...
```

- 训练集 ~500 条，测试集 ~100 条
- 数据由 `scripts/prepare_data.py` 从经典古诗构造 prompt-response 对 + 自动生成变体

## 4. LoRA 训练配置

两个 LoRA 共享基座模型 Qwen2.5-1.5B-Instruct，使用 LLaMAFactory 训练。

| 配置项 | Judge LoRA | Poet LoRA |
|--------|------------|-----------|
| LoRA rank (r) | 8 | 8 |
| LoRA alpha | 16 | 16 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj | 同左 |
| 学习率 | 5e-4 | 5e-4 |
| Epochs | 3 | 3 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Max seq length | 512 | 1024 |
| 训练时间预估 | ~30min (单卡16GB) | ~45min (单卡16GB) |

LoraMAFactory yaml 配置文件由训练脚本引用，输出 HuggingFace 标准 LoRA adapter 格式。

## 5. vLLM 多适配器推理

### 5.1 启动命令

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --enable-lora \
  --lora-modules judge=adapters/judge poet=adapters/poet \
  --max-lora-rank 8 \
  --gpu-memory-utilization 0.85
```

关键参数：
- `--enable-lora`: 开启多适配器支持
- `--lora-modules`: 注册两个适配器，命名为 `judge` 和 `poet`
- `--max-lora-rank 8`: 与训练时 rank 匹配
- vLLM 通过 OpenAI-compatible API 提供服务

### 5.2 动态切换推理

通过 OpenAI API 的 `model` 参数动态切换角色，**无需重启 vLLM 实例**：

```python
# Judge 模式
response = client.chat.completions.create(
    model="judge",
    messages=[{"role": "user", "content": "旧记忆：张三喜欢吃苹果\n新事实：张三不喜欢吃苹果\n请判断新事实与旧记忆的关系，并决定处理策略。"}]
)

# Poet 模式
response = client.chat.completions.create(
    model="poet",
    messages=[{"role": "user", "content": "请写一首关于秋天的七言绝句"}]
)

# 基座模型（无 LoRA）
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[...]
)
```

## 6. 评测对比实验

### 6.1 实验矩阵

每个领域做 **基座 vs Judge-LoRA vs Poet-LoRA** 三组对比：

| | 基座模型 | Judge LoRA | Poet LoRA |
|---|---|---|---|
| Judge 任务 | baseline_judge | **lora_judge** | 交叉验证：应该无提升 |
| Poet 任务 | baseline_poet | 交叉验证：应该无提升 | **lora_poet** |

交叉验证证明：**LoRA 是领域专用的，不是通用提升**。

### 6.2 Judge 评测指标

- **Accuracy**: decision 字段（UPDATE / KEEP）是否与标注一致
- **F1-score**: 分别计算 UPDATE 和 KEEP 的 F1
- **Reason质量**: BLEU（辅助指标，不做硬对比）

测试集 200 条，3 个模型各跑一遍，结果存 `results/judge_baseline.json` / `results/judge_lora.json`。

### 6.3 Poet 评测指标

- **格式合规率**: 是否输出一首完整古诗（行数、字数符合要求）
- **主题相关性**: 是否围绕给定主题创作（关键词命中率）
- **韵律合规率**: 平仄、押韵基本合规（脚本自动检测）
- **多样性**: 同一主题多次生成结果不重复（distinct-n）

测试集 100 条，3 个模型各跑一遍，结果存 `results/poet_baseline.json` / `results/poet_lora.json`。

### 6.4 结果输出

`results/comparison.md` 由 `scripts/evaluate.py` 自动生成对比报告：

```markdown
## Judge 评测结果
| 模型 | Accuracy | F1(UPDATE) | F1(KEEP) |
|------|----------|------------|----------|
| 基座 | 0.xx     | 0.xx       | 0.xx     |
| Judge LoRA | 0.xx | 0.xx | 0.xx |
| Poet LoRA  | 0.xx | 0.xx | 0.xx |

## Poet 评测结果
| 模型 | 格式合规率 | 主题相关性 | 韵律合规率 |
|------|-----------|-----------|-----------|
| 基座 | xx% | xx% | xx% |
| Poet LoRA | xx% | xx% | xx% |
| Judge LoRA | xx% | xx% | xx% |

## 结论
Judge LoRA 在 Judge 任务上提升 xx%，在 Poet 任务上无提升。
Poet LoRA 在 Poet 任务上提升 xx%，在 Judge 任务上无提升。
→ LoRA 微调实现了领域专用增强，且不干扰其他领域。
```

## 7. 一站式流程

`run_all.sh` 串联所有环节，可从零到评测结果一键运行：

```bash
# Step 1: 合成数据
python scripts/prepare_data.py

# Step 2: 训练两个 LoRA
python scripts/train_lora.py --task judge
python scripts/train_lora.py --task poet

# Step 3: 启动 vLLM 多适配器服务
bash scripts/start_vllm.sh

# Step 4: 等待 vLLM 就绪后，运行评测
python scripts/evaluate.py

# Step 5: 生成对比报告
python scripts/evaluate.py --report

# 结果在 results/comparison.md
```

每个环节也可单独运行。

## 8. 技术依赖

```
# 核心依赖
vllm >= 0.6.0            # 多适配器推理
llamafactory >= 0.7.0    # LoRA 训练
transformers >= 4.40.0
peft >= 0.11.0
torch >= 2.3.0

# 评测辅助
scikit-learn             # F1 计算
nltk / jieba             # 文本分析
distinct-n               # 多样性指标
```

## 9. 风险与注意事项

- **vLLM 版本**: 多适配器功能需要 vLLM >= 0.6.0，确保版本兼容
- **LoRA 格式**: LLaMAFactory 输出的 adapter 必须是 HuggingFace PEFT 格式，否则 vLLM 无法加载
- **GPU 内存**: 1.5B 模型 + 2个 LoRA adapter 需要 ~8GB，单卡 16GB 充裕
- **模型首次下载**: vLLM 启动时会从 HuggingFace 下载基座模型，需确保网络通畅或提前下载
