# Prism-LoRA 2x3 Cross-Comparison Evaluation Report

> 数据来源：`judge_base.json` / `judge_lora.json` / `poet_base.json` / `poet_lora.json` / `cross_eval.json`

## Judge Task (Memory Conflict Detection)

| 模型 | Accuracy | F1(UPDATE) | F1(KEEP) | Valid/Total |
|------|----------|------------|----------|-------------|
| base (Qwen2.5-1.5B) | 0.6067 | 0.7552 | 0.0000 | 300/300 |
| judge | 1.0000 | 1.0000 | 1.0000 | 300/300 |
| poet | 0.5800* | 0.7342* | —* | 100/100* |

> \* 诗人LoRA在Judge任务上的结果来自 `cross_eval.json` 的100样本子集，无 F1(KEEP) 数据。

**关键发现**：Judge LoRA 在测试集上达到完美准确率（1.0），高度疑似**过拟合/记忆答案**——基座模型将所有样本预测为 UPDATE（KEEP 的 precision/recall/F1 均为 0），而 LoRA 完美区分两个类别，这种极端跃升不符合正常学习曲线。

## Poet Task (Chinese Poetry Writing)

| 模型 | format_compliance | rhyme_compliance | topic_relevance | diversity |
|------|-------------------|------------------|-----------------|-----------|
| base (Qwen2.5-1.5B) | 0.0021 | 0.2117 | 0.4067 | 0.3964 |
| judge | 0.1489* | 0.1550* | 0.5937* | —* |
| poet | 0.2704 | 0.3816 | 0.3124 | 0.2892 |

> \* Judge LoRA 在Poet任务上的结果来自 `cross_eval.json` 的100样本子集，无 diversity 数据。

**关键发现**：Poet LoRA 在格式合规性（+0.2683）和押韵合规性（+0.1699）上明显优于基座模型，但 topic_relevance 反而下降（0.4067 → 0.3124），diversity 也降低（0.3964 → 0.2892），说明 LoRA 学到了格式/押韵模式，但牺牲了主题相关性和多样性。

## Per-Form Breakdown (Poet Task, Full 477 Samples)

### 基座模型

| 诗体 | count | avg_format_compliance | avg_rhyme_compliance |
|------|-------|-----------------------|----------------------|
| 五言绝句 | 120 | 0.1512 | 0.0000 |
| 七言绝句 | 120 | 0.1583 | 0.0042 |
| 五言律诗 | 117 | 0.1487 | 0.3504 |
| 七言律诗 | 120 | 0.1506 | 0.2854 |

### Poet LoRA

| 诗体 | count | avg_format_compliance | avg_rhyme_compliance |
|------|-------|-----------------------|----------------------|
| 五言绝句 | 120 | 0.6529 | 0.4333 |
| 七言绝句 | 120 | 0.3894 | 0.2000 |
| 五言律诗 | 117 | 0.1573 | 0.3483 |
| 七言律诗 | 120 | 0.3129 | 0.3083 |

**诗体差异**：Poet LoRA 对绝句（尤以五言绝句）提升最显著，律诗改善有限——格式合规性在绝句上提升约 0.23~0.50，但律诗几乎不变。

## 交叉领域特化矩阵 (Cross-Domain Specialization Matrix, 100-Sample Subset)

| Model | Conflict(Acc) | Conflict(F1) | Poet(Form) | Poet(Rhyme) |
|-------|---------------|--------------|------------|-------------|
| Base Model | 0.5800 | 0.7342 | 0.1500 | 0.1700 |
| Judge LoRA | 1.0000 | 1.0000 | 0.1489 | 0.1550 |
| Poet LoRA | 0.5800 | 0.7342 | 0.2995 | 0.3000 |

## 可视化

### Judge Task 对比

![Judge Comparison](judge_comparison.png)

### Poet Task 对比

![Poet Comparison](poet_comparison.png)

### 交叉领域热力图

![Cross-Domain Heatmap](cross_domain_heatmap.png)

## Conclusions

- Judge LoRA improvement (Accuracy delta): **+0.3933** (0.6067 → 1.0000)
- Poet LoRA improvement (format_compliance delta): **+0.2683** (0.0021 → 0.2704)
- Poet LoRA improvement (rhyme_compliance delta): **+0.1699** (0.2117 → 0.3816)

## Specialization Verdict (4 Conditions)

基于 `cross_eval.json` 的100样本子集：

- Condition 1: Judge LoRA improves on judge task (delta > 0): **PASS** (Δ = +0.4200)
  > ⚠️ 但 Judge LoRA 达到完美准确率，高度疑似过拟合/记忆训练数据
- Condition 2: Poet LoRA does NOT improve on judge task (|delta| < 0.05): **PASS** (Δ = +0.0000)
- Condition 3: Poet LoRA improves on poet task (delta > 0): **PASS** (Δ = +0.1495)
- Condition 4: Judge LoRA does NOT improve on poet task (|delta| < 0.05): **PASS** (Δ = -0.0011)

### Result: ✓ TRUE SPECIALIZATION (形式上通过)

> **重要保留意见**：虽然4个条件形式上全部通过，但 Condition 1 的结果是建立在对训练数据过拟合的基础上（完美1.0准确率），不能反映真实泛化能力。建议：
> 1. 对 Judge LoRA 使用独立的保留测试集（非训练/验证数据）重新评估
> 2. 分析训练 loss 曲线，检查是否出现记忆信号（训练 loss → 0 而验证 loss 上升）
> 3. 增加数据集规模或添加正则化（dropout、weight decay）来缓解过拟合