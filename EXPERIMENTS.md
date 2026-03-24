# C3 实验记录

> 本文档记录了 C3 (Context Cascade Compression) 项目中各组模型与实验的基本信息、训练策略和评估结果。

---

## 1. 模型架构概览

C3 采用 **编码器-解码器** 架构，将长上下文压缩为固定数量的 latent tokens，再由解码器基于这些 latent tokens 进行重建或续写。

| 组件 | 规格 |
|------|------|
| 编码器 (Encoder) | hidden_size = 1536, 基于原始 C3 压缩模块 |
| 解码器 (Decoder) | Qwen2.5-3B, hidden_size = 2048, 36 layers, 16 heads, 2 KV heads |
| Latent Tokens | 32 |
| 最大位置编码 | 32768 |
| 词表大小 | 151860 |
| 精度 | bfloat16 |

### 训练策略 (两阶段)

- **Phase 1**: 冻结解码器，仅训练编码器 + Q-Former + mm_projector，使用 100% 重建任务
- **Phase 2**: 全模型微调（可选 decoder 低学习率），使用重建 + 续写混合任务

---

## 2. 基线模型

### 2.1 qwen25-3b

| 项目 | 说明 |
|------|------|
| 来源 | HuggingFace 官方权重 (Qwen2.5-3B) |
| 路径 | `./models/qwen25-3b/` |
| 说明 | 原始 Qwen2.5-3B 预训练模型，作为解码器能力的基线参考 |

### 2.2 c3

| 项目 | 说明 |
|------|------|
| 来源 | 美团 C3 官方 HuggingFace 权重 |
| 路径 | `./models/c3/` |
| 说明 | 原始 C3 模型（编码器 + 解码器），作为压缩重建能力的基线参考 |

---

## 3. 第一轮实验 (v1)

> **注意**: v1 实验的训练数据和数据处理代码已被 v2 覆盖，以下数据量为根据 trainer_state 日志推算。

### 3.1 数据

数据来源与 v2 一致（见第 4 节），但下载数量约为 v2 的 **2/5**。

| 数据集 | 用途 |
|--------|------|
| FineWeb-Edu | 重建 |
| Cosmopedia v2 | 重建 |
| Proof-Pile-2 | 重建 |
| SlimPajama | 续写 |

根据 trainer_state 推算数据量：

| 阶段 | 推算样本数 | 计算方式 |
|------|-----------|---------|
| Phase 1 | ~550K | 17,184 steps / 8 epochs × 256 batch |
| Phase 2 | ~170K | 1,992 steps / 3 epochs × 256 batch |

Phase 2 重建/续写配比与 v2 一致 (约 7:3)。

### 3.2 c3-phase1

| 参数 | 值 |
|------|-----|
| 阶段 | Phase 1（冻结解码器） |
| 输出 | `./output/phase1/` |
| 训练数据 | ~550K（100% 重建） |
| Epochs | 8 |
| 总步数 | 17,184 |
| 有效 batch size | 8 × 4 × 8 GPU = 256 |
| 学习率 | 1e-5 |
| 调度器 | cosine |
| Warmup | 200 steps |
| Weight decay | 0.0 |
| 最大序列长度 | 8192 |
| DeepSpeed | ZeRO-2, bf16 |
| 梯度裁剪 | 1.0 |
| Gradient checkpointing | 是 |
| 训练耗时 | ~24.2h |
| 最终 loss | 0.7205 |
| 保留 checkpoints | 14000, 16000, 17184 |

### 3.3 c3-phase2

| 参数 | 值 |
|------|-----|
| 阶段 | Phase 2（全模型微调） |
| 输出 | `./output/phase2/` |
| Phase 1 来源 | `./output/phase1/` |
| 训练数据 | ~170K（70% 重建 + 30% 续写） |
| Epochs | 3 |
| 总步数 | 1,992 |
| 有效 batch size | 256 |
| 学习率 | 1e-5 |
| Decoder LR ratio | 1.0 |
| 调度器 | cosine |
| Warmup | 100 steps |
| 训练耗时 | ~3.2h |
| 最终 loss | 0.9494 |
| 保留 checkpoints | 1000, 1500, 1992 |

### 3.4 c3-phase2-recon-only

消融实验：Phase 2 仅使用重建数据，不包含续写任务。

| 参数 | 值 |
|------|-----|
| 阶段 | Phase 2（全模型微调，仅重建） |
| 输出 | `./output/phase2_recon_only/` |
| Phase 1 来源 | `./output/phase1/` |
| 训练数据 | ~170K（100% 重建） |
| Epochs | 3 |
| 总步数 | 1,992 |
| 有效 batch size | 256 |
| 学习率 | 1e-5 |
| Decoder LR ratio | 0.02 |
| 调度器 | cosine |
| Warmup | 100 steps |
| 训练耗时 | ~3.2h |
| 最终 loss | 0.4888 |
| 保留 checkpoints | 1000, 1500, 1992 |

---

## 4. 第二轮实验 (v2)

### 4.1 数据

#### 数据下载

| 数据集 | HuggingFace ID | 下载量 | Token 范围 | 筛选条件 |
|--------|----------------|--------|-----------|---------|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` (sample-10BT) | 1,500,000 | 400–1500 | score ≥ 3.0 |
| Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | 300,000 | 400–1200 | — |
| Proof-Pile-2 | `EleutherAI/proof-pile-2` (algebraic-stack) | 200,000 | 400–1500 | — |
| SlimPajama | `DKYoon/SlimPajama-6B` | 600,000 | 800–2000 | — |

**重建数据来源配比**: FineWeb-Edu 75% + Cosmopedia 15% + Proof-Pile 10%

#### 数据处理

| 数据集 | 样本数 | 构成 |
|--------|--------|------|
| `phase1_train.jsonl` | 2,000,000 | 100% 重建 |
| `phase2_train.jsonl` | 2,000,000 | 1,400,000 重建 (70%) + 600,000 续写 (30%) |

续写数据的 context/target 切分比例：0.4–0.7（随机）。

### 4.2 c3-p1-v2 (Phase 1)

| 参数 | 值 |
|------|-----|
| 阶段 | Phase 1（冻结解码器） |
| 输出 | `./output/phase1_v2/` |
| C3 编码器来源 | `./models/c3` |
| 解码器来源 | `./models/qwen25-3b` |
| 训练数据 | 2,000,000（100% 重建） |
| Epochs | 5（计划），实际在 step 20,000 / epoch 2.56 时停止 |
| 计划总步数 | 39,060 |
| 实际训练步数 | 20,000 |
| 有效 batch size | 8 × 4 × 8 GPU = 256 |
| 学习率 | 1e-5 |
| 调度器 | cosine |
| Warmup | 200 steps |
| Weight decay | 0.0 |
| 最大序列长度 | 8192 |
| DeepSpeed | ZeRO-2, bf16 |
| 梯度裁剪 | 1.0 |
| Gradient checkpointing | 是 |
| Step 20000 loss | 0.4901 |
| Step 20000 学习率 | ~4.85e-6 |
| Total FLOPs | 1.626e+20 |
| 保留 checkpoints | 10000, 15000, 20000 |

### 4.3 c3-p2-v2 (Phase 2)

| 参数 | 值 |
|------|-----|
| 阶段 | Phase 2（全模型微调） |
| 输出 | `./output/phase2_v2/` |
| Phase 1 来源 | `./output/phase1_v2/checkpoint-20000` |
| 训练数据 | 2,000,000（70% 重建 + 30% 续写） |
| Epochs | 3 |
| 总步数 | 23,436 |
| 有效 batch size | 256 |
| 学习率 | 1e-5 |
| Decoder LR ratio | 1.0（编码器与解码器同一学习率） |
| 调度器 | cosine |
| Warmup | 100 steps |
| Weight decay | 0.0 |
| 最大序列长度 | 8192 |
| DeepSpeed | ZeRO-2, bf16 |
| 训练耗时 | ~37.2h |
| 最终 loss | 0.548 |
| Total FLOPs | 1.807e+20 |
| 保留 checkpoints | 16000, 18000, 20000, 22000, 23436 |

---

## 5. 各组公共超参对照

| 参数 | Phase 1 (v1) | Phase 2 (v1) | Phase 2 recon-only (v1) | Phase 1 (v2) | Phase 2 (v2) |
|------|:---:|:---:|:---:|:---:|:---:|
| 冻结解码器 | 是 | 否 | 否 | 是 | 否 |
| 学习率 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| Decoder LR ratio | — | 1.0 | 0.02 | — | 1.0 |
| 有效 batch size | 256 | 256 | 256 | 256 | 256 |
| Epochs | 8 | 3 | 3 | 5 (停于2.56) | 3 |
| 总步数 | 17,184 | 1,992 | 1,992 | 20,000 | 23,436 |
| Warmup | 200 | 100 | 100 | 200 | 100 |
| 调度器 | cosine | cosine | cosine | cosine | cosine |
| 序列长度 | 8192 | 8192 | 8192 | 8192 | 8192 |
| 训练数据(重建) | ~550K | ~119K | ~170K | 2M | 1.4M |
| 训练数据(续写) | — | ~51K | — | — | 0.6M |
| DeepSpeed | ZeRO-2 | ZeRO-2 | ZeRO-2 | ZeRO-2 | ZeRO-2 |

---

## 6. 评估结果汇总

### 6.1 WikiText-2 Perplexity（提取的解码器）

| 模型 | PPL | 相对 qwen25-3b |
|------|-----|----------------|
| qwen25-3b | 7.0946 | — |
| c3 (美团) | 18.4433 | +159.96% |
| c3-phase1 | 7.0946 | +0.00% |
| c3-phase2 | 7.1102 | +0.22% |
| c3-phase2-recon-only | 7.1184 | +0.34% |
| c3-p2-v2 | 7.6429 | +7.73% |

> c3-phase1 PPL 与 qwen25-3b 完全一致，因为 Phase 1 冻结解码器，其权重未改变。

### 6.2 NLP Benchmarks（提取的解码器，lm_eval）

| 模型 | MMLU | HellaSwag | ARC-Easy | ARC-Challenge | WinoGrande |
|------|------|-----------|----------|---------------|------------|
| qwen25-3b | — | 73.63% | 73.15% | 47.27% | 68.82% |
| c3 (美团) | 25.94% | 45.29% | 50.13% | 27.82% | 52.64% |
| c3-phase1 | 65.23% | 73.53% | 73.11% | 47.44% | 68.51% |
| c3-phase2 | 65.10% | 73.35% | 72.14% | 47.44% | 68.82% |
| c3-phase2-recon-only | 65.12% | 73.50% | 72.22% | 47.44% | 68.67% |
| c3-p2-v2 | 62.31% | 73.03% | 75.46% | 47.53% | 69.53% |

### 6.3 重建准确率（编码器 → 解码器，C3 全模型）

**wikitext-2** (50 samples, avg 162 tokens):

| 模型 | Char Prec | ROUGE-L | Token F1 | Exact Match |
|------|-----------|---------|----------|-------------|
| c3 (美团) | 1.0000 | 1.0000 | 1.0000 | 100% |
| c3-phase1 | 0.2935 | 0.2538 | 0.2700 | 0% |
| c3-phase2 | 0.2337 | 0.2179 | 0.2318 | 0% |
| c3-phase2-recon-only | 0.3126 | 0.3303 | 0.3443 | 0% |
| c3-p1-v2 | 0.4215 | 0.2780 | 0.2895 | 0% |
| c3-p2-v2 | 0.7490 | 0.8010 | 0.8078 | 44% |

**mixed-1k** (20 samples, avg 1024 tokens):

| 模型 | Char Prec | ROUGE-L | Token F1 | Exact Match |
|------|-----------|---------|----------|-------------|
| c3 (美团) | 0.9957 | 0.9936 | 0.9945 | 40% |
| c3-phase1 | 0.2930 | 0.2416 | 0.4307 | 0% |
| c3-phase2 | 0.3064 | 0.2296 | 0.3890 | 0% |
| c3-phase2-recon-only | 0.3272 | 0.2960 | 0.4905 | 0% |
| c3-p1-v2 | 0.3111 | 0.2826 | 0.4776 | 0% |
| c3-p2-v2 | 0.9601 | 0.9524 | 0.9672 | 20% |

### 6.4 Context-Compressed NLP Benchmarks (threshold=256)

| 模型 | MMLU | HellaSwag | ARC-Easy | ARC-Challenge | WinoGrande |
|------|------|-----------|----------|---------------|------------|
| c3 (美团) | 23.20% | 35.96% | 33.54% | 24.66% | 52.33% |
| c3-phase2 | 24.58% | 52.05% | 38.30% | 28.07% | 52.25% |
| c3-phase2-recon-only | 23.11% | 37.46% | 33.33% | 24.49% | 50.59% |
| c3-p1-v2 | 22.90% | 38.30% | 32.70% | 26.10% | 49.50% |
| c3-p2-v2 | 24.80% | 60.95% | 44.78% | 31.91% | 53.28% |

---

## 7. 关键发现

1. **数据量显著影响重建质量**: v2 使用 2M 样本训练，重建指标远超 v1（~550K/~170K 样本），c3-p2-v2 在 mixed-1k 上达到 Token F1 0.967，接近美团原始 C3
2. **Phase 2 全模型微调是关键**: Phase 1 仅训练编码器，重建能力有限；Phase 2 联合训练后大幅提升
3. **续写任务有助于 NLP 能力保持**: c3-phase2 (含续写) 在 compressed NLP 上优于 c3-phase2-recon-only (纯重建)
4. **解码器 PPL 存在轻微退化**: v2 全模型微调 (decoder_lr_ratio=1.0) 导致 PPL 从 7.09 升至 7.64 (+7.7%)，而 v1 的退化极小 (+0.2%)，这可能与 v2 训练步数多和 decoder 学习率高有关
5. **v1 的 decoder_lr_ratio=0.02 更好地保护了解码器**: c3-phase2-recon-only 使用低 decoder 学习率，PPL 几乎不变 (7.12)

---

## 8. 文件路径索引

| 类别 | 路径 |
|------|------|
| 训练脚本 | `train/run_phase1.sh`, `train/run_phase2.sh`, `train/run_phase2_recon_only.sh` |
| 训练代码 | `train/train_c3v2.py`, `train/trainer_c3v2.py`, `train/dataset_c3v2.py` |
| 数据下载 | `data/download_data.py` |
| 数据处理 | `data/prepare_data.py` |
| 处理后数据 | `data/processed/phase1_train.jsonl`, `data/processed/phase2_train.jsonl` |
| DeepSpeed 配置 | `train/ds_zero2.json` |
| 模型输出 | `output/phase1/`, `output/phase2/`, `output/phase2_recon_only/`, `output/phase1_v2/`, `output/phase2_v2/` |
| 评估结果 | `eval/results/{模型名}/` |
| 评估日志 | `eval/logs/{模型名}/` |
| 评估汇总 | `eval/results/summary.md` |
