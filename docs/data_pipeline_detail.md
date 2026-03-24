# C3v2 训练数据流水线详解

## 目录

- [1. 数据流水线总览](#1-数据流水线总览)
- [2. 第一步：数据下载与过滤](#2-第一步数据下载与过滤)
  - [2.1 四大数据源详解](#21-四大数据源详解)
  - [2.2 过滤机制](#22-过滤机制)
  - [2.3 下载机制](#23-下载机制)
- [3. 第二步：数据预处理与阶段划分](#3-第二步数据预处理与阶段划分)
  - [3.1 两种任务类型](#31-两种任务类型)
  - [3.2 Phase 1 数据组成](#32-phase-1-数据组成)
  - [3.3 Phase 2 数据组成](#33-phase-2-数据组成)
  - [3.4 Phase 2 Recon-Only（消融实验）](#34-phase-2-recon-only消融实验)
- [4. 第三步：训练时数据加载与格式化](#4-第三步训练时数据加载与格式化)
  - [4.1 JSONL 格式](#41-jsonl-格式)
  - [4.2 Reconstruct 任务的数据构造](#42-reconstruct-任务的数据构造)
  - [4.3 Continuation 任务的数据构造](#43-continuation-任务的数据构造)
  - [4.4 DataCollator 批处理](#44-datacollator-批处理)
- [5. 数据量与配比总结](#5-数据量与配比总结)
- [6. Token 长度分布](#6-token-长度分布)
- [7. 完整流程图](#7-完整流程图)

---

## 1. 数据流水线总览

C3v2 的训练数据流水线分为三个阶段：

```
HuggingFace 数据集（远程）
        │
        ▼
  ① download_data.py ── 流式下载 + token 长度过滤 + 质量过滤
        │
        ▼
  data/raw/*.jsonl（本地原始 JSONL）
        │
        ▼
  ② prepare_data.py ── 合并、任务标注、采样、打乱
        │
        ▼
  data/processed/phase{1,2}_train.jsonl（训练用 JSONL）
        │
        ▼
  ③ dataset_c3v2.py ── 训练时加载，构造 encoder/decoder 输入对
        │
        ▼
  模型训练（train_c3v2.py）
```

一键运行整条流水线：

```bash
bash data/run_data_pipeline.sh
```

预计耗时：下载 3-6 小时（4 路并行），预处理约 30 分钟。

---

## 2. 第一步：数据下载与过滤

**入口脚本：** `data/download_data.py`

从 HuggingFace Hub 流式下载 4 个公开数据集，边下载边用 tokenizer 计算 token 数量，只保留符合长度区间的文本。

### 2.1 四大数据源详解

| 数据集 | HuggingFace ID | Subset / Split | 目标数量 | Token 范围 | 额外过滤 | 用途 |
|:------|:--------------|:---------------|:--------|:----------|:--------|:-----|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` / train | **1,500,000** | 400–1,500 | `score >= 3.0` | 重建 (reconstruct) |
| **Cosmopedia v2** | `HuggingFaceTB/cosmopedia-v2` | `cosmopedia-v2` / train | **300,000** | 400–1,200 | 无 | 重建 (reconstruct) |
| **Proof-Pile-2** | `EleutherAI/proof-pile-2` | `algebraic-stack` / train | **200,000** | 400–1,500 | 无 | 重建 (reconstruct) |
| **SlimPajama** | `DKYoon/SlimPajama-6B` | 无 subset / train | **600,000** | 800–2,000 | 无 | 续写 (continuation) |

**各数据集内容特点：**

- **FineWeb-Edu**：从 Common Crawl 中筛选的高教育价值网页文本。使用 10BT 子集（约 100 亿 token 的采样版）。额外要求教育质量评分 `score >= 3.0`，确保文本具有良好的知识密度和可读性。这是最大的数据来源，占重建数据的 **75%**。

- **Cosmopedia v2**：由 LLM 合成的教科书风格文本，涵盖科学、历史、数学等多个领域。文本质量高、结构规范，适合作为重建目标。限制 token 上限为 1,200，比其他数据集更短，是因为该数据集的合成文本普遍较短。占重建数据的 **15%**。

- **Proof-Pile-2 (algebraic-stack)**：数学和代码相关的文本，来自 EleutherAI 的 Proof-Pile-2 数据集中的 algebraic-stack 子集。包含数学论文、代码、形式化证明等。目的是让模型学会压缩和重建结构化/符号密集的内容。占重建数据的 **10%**。

- **SlimPajama**：通用预训练语料的去重精简版（6B token 子集），包含 Common Crawl、C4、GitHub、Wikipedia、StackExchange、ArXiv 等多种来源。token 范围设为 800–2,000，比重建数据更长，因为续写任务需要前半部分作为上下文、后半部分作为生成目标，文本需足够长才有意义。是唯一用于续写任务的数据集。

### 2.2 过滤机制

每条文本经过以下检查：

1. **非空检查**：`text` 字段不为空且为字符串
2. **额外过滤函数**（仅 FineWeb-Edu）：`score >= 3.0`
3. **Token 长度过滤**：使用 Qwen2.5-3B tokenizer 编码后的 token 数需在 `[min_tokens, max_tokens]` 区间内

过滤后的每条记录保存为：

```json
{
  "text": "文章正文...",
  "source": "fineweb-edu",
  "num_tokens": 823
}
```

### 2.3 下载机制

- **流式下载**：使用 HuggingFace `datasets` 库的 `streaming=True` 模式，不需要先下载完整数据集
- **4 路并行**：使用 Python `multiprocessing`，同时下载 4 个数据集
- **断点续传**：每收集 10,000 条数据就保存进度到 `.progress` 文件，中断后可从上次位置继续
- **本地加载**（可选）：FineWeb-Edu 支持从预下载的本地 parquet 文件加载

**输出文件：**

```
data/raw/
├── finewebedu.jsonl      # ~1,500,000 条
├── cosmopedia.jsonl      # ~300,000 条
├── proofpile.jsonl       # ~200,000 条
├── slimpajama.jsonl      # ~600,000 条
└── logs/
    ├── download_finewebedu.log
    ├── download_cosmopedia.log
    ├── download_proofpile.log
    └── download_slimpajama.log
```

---

## 3. 第二步：数据预处理与阶段划分

**入口脚本：** `data/prepare_data.py`

将 4 个原始 JSONL 合并、标注任务类型、按阶段需求采样和混合。

### 3.1 两种任务类型

| 任务 | 标识 | 含义 | 训练目标 |
|:----|:-----|:-----|:--------|
| **Reconstruct（重建）** | `"task": "reconstruct"` | 给定完整文本，模型需从压缩表征中还原原文 | 学习信息压缩和忠实还原 |
| **Continuation（续写）** | `"task": "continuation"` | 给定文本前半部分，模型从压缩表征中续写后半部分 | 学习在压缩条件下生成连贯文本 |

### 3.2 Phase 1 数据组成

**文件：** `data/processed/phase1_train.jsonl`

| 来源 | 任务类型 | 数量 | 占比 |
|:----|:-------|:-----|:----|
| FineWeb-Edu | reconstruct | 1,500,000 | 75.0% |
| Cosmopedia v2 | reconstruct | 300,000 | 15.0% |
| Proof-Pile-2 | reconstruct | 200,000 | 10.0% |
| **合计** | **100% reconstruct** | **2,000,000** | **100%** |

**构建逻辑：**

```python
# FineWeb-Edu + Cosmopedia + Proof-Pile 全部标记为 reconstruct
reconstruct_data = []
for item in finewebedu + cosmopedia + proofpile:
    reconstruct_data.append({"task": "reconstruct", "text": item["text"]})

# Phase 1 = 全部 reconstruct，随机打乱
phase1 = reconstruct_data.copy()
random.shuffle(phase1)
```

Phase 1 仅使用重建任务，因为此阶段冻结 decoder，只训练 encoder + Q-Former + mm_projector，目标是让编码器学会把长文本压缩为固定数量的 latent token。

### 3.3 Phase 2 数据组成

**文件：** `data/processed/phase2_train.jsonl`

| 来源 | 任务类型 | 数量 | 占比 |
|:----|:-------|:-----|:----|
| FineWeb-Edu / Cosmopedia / Proof-Pile（采样） | reconstruct | 1,400,000 | 70.0% |
| SlimPajama | continuation | 600,000 | 30.0% |
| **合计** | **70% 重建 + 30% 续写** | **2,000,000** | **100%** |

**构建逻辑：**

```python
# 从 200 万 reconstruct 数据中随机采样 140 万
phase2_recon = random.sample(reconstruct_data, 1_400_000)

# SlimPajama 全部标记为 continuation，附带随机切分比例
continuation_data = []
for item in slimpajama:
    split_ratio = random.uniform(0.4, 0.7)  # 前 40%-70% 作为上下文
    continuation_data.append({
        "task": "continuation",
        "text": item["text"],
        "split_ratio": round(split_ratio, 3),
    })

# Phase 2 = 140 万 reconstruct + 60 万 continuation，混合打乱
phase2 = phase2_recon + continuation_data
random.shuffle(phase2)
```

> **注意：Phase 2 的 140 万重建数据是从与 Phase 1 完全相同的 `reconstruct_data` pool（200 万条）中 `random.sample` 采样的子集。** 也就是说 Phase 1 和 Phase 2 共享同一批文本数据，Phase 2 只是少用了 60 万条重建数据，腾出空间给续写数据。

**续写任务的切分比例 `split_ratio`：**

- 范围：`[0.4, 0.7]`，均匀随机采样
- 含义：对每条文本，前 `split_ratio` 部分作为 encoder 输入（上下文），后 `1 - split_ratio` 部分作为 decoder 生成目标
- 例如 `split_ratio = 0.55`，1000 token 的文本 → 前 550 token 给 encoder 压缩，后 450 token 由 decoder 续写

Phase 2 解冻全部参数，引入续写任务是为了让模型在压缩表征上进行自由生成，而不仅仅是逐字复制。

### 3.4 Phase 2 Recon-Only（消融实验）

**文件：** `data/processed/phase2_recon_only_train.jsonl`

用于对比实验，Phase 2 仅使用 reconstruct 数据（不含 continuation），与标准 Phase 2 对照，验证续写任务的贡献。

| 来源 | 任务类型 | 数量 | 占比 |
|:----|:-------|:-----|:----|
| FineWeb-Edu / Cosmopedia / Proof-Pile（全部） | reconstruct | 2,000,000 | 100% |

通过 `--phase2_recon_only` 标志启用。

---

## 4. 第三步：训练时数据加载与格式化

**入口文件：** `train/dataset_c3v2.py` (C3V2Dataset) + `train/data_collator.py` (C3V2DataCollator)

### 4.1 JSONL 格式

训练数据每行是一个 JSON 对象：

```jsonl
{"task": "reconstruct", "text": "The quantum computing paradigm..."}
{"task": "continuation", "text": "In 1945, after the war ended...", "split_ratio": 0.55}
```

### 4.2 Reconstruct 任务的数据构造

对于一条 `task = "reconstruct"` 的数据：

```
原始文本: "The quantum computing paradigm enables..."
```

**Encoder 输入 (context_ids)：**

```
The quantum computing paradigm enables...<img><imgpad><imgpad>...<imgpad></img>
                                          │         32 个 imgpad          │
```

文本后面拼接 `<img>` + 32 个 `<imgpad>` + `</img>`，这些 placeholder 对应 latent token 的位置。

**Decoder 输入 (input_ids)：**

```
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|>
<|im_start|>user
<img><imgpad>...<imgpad></img>
Repeat the text: <|im_end|>
<|im_start|>assistant
The quantum computing paradigm enables...<|im_end|>
```

**Labels（训练目标）：**

只有 assistant 回复的部分参与 loss 计算，前面的 system/user 部分全部 mask 为 `IGNORE_INDEX = -100`。

### 4.3 Continuation 任务的数据构造

对于一条 `task = "continuation"`, `split_ratio = 0.55` 的数据：

```
原始文本 (1000 tokens): "In 1945, after the war ended, many soldiers returned..."
按 split_ratio=0.55 切分 → 前 550 tokens 作为上下文, 后 450 tokens 作为目标
```

**Encoder 输入 (context_ids)：**

```
In 1945, after the war ended, many soldiers...(前550 tokens)<img><imgpad>...<imgpad></img>
```

前半部分文本送入 encoder 压缩为 latent representation。

**Decoder 输入 (input_ids)：**

```
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|>
<|im_start|>user
<img><imgpad>...<imgpad></img><|im_end|>
<|im_start|>assistant
...returned home to find their communities transformed...(后450 tokens)<|im_end|>
```

续写任务中 **没有** "Repeat the text: " 这个 prompt，user 消息中仅包含 latent placeholder。latent placeholder 位于 **user turn 中**，代表 encoder 压缩出的上下文向量；后半部分待续写文本位于 **assistant turn 中**，作为生成目标。二者通过对话模板的 user → assistant 角色结构衔接，而非直接拼接。

### 4.4 DataCollator 批处理

`C3V2DataCollator` 将同一 batch 中的样本：

- `input_ids`：右侧 pad 到 batch 内最长序列（pad token = `<|endoftext|>`）
- `context_ids`：右侧 pad 到 batch 内最长上下文序列
- `labels`：右侧 pad 为 `IGNORE_INDEX = -100`
- `attention_mask` / `context_attention_mask`：由 pad token 位置生成
- `task_types`：`0` = reconstruct, `1` = continuation

**长度保护：** 如果 tokenize 后的 `input_ids` 长度 >= `model_max_length`（默认 8192），该样本会被丢弃，替换为另一个随机样本。

---

## 5. 数据量与配比总结

### 原始数据量

| 数据集 | 目标采样量 | Token 范围 | 平均 Token 数（估算） | 估算总 Token 量 |
|:------|:---------|:----------|:-----------------|:-------------|
| FineWeb-Edu | 1,500,000 | 400–1,500 | ~950 | ~14.25 亿 |
| Cosmopedia v2 | 300,000 | 400–1,200 | ~800 | ~2.40 亿 |
| Proof-Pile-2 | 200,000 | 400–1,500 | ~950 | ~1.90 亿 |
| SlimPajama | 600,000 | 800–2,000 | ~1,400 | ~8.40 亿 |
| **总计** | **2,600,000** | — | — | **~26.95 亿** |

### Phase 1 配比

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 1: 2,000,000 条                      │
│                    100% Reconstruct                           │
├─────────────────────────────────────────────────┬────────────┤
│         FineWeb-Edu: 1,500,000 (75%)            │Cosmo 15%   │
│                                                 ├────────────┤
│                                                 │Proof 10%   │
└─────────────────────────────────────────────────┴────────────┘
```

- FineWeb-Edu: 1,500,000 条 (75%)
- Cosmopedia v2: 300,000 条 (15%)
- Proof-Pile-2: 200,000 条 (10%)
- 训练 5 个 epoch → 有效训练样本 10,000,000

### Phase 2 配比

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 2: 2,000,000 条                      │
│              70% Reconstruct + 30% Continuation              │
├────────────────────────────────────────────┬─────────────────┤
│     Reconstruct (采样): 1,400,000 (70%)     │ Continuation:   │
│     来源: FineWeb + Cosmo + Proof 混合      │ SlimPajama      │
│                                            │ 600,000 (30%)   │
└────────────────────────────────────────────┴─────────────────┘
```

- Reconstruct 采样（从 200 万中取 140 万）: 1,400,000 条 (70%)
- Continuation (SlimPajama 全部): 600,000 条 (30%)
- 训练 3 个 epoch → 有效训练样本 6,000,000

### 两阶段训练配置对比

| 维度 | Phase 1 | Phase 2 |
|:----|:--------|:--------|
| 数据文件 | `phase1_train.jsonl` | `phase2_train.jsonl` |
| 样本总量 | 2,000,000 | 2,000,000 |
| 任务类型 | 100% reconstruct | 70% reconstruct + 30% continuation |
| 数据来源 | FineWeb-Edu + Cosmopedia + Proof-Pile | 同左（采样）+ SlimPajama |
| Epochs | 5 | 3 |
| 可训参数 | encoder + Q + mm_projector（冻结 decoder） | 全部参数（decoder 使用低学习率） |
| 学习率 | 1e-5 | 1e-5（encoder），1e-5 × ratio（decoder） |
| Effective Batch Size | 8 × 4 × 8 = 256 | 8 × 4 × 8 = 256 |
| 总训练步数 | ~39,000 | ~23,400 |
| 预计耗时（8×H200） | 20-24 小时 | 12-16 小时 |

---

## 6. Token 长度分布

### 各数据集 Token 长度窗口

```
Token 数量
0    200   400   600   800  1000  1200  1400  1600  1800  2000
│     │     │     │     │     │     │     │     │     │     │
│     │     ├─────────────────────────────────┤     │     │    FineWeb-Edu [400, 1500]
│     │     ├───────────────────────┤         │     │     │    Cosmopedia  [400, 1200]
│     │     ├─────────────────────────────────┤     │     │    Proof-Pile  [400, 1500]
│     │     │     │     ├─────────────────────────────────┤    SlimPajama  [800, 2000]
```

### 各数据集预期长度分布特征

| 数据集 | Min Token | Max Token | 预期分布 | 说明 |
|:------|:---------|:---------|:--------|:-----|
| FineWeb-Edu | 400 | 1,500 | 偏右集中（800-1200 较多） | 网页文本经 score>=3 过滤后，高质量长文占比高 |
| Cosmopedia v2 | 400 | 1,200 | 较均匀 | LLM 合成文本长度相对可控 |
| Proof-Pile-2 | 400 | 1,500 | 双峰（短代码片段 + 长论文） | 代码和数学文本长度变化大 |
| SlimPajama | 800 | 2,000 | 偏右（1200-1800 较多） | 下限较高，保证续写有足够上下文 |

### Encoder 与 Decoder 实际输入长度

训练时，原始文本经过格式化后的实际序列长度：

**Reconstruct 任务：**
- Encoder 输入 = 原文 tokens + 34 个特殊 token（`<img>` + 32×`<imgpad>` + `</img>`）
- Decoder 输入 = system prompt (~30 tok) + user prompt with placeholder (~40 tok) + 原文 tokens + 分隔符 (~5 tok)
- 约为：原文长度 × 2 + ~110 token 的 overhead

**Continuation 任务：**
- Encoder 输入 = 原文前 `split_ratio` 部分 + 34 特殊 token
- Decoder 输入 = system prompt (~30 tok) + user placeholder (~38 tok) + 原文后 `1-split_ratio` 部分 + 分隔符 (~5 tok)
- 约为：原文长度 + ~110 token 的 overhead

**`model_max_length = 8192`** 的保护机制确保超长序列不会进入训练。由于原始文本最大 2,000 token，加上 overhead 后远低于 8192，绝大多数样本都能完整保留。

---

## 7. 完整流程图

```
                         ┌──────────────────────────┐
                         │    HuggingFace Hub       │
                         └──────────┬───────────────┘
                                    │
                    ┌───────────────┼───────────────────┐
                    │               │                   │
              ┌─────▼─────┐  ┌─────▼──────┐  ┌────────▼────────┐  ┌──────────────┐
              │ FineWeb   │  │ Cosmopedia │  │  Proof-Pile-2  │  │  SlimPajama  │
              │ -Edu      │  │ v2         │  │  (alg-stack)   │  │  -6B         │
              │ 150万条    │  │ 30万条     │  │  20万条         │  │  60万条       │
              │ 400-1500t │  │ 400-1200t  │  │  400-1500t     │  │  800-2000t   │
              │ score≥3   │  │            │  │                │  │              │
              └─────┬─────┘  └─────┬──────┘  └────────┬───────┘  └──────┬───────┘
                    │               │                  │                 │
                    ▼               ▼                  ▼                 │
          ┌─────────────────────────────────────────────┐               │
          │     data/raw/ 合并为 Reconstruct Pool       │               │
          │            2,000,000 条                      │               │
          │   (task = "reconstruct")                    │               │
          └────────────────────┬────────────────────────┘               │
                               │                                        │
                    ┌──────────┴──────────┐                             │
                    │                     │                             │
                    ▼                     ▼                             ▼
          ┌─────────────────┐  ┌──────────────────┐          ┌──────────────────┐
          │   Phase 1       │  │   Phase 2        │          │  Continuation    │
          │   2,000,000 条   │  │   从 recon pool  │          │  Pool            │
          │   100% recon    │  │   采样 1,400,000  │          │  600,000 条       │
          │                 │  │                  │          │  task=continuation│
          │                 │  │    + ─────────────┼──────────┤  split_ratio     │
          │                 │  │    2,000,000 条   │          │  ∈ [0.4, 0.7]    │
          │                 │  │    70% R + 30% C │          │                  │
          └────────┬────────┘  └────────┬─────────┘          └──────────────────┘
                   │                    │
                   ▼                    ▼
          phase1_train.jsonl    phase2_train.jsonl
                   │                    │
                   ▼                    ▼
          ┌─────────────────┐  ┌──────────────────┐
          │ Phase 1 训练     │  │ Phase 2 训练      │
          │ 5 epochs        │  │ 3 epochs         │
          │ 冻结 decoder     │  │ 全参数微调         │
          │ lr = 1e-5       │  │ encoder lr = 1e-5│
          │                 │  │ decoder lr = 低   │
          └─────────────────┘  └──────────────────┘
```

### 关键设计决策

1. **为什么重建数据来源多样？** FineWeb-Edu（通用高质量网页）+ Cosmopedia（合成教科书）+ Proof-Pile（数学/代码）三者互补，让 encoder 学会压缩不同领域和风格的文本。

2. **为什么续写数据单独用 SlimPajama？** SlimPajama 包含多种来源的通用文本，且 token 范围设为 800-2000（更长），确保切分后前半部分有足够的上下文信息，后半部分有足够的生成目标。

3. **为什么 Phase 1 不含续写？** Phase 1 冻结 decoder，目标是纯粹训练 encoder 的压缩能力。重建任务要求忠实还原全文，是最直接的压缩学习信号。

4. **为什么 Phase 2 的 reconstruct 数量从 200 万降到 140 万？** 为续写数据（60 万）腾出空间，保持总量不变（200 万），同时 140:60 ≈ 7:3 的比例确保重建能力不会退化。

5. **为什么续写的 `split_ratio` 在 0.4-0.7 之间随机？** 训练时变化切分位置，让模型适应不同长度的上下文和不同长度的生成目标，避免过拟合于特定比例。
