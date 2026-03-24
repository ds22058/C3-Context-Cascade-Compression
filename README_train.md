# C3 v2 训练指南

基于 C3 论文的 encoder 权重，搭配全新 Qwen2.5-3B decoder 的两阶段训练方案。

## 架构

```
Encoder (Qwen2.5-1.5B, C3预训练) --> Q Queries (N个) --> mm_projector (1536->2048) --> Decoder (Qwen2.5-3B, 原始权重)
```

- Phase 1：冻结 decoder，只训 encoder + Q + projector（重建任务）
- Phase 2：全模型微开，decoder 使用极低学习率（重建 + 续写任务）

## 环境安装

```bash
# PyTorch (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 训练依赖
pip install -r train/requirements.txt
```

## 快速开始

### 1. 准备模型

确保以下模型已下载到本地：

```
models/
  c3/                 # C3 预训练模型 (含 encoder + decoder)
  qwen25-3b/          # 原始 Qwen2.5-3B
```

可用 `eval/download_models.py` 下载，或手动从 HuggingFace 拉取。

### 2. 下载训练数据

```bash
# 镜像加速（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 下载并按 token 长度过滤（约需 1-3 小时，取决于网速）
python data/download_data.py \
    --tokenizer_path ./models/qwen25-3b \
    --output_dir ./data/raw/
```

各数据集默认配置：


| 数据集                    | HF ID                         | 数量   | Token 范围 | 用途  |
| ---------------------- | ----------------------------- | ---- | -------- | --- |
| FineWeb-Edu (score>=3) | `HuggingFaceFW/finewebedu`    | 150万 | 400-1500 | 重建  |
| Cosmopedia v2          | `HuggingFaceTB/cosmopedia-v2` | 30万  | 400-1200 | 重建  |
| Proof-Pile-2           | `EleutherAI/proof-pile-2`     | 20万  | 400-1500 | 重建  |
| SlimPajama             | `cerebras/SlimPajama-627B`    | 60万  | 800-2000 | 续写  |


可通过参数修改：

```bash
python data/download_data.py \
    --finewebedu_count 2000000 \
    --finewebedu_min_tokens 300 \
    --finewebedu_max_tokens 2000 \
    --slimpajama_count 800000
```

### 3. 预处理数据

```bash
python data/prepare_data.py \
    --raw_dir ./data/raw/ \
    --output_dir ./data/processed/ \
    --phase2_reconstruct_count 120000
```

输出：

- `data/processed/phase1_train.jsonl` — 200万条 (100% 重建)
- `data/processed/phase2_train.jsonl` — 200万条 (70% 重建 + 30% 续写)

### 4. Phase 1 训练（冻结 Decoder）

```bash
bash train/run_phase1.sh
```

默认配置（单机 8 卡 H200）：


| 参数              | 默认值                                |
| --------------- | ---------------------------------- |
| 可训参数            | encoder + Q + mm_projector (~1.5B) |
| 冻结              | decoder (Qwen2.5-3B)               |
| 数据              | 200万 reconstruct x 5 epochs        |
| Effective batch | 2 x 16 x 8 = 256                   |
| Steps           | ~39,000                            |
| LR              | 1e-5, cosine, warmup 3%            |
| 预计耗时            | ~20-24 小时 (H200)                   |


可通过环境变量覆盖：

```bash
LATENT_TOKEN_LEN=64 NUM_EPOCHS=5 LEARNING_RATE=2e-5 bash train/run_phase1.sh
```

### 5. Phase 2 训练（全模型微开）

```bash
bash train/run_phase2.sh
```


| 参数         | 默认值                                            |
| ---------- | ---------------------------------------------- |
| 可训参数       | 全部 (~4.5B)                                     |
| Decoder LR | 1e-5 x 0.02 = 2e-7                             |
| 数据         | 140万 reconstruct + 60万 continuation x 3 epochs |
| Steps      | ~23,400                                        |
| 预计耗时       | ~12-16 小时 (H200)                               |


```bash
PHASE1_CHECKPOINT=./output/phase1 DECODER_LR_RATIO=0.01 bash train/run_phase2.sh
```

### 6. 多机训练（可选）

准备 `hostfile.txt`：

```
node1 slots=8
node2 slots=8
```

```bash
PHASE=1 NUM_NODES=2 HOSTFILE=hostfile.txt bash train/run_multinode.sh
```

多机时 effective batch = 2 x 16 x 16 = 512，建议相应调整 epochs 或 grad_accum。

## 所有可配置参数

### 模型参数


| 参数                     | 说明                 | 默认值                  |
| ---------------------- | ------------------ | -------------------- |
| `--c3_model_path`      | C3 预训练模型路径         | `./models/c3`        |
| `--decoder_model_path` | Qwen2.5-3B 路径      | `./models/qwen25-3b` |
| `--latent_token_len`   | 压缩后 latent token 数 | `32` (可选 64, 128)    |
| `--reinit_projector`   | 随机初始化 mm_projector | `False`              |


### 训练阶段参数


| 参数                    | 说明                    | 默认值    |
| --------------------- | --------------------- | ------ |
| `--phase`             | 训练阶段 (1 或 2)          | `1`    |
| `--phase1_checkpoint` | Phase 1 的输出路径         | `None` |
| `--decoder_lr_ratio`  | Phase 2 decoder lr 倍率 | `0.02` |


### 数据下载参数


| 参数                        | 说明                      | 默认值       |
| ------------------------- | ----------------------- | --------- |
| `--finewebedu_count`      | FineWeb-Edu 采样数         | `1500000` |
| `--finewebedu_min_tokens` | FineWeb-Edu 最小 token 数  | `400`     |
| `--finewebedu_max_tokens` | FineWeb-Edu 最大 token 数  | `1500`    |
| `--finewebedu_min_score`  | FineWeb-Edu 最低教育质量分     | `3.0`     |
| `--cosmopedia_count`      | Cosmopedia v2 采样数       | `300000`  |
| `--cosmopedia_min_tokens` | Cosmopedia 最小 token 数   | `400`     |
| `--cosmopedia_max_tokens` | Cosmopedia 最大 token 数   | `1200`    |
| `--proofpile_count`       | Proof-Pile-2 采样数        | `200000`  |
| `--proofpile_min_tokens`  | Proof-Pile-2 最小 token 数 | `400`     |
| `--proofpile_max_tokens`  | Proof-Pile-2 最大 token 数 | `1500`    |
| `--slimpajama_count`      | SlimPajama 采样数          | `600000`  |
| `--slimpajama_min_tokens` | SlimPajama 最小 token 数   | `800`     |
| `--slimpajama_max_tokens` | SlimPajama 最大 token 数   | `2000`    |


### 数据预处理参数


| 参数                           | 说明              | 默认值       |
| ---------------------------- | --------------- | --------- |
| `--phase2_reconstruct_count` | Phase 2 重建数据采样数 | `1400000` |
| `--continuation_split_min`   | 续写任务前半部分最小比例    | `0.4`     |
| `--continuation_split_max`   | 续写任务前半部分最大比例    | `0.7`     |


## 文件结构

```
train/
  train_c3v2.py          # 主训练入口
  trainer_c3v2.py         # 自定义 Trainer (差异化 LR, 分离保存)
  dataset_c3v2.py         # 数据集 (reconstruct + continuation)
  data_collator.py        # DataCollator
  config.py               # 常量
  ds_zero2.json           # DeepSpeed ZeRO-2 配置
  run_phase1.sh           # Phase 1 启动脚本
  run_phase2.sh           # Phase 2 启动脚本
  run_multinode.sh        # 多机启动脚本
  requirements.txt        # 依赖
data/
  download_data.py        # 数据下载 (流式 + token 过滤)
  prepare_data.py         # 数据预处理 (Phase 1/2 JSONL)
```

## 常见问题

**Q: HuggingFace 下载很慢？**

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何更改 latent token 数量？**

C3 原始模型使用 32 个 latent tokens。如需 64 或 128，需要重新初始化 Q embeddings（原始权重只有 32 维）。设置 `--latent_token_len 64` 后，Q 会自动调整大小，但无法复用原始 C3 的 Q 权重。

**Q: Phase 1 显存不够？**

- 降低 `BATCH_SIZE` 到 1
- 增大 `GRAD_ACCUM` 保持 effective batch 不变
- 启用 `--gradient_checkpointing True`（默认已开启）

**Q: Phase 2 decoder 退化了？**

- 降低 `DECODER_LR_RATIO`（如 0.01 或 0.005）
- 减少 Phase 2 的 `NUM_EPOCHS`
- 增加 Phase 2 数据中重建的比例（修改 `--phase2_reconstruct_count`）

**Q: 如何验证训练效果？**

Phase 1 完成后，可以用 `eval/` 目录下的脚本评估：

1. 提取 decoder 权重，检查 PPL 是否和原始 Qwen2.5-3B 一致（验证 decoder 未被修改）
2. 用完整 C3 模型做 "Repeat the text" 测试，检查重建质量

