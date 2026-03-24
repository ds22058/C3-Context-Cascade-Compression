# C3 模型评估

每次评估**一个模型**，结果按模型名分目录保存，最后用对比脚本横向比较多个模型。

## 模型类型与测试方法

| 模型类型 | 示例 | 检测方式 | 运行的测试 |
|---------|------|---------|-----------|
| 标准 CausalLM | Qwen2.5-3B | 无 `llm1/` 子目录 | 重建(decoder-only) + PPL + 生成质量 + lm_eval (5 个 benchmark) |
| C3 完整模型 | output/phase1（含编码器） | 有 `llm1/` 子目录 | **以上全部** + 重建(encoder压缩) + compressed lm_eval* |

> C3 完整模型会自动提取 Decoder（保存到 `<model_path>_decoder_extracted`），用 Decoder 跑 PPL / 生成 / lm_eval / 重建(decoder-only)，用完整模型跑重建(encoder) 和 compressed lm_eval。
> *compressed lm_eval 需设置 `COMPRESS_THRESHOLD` 环境变量启用。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_h200.sh` | **一键评估单模型**（自动检测类型、多卡并行） |
| `compare_results.py` | **对比多个模型**的结果，生成 Markdown 报告 |
| `eval_perplexity.py` | WikiText-2 perplexity（单模型） |
| `eval_generation.py` | 生成质量，7 个 prompt（单模型） |
| `eval_reconstruction.py` | 重建准确率（encoder 压缩→重建）：C3 完整模型，多数据集多长度 |
| `eval_reconstruction_decoder.py` | 重建准确率（decoder-only）：不经过 encoder，直接用 decoder 复述。任何 CausalLM 可用 |
| `eval_lm_eval_compressed.py` | **Context-Compressed lm_eval**：prompt 过 encoder 压缩后评估 NLP benchmark（需 C3 完整模型） |
| `download_eval_data.py` | **重建评估前置**：下载 WikiText-2 / GovReport / PG 书籍到 `eval/data/`，评估时完全离线 |
| `eval_recon_showcase.py` | 重建质量展示：7 条固定上下文（中英文/古文/代码/无意义语料）原文 vs 重建对比 |
| `extract_decoder.py` | 从 C3 safetensors 提取 decoder，保存为 Qwen2ForCausalLM |
| `download_models.py` | 下载 C3 和 Qwen2.5-3B 到 `./models/` |
| `requirements.txt` | Python 依赖 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r eval/requirements.txt
```

### 2. 准备模型 & 数据

```bash
# 下载 C3 + Qwen2.5-3B
python eval/download_models.py

# 从 C3 提取 decoder
python eval/extract_decoder.py

# 重建评估数据（一次性下载到 eval/data/，后续完全离线）
HF_ENDPOINT=https://hf-mirror.com python eval/download_eval_data.py
```

**重建评估使用的数据集（本地 `eval/data/`，运行 `download_eval_data.py` 后可用）：**

| 数据集名 | 来源 | 上下文长度 | 默认样本数 | 说明 |
|---------|------|-----------|-----------|------|
| `wikitext-2` | `eval/data/wikitext2.jsonl` | 64-512 tok | 50 | 百科短段落 |
| `mixed-1k/2k/4k/8k` | 书籍 + `govreport.jsonl` 混合切窗 | ~1k/2k/4k/8k tok | 各 20 | 书籍+政府报告各半 |

### 3. 评估（单模型 · 多卡并行）

```bash
# 用法: bash eval/run_h200.sh <model_path> <model_name> <gpu_list> [compress_threshold]
# gpu_list 为逗号分隔的 GPU 编号，支持任意数量
# compress_threshold 为 compressed lm_eval 阈值（默认 256，仅 C3 模型使用）

# 测试 Qwen2.5-3B 基线（GPU 0-3）
bash eval/run_h200.sh ./models/qwen25-3b qwen25-3b 0,1,2,3

# 测试自训练的 C3 模型（GPU 4-7，自动提取 Decoder）
bash eval/run_h200.sh ./output/phase1 c3-phase1 4,5,6,7

# 自定义压缩阈值
bash eval/run_h200.sh ./output/phase2 c3-phase2 0,1,2,3,4,5,6,7 512

# 2 个模型同时评估（各占 4 卡）
bash eval/run_h200.sh ./output/phase1 c3-phase1 0,1,2,3 &
bash eval/run_h200.sh ./output/phase2 c3-phase2 4,5,6,7 &
wait
```

**跳过特定阶段（通过环境变量控制，设为 1 则跳过）：**

| 环境变量 | 跳过的阶段 |
|---------|-----------|
| `SKIP_RECON_ENC=1` | 重建准确率 (encoder) |
| `SKIP_RECON_DEC=1` | 重建准确率 (decoder-only) |
| `SKIP_LM_EVAL=1` | NLP 基准 (MMLU/hellaswag/arc_easy/arc_challenge/winogrande) |
| `SKIP_PPL=1` | WikiText-2 Perplexity |
| `SKIP_GENERATION=1` | 生成质量 |
| `SKIP_COMPRESSED=1` | Compressed lm_eval |

```bash
# Phase 1 评估：decoder 冻结，跳过与原始模型重复的项目
SKIP_RECON_DEC=1 SKIP_LM_EVAL=1 SKIP_PPL=1 SKIP_GENERATION=1 \
  bash eval/run_h200.sh ./output/phase1/checkpoint-20000 c3-p1 0,1,2,3,4,5,6,7

# 只跑重建，跳过所有其他测试
SKIP_LM_EVAL=1 SKIP_PPL=1 SKIP_GENERATION=1 SKIP_COMPRESSED=1 \
  bash eval/run_h200.sh ./output/phase1 c3-phase1 0,1,2,3,4,5,6,7

# 只跑 NLP 基准 + PPL
SKIP_RECON_ENC=1 SKIP_RECON_DEC=1 SKIP_GENERATION=1 SKIP_COMPRESSED=1 \
  bash eval/run_h200.sh ./output/phase2 c3-phase2 0,1,2,3,4,5,6,7
```

**多阶段调度（C3 模型，以 8 卡为例）：**

| 阶段 | 任务 | GPU 分配 | 预计耗时 | 跳过变量 |
|------|------|----------|----------|----------|
| **1** | 重建准确率 (encoder) | 全部 8 卡并行 | ~4-5 min | `SKIP_RECON_ENC` |
| **2** | 重建准确率 (decoder-only) | 全部 8 卡并行 | ~3-4 min | `SKIP_RECON_DEC` |
| **3** | MMLU / hellaswag / arc_easy / arc_c / wino / PPL | 各1卡 + PPL 2卡 同时 | ~12 min | `SKIP_LM_EVAL` / `SKIP_PPL` |
| **4** | 生成质量 | 1 卡 | ~1 min | `SKIP_GENERATION` |
| **5** | Compressed lm_eval | 全部 N 卡并行 | 按任务而定 | `SKIP_COMPRESSED` |

标准 CausalLM 模型跳过阶段 1 和 5，其余照常。
lm_eval 全部单卡运行（避免 accelerate 多进程 context building 开销）。卡数不足时自动循环复用。

**自动检测模型类型：**
- 若 `<model_path>/llm1/` 存在 → C3 完整模型（全部 5 个阶段）
- 否则 → 标准 CausalLM（跳过 encoder 重建和 compressed lm_eval，共 3 个阶段）
- 若 `modeling_C3.py` 缺失 → 自动从 `models/c3/` 复制

### 4. 对比结果

```bash
# 对比任意多个模型（名称即 results/ 下的目录名）
python eval/compare_results.py qwen25-3b c3-decoder c3-full

# 指定输出路径
python eval/compare_results.py qwen25-3b c3-decoder --output ./my_report.md
```

---

## 结果目录结构

每个模型的结果独立存放：

```
eval/results/
  qwen25-3b/                   标准 CausalLM
    reconstruction_decoder.json ← 重建准确率 decoder-only
    perplexity.json
    generation.json
    lm_eval/
  c3-phase1/                   C3 完整模型（全部测试）
    reconstruction.json        ← 重建准确率 encoder（C3 完整模型）
    reconstruction_decoder.json ← 重建准确率 decoder-only（Decoder）
    perplexity.json            ← PPL（自动提取的 Decoder）
    generation.json            ← 生成质量（Decoder）
    lm_eval/                   ← NLP 基准（Decoder）
    lm_eval_compressed/        ← Compressed NLP 基准（C3 完整模型）
  summary.md                   compare_results.py 生成的对比报告

eval/logs/
  qwen25-3b/
    gpu0_ppl_gen.log
    gpu1_lm_eval_mmlu.log
    ...
  c3-phase1/
    gpu0_recon_ppl_gen.log     ← 重建 + PPL + 生成 的完整日志
    ...
```

---

## 手动分步运行

```bash
# 所有命令通过 CUDA_VISIBLE_DEVICES 指定 GPU，可多个终端同时跑

# === 重建准确率 · encoder 压缩（需要 C3 完整模型）===
CUDA_VISIBLE_DEVICES=0 python eval/eval_reconstruction.py \
  --model_path ./output/phase1 --model_name c3-phase1 --device cuda

# 只跑短文本快速验证
CUDA_VISIBLE_DEVICES=0 python eval/eval_reconstruction.py \
  --model_path ./output/phase1 --model_name c3-phase1 \
  --datasets wikitext-2 --samples_per_dataset 10 --device cuda

# 多卡并行
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/eval_reconstruction.py \
  --model_path ./output/phase1 --model_name c3-phase1 --num_gpus 4

# === 重建准确率 · decoder-only（任何 CausalLM 都可以）===
CUDA_VISIBLE_DEVICES=0 python eval/eval_reconstruction_decoder.py \
  --model_path ./models/qwen25-3b --model_name qwen25-3b --device cuda

# 多卡并行
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/eval_reconstruction_decoder.py \
  --model_path ./output/phase1_decoder_extracted --model_name c3-phase1 --num_gpus 4

# === 重建质量展示（固定上下文，肉眼对比原文 vs 重建）===
CUDA_VISIBLE_DEVICES=0 python eval/eval_recon_showcase.py \
  --model_path ./output/phase1 --model_name c3-phase1 --device cuda

# === Perplexity（标准 CausalLM 或提取的 Decoder）===
CUDA_VISIBLE_DEVICES=1 python eval/eval_perplexity.py \
  --model_path ./output/phase1_decoder_extracted --model_name c3-phase1 --device cuda

# === 生成质量 ===
CUDA_VISIBLE_DEVICES=1 python eval/eval_generation.py \
  --model_path ./output/phase1_decoder_extracted --model_name c3-phase1 --device cuda

# === lm_eval · 标准（decoder 直接推理）===
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
  --model_args "pretrained=./output/phase1_decoder_extracted,trust_remote_code=False" \
  --tasks mmlu --device cuda:0 --batch_size auto \
  --output_path ./eval/results/c3-phase1/lm_eval &

CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
  --model_args "pretrained=./output/phase1_decoder_extracted,trust_remote_code=False" \
  --tasks hellaswag,arc_easy,arc_challenge,winogrande --device cuda:0 --batch_size auto \
  --output_path ./eval/results/c3-phase1/lm_eval &

# === lm_eval · context-compressed（需要 C3 完整模型）===
# compress_threshold: prompt 前 N 个 token 过 encoder 压缩，其余直接进 decoder
CUDA_VISIBLE_DEVICES=0 python eval/eval_lm_eval_compressed.py \
  --model_path ./output/phase2 --model_name c3-phase2 \
  --compress_threshold 256 \
  --tasks hellaswag,arc_easy,arc_challenge,winogrande

# 也可以跑 MMLU（耗时较长）
CUDA_VISIBLE_DEVICES=0 python eval/eval_lm_eval_compressed.py \
  --model_path ./output/phase2 --model_name c3-phase2 \
  --compress_threshold 512 \
  --tasks mmlu

# === 对比 ===
python eval/compare_results.py qwen25-3b c3-phase1 c3
```

---

## 常见问题

**Q: HuggingFace 下载慢？**

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 提取 decoder 报 "缺失 keys"？**

只要缺失的是 `lm_head.weight`，属于正常（`tie_word_embeddings=True`）。

**Q: lm_eval 找不到？**

```bash
pip install lm-eval
```

**Q: 想测试自己训练的模型？**

见下方"评估自训练模型"。

---

## 评估自训练模型

以 Phase 1 训练输出 `output/phase1` 为例。

### 前置：补全 modeling_C3.py

训练输出目录可能缺少 `modeling_C3.py`（自定义架构代码），需从原始 C3 模型复制：

```bash
[ -f output/phase1/modeling_C3.py ] || cp models/c3/modeling_C3.py output/phase1/
```

> `config.json` 的 `auto_map` 指向 `modeling_C3.C3Config`，HuggingFace 在 `trust_remote_code=True` 时
> 会从模型目录加载此文件。它只定义网络结构，与权重无关，可直接复制。

### 一键评估

```bash
# 一条命令完成全部测试（重建 + PPL + 生成 + lm_eval）
bash eval/run_h200.sh ./output/phase1 c3-phase1 0 1 2 3
```

脚本自动完成：
1. 检测到 `llm1/` → 识别为 C3 完整模型
2. 若 `output/phase1_decoder_extracted` 不存在 → 自动提取 Decoder
3. GPU0: 重建评估（C3 完整模型）→ PPL + 生成（Decoder）
4. GPU1-3: lm_eval（Decoder）
5. 全部结果存入 `eval/results/c3-phase1/`

### Phase 1 + Phase 2 完整示例

```bash
# 准备 modeling_C3.py
[ -f output/phase1/modeling_C3.py ] || cp models/c3/modeling_C3.py output/phase1/
[ -f output/phase2/modeling_C3.py ] || cp models/c3/modeling_C3.py output/phase2/

# 同时评估（各占 4 卡）
bash eval/run_h200.sh ./output/phase1 c3-phase1 0,1,2,3 &
bash eval/run_h200.sh ./output/phase2 c3-phase2 4,5,6,7 &
wait

# 汇总对比
python eval/compare_results.py qwen25-3b c3-phase1 c3-phase2
```

### 各阶段预期表现

| 阶段 | 预期结果 |
|------|---------|
| Phase 1 | 重建 ROUGE-L 逐步上升；PPL / 基准与 Qwen2.5-3B **完全一致**（decoder 冻结） |
| Phase 2 | 重建质量进一步提升；PPL / 基准与 Qwen2.5-3B **接近**（decoder 微调，轻微偏移） |

---

## 预期结论参考

| 指标 | 表现良好 | 需关注 |
|------|---------|--------|
| 重建 ROUGE-L (wikitext-2) | > 0.7 | < 0.4 |
| 重建 ROUGE-L (mixed-1k) | > 0.5 | < 0.3 |
| 重建 ROUGE-L (mixed-4k) | > 0.3 | < 0.15 |
| WikiText-2 PPL | 差异 < 5% | 差异 > 15% |
| 各 NLP 基准 | 平均下降 < 2pp | 平均下降 > 5pp |
| 生成质量 | 语义连贯，无复读 | 出现重复片段 |

> 随上下文长度增加，重建质量自然下降（32 个 latent token 压缩 4096 token 是 128x 压缩率）。
> 关键观察点是衰减曲线是否平缓。
