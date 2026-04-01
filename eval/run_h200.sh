#!/usr/bin/env bash
# ============================================================
# C3 评估流程 - 单模型 · 多阶段调度 (Linux)
#
# 用法：
#   bash eval/run_h200.sh <model_path> <model_name> <gpu_list> [compress_threshold]
#
# 参数：
#   model_path          模型路径
#   model_name          模型名（结果目录名）
#   gpu_list            逗号分隔的 GPU 编号（如 0,1,2,3,4,5,6,7）
#   compress_threshold  compressed lm_eval 的 token 阈值（默认 1024，仅 C3 模型使用）
#
# 环境变量：
#   RECON_DATASETS      重建测试数据集，逗号分隔，默认 "mixed-4k,mixed-8k"
#                       可选值及对应上下文长度:
#                         wikitext-2  — 64-512 tok (50 samples)
#                         mixed-1k    — ~1024 tok  (20 samples)
#                         mixed-2k    — ~2048 tok  (20 samples)
#                         mixed-4k    — ~4096 tok  (20 samples)
#                         mixed-8k    — ~8192 tok  (20 samples)
#                       示例: RECON_DATASETS="mixed-1k,mixed-2k,mixed-4k,mixed-8k"
#
# 跳过阶段（通过环境变量控制，设为 1 则跳过）：
#   SKIP_RECON_ENC=1    跳过重建(encoder)
#   SKIP_RECON_DEC=1    跳过重建(decoder-only)
#   SKIP_LM_EVAL=1      跳过 NLP 基准 (MMLU/hellaswag/arc/winogrande)
#   SKIP_PPL=1           跳过 Perplexity
#   SKIP_GENERATION=1    跳过生成质量
#   SKIP_COMPRESSED=1    跳过 compressed lm_eval
#   SKIP_COMPRESSED_V2=1 跳过 compressed lm_eval v2 (few-shot 压缩)
#
# 示例：
#   bash eval/run_h200.sh ./output/phase2  c3-phase2  0,1,2,3,4,5,6,7
#   bash eval/run_h200.sh ./output/phase2  c3-phase2  0,1,2,3,4,5,6,7  512
#
#   # 只跑重建(4k,8k)，跳过其他
#   SKIP_LM_EVAL=1 SKIP_PPL=1 SKIP_GENERATION=1 SKIP_COMPRESSED=1 \
#     bash eval/run_h200.sh ./output/phase1 c3-phase1 0,1,2,3,4,5,6,7
#
#   # Phase 1 评估（只跑重建）
#   SKIP_LM_EVAL=1 SKIP_PPL=1 SKIP_GENERATION=1 SKIP_COMPRESSED=1 SKIP_COMPRESSED_V2=1 \
#     bash eval/run_h200.sh ./output/phase1_v22_4node c3-p1-v22 0,1,2,3,4,5,6,7
#
#   # 指定重建数据集
#   RECON_DATASETS="wikitext-2,mixed-1k,mixed-2k,mixed-4k,mixed-8k" \
#     bash eval/run_h200.sh ./output/phase2 c3-phase2 0,1,2,3,4,5,6,7
#
#   # 2 个模型同时评估
#   bash eval/run_h200.sh ./output/phase1  c3-phase1  0,1,2,3 &
#   bash eval/run_h200.sh ./output/phase2  c3-phase2  4,5,6,7 &
#   wait
#
# 阶段调度（C3 模型，8 卡）：
#
#   阶段 1  重建（encoder）     — 全部 N 卡并行（仅 C3）
#   阶段 2  重建（decoder-only）— 全部 N 卡并行
#   阶段 3  同时拉起（各 1 卡，无 accelerate 开销）：
#           · MMLU / hellaswag / arc_easy / arc_challenge / winogrande 各 1 卡
#           · PPL — N≥7 时 2 卡，否则 1 卡
#   阶段 4  生成质量            — 1 卡
#   阶段 5  compressed lm_eval — 全部 N 卡并行（仅 C3）
#   阶段 6  compressed lm_eval v2 (few-shot 压缩) — C3: 全部 N 卡并行
#                                                    CausalLM: 标准 lm_eval + 匹配 few-shot
#
# 标准 CausalLM 模型跳过阶段 1 和 5，其余照常。
# 结果统一保存到 eval/results/<model_name>/
# ============================================================

set -euo pipefail

# ── 离线模式：缓存存在时跳过无意义的 Hub 请求 ─────────────────
if [ -z "${HF_DATASETS_OFFLINE:-}" ]; then
    MMLU_CACHE=$(find ~/.cache/huggingface/datasets/cais___mmlu -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
    if [ -n "$MMLU_CACHE" ]; then
        export HF_DATASETS_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    fi
fi

# ── 参数解析 ──────────────────────────────────────────────────
MODEL_PATH="${1:?用法: bash eval/run_h200.sh <model_path> <model_name> <gpu_list> [compress_threshold]}"
MODEL_NAME="${2:?用法: bash eval/run_h200.sh <model_path> <model_name> <gpu_list> [compress_threshold]}"
GPU_LIST="${3:?用法: bash eval/run_h200.sh <model_path> <model_name> <gpu_list> [compress_threshold]}"
COMPRESS_THRESHOLD="${4:-1024}"

# ── 跳过控制 ──────────────────────────────────────────────────
SKIP_RECON_ENC="${SKIP_RECON_ENC:-0}"
SKIP_RECON_DEC="${SKIP_RECON_DEC:-0}"
SKIP_LM_EVAL="${SKIP_LM_EVAL:-0}"
SKIP_PPL="${SKIP_PPL:-0}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_COMPRESSED="${SKIP_COMPRESSED:-0}"
SKIP_COMPRESSED_V2="${SKIP_COMPRESSED_V2:-0}"

# ── 重建测试数据集 ────────────────────────────────────────────
RECON_DATASETS="${RECON_DATASETS:-mixed-4k,mixed-8k}"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N=${#GPUS[@]}

if [ "$N" -lt 1 ]; then
    echo "至少需要 1 张 GPU"; exit 1
fi

RESULTS_DIR="./eval/results/${MODEL_NAME}"
LOGDIR="./eval/logs/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR" "$LOGDIR"

# ── 颜色 & 辅助函数 ──────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

step()  { echo -e "\n${CYAN}$(printf '=%.0s' {1..60})${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"; }
ok()    { echo -e "  ${GREEN}[完成]${NC} $1"; }
fail()  { echo -e "  ${RED}[失败]${NC} $1"; }
warn()  { echo -e "  ${YELLOW}[警告]${NC} $1"; }
skip()  { echo -e "  ${YELLOW}[跳过]${NC} $1"; }
ts()    { date '+%H:%M:%S'; }

gpu_range() {
    local start=$1 count=$2 result=""
    for (( i=0; i<count; i++ )); do
        local idx=$(( (start + i) % N ))
        [ -n "$result" ] && result="${result},"
        result="${result}${GPUS[$idx]}"
    done
    echo "$result"
}

# ── 卡数分配策略 ──────────────────────────────────────────────
# lm_eval 全部单卡（accelerate 多卡 context building 开销远大于推理加速）
# PPL 是唯一受益于多卡的任务（窗口分片，无启动开销）
if [ "$N" -ge 7 ]; then
    N_PPL=2
else
    N_PPL=1
fi

# 6 个并行任务各分 1 张卡（MMLU / hellaswag / arc_easy / arc_challenge / winogrande / PPL）
# 卡数不足时通过 gpu_range 循环复用
IDX=0
GPUS_MMLU=${GPUS[$((IDX % N))]};       IDX=$((IDX + 1))
GPUS_HELLA=${GPUS[$((IDX % N))]};      IDX=$((IDX + 1))
GPUS_ARCE=${GPUS[$((IDX % N))]};       IDX=$((IDX + 1))
GPUS_ARCC=${GPUS[$((IDX % N))]};       IDX=$((IDX + 1))
GPUS_WINO=${GPUS[$((IDX % N))]};       IDX=$((IDX + 1))
GPUS_PPL=$(gpu_range $IDX $N_PPL)

# ── 模型类型检测 & Decoder 准备 ───────────────────────────────
[ -d "${MODEL_PATH}" ] || { echo -e "${RED}模型路径不存在: ${MODEL_PATH}${NC}"; exit 1; }

# 如果根目录没有 llm1，自动查找最新 checkpoint-*/llm1
RESOLVED_MODEL_PATH="${MODEL_PATH}"
if [ ! -d "${MODEL_PATH}/llm1" ]; then
    LATEST_CKPT=$(ls -d "${MODEL_PATH}"/checkpoint-[0-9]* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -d "${LATEST_CKPT}/llm1" ]; then
        RESOLVED_MODEL_PATH="${LATEST_CKPT}"
        warn "根目录无 llm1，自动使用最新 checkpoint: ${RESOLVED_MODEL_PATH}"
    fi
fi
MODEL_PATH="${RESOLVED_MODEL_PATH}"

if [ -d "${MODEL_PATH}/llm1" ]; then
    MODEL_TYPE="c3"
    DECODER_PATH="${MODEL_PATH}_decoder_extracted"
    STAGE_TOTAL=6

    # 自动补全 modeling_C3.py（训练 checkpoint 目录通常不含此文件）
    if [ ! -f "${MODEL_PATH}/modeling_C3.py" ]; then
        C3_SRC="./models/c3/modeling_C3.py"
        if [ -f "$C3_SRC" ]; then
            cp "$C3_SRC" "${MODEL_PATH}/modeling_C3.py"
            ok "已自动复制 modeling_C3.py → ${MODEL_PATH}/"
        else
            echo -e "${RED}缺少 modeling_C3.py 且源文件 ${C3_SRC} 不存在，请手动复制${NC}"; exit 1
        fi
    fi
else
    MODEL_TYPE="causal_lm"
    DECODER_PATH="$MODEL_PATH"
    STAGE_TOTAL=4
fi

step "评估: ${MODEL_NAME}  |  类型: ${MODEL_TYPE}  |  GPU: ${GPU_LIST} (${N}卡)"
echo "  模型路径:  ${MODEL_PATH}"
[ "$MODEL_TYPE" = "c3" ] && echo "  Decoder:   ${DECODER_PATH}"
[ "$MODEL_TYPE" = "c3" ] && echo "  压缩阈值:  ${COMPRESS_THRESHOLD} tokens"
[ "$MODEL_TYPE" = "c3" ] && echo "  重建数据集: ${RECON_DATASETS}"
echo "  结果目录:  ${RESULTS_DIR}"
echo "  日志目录:  ${LOGDIR}"

# 显示跳过项
SKIPPED=""
[ "$SKIP_RECON_ENC" = "1" ]  && SKIPPED="${SKIPPED} recon_enc"
[ "$SKIP_RECON_DEC" = "1" ]  && SKIPPED="${SKIPPED} recon_dec"
[ "$SKIP_LM_EVAL" = "1" ]   && SKIPPED="${SKIPPED} lm_eval"
[ "$SKIP_PPL" = "1" ]       && SKIPPED="${SKIPPED} ppl"
[ "$SKIP_GENERATION" = "1" ] && SKIPPED="${SKIPPED} generation"
[ "$SKIP_COMPRESSED" = "1" ]    && SKIPPED="${SKIPPED} compressed"
[ "$SKIP_COMPRESSED_V2" = "1" ] && SKIPPED="${SKIPPED} compressed_v2"
if [ -n "$SKIPPED" ]; then
    echo -e "  ${YELLOW}跳过:${NC}${SKIPPED}"
fi

# ── 前置检查 ──────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=${GPUS[0]} python -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f'  PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}')
"

# ── C3: 提取 Decoder ─────────────────────────────────────────
if [ "$MODEL_TYPE" = "c3" ]; then
    if [ ! -d "$DECODER_PATH" ]; then
        step "提取 Decoder → ${DECODER_PATH}"
        python eval/extract_decoder.py \
            --c3_path   "$MODEL_PATH" \
            --output_path "$DECODER_PATH"
        ok "Decoder 提取完成"
    else
        echo ""; echo "  Decoder 已存在: ${DECODER_PATH}（跳过提取）"
    fi
fi

T_START=$SECONDS
FAIL=0
STAGE=0
set +e

# ==============================================================
# 阶段 1: 重建准确率 — encoder 压缩（C3 · 全部 GPU 并行）
# ==============================================================
if [ "$MODEL_TYPE" = "c3" ] && [ "$SKIP_RECON_ENC" != "1" ]; then
    STAGE=$((STAGE + 1))
    step "阶段 ${STAGE}/${STAGE_TOTAL}: 重建准确率 encoder（${N} 卡并行）"

    CUDA_VISIBLE_DEVICES=${GPU_LIST} python eval/eval_reconstruction.py \
        --model_path  "$MODEL_PATH" \
        --model_name  "$MODEL_NAME" \
        --num_gpus    "$N" \
        --datasets    "${RECON_DATASETS}" \
        --output      "${RESULTS_DIR}/reconstruction.json" \
        2>&1 | tee "${LOGDIR}/recon.log"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -eq 0 ]; then ok "重建(encoder) ($(( SECONDS - T_START ))s)"
    else fail "重建(encoder) → ${LOGDIR}/recon.log"; FAIL=1; fi
elif [ "$MODEL_TYPE" = "c3" ]; then
    skip "重建(encoder) (SKIP_RECON_ENC=1)"
fi

# ==============================================================
# 阶段 2: 重建准确率 — decoder-only（全部 GPU 并行）
# ==============================================================
# [已暂时禁用] decoder-only 复述评估暂不纳入对比报告
# if [ "$SKIP_RECON_DEC" != "1" ]; then
#     STAGE=$((STAGE + 1))
#     step "阶段 ${STAGE}/${STAGE_TOTAL}: 重建准确率 decoder-only（${N} 卡并行）"
#
#     # 需要 mixed-4k 时改为 "wikitext-2,mixed-1k,mixed-2k,mixed-4k"
#     CUDA_VISIBLE_DEVICES=${GPU_LIST} python eval/eval_reconstruction_decoder.py \
#         --model_path  "$DECODER_PATH" \
#         --model_name  "$MODEL_NAME" \
#         --num_gpus    "$N" \
#         --datasets    "wikitext-2,mixed-1k,mixed-2k" \
#         --output      "${RESULTS_DIR}/reconstruction_decoder.json" \
#         2>&1 | tee "${LOGDIR}/recon_decoder.log"
#     RC=${PIPESTATUS[0]}
#     if [ "$RC" -eq 0 ]; then ok "重建(decoder-only) ($(( SECONDS - T_START ))s)"
#     else fail "重建(decoder-only) → ${LOGDIR}/recon_decoder.log"; FAIL=1; fi
# else
#     skip "重建(decoder-only) (SKIP_RECON_DEC=1)"
# fi

# ==============================================================
# 阶段 3: lm_eval (5任务) + PPL（全部同时拉起，各 1 卡）
# ==============================================================
if [ "$SKIP_LM_EVAL" != "1" ] || [ "$SKIP_PPL" != "1" ]; then
    STAGE=$((STAGE + 1))
    step "阶段 ${STAGE}/${STAGE_TOTAL}: lm_eval + PPL"

    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    [ "$SKIP_LM_EVAL" != "1" ] && echo "  │ MMLU             GPU ${GPUS_MMLU}  (1卡, ~12 min)   │"
    [ "$SKIP_LM_EVAL" != "1" ] && echo "  │ hellaswag        GPU ${GPUS_HELLA}  (1卡, ~10 min)   │"
    [ "$SKIP_LM_EVAL" != "1" ] && echo "  │ arc_easy         GPU ${GPUS_ARCE}  (1卡, ~2 min)    │"
    [ "$SKIP_LM_EVAL" != "1" ] && echo "  │ arc_challenge    GPU ${GPUS_ARCC}  (1卡, ~2 min)    │"
    [ "$SKIP_LM_EVAL" != "1" ] && echo "  │ winogrande       GPU ${GPUS_WINO}  (1卡, ~2 min)    │"
    [ "$SKIP_PPL" != "1" ]     && echo "  │ PPL              GPU ${GPUS_PPL}  (${N_PPL}卡, ~2 min)    │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    PIDS=(); LABELS=()

    if [ "$SKIP_LM_EVAL" != "1" ]; then
        # ── MMLU ──
        (
            echo "[$(ts)] MMLU: 1卡 (GPU ${GPUS_MMLU})..."
            CUDA_VISIBLE_DEVICES=${GPUS_MMLU} lm_eval \
                --model       hf \
                --model_args  "pretrained=${DECODER_PATH},trust_remote_code=False" \
                --tasks       mmlu \
                --device      cuda:0 \
                --batch_size  auto \
                --output_path "${RESULTS_DIR}/lm_eval"
            echo "[$(ts)] MMLU 完成"
        ) > "${LOGDIR}/lm_eval_mmlu.log" 2>&1 &
        PIDS+=($!); LABELS+=("MMLU (GPU ${GPUS_MMLU})")

        # ── hellaswag ──
        (
            echo "[$(ts)] hellaswag (GPU ${GPUS_HELLA})..."
            CUDA_VISIBLE_DEVICES=${GPUS_HELLA} lm_eval \
                --model       hf \
                --model_args  "pretrained=${DECODER_PATH},trust_remote_code=False" \
                --tasks       hellaswag \
                --device      cuda:0 \
                --batch_size  auto \
                --output_path "${RESULTS_DIR}/lm_eval"
            echo "[$(ts)] hellaswag 完成"
        ) > "${LOGDIR}/lm_eval_hellaswag.log" 2>&1 &
        PIDS+=($!); LABELS+=("hellaswag (GPU ${GPUS_HELLA})")

        # ── arc_easy ──
        (
            echo "[$(ts)] arc_easy (GPU ${GPUS_ARCE})..."
            CUDA_VISIBLE_DEVICES=${GPUS_ARCE} lm_eval \
                --model       hf \
                --model_args  "pretrained=${DECODER_PATH},trust_remote_code=False" \
                --tasks       arc_easy \
                --device      cuda:0 \
                --batch_size  auto \
                --output_path "${RESULTS_DIR}/lm_eval"
            echo "[$(ts)] arc_easy 完成"
        ) > "${LOGDIR}/lm_eval_arc_easy.log" 2>&1 &
        PIDS+=($!); LABELS+=("arc_easy (GPU ${GPUS_ARCE})")

        # ── arc_challenge ──
        (
            echo "[$(ts)] arc_challenge (GPU ${GPUS_ARCC})..."
            CUDA_VISIBLE_DEVICES=${GPUS_ARCC} lm_eval \
                --model       hf \
                --model_args  "pretrained=${DECODER_PATH},trust_remote_code=False" \
                --tasks       arc_challenge \
                --device      cuda:0 \
                --batch_size  auto \
                --output_path "${RESULTS_DIR}/lm_eval"
            echo "[$(ts)] arc_challenge 完成"
        ) > "${LOGDIR}/lm_eval_arc_challenge.log" 2>&1 &
        PIDS+=($!); LABELS+=("arc_challenge (GPU ${GPUS_ARCC})")

        # ── winogrande ──
        (
            echo "[$(ts)] winogrande (GPU ${GPUS_WINO})..."
            CUDA_VISIBLE_DEVICES=${GPUS_WINO} lm_eval \
                --model       hf \
                --model_args  "pretrained=${DECODER_PATH},trust_remote_code=False" \
                --tasks       winogrande \
                --device      cuda:0 \
                --batch_size  auto \
                --output_path "${RESULTS_DIR}/lm_eval"
            echo "[$(ts)] winogrande 完成"
        ) > "${LOGDIR}/lm_eval_winogrande.log" 2>&1 &
        PIDS+=($!); LABELS+=("winogrande (GPU ${GPUS_WINO})")
    else
        skip "lm_eval (SKIP_LM_EVAL=1)"
    fi

    if [ "$SKIP_PPL" != "1" ]; then
        # ── PPL ──
        (
            echo "[$(ts)] PPL: ${N_PPL}卡 (GPU ${GPUS_PPL})..."
            CUDA_VISIBLE_DEVICES=${GPUS_PPL} python eval/eval_perplexity.py \
                --model_path  "$DECODER_PATH" \
                --model_name  "$MODEL_NAME" \
                --num_gpus    "$N_PPL" \
                --output      "${RESULTS_DIR}/perplexity.json"
            echo "[$(ts)] PPL 完成"
        ) > "${LOGDIR}/ppl.log" 2>&1 &
        PIDS+=($!); LABELS+=("PPL (${N_PPL}卡 GPU ${GPUS_PPL})")
    else
        skip "PPL (SKIP_PPL=1)"
    fi

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "  后台 PID: ${PIDS[*]}"
        echo "  日志: tail -f ${LOGDIR}/*.log"
        echo ""; echo "  等待中..."; echo ""

        for i in "${!PIDS[@]}"; do
            wait ${PIDS[$i]}
            if [ $? -eq 0 ]; then ok "${LABELS[$i]}  ($(( SECONDS - T_START ))s)"
            else fail "${LABELS[$i]} → ${LOGDIR}/"; FAIL=1; fi
        done
    fi
else
    skip "lm_eval + PPL (SKIP_LM_EVAL=1, SKIP_PPL=1)"
fi

# ==============================================================
# 阶段 4: 生成质量（1 卡）— 暂时注释
# ==============================================================
# if [ "$SKIP_GENERATION" != "1" ]; then
#     STAGE=$((STAGE + 1))
#     step "阶段 ${STAGE}/${STAGE_TOTAL}: 生成质量 (1卡 GPU ${GPUS[0]})"
#
#     CUDA_VISIBLE_DEVICES=${GPUS[0]} python eval/eval_generation.py \
#         --model_path  "$DECODER_PATH" \
#         --model_name  "$MODEL_NAME" \
#         --device      cuda \
#         --output      "${RESULTS_DIR}/generation.json" \
#         2>&1 | tee "${LOGDIR}/generation.log"
#     GEN_RC=${PIPESTATUS[0]}
#     if [ "$GEN_RC" -eq 0 ]; then ok "生成质量 ($(( SECONDS - T_START ))s)"
#     else fail "生成质量 → ${LOGDIR}/generation.log"; FAIL=1; fi
# else
#     skip "生成质量 (SKIP_GENERATION=1)"
# fi

# ==============================================================
# 阶段 5: Context-Compressed lm_eval（仅 C3）
# ==============================================================
if [ "$MODEL_TYPE" = "c3" ] && [ "$SKIP_COMPRESSED" != "1" ]; then
    STAGE=$((STAGE + 1))
    step "阶段 ${STAGE}/${STAGE_TOTAL}: Compressed lm_eval (threshold=${COMPRESS_THRESHOLD})"

    CUDA_VISIBLE_DEVICES=${GPU_LIST} python eval/eval_lm_eval_compressed.py \
        --model_path         "$MODEL_PATH" \
        --model_name         "$MODEL_NAME" \
        --compress_threshold "$COMPRESS_THRESHOLD" \
        --num_gpus           "$N" \
        --tasks              "mmlu,hellaswag,arc_easy,arc_challenge,winogrande" \
        --output_path        "${RESULTS_DIR}/lm_eval_compressed" \
        2>&1 | tee "${LOGDIR}/lm_eval_compressed.log"
    RC=${PIPESTATUS[0]}
    if [ "$RC" -eq 0 ]; then ok "Compressed lm_eval ($(( SECONDS - T_START ))s)"
    else fail "Compressed lm_eval → ${LOGDIR}/lm_eval_compressed.log"; FAIL=1; fi
elif [ "$MODEL_TYPE" = "c3" ]; then
    skip "Compressed lm_eval (SKIP_COMPRESSED=1)"
fi

# ==============================================================
# 阶段 6: Compressed Benchmark v2 (预处理数据 · 严格 600-1000 tok)
#
#   前置: eval/benchmark_data/processed/ 下需有预处理数据
#         (运行 prepare.py 生成，每条样本 few-shot + 题干严格在 600-1000 tok)
#
#   C3 模型:     few-shot+题干 过 encoder 压缩, 选项不压缩
#   CausalLM:    同样数据, 不压缩, 作为公平基线
# ==============================================================

V2_DATA_DIR="./eval/benchmark_data/processed"

if [ "$SKIP_COMPRESSED_V2" != "1" ]; then
    STAGE=$((STAGE + 1))

    # 检查预处理数据
    if [ ! -d "$V2_DATA_DIR" ] || [ -z "$(ls ${V2_DATA_DIR}/*.jsonl 2>/dev/null)" ]; then
        warn "预处理数据不存在: ${V2_DATA_DIR}/"
        warn "请先运行: python eval/benchmark_data/download.py && python eval/benchmark_data/prepare.py"
        skip "Compressed Benchmark v2 (数据缺失)"
    else
        if [ "$MODEL_TYPE" = "c3" ]; then
            step "阶段 ${STAGE}/${STAGE_TOTAL}: Compressed Benchmark v2 (C3 压缩, ${N}卡)"
            CUDA_VISIBLE_DEVICES=${GPU_LIST} python eval/eval_lm_eval_compressed_v2.py \
                --model_path  "$MODEL_PATH" \
                --model_name  "$MODEL_NAME" \
                --num_gpus    "$N" \
                --data_dir    "$V2_DATA_DIR" \
                --output_path "${RESULTS_DIR}/lm_eval_compressed_v2" \
                2>&1 | tee "${LOGDIR}/lm_eval_compressed_v2.log"
        else
            step "阶段 ${STAGE}/${STAGE_TOTAL}: Compressed Benchmark v2 (基线, 不压缩, ${N}卡)"
            CUDA_VISIBLE_DEVICES=${GPU_LIST} python eval/eval_lm_eval_compressed_v2.py \
                --model_path  "$DECODER_PATH" \
                --model_name  "$MODEL_NAME" \
                --baseline \
                --num_gpus    "$N" \
                --data_dir    "$V2_DATA_DIR" \
                --output_path "${RESULTS_DIR}/lm_eval_compressed_v2" \
                2>&1 | tee "${LOGDIR}/lm_eval_compressed_v2.log"
        fi
        RC=${PIPESTATUS[0]}
        if [ "$RC" -eq 0 ]; then ok "Compressed Benchmark v2 ($(( SECONDS - T_START ))s)"
        else fail "Compressed Benchmark v2 → ${LOGDIR}/lm_eval_compressed_v2.log"; FAIL=1; fi
    fi
else
    skip "Compressed Benchmark v2 (SKIP_COMPRESSED_V2=1)"
fi

set -e

# ── 汇总 ─────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - T_START ))
echo ""
echo "  总耗时: ${ELAPSED}s ($(( ELAPSED / 60 ))min)"
[ "$FAIL" -ne 0 ] && warn "部分任务失败，请查看上方日志路径"
echo ""
echo -e "${GREEN}完成！结果: ${RESULTS_DIR}/${NC}"
echo -e "${GREEN}日志: ${LOGDIR}/${NC}"
