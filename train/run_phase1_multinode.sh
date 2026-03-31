#!/bin/bash
# ============================================================================
# C3 v2.2 Phase 1: Encoder + Q + MM-Projector Training (Decoder frozen)
# 4 nodes × 8 GPUs = 32 GPUs, latent_token_len=512, max_length=12288
#
# 训练平台用法: 在 Executable Command 填写本脚本的绝对路径即可, 例如:
#   /public/longteng/C3-Context-Cascade-Compression/train/run_phase1_multinode.sh
#
# 平台会在所有节点上执行同一脚本, 脚本自动探测平台注入的分布式环境变量.
# ============================================================================

set -e

# ==================== 激活 Conda 环境 ====================
CONDA_ENV=${CONDA_ENV:-"megatron-lm-014"}

if ! command -v torchrun &>/dev/null; then
    for CONDA_BASE in /home/miniconda3 /opt/conda /root/miniconda3 /home/*/miniconda3; do
        if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            source "${CONDA_BASE}/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV}"
            echo "[INFO] Activated conda env: ${CONDA_ENV} (from ${CONDA_BASE})"
            break
        fi
    done
fi

if ! command -v torchrun &>/dev/null; then
    echo "[ERROR] torchrun not found. Please set CONDA_ENV or ensure PyTorch is in PATH."
    exit 1
fi

# ==================== 项目根目录 (从脚本位置自动推导) ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

echo "[INFO] Script:       ${SCRIPT_DIR}/$(basename "$0")"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Hostname:     $(hostname)"
echo "[INFO] Date:         $(date '+%Y-%m-%d %H:%M:%S')"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ==================== 自动探测分布式环境变量 ====================

# --- Master Address ---
if [ -z "${MASTER_ADDR}" ]; then
    for var in MASTER_IP CHIEF_IP ARNOLD_WORKER_0_HOST; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export MASTER_ADDR="${val}"
            echo "[INFO] MASTER_ADDR from \$${var} = ${val}"
            break
        fi
    done
fi

# --- Master Port (避免冲突, 使用非常用端口) ---
if [ -z "${MASTER_PORT}" ]; then
    export MASTER_PORT=29501
fi

# --- Node Rank ---
if [ -z "${NODE_RANK}" ]; then
    for var in RANK INDEX WORKER_ID ARNOLD_ID; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export NODE_RANK="${val}"
            echo "[INFO] NODE_RANK from \$${var} = ${val}"
            break
        fi
    done
fi
NODE_RANK=${NODE_RANK:-0}

# --- Number of Nodes ---
if [ -z "${NUM_NODES}" ]; then
    for var in NNODES NODE_NUM ARNOLD_WORKER_NUM; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export NUM_NODES="${val}"
            echo "[INFO] NUM_NODES from \$${var} = ${val}"
            break
        fi
    done
fi
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS=${NUM_GPUS:-8}

# ==================== 校验 MASTER_ADDR ====================
if [ -z "${MASTER_ADDR}" ]; then
    echo "[ERROR] 无法检测 MASTER_ADDR, 请确保平台注入了分布式环境变量,"
    echo "        或手动设置: export MASTER_ADDR=<master_ip>"
    echo "  已检查变量: MASTER_ADDR, MASTER_IP, CHIEF_IP, ARNOLD_WORKER_0_HOST"
    exit 1
fi

# ==================== 打印分布式环境 (调试) ====================
echo ""
echo "===== Distributed Environment ====="
echo "  MASTER_ADDR:  ${MASTER_ADDR}"
echo "  MASTER_PORT:  ${MASTER_PORT}"
echo "  NODE_RANK:    ${NODE_RANK}"
echo "  NUM_NODES:    ${NUM_NODES}"
echo "  NUM_GPUS:     ${NUM_GPUS}"
echo "  WORLD_SIZE:   $((NUM_NODES * NUM_GPUS))"
echo "====================================="
env | grep -iE "MASTER|RANK|WORLD|NODE|WORKER|CHIEF|ARNOLD|NCCL|INDEX|NNODES" | sort 2>/dev/null || true
echo ""

# ==================== Paths (绝对路径, 各节点通过共享存储访问) ====================
C3_MODEL_PATH="${PROJECT_ROOT}/models/c3"
DECODER_MODEL_PATH="${PROJECT_ROOT}/models/qwen25-3b"
DATA_PATH="${PROJECT_ROOT}/data/processed_v22/phase1_train.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/output/phase1_v22_4node"
DS_CONFIG="${PROJECT_ROOT}/train/ds_zero2.json"

# ==================== 校验关键文件 ====================
for f in "${C3_MODEL_PATH}/config.json" "${DECODER_MODEL_PATH}/config.json" "${DATA_PATH}" "${DS_CONFIG}"; do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] 文件不存在: ${f}"
        exit 1
    fi
done

# ==================== Hyperparameters ====================
# 单机: batch=4 × grad_accum=8 × 8gpu = 256
# 四机: batch=4 × grad_accum=2 × 8gpu × 4node = 256 (保持 effective batch 一致)
LATENT_TOKEN_LEN=512
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=4
GRAD_ACCUM=2
MAX_LENGTH=12288

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * NUM_NODES))

# ==================== NCCL ====================
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-13}

# ==================== Logging ====================
export WANDB_PROJECT=${WANDB_PROJECT:-"c3v2-train"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase1-v22-${NUM_NODES}x${NUM_GPUS}gpu-latent${LATENT_TOKEN_LEN}-lr${LEARNING_RATE}-ep${NUM_EPOCHS}"}

echo "============================================"
echo "  C3 v2.2 Phase 1: Multi-Node Training"
echo "============================================"
echo "  C3 model:     ${C3_MODEL_PATH}"
echo "  Decoder:      ${DECODER_MODEL_PATH}"
echo "  Data:         ${DATA_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  DS config:    ${DS_CONFIG}"
echo "  Latent len:   ${LATENT_TOKEN_LEN}"
echo "  Max length:   ${MAX_LENGTH}"
echo "  Epochs:       ${NUM_EPOCHS}"
echo "  LR:           ${LEARNING_RATE}"
echo "  Batch:        ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS}gpu x ${NUM_NODES}node = ${EFFECTIVE_BATCH}"
echo "  Node rank:    ${NODE_RANK} / ${NUM_NODES}"
echo "  Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=${NODE_RANK} \
    --rdzv_id=${RDZV_ID:-"c3v22_phase1"} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    "${PROJECT_ROOT}/train/train_c3v2.py" \
    --deepspeed "${DS_CONFIG}" \
    --phase 1 \
    --c3_model_path "${C3_MODEL_PATH}" \
    --decoder_model_path "${DECODER_MODEL_PATH}" \
    --latent_token_len ${LATENT_TOKEN_LEN} \
    --data_path "${DATA_PATH}" \
    --bf16 True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --model_max_length ${MAX_LENGTH} \
    --gradient_checkpointing True \
    --group_by_length True \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}"
