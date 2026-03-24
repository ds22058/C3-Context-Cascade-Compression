#!/bin/bash
# Phase 1: Freeze decoder, train encoder + Q + mm_projector
# Multi-node training (default: 4 nodes × 8 GPUs = 32 GPUs)
#
# 适配训练平台: 平台在所有节点上执行同一个脚本, 自动注入分布式环境变量.
# 在平台 Executable Command 填写:
#   /public/longteng/C3-Context-Cascade-Compression/train/run_phase1_multinode.sh

set -e

# ==================== 自动推导项目根目录 (从脚本自身位置) ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "[INFO] Script location: ${SCRIPT_DIR}"
echo "[INFO] Project root:    ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ==================== 自动探测平台注入的分布式环境变量 ====================
# 不同平台变量名不同, 按优先级逐一探测

# --- Master Address ---
if [ -z "${MASTER_ADDR}" ]; then
    # 常见备选变量名
    for var in MASTER_IP CHIEF_IP ARNOLD_WORKER_0_HOST; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export MASTER_ADDR="${val}"
            echo "[INFO] MASTER_ADDR detected from \$${var} = ${val}"
            break
        fi
    done
fi

# --- Master Port ---
MASTER_PORT=${MASTER_PORT:-29500}

# --- Node Rank ---
if [ -z "${NODE_RANK}" ]; then
    for var in RANK INDEX WORKER_ID ARNOLD_ID; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export NODE_RANK="${val}"
            echo "[INFO] NODE_RANK detected from \$${var} = ${val}"
            break
        fi
    done
fi
NODE_RANK=${NODE_RANK:-0}

# --- Number of Nodes ---
if [ -z "${NUM_NODES}" ]; then
    for var in NNODES WORLD_SIZE NODE_NUM ARNOLD_WORKER_NUM; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export NUM_NODES="${val}"
            echo "[INFO] NUM_NODES detected from \$${var} = ${val}"
            break
        fi
    done
fi
NUM_NODES=${NUM_NODES:-4}

NUM_GPUS=${NUM_GPUS:-8}

# ==================== 打印平台注入的所有环境变量 (调试用) ====================
echo ""
echo "===== Platform Environment (distributed-related) ====="
env | grep -iE "MASTER|RANK|WORLD|NODE|WORKER|CHIEF|ARNOLD|NCCL|INDEX|NNODES" | sort || true
echo "======================================================="
echo ""

# ==================== Model / Data Paths (absolute) ====================
C3_MODEL_PATH=${C3_MODEL_PATH:-"${PROJECT_ROOT}/models/c3"}
DECODER_MODEL_PATH=${DECODER_MODEL_PATH:-"${PROJECT_ROOT}/models/qwen25-3b"}
DATA_PATH=${DATA_PATH:-"${PROJECT_ROOT}/data/processed/phase1_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/output/phase1_v2"}
DS_CONFIG=${DS_CONFIG:-"${PROJECT_ROOT}/train/ds_zero2.json"}

# ==================== Training Hyperparams ====================
LATENT_TOKEN_LEN=${LATENT_TOKEN_LEN:-32}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-8}
# 单机8卡: GRAD_ACCUM=4, effective_batch = 8*4*8 = 256
# 四机32卡: GRAD_ACCUM=1, effective_batch = 8*1*32 = 256 (保持一致)
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_STEPS=${MAX_STEPS:-100}

# ==================== NCCL Settings ====================
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-13}
# 根据集群网络环境取消注释并修改:
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1          # 没有 InfiniBand 时启用
# export NCCL_P2P_DISABLE=1         # 跨机时可能需要禁用 P2P

# ==================== Logging ====================
export WANDB_PROJECT=${WANDB_PROJECT:-"c3v2-train"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase1-${NUM_NODES}node-latent${LATENT_TOKEN_LEN}-lr${LEARNING_RATE}-test${MAX_STEPS}steps"}

# ==================== Validation ====================
if [ -z "${MASTER_ADDR}" ]; then
    echo "[ERROR] Cannot detect MASTER_ADDR. Please ensure the platform injects it,"
    echo "        or set it manually: export MASTER_ADDR=<master_ip>"
    echo ""
    echo "  Checked variables: MASTER_ADDR, MASTER_IP, CHIEF_IP, ARNOLD_WORKER_0_HOST"
    exit 1
fi

if [ ! -f "${DS_CONFIG}" ]; then
    echo "[ERROR] DeepSpeed config not found: ${DS_CONFIG}"
    exit 1
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * NUM_NODES))

echo "============================================"
echo "  C3 v2 Phase 1: Multi-Node Training"
echo "============================================"
echo "  Project root: ${PROJECT_ROOT}"
echo "  C3 model:     ${C3_MODEL_PATH}"
echo "  Decoder:      ${DECODER_MODEL_PATH}"
echo "  Data:         ${DATA_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  DS config:    ${DS_CONFIG}"
echo "  Latent len:   ${LATENT_TOKEN_LEN}"
echo "  Epochs:       ${NUM_EPOCHS} (max_steps=${MAX_STEPS})"
echo "  LR:           ${LEARNING_RATE}"
echo "  Batch:        ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS}gpu x ${NUM_NODES}node = ${EFFECTIVE_BATCH}"
echo "  Nodes:        ${NUM_NODES} | Node rank: ${NODE_RANK}"
echo "  Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

deepspeed \
    --num_nodes=${NUM_NODES} \
    --num_gpus=${NUM_GPUS} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
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
    --max_steps ${MAX_STEPS} \
    --warmup_steps 10 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --model_max_length ${MAX_LENGTH} \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}"
