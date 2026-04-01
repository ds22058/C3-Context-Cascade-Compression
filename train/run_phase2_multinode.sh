#!/bin/bash
# ============================================================================
# C3 v2.2 Phase 2: Full Model Training (all params unfrozen)
# 4 nodes × 8 GPUs = 32 GPUs, latent_token_len=512, max_length=12288
#
# 数据: phase2_train.jsonl (821K samples)
#   - reconstruct 550K (67%)  + continuation 240K (29%)  + instruction 31K (4%)
#
# 训练平台用法:
#   /public/longteng/C3-Context-Cascade-Compression/train/run_phase2_multinode.sh
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

# ==================== 项目根目录 ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

echo "[INFO] Script:       ${SCRIPT_DIR}/$(basename "$0")"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Hostname:     $(hostname)"
echo "[INFO] Date:         $(date '+%Y-%m-%d %H:%M:%S')"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ==================== 自动探测分布式环境变量 ====================

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

if [ -z "${MASTER_PORT}" ]; then
    export MASTER_PORT=29501
fi

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
    exit 1
fi

# ==================== 打印分布式环境 ====================
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

# ==================== Paths ====================
PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-"${PROJECT_ROOT}/output/phase1_v22_4node"}
DATA_PATH="${PROJECT_ROOT}/data/processed_v22/phase2_train.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/output/phase2_v22_4node"
DS_CONFIG="${PROJECT_ROOT}/train/ds_zero2.json"

# ==================== 校验关键文件 ====================
for f in "${PHASE1_CHECKPOINT}/config.json" "${DATA_PATH}" "${DS_CONFIG}"; do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] 文件不存在: ${f}"
        exit 1
    fi
done

# ==================== Hyperparameters ====================
# Phase 2: 全参训练 (~4.5B), decoder 和 encoder 同 LR
# effective batch = 4 × 2 × 8gpu × 4node = 256
DECODER_LR_RATIO=${DECODER_LR_RATIO:-1.0}
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
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase2-v22-${NUM_NODES}x${NUM_GPUS}gpu-dlr${DECODER_LR_RATIO}-lr${LEARNING_RATE}-ep${NUM_EPOCHS}"}

echo "============================================"
echo "  C3 v2.2 Phase 2: Multi-Node Full Training"
echo "============================================"
echo "  Phase 1 ckpt: ${PHASE1_CHECKPOINT}"
echo "  Data:         ${DATA_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Decoder LR:   ${LEARNING_RATE} x ${DECODER_LR_RATIO}"
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
    --rdzv_id=${RDZV_ID:-"c3v22_phase2"} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    "${PROJECT_ROOT}/train/train_c3v2.py" \
    --deepspeed "${DS_CONFIG}" \
    --phase 2 \
    --phase1_checkpoint "${PHASE1_CHECKPOINT}" \
    --decoder_lr_ratio ${DECODER_LR_RATIO} \
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
    --save_steps 1000 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}"
