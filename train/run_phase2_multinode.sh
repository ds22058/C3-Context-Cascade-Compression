#!/bin/bash
# C3 v2.2 Phase 2: Full model training — Multi-node
#
# Adapts to training platforms that inject distributed env vars.
# Usage: set as the Executable Command on each node.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "[INFO] Script location: ${SCRIPT_DIR}"
echo "[INFO] Project root:    ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ==================== Auto-detect distributed env vars ====================

if [ -z "${MASTER_ADDR}" ]; then
    for var in MASTER_IP CHIEF_IP ARNOLD_WORKER_0_HOST; do
        val=$(eval echo "\$${var}" 2>/dev/null)
        if [ -n "${val}" ]; then
            export MASTER_ADDR="${val}"
            echo "[INFO] MASTER_ADDR detected from \$${var} = ${val}"
            break
        fi
    done
fi

MASTER_PORT=${MASTER_PORT:-29500}

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

echo ""
echo "===== Platform Environment (distributed-related) ====="
env | grep -iE "MASTER|RANK|WORLD|NODE|WORKER|CHIEF|ARNOLD|NCCL|INDEX|NNODES" | sort || true
echo "======================================================="
echo ""

# ==================== Paths ====================

PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-"${PROJECT_ROOT}/output/phase1_v22/checkpoint-20000"}
DATA_PATH=${DATA_PATH:-"${PROJECT_ROOT}/data/processed_v22/phase2_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/output/phase2_v22"}
DS_CONFIG=${DS_CONFIG:-"${PROJECT_ROOT}/train/ds_zero2.json"}

# ==================== Hyperparams ====================

DECODER_LR_RATIO=${DECODER_LR_RATIO:-1.0}
LATENT_TOKEN_LEN=${LATENT_TOKEN_LEN:-512}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-4}
# 4 nodes x 8 GPUs: GRAD_ACCUM=2 → effective batch = 4*2*8*4 = 256
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_LENGTH=${MAX_LENGTH:-12288}

# ==================== NCCL ====================

export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-13}

# ==================== Logging ====================

export WANDB_PROJECT=${WANDB_PROJECT:-"c3v2-train"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase2-v22-${NUM_NODES}node-dlr${DECODER_LR_RATIO}-lr${LEARNING_RATE}"}

# ==================== Validation ====================

if [ -z "${MASTER_ADDR}" ]; then
    echo "[ERROR] Cannot detect MASTER_ADDR."
    echo "  Checked: MASTER_ADDR, MASTER_IP, CHIEF_IP, ARNOLD_WORKER_0_HOST"
    exit 1
fi

if [ ! -f "${DS_CONFIG}" ]; then
    echo "[ERROR] DeepSpeed config not found: ${DS_CONFIG}"
    exit 1
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * NUM_NODES))

echo "============================================"
echo "  C3 v2.2 Phase 2: Multi-Node Training"
echo "============================================"
echo "  Phase 1 ckpt: ${PHASE1_CHECKPOINT}"
echo "  Data:         ${DATA_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Decoder LR:   ${LEARNING_RATE} x ${DECODER_LR_RATIO}"
echo "  Latent len:   ${LATENT_TOKEN_LEN}"
echo "  Max length:   ${MAX_LENGTH}"
echo "  Epochs:       ${NUM_EPOCHS}"
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
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}"
