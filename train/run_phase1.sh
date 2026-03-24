#!/bin/bash
# Phase 1: Freeze decoder, train encoder + Q + mm_projector
# Single node, 8 GPUs (H200)

set -e

export PYTHONPATH="$(dirname "$(dirname "$(realpath "$0")")")":$PYTHONPATH

C3_MODEL_PATH=${C3_MODEL_PATH:-"./models/c3"}
DECODER_MODEL_PATH=${DECODER_MODEL_PATH:-"./models/qwen25-3b"}
DATA_PATH=${DATA_PATH:-"./data/processed/phase1_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/phase1_v2"}
LATENT_TOKEN_LEN=${LATENT_TOKEN_LEN:-32}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_LENGTH=${MAX_LENGTH:-8192}
NUM_GPUS=${NUM_GPUS:-8}

# --- Logging ---
export WANDB_PROJECT=${WANDB_PROJECT:-"c3v2-train"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase1-latent${LATENT_TOKEN_LEN}-lr${LEARNING_RATE}-ep${NUM_EPOCHS}"}

echo "============================================"
echo "  C3 v2 Phase 1: Freeze Decoder Training"
echo "============================================"
echo "  C3 model:    ${C3_MODEL_PATH}"
echo "  Decoder:     ${DECODER_MODEL_PATH}"
echo "  Data:        ${DATA_PATH}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Latent len:  ${LATENT_TOKEN_LEN}"
echo "  Epochs:      ${NUM_EPOCHS}"
echo "  LR:          ${LEARNING_RATE}"
echo "  Batch:       ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS}"
echo "============================================"

deepspeed --num_gpus=${NUM_GPUS} train/train_c3v2.py \
    --deepspeed train/ds_zero2.json \
    --phase 1 \
    --c3_model_path ${C3_MODEL_PATH} \
    --decoder_model_path ${DECODER_MODEL_PATH} \
    --latent_token_len ${LATENT_TOKEN_LEN} \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps 200 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --model_max_length ${MAX_LENGTH} \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir ${OUTPUT_DIR}
