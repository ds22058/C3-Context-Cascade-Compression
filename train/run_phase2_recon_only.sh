#!/bin/bash
# Phase 2 Ablation: 100% reconstruct data (no continuation)
# Compare with standard phase2 (70% recon + 30% continuation)

set -e

export PYTHONPATH="$(dirname "$(dirname "$(realpath "$0")")")":$PYTHONPATH

PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-"./output/phase1"}
DATA_PATH=${DATA_PATH:-"./data/processed/phase2_recon_only_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/phase2_recon_only"}
DECODER_LR_RATIO=${DECODER_LR_RATIO:-0.02}
LATENT_TOKEN_LEN=${LATENT_TOKEN_LEN:-32}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_LENGTH=${MAX_LENGTH:-8192}
NUM_GPUS=${NUM_GPUS:-8}

# --- Logging ---
export WANDB_PROJECT=${WANDB_PROJECT:-"c3v2-train"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase2-recon-only-dlr${DECODER_LR_RATIO}-lr${LEARNING_RATE}-ep${NUM_EPOCHS}"}

echo "============================================"
echo "  C3 v2 Phase 2 Ablation: Recon-Only"
echo "============================================"
echo "  Phase 1 ckpt: ${PHASE1_CHECKPOINT}"
echo "  Data:         ${DATA_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Decoder LR:   ${LEARNING_RATE} x ${DECODER_LR_RATIO}"
echo "  Epochs:       ${NUM_EPOCHS}"
echo "============================================"

deepspeed --num_gpus=${NUM_GPUS} train/train_c3v2.py \
    --deepspeed train/ds_zero2.json \
    --phase 2 \
    --phase1_checkpoint ${PHASE1_CHECKPOINT} \
    --decoder_lr_ratio ${DECODER_LR_RATIO} \
    --latent_token_len ${LATENT_TOKEN_LEN} \
    --data_path ${DATA_PATH} \
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
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir ${OUTPUT_DIR}
