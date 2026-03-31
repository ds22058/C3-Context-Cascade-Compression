#!/bin/bash
# C3 v2.2 Phase 2: Unfreeze all, decoder with low lr
# Single node, 8 GPUs (H200) — 8k context, latent_token_len=512
set -e

export PYTHONPATH="$(dirname "$(dirname "$(realpath "$0")")")":$PYTHONPATH

PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-"./output/phase1_v22/checkpoint-20000"}
DATA_PATH=${DATA_PATH:-"./data/processed_v22/phase2_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/phase2_v22"}
DECODER_LR_RATIO=${DECODER_LR_RATIO:-1.0}
LATENT_TOKEN_LEN=${LATENT_TOKEN_LEN:-512}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-12288}
NUM_GPUS=${NUM_GPUS:-8}

export WANDB_PROJECT=${WANDB_PROJECT:-"c3v2-train"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"phase2-v22-dlr${DECODER_LR_RATIO}-lr${LEARNING_RATE}-ep${NUM_EPOCHS}"}

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

echo "============================================"
echo "  C3 v2.2 Phase 2: Full Model (8k)"
echo "============================================"
echo "  Phase 1 ckpt: ${PHASE1_CHECKPOINT}"
echo "  Data:         ${DATA_PATH}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Decoder LR:   ${LEARNING_RATE} x ${DECODER_LR_RATIO}"
echo "  Latent len:   ${LATENT_TOKEN_LEN}"
echo "  Max length:   ${MAX_LENGTH}"
echo "  Epochs:       ${NUM_EPOCHS}"
echo "  Batch:        ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = ${EFFECTIVE_BATCH}"
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
    --group_by_length True \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --output_dir ${OUTPUT_DIR}
