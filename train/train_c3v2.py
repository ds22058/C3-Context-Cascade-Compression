"""
C3 v2 Training Script — two-phase training with frozen/unfrozen decoder.

Usage:
  Phase 1 (freeze decoder):
    deepspeed --num_gpus=8 train/train_c3v2.py --phase 1 ...
  Phase 2 (unfreeze decoder with low lr):
    deepspeed --num_gpus=8 train/train_c3v2.py --phase 2 --phase1_checkpoint ./output/phase1 ...
"""

import os
import sys
import pathlib
import logging
import json
import shutil
import subprocess
import socket
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2ForCausalLM,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "C3-master"))
from C3.model.C3 import C3QwenForCausalLM, C3Config

from train.config import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from train.dataset_c3v2 import C3V2Dataset
from train.data_collator import C3V2DataCollator
from train.trainer_c3v2 import C3V2Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    c3_model_path: str = field(default="./models/c3")
    decoder_model_path: str = field(default="./models/qwen25-3b")
    latent_token_len: int = field(default=32, metadata={"help": "32, 64, or 128"})
    reinit_projector: bool = field(default=False)


@dataclass
class TrainingPipelineArguments:
    phase: int = field(default=1, metadata={"help": "1 = freeze decoder, 2 = full"})
    phase1_checkpoint: Optional[str] = field(default=None)
    decoder_lr_ratio: float = field(
        default=0.02, metadata={"help": "decoder_lr = lr * ratio (Phase 2)"}
    )


@dataclass
class DataArguments:
    data_path: str = field(default="./data/processed/phase1_train.jsonl")


@dataclass
class C3V2TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=8192)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def load_model(model_args: ModelArguments, pipeline_args: TrainingPipelineArguments, dtype):
    """Load C3 encoder + fresh Qwen2.5-3B decoder, or resume from Phase 1."""

    if pipeline_args.phase == 2 and pipeline_args.phase1_checkpoint:
        logger.info(f"Phase 2: loading from Phase 1 checkpoint {pipeline_args.phase1_checkpoint}")
        model = C3QwenForCausalLM.from_pretrained(
            pipeline_args.phase1_checkpoint,
            torch_dtype=dtype,
            use_safetensors=True,
            device_map=None,
        )
        return model

    logger.info(f"Loading C3 model from {model_args.c3_model_path}")
    model = C3QwenForCausalLM.from_pretrained(
        model_args.c3_model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        device_map=None,
    )

    logger.info(f"Loading fresh decoder from {model_args.decoder_model_path}")
    fresh_decoder = Qwen2ForCausalLM.from_pretrained(
        model_args.decoder_model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        device_map=None,
    )

    fresh_vocab_size = fresh_decoder.config.vocab_size
    if model.config.vocab_size != fresh_vocab_size:
        logger.info(
            f"Resizing C3 decoder embeddings {model.config.vocab_size} -> {fresh_vocab_size} "
            f"to match fresh decoder before weight copy"
        )
        model.resize_token_embeddings(fresh_vocab_size)

    fresh_sd = fresh_decoder.state_dict()
    model_sd = model.state_dict()

    replaced = 0
    skipped = []
    for key in fresh_sd:
        prefixed = f"model.{key}" if not key.startswith("model.") else key
        if "llm1" in prefixed or "mm_projector" in prefixed or ".Q." in prefixed:
            continue
        if prefixed in model_sd:
            if fresh_sd[key].shape != model_sd[prefixed].shape:
                skipped.append((prefixed, model_sd[prefixed].shape, fresh_sd[key].shape))
                continue
            model_sd[prefixed] = fresh_sd[key]
            replaced += 1

    model.load_state_dict(model_sd, strict=False)
    logger.info(f"Replaced {replaced} decoder weight tensors with fresh Qwen2.5-3B")
    if skipped:
        for name, old_shape, new_shape in skipped:
            logger.warning(f"Skipped {name}: {old_shape} vs {new_shape}")
    del fresh_decoder, fresh_sd
    torch.cuda.empty_cache()

    if model_args.reinit_projector:
        logger.info("Re-initializing mm_projector weights")
        model.model.mm_projector.reset_parameters()

    return model


def apply_freeze(model, phase: int):
    """Apply freezing strategy based on training phase."""
    if phase == 1:
        model.requires_grad_(False)
        for p in model.model.llm1.parameters():
            p.requires_grad = True
        for p in model.model.Q.parameters():
            p.requires_grad = True
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Phase 1 freeze: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total"
        )
    else:
        model.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Phase 2: all {trainable / 1e6:.1f}M params trainable")


def setup_tokenizer(model_args: ModelArguments):
    """Load tokenizer and add special tokens if needed."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.decoder_model_path,
        trust_remote_code=True,
        padding_side="right",
    )

    special_tokens = [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in special_tokens if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        logger.info(f"Added special tokens: {to_add}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = DEFAULT_PAD_TOKEN

    return tokenizer


# ---------------------------------------------------------------------------
# Run snapshot — persist code, config, and metadata for reproducibility
# ---------------------------------------------------------------------------
def _json_serializable(obj):
    """Convert dataclass / non-serializable objects to JSON-friendly types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _json_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def save_run_snapshot(
    output_dir: str,
    model_args: "ModelArguments",
    pipeline_args: "TrainingPipelineArguments",
    data_args: "DataArguments",
    training_args: "C3V2TrainingArguments",
):
    """Save a full snapshot of code, args, and environment at training start.

    Written to ``{output_dir}/run_snapshot/``.  Only executed on local_rank 0.
    """
    snap_dir = os.path.join(output_dir, "run_snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    # ---- 1. All arguments as readable JSON ----
    all_args = {
        "model_args": _json_serializable(model_args),
        "pipeline_args": _json_serializable(pipeline_args),
        "data_args": _json_serializable(data_args),
        "training_args": {
            k: _json_serializable(v)
            for k, v in training_args.to_dict().items()
        },
    }
    with open(os.path.join(snap_dir, "all_args.json"), "w") as f:
        json.dump(all_args, f, indent=2, default=str)

    # ---- 2. Copy training source code ----
    project_root = pathlib.Path(__file__).resolve().parent.parent
    train_src = project_root / "train"
    src_dst = pathlib.Path(snap_dir) / "source_code" / "train"
    if train_src.exists():
        shutil.copytree(
            train_src, src_dst, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    model_src = project_root / "C3-master" / "C3"
    model_dst = pathlib.Path(snap_dir) / "source_code" / "C3-master" / "C3"
    if model_src.exists():
        shutil.copytree(
            model_src, model_dst, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )

    # ---- 3. DeepSpeed config ----
    ds_cfg_path = getattr(training_args, "deepspeed", None)
    if ds_cfg_path and os.path.isfile(ds_cfg_path):
        shutil.copy2(ds_cfg_path, os.path.join(snap_dir, "deepspeed_config.json"))

    # ---- 4. Launch command ----
    with open(os.path.join(snap_dir, "launch_cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # ---- 5. Environment variables (training-relevant) ----
    relevant_prefixes = (
        "CUDA", "NCCL", "WANDB", "DEEPSPEED", "PYTHONPATH",
        "MASTER", "RANK", "WORLD_SIZE", "LOCAL_RANK",
        "OMP", "MKL", "NUM_GPUS", "OUTPUT_DIR",
        "C3_", "DECODER_", "DATA_PATH", "LEARNING_RATE",
        "BATCH_SIZE", "GRAD_ACCUM", "MAX_LENGTH", "NUM_EPOCHS",
        "PHASE", "LATENT",
    )
    env_snapshot = {
        k: v for k, v in sorted(os.environ.items())
        if any(k.startswith(p) or k == p for p in relevant_prefixes)
    }
    with open(os.path.join(snap_dir, "env_vars.json"), "w") as f:
        json.dump(env_snapshot, f, indent=2)

    # ---- 6. Git info (best-effort) ----
    git_info = {}
    try:
        git_info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff", "--stat"], cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_info["dirty"] = bool(diff)
        if diff:
            git_info["diff_stat"] = diff
            full_diff = subprocess.check_output(
                ["git", "diff"], cwd=str(project_root),
                stderr=subprocess.DEVNULL,
            ).decode()
            with open(os.path.join(snap_dir, "git_diff.patch"), "w") as f:
                f.write(full_diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_info["note"] = "not a git repo or git unavailable"
    with open(os.path.join(snap_dir, "git_info.json"), "w") as f:
        json.dump(git_info, f, indent=2)

    # ---- 7. Metadata ----
    meta = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": torch.cuda.device_count(),
    }
    with open(os.path.join(snap_dir, "snapshot_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Run snapshot saved to {snap_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingPipelineArguments, DataArguments, C3V2TrainingArguments)
    )
    model_args, pipeline_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Save full run snapshot (code + config + env) for reproducibility
    if training_args.local_rank in (-1, 0):
        save_run_snapshot(
            training_args.output_dir,
            model_args, pipeline_args, data_args, training_args,
        )

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    tokenizer = setup_tokenizer(model_args)

    model = load_model(model_args, pipeline_args, dtype)
    model.resize_token_embeddings(len(tokenizer))

    # Sync special token IDs between tokenizer and model config
    # (C3 config may have stale IDs from a different tokenizer)
    model.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    model.config.im_start_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0]
    model.config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])[0]

    model.to(dtype=dtype)

    apply_freeze(model, pipeline_args.phase)

    train_dataset = C3V2Dataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        latent_token_len=model_args.latent_token_len,
        model_max_length=training_args.model_max_length,
    )

    data_collator = C3V2DataCollator(tokenizer=tokenizer)

    decoder_lr_ratio = pipeline_args.decoder_lr_ratio if pipeline_args.phase == 2 else 1.0

    # --- Log config to wandb ---
    if training_args.local_rank in (-1, 0):
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "phase": pipeline_args.phase,
                    "latent_token_len": model_args.latent_token_len,
                    "c3_model_path": model_args.c3_model_path,
                    "decoder_model_path": model_args.decoder_model_path,
                    "decoder_lr_ratio": decoder_lr_ratio,
                    "data_path": data_args.data_path,
                    "dataset_size": len(train_dataset),
                    "model_max_length": training_args.model_max_length,
                    "trainable_params_M": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ) / 1e6,
                    "total_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                }, allow_val_change=True)
        except ImportError:
            pass

    trainer = C3V2Trainer(
        decoder_lr_ratio=decoder_lr_ratio,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    ckpt_path = None
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        ckpt_path = True
    trainer.train(resume_from_checkpoint=ckpt_path)

    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)
    logger.info(f"Training complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
