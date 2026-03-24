"""
Smoke test for C3 v2 training pipeline.

Tests:
  1. Convert dataset/train_test_data.json → temp JSONL (new format)
  2. C3V2Dataset loading & __getitem__
  3. C3V2DataCollator batching
  4. Model loading (C3 encoder + Qwen2.5-3B decoder)
  5. Single forward pass with loss computation
  6. Single backward pass (gradient flow)

Usage:
  python tests/smoke_test_train.py
"""

import os
import sys
import json
import tempfile
import logging

import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "C3-master"))

from transformers import AutoTokenizer
from C3.model.C3 import C3QwenForCausalLM, C3Config

from train.config import (
    DEFAULT_PAD_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from train.dataset_c3v2 import C3V2Dataset
from train.data_collator import C3V2DataCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def convert_test_data(src_path: str, dst_path: str, max_samples: int = 10):
    """Convert old-format JSON (with conversations) to new JSONL format."""
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(dst_path, "w", encoding="utf-8") as f:
        for item in data[:max_samples]:
            convs = item.get("conversations", [])
            # Extract context text (the "from": "context" entry)
            context_text = ""
            for c in convs:
                if c["from"] == "context":
                    context_text = c["value"]
                    break

            if not context_text:
                continue

            # Write as reconstruct task
            record = {"task": "reconstruct", "text": context_text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            # Also create a continuation variant from longer texts
            if len(context_text) > 500:
                record_cont = {
                    "task": "continuation",
                    "text": context_text,
                    "split_ratio": 0.55,
                }
                f.write(json.dumps(record_cont, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Converted {count} samples to {dst_path}")
    return count


def test_dataset(data_path: str, tokenizer, latent_token_len: int = 32):
    """Test C3V2Dataset loading and item access."""
    logger.info("--- Test: C3V2Dataset ---")
    ds = C3V2Dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        latent_token_len=latent_token_len,
        model_max_length=2048,
    )
    logger.info(f"Dataset size: {len(ds)}")
    assert len(ds) > 0, "Dataset is empty!"

    item = ds[0]
    assert "input_ids" in item, "Missing input_ids"
    assert "labels" in item, "Missing labels"
    assert "context_ids" in item, "Missing context_ids"

    logger.info(f"  input_ids shape: {item['input_ids'].shape}")
    logger.info(f"  labels shape:    {item['labels'].shape}")
    logger.info(f"  context_ids shape: {item['context_ids'].shape}")

    # Verify labels masking: some should be IGNORE_INDEX (-100)
    n_masked = (item["labels"] == -100).sum().item()
    n_total = item["labels"].shape[0]
    logger.info(f"  Labels: {n_masked}/{n_total} masked (prompt region)")
    assert n_masked > 0, "No labels masked — prompt masking may be broken"
    assert n_masked < n_total, "All labels masked — no training signal"

    logger.info("  PASSED: Dataset loading and item structure OK")
    return ds


def test_collator(ds, tokenizer):
    """Test C3V2DataCollator batching."""
    logger.info("--- Test: C3V2DataCollator ---")
    collator = C3V2DataCollator(tokenizer=tokenizer)

    batch_items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collator(batch_items)

    expected_keys = {"input_ids", "labels", "context_ids", "attention_mask", "context_attention_mask"}
    assert expected_keys == set(batch.keys()), f"Unexpected batch keys: {batch.keys()}"

    bs = batch["input_ids"].shape[0]
    logger.info(f"  Batch size: {bs}")
    logger.info(f"  input_ids: {batch['input_ids'].shape}")
    logger.info(f"  context_ids: {batch['context_ids'].shape}")
    logger.info(f"  labels: {batch['labels'].shape}")
    logger.info(f"  attention_mask: {batch['attention_mask'].shape}")
    logger.info(f"  context_attention_mask: {batch['context_attention_mask'].shape}")

    # Shapes should match
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["context_ids"].shape == batch["context_attention_mask"].shape

    logger.info("  PASSED: Collator produces correct batch structure")
    return batch


def test_model_load(model_path: str, decoder_path: str, tokenizer, dtype):
    """Test model loading: C3 encoder + fresh Qwen2.5-3B decoder."""
    logger.info("--- Test: Model Loading ---")

    from transformers import Qwen2ForCausalLM

    logger.info(f"  Loading C3 model from {model_path}")
    model = C3QwenForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, use_safetensors=True,
        device_map={"": "cpu"},
    )

    logger.info(f"  Loading fresh decoder from {decoder_path}")
    fresh_decoder = Qwen2ForCausalLM.from_pretrained(
        decoder_path, torch_dtype=dtype, use_safetensors=True
    )

    # Replace decoder weights (same logic as train_c3v2.py)
    fresh_sd = fresh_decoder.state_dict()
    model_sd = model.state_dict()
    replaced = 0
    skipped = 0
    for key in fresh_sd:
        prefixed = f"model.{key}" if not key.startswith("model.") else key
        if "llm1" in prefixed or "mm_projector" in prefixed or ".Q." in prefixed:
            continue
        if prefixed in model_sd:
            if fresh_sd[key].shape != model_sd[prefixed].shape:
                skipped += 1
                continue
            model_sd[prefixed] = fresh_sd[key]
            replaced += 1
    model.load_state_dict(model_sd, strict=False)
    if skipped:
        logger.info(f"  Skipped {skipped} tensors due to shape mismatch (vocab size diff)")
    logger.info(f"  Replaced {replaced} decoder weight tensors")
    del fresh_decoder, fresh_sd

    # Resize embeddings for special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Sync special token IDs between tokenizer and model config
    model.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    model.config.im_start_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0]
    model.config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])[0]

    # Verify key components exist
    assert hasattr(model.model, "llm1"), "Missing llm1 (encoder)"
    assert hasattr(model.model, "Q"), "Missing Q (latent queries)"
    assert hasattr(model.model, "mm_projector"), "Missing mm_projector"
    assert model.model.llm1 is not None, "llm1 is None"

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total params: {total_params / 1e6:.1f}M")

    # Test Phase 1 freeze
    model.requires_grad_(False)
    for p in model.model.llm1.parameters():
        p.requires_grad = True
    for p in model.model.Q.parameters():
        p.requires_grad = True
    for p in model.model.mm_projector.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Phase 1 trainable: {trainable / 1e6:.1f}M / {total_params / 1e6:.1f}M")
    assert trainable > 0, "No trainable parameters in Phase 1!"
    assert trainable < total_params, "All params trainable — freeze not working"

    logger.info("  PASSED: Model loading and freeze OK")
    return model


def test_forward_backward(model, batch, device):
    """Test forward pass with loss and backward pass."""
    logger.info("--- Test: Forward + Backward ---")

    model.to(device)
    model.train()

    # Move batch to device
    batch_gpu = {k: v.to(device) for k, v in batch.items()}

    logger.info("  Running forward pass...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(
            input_ids=batch_gpu["input_ids"],
            context_ids=batch_gpu["context_ids"],
            attention_mask=batch_gpu["attention_mask"],
            context_attention_mask=batch_gpu["context_attention_mask"],
            labels=batch_gpu["labels"],
        )

    loss = outputs.loss
    logits = outputs.logits
    logger.info(f"  Loss: {loss.item():.4f}")
    logger.info(f"  Logits shape: {logits.shape}")

    assert loss is not None, "Loss is None"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.item() > 0, "Loss should be positive"

    logger.info("  Running backward pass...")
    loss.backward()

    # Check gradients exist for trainable params
    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().sum() > 0:
                has_grad = True
                break

    assert has_grad, "No gradients flowing — backward pass may be broken"
    logger.info("  PASSED: Forward + backward OK, gradients flowing")


def main():
    c3_model_path = os.path.join(PROJECT_ROOT, "models", "c3")
    decoder_model_path = os.path.join(PROJECT_ROOT, "models", "qwen25-3b")
    test_data_path = os.path.join(PROJECT_ROOT, "dataset", "train_test_data.json")
    latent_token_len = 32

    assert os.path.isdir(c3_model_path), f"C3 model not found: {c3_model_path}"
    assert os.path.isdir(decoder_model_path), f"Decoder not found: {decoder_model_path}"
    assert os.path.isfile(test_data_path), f"Test data not found: {test_data_path}"

    dtype = torch.bfloat16
    device = "cuda:0"

    # Step 1: Convert test data to JSONL
    logger.info("=" * 60)
    logger.info("Step 1: Convert test data")
    logger.info("=" * 60)
    tmp_jsonl = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir="/tmp"
    )
    tmp_jsonl.close()
    convert_test_data(test_data_path, tmp_jsonl.name, max_samples=10)

    # Step 2: Load tokenizer
    logger.info("=" * 60)
    logger.info("Step 2: Load tokenizer")
    logger.info("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(
        decoder_model_path, trust_remote_code=True, padding_side="right"
    )
    special_tokens = [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in special_tokens if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = DEFAULT_PAD_TOKEN
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Step 3: Test Dataset
    logger.info("=" * 60)
    logger.info("Step 3: Test Dataset")
    logger.info("=" * 60)
    ds = test_dataset(tmp_jsonl.name, tokenizer, latent_token_len)

    # Step 4: Test Collator
    logger.info("=" * 60)
    logger.info("Step 4: Test Collator")
    logger.info("=" * 60)
    batch = test_collator(ds, tokenizer)

    # Step 5: Test Model Loading
    logger.info("=" * 60)
    logger.info("Step 5: Test Model Loading")
    logger.info("=" * 60)
    model = test_model_load(c3_model_path, decoder_model_path, tokenizer, dtype)

    # Step 6: Test Forward + Backward
    logger.info("=" * 60)
    logger.info("Step 6: Test Forward + Backward")
    logger.info("=" * 60)
    test_forward_backward(model, batch, device)

    # Cleanup
    os.unlink(tmp_jsonl.name)

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
