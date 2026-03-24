"""
找到 wikitext2.jsonl 中最长的样本，用 C3 模型进行 encoder 压缩→重建，
打印原文、词数、token 数、重建文本和准确率指标。
"""

import os
import sys
import json
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModel, AutoTokenizer

from train.config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
    ROLE_USER,
    ROLE_ASSISTANT,
    SEP_TOKEN,
    RECONSTRUCT_PROMPT,
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
WIKITEXT2_JSONL = os.path.join(os.path.dirname(__file__), "data", "wikitext2.jsonl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "c3")

SPECIAL_TOKENS_TO_SANITIZE = [
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
]


def sanitize(text):
    for tok in SPECIAL_TOKENS_TO_SANITIZE:
        text = text.replace(tok, "")
    return text


# ── Metrics ──

def compute_rouge_l(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)


def compute_token_f1(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    common = sum((ref_counter & hyp_counter).values())
    if common == 0:
        return 0.0
    precision = common / sum(hyp_counter.values())
    recall = common / sum(ref_counter.values())
    return 2 * precision * recall / (precision + recall)


def char_accuracy(ref, hyp):
    """逐字符比较的准确率。"""
    matches = sum(a == b for a, b in zip(ref, hyp))
    return matches / max(len(ref), len(hyp), 1)


# ── Data ──

def find_longest_sample(tokenizer):
    """遍历 wikitext2.jsonl，找到 token 数最多的那条。"""
    print(f"正在读取 {WIKITEXT2_JSONL} ...")
    best_text = ""
    best_token_count = 0
    best_line_idx = -1
    total_lines = 0

    with open(WIKITEXT2_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = sanitize((obj.get("text") or "").strip())
            if not text:
                continue
            total_lines += 1
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) > best_token_count:
                best_token_count = len(ids)
                best_text = text
                best_line_idx = i

    print(f"共 {total_lines} 条有效样本，最长样本在第 {best_line_idx + 1} 行")
    return best_text, best_line_idx


# ── Model helpers ──

def prepare_inputs(tokenizer, context_text, latent_token_len, device):
    placeholder = (
        DEFAULT_IM_START_TOKEN
        + DEFAULT_IMAGE_PATCH_TOKEN * latent_token_len
        + DEFAULT_IM_END_TOKEN
    )
    context_str = context_text + placeholder
    user_content = placeholder + "\n" + RECONSTRUCT_PROMPT
    conversation_str = (
        SYSTEM_MESSAGE + SEP_TOKEN
        + ROLE_USER + user_content + SEP_TOKEN
        + ROLE_ASSISTANT
    )
    context_ids = tokenizer(
        context_str, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    input_ids = tokenizer(
        conversation_str, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    return input_ids, context_ids


def reconstruct_text(model, tokenizer, context_text, latent_token_len,
                     device, max_new_tokens, stop_token_id):
    input_ids, context_ids = prepare_inputs(
        tokenizer, context_text, latent_token_len, device,
    )
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids, context_ids=context_ids,
                do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_id,
            )
    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ── Main ──

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 70)
    print("  C3 最长 WikiText-2 样本重建测试")
    print("=" * 70)
    print(f"模型路径: {os.path.abspath(MODEL_PATH)}")
    print(f"设备: {device}  |  精度: {dtype}")

    # 1. 加载 tokenizer
    print("\n[1/4] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 2. 找到最长样本
    print("\n[2/4] 查找最长样本 ...")
    longest_text, line_idx = find_longest_sample(tokenizer)

    token_ids = tokenizer(longest_text, add_special_tokens=False).input_ids
    num_tokens = len(token_ids)
    num_words = len(longest_text.split())

    print(f"\n{'─' * 70}")
    print(f"  最长样本信息")
    print(f"{'─' * 70}")
    print(f"  行号 (1-based): {line_idx + 1}")
    print(f"  Word 数:        {num_words}")
    print(f"  Token 数:       {num_tokens}")
    print(f"  字符数:         {len(longest_text)}")
    print(f"{'─' * 70}")
    print(f"  原始文本:")
    print(f"{'─' * 70}")
    print(longest_text)
    print(f"{'─' * 70}")

    # 3. 加载模型
    print("\n[3/4] 加载 C3 模型 ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=dtype, device_map={"": device},
        low_cpu_mem_usage=True, use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    model.eval()
    model.initialize_special_tokenizer(tokenizer, device=str(device))
    latent_token_len = model.get_model().config.latent_token_len
    stop_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f"模型加载耗时: {time.time() - t0:.1f}s")
    print(f"latent_token_len: {latent_token_len}")

    # 4. 重建
    print(f"\n[4/4] 执行 encoder 压缩 → 重建 ...")
    max_new_tokens = min(int(num_tokens * 1.5), 8192)
    print(f"max_new_tokens: {max_new_tokens}")

    t0 = time.time()
    generated = reconstruct_text(
        model, tokenizer, longest_text, latent_token_len,
        device, max_new_tokens, stop_token_id,
    )
    elapsed = time.time() - t0

    # 评估
    ref_tokens = tokenizer.tokenize(longest_text)
    hyp_tokens = tokenizer.tokenize(generated)
    rouge_l = compute_rouge_l(ref_tokens, hyp_tokens)
    token_f1 = compute_token_f1(ref_tokens, hyp_tokens)
    exact_match = 1.0 if longest_text.strip() == generated.strip() else 0.0
    char_acc = char_accuracy(longest_text, generated)

    gen_token_ids = tokenizer(generated, add_special_tokens=False).input_ids
    gen_num_words = len(generated.split())

    print(f"\n{'─' * 70}")
    print(f"  重建结果")
    print(f"{'─' * 70}")
    print(f"  重建文本:")
    print(f"{'─' * 70}")
    print(generated)
    print(f"{'─' * 70}")

    print(f"\n{'═' * 70}")
    print(f"  评估指标")
    print(f"{'═' * 70}")
    print(f"  原文 word 数:     {num_words}")
    print(f"  原文 token 数:    {num_tokens}")
    print(f"  重建 word 数:     {gen_num_words}")
    print(f"  重建 token 数:    {len(gen_token_ids)}")
    print(f"  ─────────────────────────────────")
    print(f"  ROUGE-L:          {rouge_l:.4f}")
    print(f"  Token F1:         {token_f1:.4f}")
    print(f"  Exact Match:      {'YES' if exact_match == 1.0 else 'NO'}")
    print(f"  Char Accuracy:    {char_acc:.4f} ({char_acc * 100:.2f}%)")
    print(f"  ─────────────────────────────────")
    print(f"  生成耗时:         {elapsed:.2f}s")
    print(f"{'═' * 70}")

    if exact_match != 1.0:
        print(f"\n  [差异分析]")
        ref_chars = list(longest_text)
        hyp_chars = list(generated)
        diff_positions = []
        for i in range(min(len(ref_chars), len(hyp_chars))):
            if ref_chars[i] != hyp_chars[i]:
                diff_positions.append(i)
                if len(diff_positions) >= 10:
                    break
        if len(ref_chars) != len(hyp_chars):
            print(f"  长度不同: 原文 {len(ref_chars)} 字符 vs 重建 {len(hyp_chars)} 字符")
        if diff_positions:
            print(f"  前 {len(diff_positions)} 个不同位置:")
            for pos in diff_positions:
                ctx_start = max(0, pos - 15)
                ctx_end = min(len(ref_chars), pos + 15)
                ref_snippet = "".join(ref_chars[ctx_start:ctx_end])
                hyp_ctx_end = min(len(hyp_chars), pos + 15)
                hyp_snippet = "".join(hyp_chars[ctx_start:hyp_ctx_end])
                print(f"    位置 {pos}: 原文[...{ref_snippet}...]")
                print(f"    {'':8s} 重建[...{hyp_snippet}...]")


if __name__ == "__main__":
    main()
