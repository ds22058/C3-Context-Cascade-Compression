"""
为 MMLU 和 HotpotQA 构建评测数据。

两类评测范式：
  A) MMLU: few-shot 压缩 (600-1000 tok)，题干不压缩，log-prob 多选评测
  B) HotpotQA: 段落压缩，问题不压缩，生成式评测 (EM/F1)
     按段落长度分 3 个子集:
       - hotpotqa_short  (<1200 tok): 压缩 1 次 → 32 latent
       - hotpotqa_medium (1200-2400 tok): 对半分压 2 次 → 64 latent
       - hotpotqa_long   (>2400 tok): 三等分压 3 次 → 96 latent

用法：
    python eval/benchmark_data/prepare.py --tokenizer ./models/qwen25-3b
    python eval/benchmark_data/prepare.py --tasks mmlu,hotpotqa
"""

import argparse
import json
import os
import random

_this_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_this_dir, "raw")
PROCESSED_DIR = os.path.join(_this_dir, "processed")


# ════════════════════════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════════════════════════

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tok_len(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


# ════════════════════════════════════════════════════════════════
# MMLU: few-shot 组装 (不含 stem)
# ════════════════════════════════════════════════════════════════

def assemble_fewshot(instruction, fewshot_pool, tokenizer, min_tokens, max_tokens, rng):
    """组装 instruction + few-shot，严格控制在 [min_tokens, max_tokens]。"""
    inst_toks = tok_len(tokenizer, instruction) if instruction else 0
    if inst_toks > max_tokens:
        return None

    budget = max_tokens - inst_toks
    need = max(0, min_tokens - inst_toks)

    pool = list(fewshot_pool)
    rng.shuffle(pool)

    selected = []
    approx_sum = 0
    for fs_text, fs_toks in pool:
        if approx_sum + fs_toks > budget:
            if approx_sum >= need:
                break
            continue
        selected.append(fs_text)
        approx_sum += fs_toks
        if approx_sum >= need:
            break

    context = instruction + "".join(selected)
    actual = tok_len(tokenizer, context)

    while actual > max_tokens and selected:
        selected.pop(0)
        context = instruction + "".join(selected)
        actual = tok_len(tokenizer, context)

    if actual < min_tokens:
        return None
    return context, actual, len(selected)


def mmlu_fewshot_text(s):
    c = s["choices"]
    letter = "ABCD"[s["answer"]]
    return (
        f"{s['question'].strip()}\n"
        f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
        f"Answer: {letter}\n\n"
    )

def mmlu_instruction(subject):
    return f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"


def prepare_mmlu(tokenizer, min_tokens, max_tokens, seed, max_fewshot_pool):
    test_path = os.path.join(RAW_DIR, "mmlu/test.jsonl")
    fs_path = os.path.join(RAW_DIR, "mmlu/dev.jsonl")
    if not os.path.exists(test_path) or not os.path.exists(fs_path):
        print("  [跳过] mmlu: 原始数据不存在")
        return [], 0, 0

    test_data = load_jsonl(test_path)
    fewshot_raw = load_jsonl(fs_path)

    pool_by_group = {}
    for s in fewshot_raw:
        g = s.get("subject", "")
        text = mmlu_fewshot_text(s)
        toks = tok_len(tokenizer, text)
        pool_by_group.setdefault(g, []).append((text, toks))

    rng = random.Random(seed)
    results = []
    skipped = 0

    for sample in test_data:
        subject = sample.get("subject", "general")
        instruction = mmlu_instruction(subject)
        pool = pool_by_group.get(subject, [])
        ret = assemble_fewshot(instruction, pool, tokenizer, min_tokens, max_tokens, rng)
        if ret is None:
            skipped += 1
            continue

        fewshot_ctx, fewshot_toks, n_fs = ret
        c = sample["choices"]
        results.append({
            "task": "mmlu",
            "subtask": subject,
            "context_to_compress": fewshot_ctx,
            "context_tokens": fewshot_toks,
            "num_segments": 1,
            "stem_text": sample["question"].strip() + "\n",
            "options_text": f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:",
            "continuations": [" A", " B", " C", " D"],
            "answer_idx": sample["answer"],
            "num_fewshot": n_fs,
            "eval_mode": "logprob",
            "use_acc_norm": False,
        })

    return results, skipped, len(test_data)


# ════════════════════════════════════════════════════════════════
# HotpotQA: 段落压缩，生成式评测
# ════════════════════════════════════════════════════════════════

def prepare_hotpotqa(tokenizer, seed):
    """处理全部 HotpotQA 样本，按段落 token 长度分 3 个子集。"""
    test_path = os.path.join(RAW_DIR, "hotpotqa/test.jsonl")
    if not os.path.exists(test_path):
        print("  [跳过] hotpotqa: 原始数据不存在")
        return {}, 0

    test_data = load_jsonl(test_path)
    rng = random.Random(seed)

    buckets = {"hotpotqa_short": [], "hotpotqa_medium": [], "hotpotqa_long": []}
    total = len(test_data)

    for sample in test_data:
        ctx = sample["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]

        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(f"{title}: {''.join(sents).strip()}")
        rng.shuffle(parts)
        passage = "\n\n".join(parts)
        passage_toks = tok_len(tokenizer, passage)

        if passage_toks < 100:
            continue

        if passage_toks < 1200:
            bucket = "hotpotqa_short"
            num_seg = 1
        elif passage_toks < 2400:
            bucket = "hotpotqa_medium"
            num_seg = 2
        else:
            bucket = "hotpotqa_long"
            num_seg = 3

        answer = sample["answer"].strip()
        stem = f"\nQuestion: {sample['question']}\nAnswer:"

        buckets[bucket].append({
            "task": bucket,
            "subtask": sample.get("type", ""),
            "context_to_compress": passage,
            "context_tokens": passage_toks,
            "num_segments": num_seg,
            "stem_text": stem,
            "answer_text": answer,
            "eval_mode": "generate",
            "use_acc_norm": False,
        })

    return buckets, total


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="构建评测数据 (MMLU + HotpotQA)")
    ap.add_argument("--tokenizer", default="./models/qwen25-3b")
    ap.add_argument("--tasks", default=None, help="逗号分隔 (默认: mmlu,hotpotqa)")
    ap.add_argument("--min_tokens", type=int, default=600, help="MMLU few-shot 最小 tok")
    ap.add_argument("--max_tokens", type=int, default=1000, help="MMLU few-shot 最大 tok")
    ap.add_argument("--max_fewshot_pool", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    task_list = [t.strip() for t in args.tasks.split(",")] if args.tasks else ["mmlu", "hotpotqa"]
    print(f"任务: {', '.join(task_list)}\n")

    for task_name in task_list:
        if task_name == "mmlu":
            print(f"── mmlu ──")
            results, skipped, total = prepare_mmlu(
                tokenizer, args.min_tokens, args.max_tokens, args.seed, args.max_fewshot_pool
            )
            if not results:
                continue
            out_path = os.path.join(PROCESSED_DIR, "mmlu.jsonl")
            save_jsonl(results, out_path)
            toks = [r["context_tokens"] for r in results]
            print(f"  保留: {len(results)}/{total}  丢弃: {skipped}")
            print(f"  压缩部分 tokens: avg={sum(toks)//len(toks)}  min={min(toks)}  max={max(toks)}")
            print(f"  输出: {out_path}\n")

        elif task_name == "hotpotqa":
            print(f"── hotpotqa ──")
            buckets, total = prepare_hotpotqa(tokenizer, args.seed)
            if not buckets:
                continue
            for bname, rows in buckets.items():
                if not rows:
                    print(f"  {bname}: 0 条")
                    continue
                out_path = os.path.join(PROCESSED_DIR, f"{bname}.jsonl")
                save_jsonl(rows, out_path)
                toks = [r["context_tokens"] for r in rows]
                ns = rows[0]["num_segments"]
                print(f"  {bname}: {len(rows)} 条  "
                      f"tokens: avg={sum(toks)//len(toks)} min={min(toks)} max={max(toks)}  "
                      f"segments={ns}")
            print(f"  总计: {sum(len(v) for v in buckets.values())}/{total}")
            print()
        else:
            print(f"  [跳过] 未知任务: {task_name}")

    print("完成。")


if __name__ == "__main__":
    main()
