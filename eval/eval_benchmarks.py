"""
自定义 NLP 基准评估 - 替代 lm-eval-harness CLI
评估 ARC-Easy 和 WinoGrande，使用 log-likelihood 比较（与 lm-eval 完全相同的方法）。
解决 lm-eval 内部 dtype 参数与 transformers 4.49.0 不兼容的问题。

输出格式与 lm-eval-harness 兼容，可直接被 analyze_results.py 解析。

用法：
    python eval/eval_benchmarks.py --model_path ./models/qwen25-3b --output_path ./eval/results/lm_eval_qwen25_3b_cpu
    python eval/eval_benchmarks.py --model_path ./models/c3_decoder_extracted --tokenizer_path ./models/qwen25-3b --output_path ./eval/results/lm_eval_c3_decoder_cpu
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ──────────────────────────────────────────────────────────────
# 核心：log-likelihood 打分
# ──────────────────────────────────────────────────────────────

def score_choices(
    model: torch.nn.Module,
    tokenizer,
    context: str,
    choices: list[str],
    device: torch.device,
) -> list[float]:
    """
    计算每个 choice 在 context 条件下的归一化 log-likelihood。
    logits[i] 预测第 i+1 个 token，因此 choice 的第 j 个 token 对应
    full_ids[ctx_len + j]，由 logits[ctx_len - 1 + j] 给出概率。
    归一化除以 choice 的 token 数，防止较长 choice 被错误惩罚。
    """
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    scores = []

    for choice in choices:
        choice_ids = tokenizer(
            choice, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        if choice_ids.shape[1] == 0:
            scores.append(float("-inf"))
            continue

        full_ids = torch.cat([ctx_ids, choice_ids], dim=1)

        with torch.no_grad():
            logits = model(full_ids).logits  # [1, seq_len, vocab]

        # 取 choice 对应的 logit 切片
        start = ctx_ids.shape[1] - 1
        end = start + choice_ids.shape[1]
        cont_logits = logits[0, start:end, :]           # [choice_len, vocab]
        log_probs = F.log_softmax(cont_logits, dim=-1)  # [choice_len, vocab]
        ll = log_probs[range(choice_ids.shape[1]), choice_ids[0]].sum().item()

        scores.append(ll / choice_ids.shape[1])  # 长度归一化

    return scores


# ──────────────────────────────────────────────────────────────
# ARC-Easy
# ──────────────────────────────────────────────────────────────

def format_arc_context(example: dict) -> tuple[str, int]:
    """返回 (prompt, 正确答案的 index)"""
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    answer_key = example["answerKey"]

    prompt = f"Question: {example['question']}\n"
    for label, text in zip(labels, texts):
        prompt += f"({label}) {text}\n"
    prompt += "Answer:"

    # answerKey 可能是 "A"/"B"/"C"/"D" 或 "1"/"2"/"3"/"4"
    try:
        correct_idx = labels.index(answer_key)
    except ValueError:
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        correct_idx = labels.index(mapping.get(answer_key, "A"))

    return prompt, correct_idx


def eval_arc_easy(
    model, tokenizer, device: torch.device, limit: int
) -> dict:
    print("\n  加载 ARC-Easy...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    for ex in tqdm(ds, desc="  ARC-Easy", ncols=80):
        prompt, correct_idx = format_arc_context(ex)
        labels = ex["choices"]["label"]
        # choice 是单个字母，前面加空格匹配自然语言习惯
        choices = [f" {l}" for l in labels]
        scores = score_choices(model, tokenizer, prompt, choices, device)
        if scores.index(max(scores)) == correct_idx:
            correct += 1

    acc = correct / len(ds)
    print(f"  结果: {correct}/{len(ds)} = {acc*100:.2f}%")
    return {"acc,none": acc, "acc_stderr,none": 0.0, "alias": "arc_easy", "n": len(ds)}


# ──────────────────────────────────────────────────────────────
# WinoGrande
# ──────────────────────────────────────────────────────────────

def eval_winogrande(
    model, tokenizer, device: torch.device, limit: int
) -> dict:
    print("\n  加载 WinoGrande...")
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    for ex in tqdm(ds, desc="  WinoGrande", ncols=80):
        sentence = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        answer = ex["answer"]  # "1" 或 "2"

        blank_pos = sentence.index("_")
        context = sentence[:blank_pos]
        suffix = sentence[blank_pos + 1:]

        cont1 = opt1 + suffix
        cont2 = opt2 + suffix

        scores = score_choices(model, tokenizer, context, [cont1, cont2], device)
        pred = "1" if scores[0] >= scores[1] else "2"
        if pred == answer:
            correct += 1

    acc = correct / len(ds)
    print(f"  结果: {correct}/{len(ds)} = {acc*100:.2f}%")
    return {"acc,none": acc, "acc_stderr,none": 0.0, "alias": "winogrande", "n": len(ds)}


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

TASK_FN = {
    "arc_easy": eval_arc_easy,
    "winogrande": eval_winogrande,
}


def main():
    parser = argparse.ArgumentParser(description="ARC-Easy + WinoGrande 基准评估（无需 lm-eval）")
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--tokenizer_path", default=None,
        help="tokenizer 路径，默认与 model_path 相同。C3-Decoder 建议传 ./models/qwen25-3b",
    )
    parser.add_argument("--output_path", required=True, help="结果保存目录")
    parser.add_argument("--tasks", default="arc_easy,winogrande")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    tokenizer_path = args.tokenizer_path or args.model_path
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"模型:      {args.model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"设备:      {device}")
    print(f"任务:      {tasks}  (各限 {args.limit} 条)")

    # 加载模型
    print("\n加载模型...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"加载耗时: {time.time() - t0:.1f}s")

    # 评估各任务
    results = {}
    t_eval = time.time()
    for task in tasks:
        if task not in TASK_FN:
            print(f"[跳过] 未知任务: {task}")
            continue
        results[task] = TASK_FN[task](model, tokenizer, device, args.limit)

    elapsed = time.time() - t_eval

    # 保存（与 lm-eval 兼容的 JSON 格式）
    os.makedirs(args.output_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output_path, f"results_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "model": args.model_path,
             "limit": args.limit, "elapsed_s": round(elapsed, 1)},
            f, indent=2, ensure_ascii=False,
        )

    # 汇总
    print(f"\n{'='*50}")
    print("评估结果汇总")
    print(f"{'='*50}")
    for task, r in results.items():
        print(f"  {task:20s}: {r['acc,none']*100:.2f}%  (n={r['n']})")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"\n结果已保存到: {out_file}")


if __name__ == "__main__":
    main()
