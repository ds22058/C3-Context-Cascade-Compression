"""
C3 上下文压缩→重建准确率评估（多数据集 / 多长度）。

数据全部从本地 eval/data/ 读取，无需联网。需先运行:
  python eval/download_eval_data.py

支持的数据集（--datasets 参数选择，逗号分隔）:
  wikitext-2   WikiText-2 短段落          64-512 tok, 50 samples
  mixed-1k/2k/4k/8k  书籍+GovReport 混合  ~1k/2k/4k/8k tok, 各 20 samples

用法：
    # 单卡
    python eval/eval_reconstruction.py \
        --model_path ./output/phase1 --model_name c3-phase1 --device cuda

    # 4卡并行
    python eval/eval_reconstruction.py \
        --model_path ./output/phase1 --model_name c3-phase1 --num_gpus 4

    # 自动检测可用 GPU 数量
    python eval/eval_reconstruction.py \
        --model_path ./output/phase1 --model_name c3-phase1 --num_gpus auto
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
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

EVAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WIKITEXT2_JSONL = os.path.join(EVAL_DATA_DIR, "wikitext2.jsonl")
GOVREPORT_JSONL = os.path.join(EVAL_DATA_DIR, "govreport.jsonl")
LONG_TEXTS_DIR = os.path.join(EVAL_DATA_DIR, "long_texts")

SPECIAL_TOKENS_TO_SANITIZE = [
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
]

DATASET_CONFIGS = {
    "wikitext-2": {
        "description": "WikiText-2 短文本 (64-512 tok)",
        "source": "wikitext-2",
        "min_tokens": 64,
        "max_tokens": 512,
        "default_samples": 50,
    },
    "mixed-1k": {
        "description": "书籍+GovReport 混合 (~1024 tok)",
        "source": "mixed",
        "target_tokens": 1024,
        "default_samples": 20,
    },
    "mixed-2k": {
        "description": "书籍+GovReport 混合 (~2048 tok)",
        "source": "mixed",
        "target_tokens": 2048,
        "default_samples": 20,
    },
    "mixed-4k": {
        "description": "书籍+GovReport 混合 (~4096 tok)",
        "source": "mixed",
        "target_tokens": 4096,
        "default_samples": 20,
    },
    "mixed-8k": {
        "description": "书籍+GovReport 混合 (~8192 tok)",
        "source": "mixed",
        "target_tokens": 8192,
        "default_samples": 20,
    },
}


# ── Metric helpers ─────────────────────────────────────────────


def compute_char_precision(reference: str, hypothesis: str) -> float:
    """Character-level precision: 1 - NED (Normalized Edit Distance).

    This is the metric used in the C3 paper (and DeepSeek-OCR / GOT-OCR).
    NED = Levenshtein(pred, ref) / max(len(pred), len(ref))
    Precision = 1 - NED, in range [0, 1].
    """
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    max_len = max(len(reference), len(hypothesis))
    try:
        from rapidfuzz.distance import Levenshtein
        dist = Levenshtein.distance(reference, hypothesis)
    except ImportError:
        dist = _levenshtein_dp(reference, hypothesis)
    return 1.0 - dist / max_len


def _levenshtein_dp(s1: str, s2: str) -> int:
    """Pure-Python two-row Levenshtein distance (fallback when rapidfuzz is absent)."""
    if len(s1) < len(s2):
        return _levenshtein_dp(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                curr.append(prev[j])
            else:
                curr.append(1 + min(prev[j], prev[j + 1], curr[j]))
        prev = curr
    return prev[-1]


def compute_rouge_l(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    m, n = len(reference_tokens), len(hypothesis_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
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


def compute_token_f1(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    ref_counter = Counter(reference_tokens)
    hyp_counter = Counter(hypothesis_tokens)
    common = sum((ref_counter & hyp_counter).values())
    if common == 0:
        return 0.0
    precision = common / sum(hyp_counter.values())
    recall = common / sum(ref_counter.values())
    return 2 * precision * recall / (precision + recall)


# ── Data loading ───────────────────────────────────────────────


def _sanitize(text):
    for tok in SPECIAL_TOKENS_TO_SANITIZE:
        text = text.replace(tok, "")
    return text


def load_wikitext2_samples(tokenizer, cfg, num_samples):
    """从 eval/data/wikitext2.jsonl 读取，一行一条 {"text": "..."}。"""
    if not os.path.isfile(WIKITEXT2_JSONL):
        raise FileNotFoundError(
            f"未找到 {WIKITEXT2_JSONL}，请先运行: python eval/download_eval_data.py"
        )
    samples = []
    min_tok, max_tok = cfg["min_tokens"], cfg["max_tokens"]
    with open(WIKITEXT2_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _sanitize((obj.get("text") or "").strip())
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) < min_tok:
                continue
            if len(ids) > max_tok:
                text = tokenizer.decode(ids[:max_tok], skip_special_tokens=False)
            samples.append(text)
            if len(samples) >= num_samples:
                break
    return samples


def _extract_windows(doc_token_ids_list, tokenizer, target_tokens, num_samples):
    """从 tokenized 文档列表中截取不重叠的固定长度窗口。"""
    samples = []
    for doc_ids in doc_token_ids_list:
        for start in range(0, len(doc_ids) - target_tokens + 1, target_tokens):
            window_ids = doc_ids[start : start + target_tokens]
            text = tokenizer.decode(window_ids, skip_special_tokens=False).strip()
            text = _sanitize(text)
            if text and len(text) > 100:
                samples.append(text)
            if len(samples) >= num_samples:
                return samples
    return samples


def _load_book_token_ids(tokenizer):
    """从 eval/data/long_texts/ 读取英文书籍 .txt，返回 token id 列表的列表。"""
    if not os.path.isdir(LONG_TEXTS_DIR):
        return []
    docs = []
    for fname in sorted(os.listdir(LONG_TEXTS_DIR)):
        if not fname.endswith(".txt") or fname.startswith("zh_"):
            continue
        path = os.path.join(LONG_TEXTS_DIR, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = _sanitize(f.read())
        if len(text) < 5000:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        docs.append(ids)
    return docs


def _load_govreport_token_ids(tokenizer):
    """从 eval/data/govreport.jsonl 读取，每行 {"report": "..."}。"""
    if not os.path.isfile(GOVREPORT_JSONL):
        return []
    docs = []
    with open(GOVREPORT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _sanitize((obj.get("report") or "").strip())
            if len(text) < 2000:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            docs.append(ids)
    return docs


def load_long_samples(tokenizer, target_tokens, num_samples):
    """书籍 + GovReport 混合：各取一半窗口，合并返回。"""
    num_from_books = num_samples // 2
    num_from_gov = num_samples - num_from_books
    samples = []

    book_docs = _load_book_token_ids(tokenizer)
    if book_docs:
        print(f"    书籍: {len(book_docs)} 本")
        samples.extend(
            _extract_windows(book_docs, tokenizer, target_tokens, num_from_books)
        )
    needed = num_from_books - len(samples)
    if needed > 0:
        print(f"    注意: 书籍窗口不足，仅取到 {len(samples)}/{num_from_books}")

    gov_docs = _load_govreport_token_ids(tokenizer)
    if gov_docs:
        print(f"    报告: {len(gov_docs)} 篇")
        from_gov = _extract_windows(gov_docs, tokenizer, target_tokens, num_from_gov)
        samples.extend(from_gov)
    if len(samples) < num_samples:
        print(f"  注意: 混合长文本共 {len(samples)}/{num_samples} 条")
    return samples[:num_samples]


def load_samples(ds_name, tokenizer, num_samples):
    cfg = DATASET_CONFIGS[ds_name]
    n = num_samples or cfg["default_samples"]
    src = cfg["source"]

    if src == "wikitext-2":
        return load_wikitext2_samples(tokenizer, cfg, n)
    elif src == "mixed":
        target = cfg["target_tokens"]
        return load_long_samples(tokenizer, target, n)
    else:
        raise ValueError(f"unknown source: {src}")


# ── Model helpers ──────────────────────────────────────────────


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
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids, context_ids=context_ids,
                    do_sample=False, num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=stop_token_id,
                )
        else:
            output_ids = model.generate(
                input_ids, context_ids=context_ids,
                do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_id,
            )
    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_sample(reference, hypothesis, tokenizer):
    ref_tokens = tokenizer.tokenize(reference)
    hyp_tokens = tokenizer.tokenize(hypothesis)
    return {
        "char_precision": compute_char_precision(reference.strip(), hypothesis.strip()),
        "rouge_l": compute_rouge_l(ref_tokens, hyp_tokens),
        "token_f1": compute_token_f1(ref_tokens, hyp_tokens),
        "exact_match": 1.0 if reference.strip() == hypothesis.strip() else 0.0,
        "ref_token_count": len(ref_tokens),
        "hyp_token_count": len(hyp_tokens),
    }


# ── Multi-GPU worker ───────────────────────────────────────────


def _gpu_worker(rank, model_path, dtype, work_items, result_queue, save_samples):
    """Multi-GPU worker: 在指定 GPU 上加载模型并处理样本分片。"""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=dtype, device_map={"": device},
        low_cpu_mem_usage=True, use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    model.eval()
    model.initialize_special_tokenizer(tokenizer, device=str(device))
    latent_token_len = model.get_model().config.latent_token_len
    stop_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    print(f"  [GPU {rank}] 模型加载完成，{len(work_items)} 个样本待处理", flush=True)

    for ds_name, sample_idx, text, max_new_tokens in work_items:
        t_start = time.time()
        try:
            generated = reconstruct_text(
                model, tokenizer, text, latent_token_len,
                device, max_new_tokens, stop_token_id,
            )
        except Exception as e:
            result_queue.put((ds_name, sample_idx, None, str(e), None))
            continue
        elapsed = time.time() - t_start
        metrics = evaluate_sample(text, generated, tokenizer)
        metrics["elapsed_s"] = round(elapsed, 2)

        detail = None
        if save_samples:
            detail = {"reference": text[:2000], "generated": generated[:2000]}
        result_queue.put((ds_name, sample_idx, metrics, None, detail))

    result_queue.put(None)
    del model
    torch.cuda.empty_cache()


# ── Main ───────────────────────────────────────────────────────


ALL_DATASETS = ",".join(DATASET_CONFIGS.keys())


def _aggregate_dataset(results_for_ds):
    """对单个数据集的 metrics 列表做聚合。"""
    metric_keys = ["char_precision", "rouge_l", "token_f1", "exact_match"]
    avg = {}
    for k in metric_keys:
        vals = [m[k] for m in results_for_ds]
        avg[k] = round(sum(vals) / len(vals), 4)
    avg["avg_ref_tokens"] = round(
        sum(m["ref_token_count"] for m in results_for_ds) / len(results_for_ds), 1
    )
    avg["avg_hyp_tokens"] = round(
        sum(m["hyp_token_count"] for m in results_for_ds) / len(results_for_ds), 1
    )
    avg["num_samples"] = len(results_for_ds)
    avg["avg_time_s"] = round(
        sum(m["elapsed_s"] for m in results_for_ds) / len(results_for_ds), 2
    )
    return avg


def _print_dataset_result(ds_name, avg):
    metric_keys = ["char_precision", "rouge_l", "token_f1", "exact_match"]
    print(f"\n  ── {ds_name} 结果 ──")
    print(f"  样本数: {avg['num_samples']}  |  平均 ref tokens: {avg['avg_ref_tokens']}")
    for k in metric_keys:
        label = {"char_precision": "Char Prec", "rouge_l": "ROUGE-L",
                 "token_f1": "Token F1", "exact_match": "Exact Match"}[k]
        val = avg.get(k)
        if val is None:
            continue
        if k == "exact_match":
            print(f"  {label:12s}  {val*100:.1f}%")
        else:
            print(f"  {label:12s}  {val:.4f}")


def _print_summary(model_name, all_dataset_results):
    print(f"\n{'=' * 65}")
    print(f"  重建准确率总结  |  {model_name}")
    print(f"{'=' * 65}")
    header = f"  {'数据集':<20s} {'样本':>4s} {'avg tok':>7s} {'Char Prec':>9s} {'ROUGE-L':>8s} {'Token F1':>8s} {'EM':>6s}"
    print(header)
    print(f"  {'─' * 68}")
    for ds, avg in all_dataset_results.items():
        em_str = f"{avg['exact_match']*100:.0f}%"
        cp_str = f"{avg['char_precision']:.4f}" if 'char_precision' in avg else "N/A"
        print(
            f"  {ds:<20s} {avg['num_samples']:>4d} {avg['avg_ref_tokens']:>7.0f} "
            f"{cp_str:>9s} {avg['rouge_l']:>8.4f} {avg['token_f1']:>8.4f} {em_str:>6s}"
        )
    print()


def main():
    ap = argparse.ArgumentParser(description="C3 重建准确率评估（多数据集 / 多长度，支持多GPU并行）")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument(
        "--datasets", default=ALL_DATASETS,
        help=f"逗号分隔 (可选: {ALL_DATASETS})",
    )
    ap.add_argument("--samples_per_dataset", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num_gpus", default="1",
                    help="并行GPU数: 数字 或 'auto'自动检测 (默认1)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--save_samples", action="store_true",
                    help="额外保存逐条详情到 _samples.json")
    args = ap.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    output = args.output or f"./eval/results/{model_name}/reconstruction.json"

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    for d in dataset_names:
        if d not in DATASET_CONFIGS:
            print(f"未知数据集: {d}，可选: {ALL_DATASETS}")
            raise SystemExit(1)

    # ── 确定设备和 GPU 数量 ──
    use_cuda = (args.device == "cuda") or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        avail_gpus = torch.cuda.device_count()
        if args.num_gpus == "auto":
            num_gpus = avail_gpus
        else:
            num_gpus = int(args.num_gpus)
        num_gpus = max(1, min(num_gpus, avail_gpus))
    else:
        num_gpus = 0

    dtype = torch.bfloat16 if use_cuda else torch.float32
    if num_gpus > 1:
        device_str = f"cuda x{num_gpus}"
    elif num_gpus == 1:
        device_str = "cuda"
    else:
        device_str = "cpu"

    print(f"模型: {model_name}  |  路径: {args.model_path}")
    print(f"设备: {device_str}  |  精度: {dtype}")
    print(f"数据集: {', '.join(dataset_names)}")

    if not os.path.isdir(args.model_path):
        print(f"错误：模型路径不存在: {args.model_path}")
        raise SystemExit(1)

    # ── 加载 tokenizer 用于数据准备（轻量） ──
    print(f"\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ── 准备全部样本 ──
    all_work = []  # [(ds_name, sample_idx, text, max_new_tokens), ...]
    ds_sample_counts = {}
    for ds_name in dataset_names:
        ds_cfg = DATASET_CONFIGS[ds_name]
        print(f"\n{'─' * 55}")
        print(f"数据集: {ds_name}  ({ds_cfg['description']})")
        samples = load_samples(ds_name, tokenizer, args.samples_per_dataset)
        if not samples:
            print(f"  没有符合条件的样本，跳过")
            continue
        print(f"  样本数: {len(samples)}")
        ds_sample_counts[ds_name] = len(samples)
        context_max = ds_cfg.get("target_tokens", ds_cfg.get("max_tokens", 512))
        max_new_tokens = min(int(context_max * 1.3), 8192)
        for idx, text in enumerate(samples):
            all_work.append((ds_name, idx, text, max_new_tokens))

    if not all_work:
        print("没有可评估的样本")
        return

    total_samples = len(all_work)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # ── 分支：多 GPU 并行 vs 单卡/CPU ──
    if num_gpus > 1:
        _run_multi_gpu(
            num_gpus, args.model_path, dtype, all_work,
            ds_sample_counts, args.save_samples,
            model_name, output,
        )
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        _run_single(
            device, dtype, args.model_path, tokenizer, all_work,
            ds_sample_counts, args.save_samples,
            model_name, output,
        )


def _run_multi_gpu(num_gpus, model_path, dtype, all_work,
                   ds_sample_counts, save_samples, model_name, output):
    """多 GPU 并行执行。"""
    total = len(all_work)
    print(f"\n多GPU并行: {num_gpus} 卡，共 {total} 个样本")
    print(f"加载模型到 {num_gpus} 张卡（每卡独立加载）...\n")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    shards = [[] for _ in range(num_gpus)]
    for i, item in enumerate(all_work):
        shards[i % num_gpus].append(item)

    t0 = time.time()
    processes = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(rank, model_path, dtype, shards[rank],
                  result_queue, save_samples),
        )
        p.start()
        processes.append(p)

    results_by_ds = {}
    details_by_ds = {}
    processed = 0
    done_count = 0

    while done_count < num_gpus:
        item = result_queue.get()
        if item is None:
            done_count += 1
            continue
        ds_name, sample_idx, metrics, error, detail = item
        if error:
            print(f"  样本 {ds_name}#{sample_idx} 失败: {error}")
            continue
        results_by_ds.setdefault(ds_name, []).append(metrics)
        if detail:
            details_by_ds.setdefault(ds_name, []).append(
                {**detail, "metrics": metrics}
            )
        processed += 1
        if processed % 10 == 0 or processed == total:
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            print(
                f"  进度: {processed}/{total}  "
                f"({elapsed:.0f}s, {speed:.1f} samples/s)",
                flush=True,
            )

    for p in processes:
        p.join()

    wall_time = time.time() - t0
    print(f"\n全部完成，总耗时 {wall_time:.1f}s")

    all_dataset_results = {}
    for ds_name in ds_sample_counts:
        if ds_name not in results_by_ds:
            continue
        avg = _aggregate_dataset(results_by_ds[ds_name])
        all_dataset_results[ds_name] = avg
        _print_dataset_result(ds_name, avg)

    output_data = {
        "model_name": model_name,
        "model_path": model_path,
        "device": f"cuda x{num_gpus}",
        "dtype": str(dtype),
        "wall_time_s": round(wall_time, 1),
        "datasets": all_dataset_results,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  → 结果已保存到 {output}")

    if save_samples and details_by_ds:
        samples_path = output.replace(".json", "_samples.json")
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(details_by_ds, f, indent=2, ensure_ascii=False)
        print(f"逐条详情已保存到: {samples_path}")

    _print_summary(model_name, all_dataset_results)


def _run_single(device, dtype, model_path, tokenizer, all_work,
                ds_sample_counts, save_samples, model_name, output):
    """单卡 / CPU 执行（原有逻辑）。"""
    print(f"\n加载模型...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=dtype, device_map=device.type,
        low_cpu_mem_usage=True, use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    model.eval()
    model.initialize_special_tokenizer(tokenizer, device=str(device))
    latent_token_len = model.get_model().config.latent_token_len
    stop_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f"模型加载耗时: {time.time() - t0:.1f}s  |  latent_token_len: {latent_token_len}")

    work_by_ds = {}
    for ds_name, idx, text, max_new in all_work:
        work_by_ds.setdefault(ds_name, []).append((idx, text, max_new))

    all_dataset_results = {}
    all_sample_details = {}

    for ds_name in ds_sample_counts:
        if ds_name not in work_by_ds:
            continue
        items = work_by_ds[ds_name]
        ds_cfg = DATASET_CONFIGS[ds_name]
        print(f"\n{'─' * 55}")
        print(f"评估: {ds_name}  ({ds_cfg['description']})  样本数: {len(items)}")

        results_for_ds = []
        for idx, text, max_new_tokens in tqdm(items, desc=ds_name, ncols=80):
            t_start = time.time()
            try:
                generated = reconstruct_text(
                    model, tokenizer, text, latent_token_len,
                    device, max_new_tokens, stop_token_id,
                )
            except Exception as e:
                print(f"\n  样本 {idx} 生成失败: {e}")
                continue

            elapsed = time.time() - t_start
            metrics = evaluate_sample(text, generated, tokenizer)
            metrics["elapsed_s"] = round(elapsed, 2)
            results_for_ds.append(metrics)

            if save_samples:
                all_sample_details.setdefault(ds_name, []).append({
                    "reference": text[:2000],
                    "generated": generated[:2000],
                    "metrics": metrics,
                })

            if (idx + 1) % 5 == 0 or idx == 0:
                avg_cp = sum(m["char_precision"] for m in results_for_ds) / len(results_for_ds)
                avg_r = sum(m["rouge_l"] for m in results_for_ds) / len(results_for_ds)
                tqdm.write(
                    f"  [{idx+1}/{len(items)}] "
                    f"CharPrec={avg_cp:.4f}  ROUGE-L={avg_r:.4f}"
                )

        if not results_for_ds:
            continue

        avg = _aggregate_dataset(results_for_ds)
        all_dataset_results[ds_name] = avg
        _print_dataset_result(ds_name, avg)

        output_data = {
            "model_name": model_name,
            "model_path": model_path,
            "device": str(device),
            "dtype": str(dtype),
            "datasets": all_dataset_results,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  → 已更新 {output}")

    if save_samples and all_sample_details:
        samples_path = output.replace(".json", "_samples.json")
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(all_sample_details, f, indent=2, ensure_ascii=False)
        print(f"逐条详情已保存到: {samples_path}")

    _print_summary(model_name, all_dataset_results)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
