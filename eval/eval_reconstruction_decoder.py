"""
Decoder-only 重建评估：不经过 encoder 压缩，直接将完整文本放入 prompt 让模型复述。

与 eval_reconstruction.py（encoder 压缩→重建）对比，衡量压缩带来的信息损失。
任何 CausalLM 模型都可以测试（基线模型、提取的 decoder 等），不需要 C3 encoder。

用法：
    # 单卡
    python eval/eval_reconstruction_decoder.py \
        --model_path ./models/qwen25-3b --model_name qwen25-3b --device cuda

    # 多卡并行
    python eval/eval_reconstruction_decoder.py \
        --model_path ./output/phase2_decoder_extracted --model_name c3-phase2 --num_gpus 4
"""

import argparse
import json
import os
import sys
import time

_eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _eval_dir)
sys.path.insert(0, os.path.join(_eval_dir, ".."))

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.config import (
    RECONSTRUCT_PROMPT,
    SYSTEM_MESSAGE,
    ROLE_USER,
    ROLE_ASSISTANT,
    SEP_TOKEN,
)
from eval_reconstruction import (
    compute_rouge_l,
    compute_token_f1,
    evaluate_sample,
    load_samples,
    DATASET_CONFIGS,
    ALL_DATASETS,
    _aggregate_dataset,
    _print_dataset_result,
    _print_summary,
)


def build_decoder_prompt(text: str) -> str:
    """构建 decoder-only 重建 prompt：完整文本 + "Repeat the text: "。"""
    user_content = text + "\n" + RECONSTRUCT_PROMPT
    return (
        SYSTEM_MESSAGE + SEP_TOKEN
        + ROLE_USER + user_content + SEP_TOKEN
        + ROLE_ASSISTANT
    )


def reconstruct_text_decoder(model, tokenizer, text, device,
                             max_new_tokens, stop_token_id):
    prompt = build_decoder_prompt(text)
    input_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    do_sample=False, num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=stop_token_id,
                )
        else:
            output_ids = model.generate(
                input_ids,
                do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_id,
            )

    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ── Multi-GPU worker ───────────────────────────────────────────


def _gpu_worker(rank, model_path, dtype, work_items, result_queue, save_samples):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map={"": device},
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    stop_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    print(f"  [GPU {rank}] 模型加载完成，{len(work_items)} 个样本待处理", flush=True)

    for ds_name, sample_idx, text, max_new_tokens in work_items:
        t_start = time.time()
        try:
            generated = reconstruct_text_decoder(
                model, tokenizer, text, device, max_new_tokens, stop_token_id,
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


# ── Runners ────────────────────────────────────────────────────


def _run_multi_gpu(num_gpus, model_path, dtype, all_work,
                   ds_sample_counts, save_samples, model_name, output):
    total = len(all_work)
    print(f"\n多GPU并行: {num_gpus} 卡，共 {total} 个样本（decoder-only）")

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
            args=(rank, model_path, dtype, shards[rank], result_queue, save_samples),
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
            details_by_ds.setdefault(ds_name, []).append({**detail, "metrics": metrics})
        processed += 1
        if processed % 10 == 0 or processed == total:
            elapsed = time.time() - t0
            speed = processed / elapsed if elapsed > 0 else 0
            print(f"  进度: {processed}/{total}  ({elapsed:.0f}s, {speed:.1f} samples/s)", flush=True)

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
        "mode": "decoder_only",
        "device": f"cuda x{num_gpus}",
        "dtype": str(dtype),
        "wall_time_s": round(wall_time, 1),
        "datasets": all_dataset_results,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  → 结果已保存到 {output}")

    if save_samples and details_by_ds:
        sp = output.replace(".json", "_samples.json")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(details_by_ds, f, indent=2, ensure_ascii=False)

    _print_summary(f"{model_name} (decoder-only)", all_dataset_results)


def _run_single(device, dtype, model_path, tokenizer, all_work,
                ds_sample_counts, save_samples, model_name, output):
    print(f"\n加载模型 (decoder-only)...")
    t0 = time.time()
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": device},
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
    model.eval()
    stop_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f"模型加载耗时: {time.time() - t0:.1f}s")

    work_by_ds = {}
    for ds_name, idx, text, max_new in all_work:
        work_by_ds.setdefault(ds_name, []).append((idx, text, max_new))

    all_dataset_results = {}

    for ds_name in ds_sample_counts:
        if ds_name not in work_by_ds:
            continue
        items = work_by_ds[ds_name]
        ds_cfg = DATASET_CONFIGS[ds_name]
        print(f"\n{'─' * 55}")
        print(f"评估 (decoder-only): {ds_name}  ({ds_cfg['description']})  样本数: {len(items)}")

        results_for_ds = []
        for idx, text, max_new_tokens in tqdm(items, desc=ds_name, ncols=80):
            t_start = time.time()
            try:
                generated = reconstruct_text_decoder(
                    model, tokenizer, text, device, max_new_tokens, stop_token_id,
                )
            except Exception as e:
                print(f"\n  样本 {idx} 失败: {e}")
                continue
            elapsed = time.time() - t_start
            metrics = evaluate_sample(text, generated, tokenizer)
            metrics["elapsed_s"] = round(elapsed, 2)
            results_for_ds.append(metrics)

            if (idx + 1) % 5 == 0 or idx == 0:
                avg_cp = sum(m["char_precision"] for m in results_for_ds) / len(results_for_ds)
                avg_r = sum(m["rouge_l"] for m in results_for_ds) / len(results_for_ds)
                tqdm.write(f"  [{idx+1}/{len(items)}] CharPrec={avg_cp:.4f}  ROUGE-L={avg_r:.4f}")

        if not results_for_ds:
            continue

        avg = _aggregate_dataset(results_for_ds)
        all_dataset_results[ds_name] = avg
        _print_dataset_result(ds_name, avg)

        output_data = {
            "model_name": model_name,
            "model_path": model_path,
            "mode": "decoder_only",
            "device": str(device),
            "dtype": str(dtype),
            "datasets": all_dataset_results,
        }
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    _print_summary(f"{model_name} (decoder-only)", all_dataset_results)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Main ───────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Decoder-only 重建评估（不经过 encoder，直接用 decoder 复述）"
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--datasets", default=ALL_DATASETS,
                    help=f"逗号分隔 (可选: {ALL_DATASETS})")
    ap.add_argument("--samples_per_dataset", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num_gpus", default="1",
                    help="并行GPU数: 数字 或 'auto' (默认1)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--save_samples", action="store_true")
    args = ap.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    output = args.output or f"./eval/results/{model_name}/reconstruction_decoder.json"

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    for d in dataset_names:
        if d not in DATASET_CONFIGS:
            print(f"未知数据集: {d}"); raise SystemExit(1)

    use_cuda = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        avail = torch.cuda.device_count()
        num_gpus = avail if args.num_gpus == "auto" else max(1, min(int(args.num_gpus), avail))
    else:
        num_gpus = 0

    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_str = f"cuda x{num_gpus}" if num_gpus > 1 else ("cuda" if use_cuda else "cpu")

    print(f"模型: {model_name}  |  路径: {args.model_path}")
    print(f"模式: decoder-only (不经过 encoder)")
    print(f"设备: {device_str}  |  精度: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    all_work = []
    ds_sample_counts = {}
    for ds_name in dataset_names:
        ds_cfg = DATASET_CONFIGS[ds_name]
        print(f"\n{'─' * 55}")
        print(f"数据集: {ds_name}  ({ds_cfg['description']})")
        samples = load_samples(ds_name, tokenizer, args.samples_per_dataset)
        if not samples:
            continue
        print(f"  样本数: {len(samples)}")
        ds_sample_counts[ds_name] = len(samples)
        context_max = ds_cfg.get("target_tokens", ds_cfg.get("max_tokens", 512))
        max_new_tokens = min(int(context_max * 1.3), 8192)
        for idx, text in enumerate(samples):
            all_work.append((ds_name, idx, text, max_new_tokens))

    if not all_work:
        print("没有可评估的样本"); return

    os.makedirs(os.path.dirname(output), exist_ok=True)

    if num_gpus > 1:
        _run_multi_gpu(
            num_gpus, args.model_path, dtype, all_work,
            ds_sample_counts, args.save_samples, model_name, output,
        )
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        _run_single(
            device, dtype, args.model_path, tokenizer, all_work,
            ds_sample_counts, args.save_samples, model_name, output,
        )


if __name__ == "__main__":
    main()
