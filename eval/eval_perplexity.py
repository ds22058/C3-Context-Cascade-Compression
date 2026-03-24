"""
在 WikiText-2 测试集上计算单个模型的 perplexity。
使用滑动窗口法（stride=512, max_length=2048）。

用法：
    # 单卡
    python eval/eval_perplexity.py --model_path ./models/qwen25-3b --model_name qwen25-3b

    # 多卡并行（窗口分片）
    python eval/eval_perplexity.py --model_path ./models/qwen25-3b --model_name qwen25-3b --num_gpus 4
"""

import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_windows(seq_len, max_length, stride):
    """预计算所有滑动窗口的 (begin, end, trg_len) 列表。"""
    windows = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        windows.append((begin, end, trg_len))
        prev_end = end
        if end == seq_len:
            break
    return windows


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: list,
    device: torch.device,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """滑动窗口 perplexity（标准做法，与 HuggingFace 官方示例一致）"""
    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"    总 token 数: {seq_len:,}")

    windows = _build_windows(seq_len, max_length, stride)
    nlls = []
    pbar = tqdm(windows, desc="    滑动窗口", unit="window", ncols=80)

    for begin, end, trg_len in pbar:
        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.float().cpu())

    total_len = windows[-1][1]
    ppl = torch.exp(torch.stack(nlls).sum() / total_len).item()
    return ppl


def _gpu_worker(rank, model_path, dtype, input_ids, my_windows, result_queue):
    """多卡 worker：加载模型到指定 GPU，处理分配的窗口。"""
    device = torch.device(f"cuda:{rank}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True,
    )
    model.eval()

    for idx, begin, end, trg_len in my_windows:
        ids = input_ids[:, begin:end].to(device)
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(ids, labels=targets).loss
            nll = (loss * trg_len).float().cpu()
        result_queue.put((idx, nll))

    result_queue.put(None)
    del model
    torch.cuda.empty_cache()


def compute_perplexity_multi_gpu(
    model_path: str,
    tokenizer,
    texts: list,
    num_gpus: int,
    dtype: torch.dtype,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """多卡并行滑动窗口 PPL：按窗口分片到多 GPU。"""
    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"    总 token 数: {seq_len:,}  |  {num_gpus} 卡并行")

    windows = _build_windows(seq_len, max_length, stride)
    indexed_windows = [(i, b, e, t) for i, (b, e, t) in enumerate(windows)]
    print(f"    窗口数: {len(windows)}, 每卡 ~{len(windows) // num_gpus} 个")

    shards = [[] for _ in range(num_gpus)]
    for i, w in enumerate(indexed_windows):
        shards[i % num_gpus].append(w)

    input_ids = encodings.input_ids.share_memory_()

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(rank, model_path, dtype, input_ids, shards[rank], result_queue),
        )
        p.start()
        processes.append(p)

    nlls = [None] * len(windows)
    done = 0
    with tqdm(total=len(windows), desc="    多卡滑动窗口", unit="window", ncols=80) as pbar:
        while done < num_gpus:
            item = result_queue.get()
            if item is None:
                done += 1
                continue
            idx, nll = item
            nlls[idx] = nll
            pbar.update(1)

    for p in processes:
        p.join()

    total_len = windows[-1][1]
    ppl = torch.exp(torch.stack(nlls).sum() / total_len).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="WikiText-2 Perplexity 评估（单模型）")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument(
        "--model_name", default=None,
        help="模型名称（用于输出标注，默认取目录名）",
    )
    parser.add_argument("--output", default=None, help="结果路径（默认 eval/results/<model_name>/perplexity.json）")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_gpus", default="1",
                        help="并行GPU数: 数字 或 'auto'自动检测 (默认1)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多使用前 N 条文本（CPU 快速验证用）")
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    output = args.output or f"./eval/results/{model_name}/perplexity.json"

    use_cuda = (args.device == "cuda") or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        device = torch.device("cuda")
        avail = torch.cuda.device_count()
        if args.num_gpus == "auto":
            num_gpus = avail
        else:
            num_gpus = max(1, min(int(args.num_gpus), avail))
    else:
        device = torch.device("cpu")
        num_gpus = 0

    if args.dtype == "auto":
        dtype = torch.bfloat16 if use_cuda else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    device_str = f"cuda x{num_gpus}" if num_gpus > 1 else str(device)
    print(f"模型: {model_name}")
    print(f"路径: {args.model_path}")
    print(f"设备: {device_str}  精度: {dtype}")

    if not os.path.isdir(args.model_path):
        print(f"错误：模型路径不存在: {args.model_path}")
        raise SystemExit(1)

    # 加载数据集
    print("\n加载 WikiText-2 测试集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if t.strip()]
    if args.max_samples is not None:
        texts = texts[: args.max_samples]
        print(f"  使用前 {args.max_samples} 条文本")
    else:
        print(f"  共 {len(texts)} 条有效文本")

    t1 = time.time()

    if num_gpus > 1:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        ppl = compute_perplexity_multi_gpu(
            args.model_path, tokenizer, texts, num_gpus, dtype,
            args.max_length, args.stride,
        )
    else:
        # 单卡 / CPU
        print(f"\n加载模型: {model_name}")
        t0 = time.time()
        if device.type == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True,
            )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"  加载耗时: {time.time() - t0:.1f}s")
        ppl = compute_perplexity(model, tokenizer, texts, device, args.max_length, args.stride)
        del model

    elapsed = time.time() - t1
    print(f"\n  Perplexity = {ppl:.4f}  (计算耗时: {elapsed:.1f}s)")

    # 保存结果
    result = {
        "model_name": model_name,
        "model_path": args.model_path,
        "perplexity": ppl,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "stride": args.stride,
        "device": device_str,
        "dtype": str(dtype),
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存到: {output}")

    if use_cuda:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
