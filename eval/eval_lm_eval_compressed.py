"""
Context-Compressed lm_eval：prompt 部分经过 C3 encoder 压缩后与剩余 token 拼接，
在 decoder 中预测下一个 token，模拟 context 压缩场景下的 NLP benchmark 评估。

压缩策略（由 --compress_threshold 控制）：
  · prompt tokens <= threshold: 全部过 encoder → latent_token_len latent tokens → decoder
  · prompt tokens >  threshold: 前 threshold 个 token 过 encoder → latent_token_len latent tokens,
                                 剩余 token 直接拼在 latent tokens 后面 → decoder
  注意：compress_threshold 应 > latent_token_len，否则压缩后反而变长。

需要 C3 完整模型（含 llm1 encoder），若无 encoder 权重则无法运行。
支持多 GPU 并行（每卡加载独立模型，请求 round-robin 分发）。

用法：
    # 单卡
    python eval/eval_lm_eval_compressed.py \
        --model_path ./output/phase2 \
        --compress_threshold 256 \
        --tasks hellaswag,arc_easy,arc_challenge,winogrande

    # 8 卡并行
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval/eval_lm_eval_compressed.py \
        --model_path ./output/phase2 \
        --compress_threshold 256 \
        --num_gpus 8 \
        --tasks hellaswag,arc_easy,arc_challenge,winogrande
"""

import argparse
import json
import os
import sys
import time
import threading

_eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _eval_dir)
sys.path.insert(0, os.path.join(_eval_dir, ".."))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Qwen2Model

import lm_eval.api.model
import lm_eval.api.registry
from lm_eval import evaluator, tasks as lm_tasks

from train.config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


@lm_eval.api.registry.register_model("c3_compressed")
class C3CompressedLM(lm_eval.api.model.LM):
    """lm_eval model wrapper: C3 encoder 压缩 context 后由 decoder 预测。
    支持 batched 多 GPU 并行推理。
    """

    def __init__(
        self,
        model_path: str,
        compress_threshold: int = 256,
        num_gpus: int = 1,
        batch_size: int = 32,
    ):
        super().__init__()
        self._batch_size = batch_size
        self.threshold = compress_threshold
        self.num_gpus = num_gpus
        self.model_path = model_path

        print(f"加载 C3 模型: {model_path}")
        print(f"压缩阈值: {compress_threshold} tokens")
        print(f"GPU 数量: {num_gpus}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._pad_id = self.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16

        self._models = []
        self._devices = []
        for rank in range(num_gpus):
            device = torch.device(f"cuda:{rank}")
            print(f"  加载模型到 GPU {rank}...", flush=True)
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=dtype, device_map={"": device},
                low_cpu_mem_usage=True, use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            model.eval()
            model.initialize_special_tokenizer(self.tokenizer, device=str(device))
            self._models.append(model)
            self._devices.append(device)

        config = self._models[0].get_model().config
        self.latent_len = config.latent_token_len
        self.im_patch_id = config.im_patch_token
        self.im_start_id = config.im_start_token
        self.im_end_id = config.im_end_token

        self._placeholder_ids = (
            [self.im_start_id]
            + [self.im_patch_id] * self.latent_len
            + [self.im_end_id]
        )

        self._device = self._devices[0]
        self.model = self._models[0]

        print(f"latent_token_len: {self.latent_len}  |  {num_gpus} 卡就绪  |  batch_size: {batch_size}")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._models[0].config, "max_position_embeddings", 32768)

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        add_special = add_special_tokens if add_special_tokens is not None else False
        ids = self.tokenizer.encode(string, add_special_tokens=add_special)
        if left_truncate_len:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    # ── 预处理 ──────────────────────────────────────────────────

    def _preprocess(self, ctx_str, cont_str):
        """Pre-tokenize and classify a single (context, continuation) pair."""
        full_ids = self.tokenizer.encode(ctx_str + cont_str, add_special_tokens=False)
        ctx_ids = self.tokenizer.encode(ctx_str, add_special_tokens=False)
        cont_ids = full_ids[len(ctx_ids):]

        if len(cont_ids) == 0:
            return {"type": "skip"}

        if len(ctx_ids) == 0:
            return {
                "type": "no_ctx",
                "full_ids": full_ids,
                "cont_ids": cont_ids,
            }

        if len(ctx_ids) <= self.threshold:
            compressed, remaining = ctx_ids, []
        else:
            compressed, remaining = ctx_ids[: self.threshold], ctx_ids[self.threshold :]

        context_input = compressed + self._placeholder_ids
        decoder_input = self._placeholder_ids + remaining + cont_ids
        cont_start = len(self._placeholder_ids) + len(remaining)

        return {
            "type": "compressed",
            "context_ids": context_input,
            "decoder_ids": decoder_input,
            "cont_ids": cont_ids,
            "cont_start": cont_start,
            "sort_key": len(decoder_input),
        }

    # ── Batched forward ─────────────────────────────────────────

    def _run_batch(self, rank, items):
        """Forward a batch of 'compressed' items on one GPU.
        Returns list of (total_logprob, is_greedy).
        """
        model = self._models[rank]
        device = self._devices[rank]
        pad = self._pad_id

        max_ctx = max(len(it["context_ids"]) for it in items)
        max_dec = max(len(it["decoder_ids"]) for it in items)

        ctx_batch, dec_batch = [], []
        ctx_masks, dec_masks = [], []

        for it in items:
            c, d = it["context_ids"], it["decoder_ids"]
            ctx_batch.append(c + [pad] * (max_ctx - len(c)))
            dec_batch.append(d + [pad] * (max_dec - len(d)))
            ctx_masks.append([1] * len(c) + [0] * (max_ctx - len(c)))
            dec_masks.append([1] * len(d) + [0] * (max_dec - len(d)))

        ctx_t = torch.tensor(ctx_batch, device=device)
        dec_t = torch.tensor(dec_batch, device=device)
        ctx_m = torch.tensor(ctx_masks, device=device)
        dec_m = torch.tensor(dec_masks, device=device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=dec_t,
                context_ids=ctx_t,
                attention_mask=dec_m,
                context_attention_mask=ctx_m,
            ).logits  # [B, max_dec, vocab]

        results = []
        for j, it in enumerate(items):
            n = len(it["cont_ids"])
            cs = it["cont_start"]
            positions = torch.arange(n, device=device) + cs - 1
            positions.clamp_(min=0)

            rel = logits[j, positions].float()
            cids = torch.tensor(it["cont_ids"], device=device)
            lps = F.log_softmax(rel, dim=-1)
            total = lps[torch.arange(n, device=device), cids].sum().item()
            greedy = bool((rel.argmax(dim=-1) == cids).all().item())
            results.append((total, greedy))

        return results

    def _run_single_no_ctx(self, rank, item):
        """Handle the rare case where context is empty (bypass C3 encoder)."""
        model = self._models[rank]
        device = self._devices[rank]

        ids_t = torch.tensor([item["full_ids"]], device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model.get_model().embed_tokens(ids_t)
            base_out = Qwen2Model.forward(model.get_model(), inputs_embeds=embeds)
            logits = model.lm_head(base_out.last_hidden_state)[0]

        cont_ids = item["cont_ids"]
        n = len(cont_ids)
        positions = torch.arange(n, device=device) - 1
        positions.clamp_(min=0)

        rel = logits[positions].float()
        cids = torch.tensor(cont_ids, device=device)
        lps = F.log_softmax(rel, dim=-1)
        total = lps[torch.arange(n, device=device), cids].sum().item()
        greedy = bool((rel.argmax(dim=-1) == cids).all().item())
        return total, greedy

    # ── loglikelihood (main entry point) ────────────────────────

    def loglikelihood(self, requests):
        total_reqs = len(requests)

        # 1. Pre-tokenize all requests
        prep = []
        for req in requests:
            ctx_str, cont_str = req.args
            prep.append(self._preprocess(ctx_str, cont_str))

        results = [None] * total_reqs

        # 2. Handle skip / no_ctx; collect compressed items
        compute_indices = []
        for i, p in enumerate(prep):
            if p["type"] == "skip":
                results[i] = (0.0, True)
            elif p["type"] == "no_ctx":
                results[i] = self._run_single_no_ctx(0, p)
            else:
                compute_indices.append(i)

        if not compute_indices:
            return results

        # 3. Sort by decoder length to minimize padding waste
        compute_indices.sort(key=lambda i: prep[i]["sort_key"])

        # 4. Create batches
        bs = self._batch_size
        batches = []
        for start in range(0, len(compute_indices), bs):
            batches.append(compute_indices[start : start + bs])

        n_compute = len(compute_indices)
        counter = [0]
        lock = threading.Lock()
        t0 = time.time()

        def gpu_worker(rank, my_batches):
            for batch_indices in my_batches:
                items = [prep[i] for i in batch_indices]
                batch_results = self._run_batch(rank, items)
                for idx, res in zip(batch_indices, batch_results):
                    results[idx] = res
                with lock:
                    prev = counter[0]
                    counter[0] += len(batch_indices)
                    done = counter[0]
                    if done >= n_compute or prev // 2000 < done // 2000:
                        elapsed = time.time() - t0
                        speed = done / elapsed if elapsed > 0 else 0
                        eta = (n_compute - done) / speed if speed > 0 else 0
                        print(
                            f"  compressed loglikelihood: {done}/{n_compute}"
                            f"  ({speed:.0f} it/s, ETA {eta:.0f}s)"
                            f"  [{self.num_gpus} GPUs, bs={bs}]",
                            flush=True,
                        )

        if self.num_gpus <= 1:
            gpu_worker(0, batches)
        else:
            gpu_batches = [[] for _ in range(self.num_gpus)]
            for i, batch in enumerate(batches):
                gpu_batches[i % self.num_gpus].append(batch)

            threads = []
            for rank in range(self.num_gpus):
                t = threading.Thread(
                    target=gpu_worker,
                    args=(rank, gpu_batches[rank]),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        elapsed = time.time() - t0
        print(
            f"  compressed loglikelihood 完成: {n_compute} 请求, "
            f"{elapsed:.1f}s ({n_compute / elapsed:.0f} it/s, {self.num_gpus} GPUs)",
            flush=True,
        )
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "Rolling loglikelihood 不支持 compressed 模式，请使用 eval_perplexity.py"
        )

    def generate_until(self, requests):
        raise NotImplementedError(
            "generate_until 暂不支持 compressed 模式"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Context-Compressed lm_eval: prompt 过 C3 encoder 压缩后评估 NLP benchmark"
    )
    ap.add_argument("--model_path", required=True, help="C3 完整模型路径（需含 llm1/）")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--compress_threshold", type=int, default=256,
                    help="压缩阈值 (默认 256)")
    ap.add_argument("--tasks", required=True,
                    help="lm_eval 任务，逗号分隔 (如 mmlu,hellaswag)")
    ap.add_argument("--num_gpus", default="1",
                    help="并行 GPU 数: 数字 或 'auto' (默认 1)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="每 GPU batch 大小 (默认 32)")
    ap.add_argument("--num_fewshot", type=int, default=None,
                    help="Few-shot 数量 (默认由任务决定)")
    ap.add_argument("--output_path", default=None,
                    help="结果输出目录 (默认 eval/results/<model_name>/lm_eval_compressed)")
    args = ap.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    output_path = args.output_path or f"./eval/results/{model_name}/lm_eval_compressed"

    if not os.path.isdir(os.path.join(args.model_path, "llm1")):
        print("错误: 模型路径中未找到 llm1/ 目录，此评估需要 C3 完整模型（含 encoder）")
        raise SystemExit(1)

    if args.num_gpus == "auto":
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = int(args.num_gpus)
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count()))

    t0 = time.time()

    lm = C3CompressedLM(
        model_path=args.model_path,
        compress_threshold=args.compress_threshold,
        num_gpus=num_gpus,
        batch_size=args.batch_size,
    )

    task_list = [t.strip() for t in args.tasks.split(",")]
    task_manager = lm_tasks.TaskManager()

    print(f"\n运行 lm_eval: {task_list}")
    print(f"compress_threshold: {args.compress_threshold}  |  GPU: {num_gpus}  |  batch_size: {args.batch_size}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        task_manager=task_manager,
    )

    elapsed = time.time() - t0

    os.makedirs(output_path, exist_ok=True)
    results_data = results.get("results", {})

    meta = {
        "model_name": model_name,
        "model_path": args.model_path,
        "compress_threshold": args.compress_threshold,
        "mode": "context_compressed",
        "num_gpus": num_gpus,
        "batch_size": args.batch_size,
        "elapsed_s": round(elapsed, 1),
    }

    output_file = os.path.join(
        output_path,
        f"results_compressed_t{args.compress_threshold}.json",
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results_data}, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Context-Compressed lm_eval 结果  |  threshold={args.compress_threshold}")
    print(f"{'=' * 60}")
    for task_name, task_results in results_data.items():
        metrics_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in task_results.items()
            if not k.endswith("_stderr") and k != "alias"
        )
        print(f"  {task_name}: {metrics_str}")
    print(f"\n  耗时: {elapsed:.0f}s  |  GPU: {num_gpus}  |  batch_size: {args.batch_size}")
    print(f"  结果: {output_file}")


if __name__ == "__main__":
    main()
