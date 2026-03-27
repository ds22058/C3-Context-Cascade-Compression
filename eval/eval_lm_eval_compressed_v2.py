"""
Context-Compressed Benchmark v2

两种评测模式：
  logprob  — MMLU 等多选题，计算 continuation log-prob 选最优
  generate — HotpotQA 等生成式 QA，greedy decode 生成答案，EM/F1 评分

多段压缩 (C3 模型):
  num_segments=1: context → encoder → 32 latent
  num_segments=2: context 对半分，各过 encoder → 64 latent
  num_segments=3: context 三等分，各过 encoder → 96 latent
  decoder 输入: [latent_1][latent_2]...[latent_N] + stem_text + continuation/generate

用法：
    # C3
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/eval_lm_eval_compressed_v2.py \\
        --model_path ./output/phase2_v2 --model_name c3-p2-v2 --num_gpus 4

    # Baseline
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/eval_lm_eval_compressed_v2.py \\
        --model_path ./models/qwen25-3b --model_name qwen25-3b --baseline --num_gpus 4
"""

import argparse
import json
import os
import re
import string
import sys
import time
import threading

_eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _eval_dir)
sys.path.insert(0, os.path.join(_eval_dir, ".."))

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

DATA_DIR = os.path.join(_eval_dir, "benchmark_data", "processed")
ALL_TASKS = ["mmlu", "mmlu_cstem", "hotpotqa_short", "hotpotqa_fewshot", "hotpotqa_inst"]

HOTPOTQA_FEWSHOT_PREFIX = (
    "Answer each question with a short phrase.\n\n"
    "Question: What is the capital of France?\nAnswer: Paris\n\n"
    "Question: Who directed the movie Jaws?\nAnswer: Steven Spielberg\n\n"
)


# ════════════════════════════════════════════════════════════════
# EM / F1
# ════════════════════════════════════════════════════════════════

def _normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_em_f1(pred, gold):
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    em = int(pred_n == gold_n)

    pred_toks = pred_n.split()
    gold_toks = gold_n.split()
    if not pred_toks or not gold_toks:
        return em, float(pred_toks == gold_toks)

    from collections import Counter
    common = Counter(pred_toks) & Counter(gold_toks)
    num = sum(common.values())
    if num == 0:
        return em, 0.0
    prec = num / len(pred_toks)
    rec = num / len(gold_toks)
    f1 = 2 * prec * rec / (prec + rec)
    return em, f1


# ════════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════════

def load_benchmark(task_name, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{task_name}.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ════════════════════════════════════════════════════════════════
# 聚合
# ════════════════════════════════════════════════════════════════

def aggregate_logprob(results, samples):
    scores = {}
    for si, ci, lp, n_tok in results:
        scores.setdefault(si, []).append((ci, lp, n_tok))
    correct, correct_norm, total = 0, 0, 0
    for si, s in enumerate(samples):
        if si not in scores:
            continue
        conts = sorted(scores[si], key=lambda x: x[0])
        pred = max(conts, key=lambda x: x[1])[0]
        pred_norm = max(conts, key=lambda x: x[1] / max(x[2], 1))[0]
        ans = s["answer_idx"]
        if pred == ans:
            correct += 1
        if pred_norm == ans:
            correct_norm += 1
        total += 1
    return {
        "acc": correct / total if total else 0,
        "acc_norm": correct_norm / total if total else 0,
        "total": total, "correct": correct, "correct_norm": correct_norm,
    }


def aggregate_generate(results, samples):
    em_sum, f1_sum, total = 0.0, 0.0, 0
    for si, pred_text in results:
        gold = samples[si]["answer_text"]
        em, f1 = compute_em_f1(pred_text, gold)
        em_sum += em
        f1_sum += f1
        total += 1
    return {
        "em": em_sum / total if total else 0,
        "f1": f1_sum / total if total else 0,
        "total": total,
    }


# ════════════════════════════════════════════════════════════════
# C3 压缩评估器
# ════════════════════════════════════════════════════════════════

class C3Evaluator:

    def __init__(self, model_path, num_gpus=1, batch_size=32):
        from train.config import (
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._eos_id = self.tokenizer.eos_token_id
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self._models, self._devices = [], []
        for rank in range(num_gpus):
            dev = torch.device(f"cuda:{rank}")
            print(f"  加载 C3 模型到 GPU {rank}...", flush=True)
            m = AutoModel.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map={"": dev},
                low_cpu_mem_usage=True, use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            m.eval()
            m.initialize_special_tokenizer(self.tokenizer, device=str(dev))
            self._models.append(m)
            self._devices.append(dev)

        cfg = self._models[0].get_model().config
        self.latent_len = cfg.latent_token_len
        self._placeholder = (
            [cfg.im_start_token]
            + [cfg.im_patch_token] * self.latent_len
            + [cfg.im_end_token]
        )
        self._newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        print(f"  latent_len={self.latent_len}  |  {num_gpus} GPUs  |  bs={batch_size}")

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ── 多段 encoder ──────────────────────────────────────────

    def _encode_segment(self, model, device, seg_ids):
        """单段 context 过 encoder，返回 projected latent [latent_len, hidden]。"""
        inner = model.get_model()
        ctx_input = seg_ids + list(self._placeholder)
        ctx_t = torch.tensor([ctx_input], device=device)
        ctx_mask = torch.ones(1, len(ctx_input), device=device, dtype=torch.long)

        ctx_embeds = inner.llm1.model.embed_tokens(ctx_t)
        q_start = len(seg_ids) + 1
        Q = inner.Q.weight.to(device=device, dtype=ctx_embeds.dtype)
        ctx_embeds = ctx_embeds.clone()
        ctx_embeds[0, q_start:q_start + self.latent_len] = Q

        enc_out = inner.llm1.forward(
            input_ids=None, attention_mask=ctx_mask,
            inputs_embeds=ctx_embeds, output_hidden_states=True,
            return_dict=True, use_cache=False,
        )
        hidden = enc_out["hidden_states"][-1]
        latent = hidden[0, q_start:q_start + self.latent_len]
        return inner.mm_projector(latent)

    def _encode_multi(self, model, device, ctx_ids, num_segments):
        """将 ctx_ids 等分为 num_segments 段，各过 encoder，返回 latent list。"""
        seg_len = len(ctx_ids) // num_segments
        latents = []
        for i in range(num_segments):
            start = i * seg_len
            end = start + seg_len if i < num_segments - 1 else len(ctx_ids)
            latent = self._encode_segment(model, device, ctx_ids[start:end])
            latents.append(latent)
        return latents

    def _build_decoder_embeds(self, model, device, latents, suffix_ids):
        """构建 decoder input_embeds: [latent_1][latent_2]...[suffix]。"""
        inner = model.get_model()
        parts = []
        for latent in latents:
            start_emb = inner.embed_tokens(torch.tensor([[self._placeholder[0]]], device=device))
            end_emb = inner.embed_tokens(torch.tensor([[self._placeholder[-1]]], device=device))
            parts.extend([start_emb, latent.unsqueeze(0), end_emb])
        if suffix_ids:
            suffix_emb = inner.embed_tokens(torch.tensor([suffix_ids], device=device))
            parts.append(suffix_emb)
        return torch.cat(parts, dim=1)

    def _greedy_decode(self, model, device, input_embeds, max_new_tokens=50):
        inner = model.get_model()
        past_kv = None
        generated = []
        cur_embeds = input_embeds
        attn_len = input_embeds.shape[1]

        for _ in range(max_new_tokens):
            attn_mask = torch.ones(1, attn_len, device=device, dtype=torch.long)
            out = Qwen2Model.forward(
                inner,
                input_ids=None, inputs_embeds=cur_embeds,
                attention_mask=attn_mask, past_key_values=past_kv,
                use_cache=True, return_dict=True,
            )
            past_kv = out.past_key_values
            logits = model.lm_head(out.last_hidden_state[:, -1:]).float()
            next_id = logits.argmax(dim=-1).squeeze().item()
            if next_id == self._eos_id:
                break
            generated.append(next_id)
            cur_embeds = inner.embed_tokens(torch.tensor([[next_id]], device=device))
            attn_len += 1

        return generated

    # ── logprob 评测 (MMLU) ────────────────────────────────────

    def build_logprob_requests(self, samples):
        requests = []
        for si, s in enumerate(samples):
            ctx_ids = self._encode(s["context_to_compress"])
            context_input = ctx_ids + list(self._placeholder)
            stem_text = s.get("stem_text", "")
            opts_text = s.get("options_text", "")
            prefix_str = stem_text + opts_text

            for ci, cont_str in enumerate(s["continuations"]):
                suffix_str = prefix_str + cont_str
                suffix_ids = self._encode(suffix_str)
                prefix_ids = self._encode(prefix_str)
                cont_ids = suffix_ids[len(prefix_ids):]
                if not cont_ids:
                    cont_ids = self._encode(cont_str)
                decoder_input = list(self._placeholder) + suffix_ids
                cont_start = len(self._placeholder) + len(prefix_ids)
                requests.append({
                    "context_ids": context_input, "decoder_ids": decoder_input,
                    "cont_ids": cont_ids, "cont_start": cont_start,
                    "sample_idx": si, "cont_idx": ci,
                })
        return requests

    def score_batch(self, rank, items):
        model = self._models[rank]
        device = self._devices[rank]
        pad = self._pad_id
        max_ctx = max(len(it["context_ids"]) for it in items)
        max_dec = max(len(it["decoder_ids"]) for it in items)

        ctx_b, dec_b, ctx_m, dec_m = [], [], [], []
        for it in items:
            c, d = it["context_ids"], it["decoder_ids"]
            ctx_b.append(c + [pad] * (max_ctx - len(c)))
            dec_b.append(d + [pad] * (max_dec - len(d)))
            ctx_m.append([1]*len(c) + [0]*(max_ctx - len(c)))
            dec_m.append([1]*len(d) + [0]*(max_dec - len(d)))

        ctx_t = torch.tensor(ctx_b, device=device)
        dec_t = torch.tensor(dec_b, device=device)
        cm = torch.tensor(ctx_m, device=device)
        dm = torch.tensor(dec_m, device=device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=dec_t, context_ids=ctx_t,
                attention_mask=dm, context_attention_mask=cm,
            ).logits

        results = []
        for j, it in enumerate(items):
            n = len(it["cont_ids"])
            cs = it["cont_start"]
            positions = torch.arange(n, device=device) + cs - 1
            positions.clamp_(min=0)
            rel = logits[j, positions].float()
            cids = torch.tensor(it["cont_ids"], device=device)
            lps = F.log_softmax(rel, dim=-1)
            total_lp = lps[torch.arange(n, device=device), cids].sum().item()
            results.append((it["sample_idx"], it["cont_idx"], total_lp, n))
        return results

    def evaluate_logprob(self, samples):
        requests = self.build_logprob_requests(samples)
        bs = self.batch_size
        batches = [requests[i:i+bs] for i in range(0, len(requests), bs)]
        counter, lock, all_results = [0], threading.Lock(), []
        t0 = time.time()

        def worker(rank, my_batches):
            for batch in my_batches:
                res = self.score_batch(rank, batch)
                with lock:
                    all_results.extend(res)
                    counter[0] += len(batch)
                    if counter[0] >= len(requests) or (counter[0] % 2000) < bs:
                        print(f"    {counter[0]}/{len(requests)}  ({counter[0]/(time.time()-t0):.0f} it/s)", flush=True)

        if self.num_gpus <= 1:
            worker(0, batches)
        else:
            gpu_b = [[] for _ in range(self.num_gpus)]
            for i, b in enumerate(batches):
                gpu_b[i % self.num_gpus].append(b)
            threads = [threading.Thread(target=worker, args=(r, gpu_b[r]), daemon=True) for r in range(self.num_gpus)]
            for t in threads: t.start()
            for t in threads: t.join()

        return aggregate_logprob(all_results, samples)

    # ── generate 评测 (HotpotQA) ──────────────────────────────

    def _generate_one(self, model, device, s):
        """用模型原生 forward + generate 生成答案。"""
        ctx_ids = self._encode(s["context_to_compress"])
        raw_stem = s.get("stem_text", "")
        task = s.get("task", "")
        if task in ("hotpotqa_fewshot", "hotpotqa_inst"):
            stem_text = raw_stem
        else:
            stem_text = HOTPOTQA_FEWSHOT_PREFIX + raw_stem
        stem_ids = self._encode(stem_text)

        context_input = ctx_ids + list(self._placeholder)
        decoder_input = list(self._placeholder) + stem_ids

        ctx_t = torch.tensor([context_input], device=device)
        dec_t = torch.tensor([decoder_input], device=device)
        cm = torch.ones_like(ctx_t)
        dm = torch.ones_like(dec_t)

        stop_ids = [self._eos_id] + self._newline_ids

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(
                input_ids=dec_t, context_ids=ctx_t,
                attention_mask=dm, context_attention_mask=cm,
                max_new_tokens=50, do_sample=False,
                pad_token_id=self._pad_id,
                eos_token_id=stop_ids,
            )

        gen_ids = output[0, len(decoder_input):]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = text.split("\n")[0].strip()
        return text

    def evaluate_generate(self, samples):
        all_results = []
        lock = threading.Lock()
        counter = [0]
        t0 = time.time()

        def worker(rank, my_items):
            model = self._models[rank]
            device = self._devices[rank]
            for si, s in my_items:
                answer = self._generate_one(model, device, s)
                with lock:
                    all_results.append((si, answer))
                    counter[0] += 1
                    if counter[0] % 200 == 0 or counter[0] >= len(samples):
                        print(f"    {counter[0]}/{len(samples)}  ({counter[0]/(time.time()-t0):.1f} it/s)", flush=True)

        gpu_items = [[] for _ in range(self.num_gpus)]
        for i, s in enumerate(samples):
            gpu_items[i % self.num_gpus].append((i, s))

        if self.num_gpus <= 1:
            worker(0, gpu_items[0])
        else:
            threads = [threading.Thread(target=worker, args=(r, gpu_items[r]), daemon=True) for r in range(self.num_gpus)]
            for t in threads: t.start()
            for t in threads: t.join()

        return aggregate_generate(all_results, samples)

    def evaluate(self, samples):
        mode = samples[0].get("eval_mode", "logprob")
        if mode == "generate":
            return self.evaluate_generate(samples)
        return self.evaluate_logprob(samples)


# ════════════════════════════════════════════════════════════════
# 标准 CausalLM 基线评估器 (多 GPU)
# ════════════════════════════════════════════════════════════════

class BaselineEvaluator:

    def __init__(self, model_path, num_gpus=1, batch_size=16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._eos_id = self.tokenizer.eos_token_id
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self._models, self._devices = [], []
        for rank in range(num_gpus):
            dev = torch.device(f"cuda:{rank}")
            print(f"  加载基线模型到 GPU {rank}...", flush=True)
            m = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map={"": dev},
                low_cpu_mem_usage=True,
            )
            m.eval()
            self._models.append(m)
            self._devices.append(dev)
        print(f"  基线模型就绪  |  {num_gpus} GPUs  |  bs={batch_size}")

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ── logprob ────────────────────────────────────────────────

    def score_batch(self, rank, items):
        model = self._models[rank]
        device = self._devices[rank]
        pad = self._pad_id
        max_len = max(len(it["full_ids"]) for it in items)
        input_ids, attn_masks = [], []
        for it in items:
            ids = it["full_ids"]
            input_ids.append(ids + [pad] * (max_len - len(ids)))
            attn_masks.append([1]*len(ids) + [0]*(max_len - len(ids)))
        inp = torch.tensor(input_ids, device=device)
        mask = torch.tensor(attn_masks, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp, attention_mask=mask).logits
        results = []
        for j, it in enumerate(items):
            n = len(it["cont_ids"])
            plen = it["prefix_len"]
            positions = torch.arange(n, device=device) + plen - 1
            positions.clamp_(min=0)
            rel = logits[j, positions].float()
            cids = torch.tensor(it["cont_ids"], device=device)
            lps = F.log_softmax(rel, dim=-1)
            total_lp = lps[torch.arange(n, device=device), cids].sum().item()
            results.append((it["sample_idx"], it["cont_idx"], total_lp, n))
        return results

    def evaluate_logprob(self, samples):
        requests = []
        for si, s in enumerate(samples):
            ctx_str = s["context_to_compress"]
            stem_str = s.get("stem_text", "")
            opts_str = s.get("options_text", "")
            prefix_str = ctx_str + stem_str + opts_str
            prefix_ids = self._encode(prefix_str)
            for ci, cont_str in enumerate(s["continuations"]):
                full_str = prefix_str + cont_str
                full_ids = self._encode(full_str)
                cont_ids = full_ids[len(prefix_ids):]
                if not cont_ids:
                    cont_ids = self._encode(cont_str)
                requests.append({
                    "full_ids": prefix_ids + cont_ids, "cont_ids": cont_ids,
                    "prefix_len": len(prefix_ids), "sample_idx": si, "cont_idx": ci,
                })

        bs = self.batch_size
        batches = [requests[i:i+bs] for i in range(0, len(requests), bs)]
        counter, lock, all_results = [0], threading.Lock(), []
        t0 = time.time()

        def worker(rank, my_batches):
            for batch in my_batches:
                res = self.score_batch(rank, batch)
                with lock:
                    all_results.extend(res)
                    counter[0] += len(batch)
                    if counter[0] >= len(requests) or (counter[0] % 2000) < bs:
                        print(f"    {counter[0]}/{len(requests)}  ({counter[0]/(time.time()-t0):.0f} it/s)", flush=True)

        if self.num_gpus <= 1:
            worker(0, batches)
        else:
            gpu_b = [[] for _ in range(self.num_gpus)]
            for i, b in enumerate(batches):
                gpu_b[i % self.num_gpus].append(b)
            threads = [threading.Thread(target=worker, args=(r, gpu_b[r]), daemon=True) for r in range(self.num_gpus)]
            for t in threads: t.start()
            for t in threads: t.join()

        return aggregate_logprob(all_results, samples)

    # ── generate ───────────────────────────────────────────────

    def evaluate_generate(self, samples):
        all_results = []
        lock = threading.Lock()
        counter = [0]
        t0 = time.time()

        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        stop_ids = [self._eos_id] + newline_ids

        def worker(rank, my_items):
            model = self._models[rank]
            device = self._devices[rank]
            for si, s in my_items:
                ctx_str = s["context_to_compress"]
                raw_stem = s.get("stem_text", "")
                task = s.get("task", "")
                if task in ("hotpotqa_fewshot", "hotpotqa_inst"):
                    stem_str = raw_stem
                else:
                    stem_str = HOTPOTQA_FEWSHOT_PREFIX + raw_stem
                input_str = ctx_str + stem_str
                input_ids = self._encode(input_str)
                inp = torch.tensor([input_ids], device=device)
                attn = torch.ones_like(inp)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.generate(
                        input_ids=inp, attention_mask=attn,
                        max_new_tokens=50, do_sample=False,
                        pad_token_id=self._pad_id,
                        eos_token_id=stop_ids,
                    )
                gen = out[0, len(input_ids):]
                answer = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                answer = answer.split("\n")[0].strip()
                with lock:
                    all_results.append((si, answer))
                    counter[0] += 1
                    if counter[0] % 200 == 0 or counter[0] >= len(samples):
                        print(f"    {counter[0]}/{len(samples)}  ({counter[0]/(time.time()-t0):.1f} it/s)", flush=True)

        gpu_items = [[] for _ in range(self.num_gpus)]
        for i, s in enumerate(samples):
            gpu_items[i % self.num_gpus].append((i, s))

        if self.num_gpus <= 1:
            worker(0, gpu_items[0])
        else:
            threads = [threading.Thread(target=worker, args=(r, gpu_items[r]), daemon=True) for r in range(self.num_gpus)]
            for t in threads: t.start()
            for t in threads: t.join()

        return aggregate_generate(all_results, samples)

    def evaluate(self, samples):
        mode = samples[0].get("eval_mode", "logprob")
        if mode == "generate":
            return self.evaluate_generate(samples)
        return self.evaluate_logprob(samples)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--num_gpus", default="1")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--output_path", default=None)
    args = ap.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    data_dir = args.data_dir or DATA_DIR
    out_dir = args.output_path or f"./eval/results/{model_name}/lm_eval_compressed_v2"

    task_list = [t.strip() for t in args.tasks.split(",")] if args.tasks else ALL_TASKS

    num_gpus = int(args.num_gpus) if args.num_gpus != "auto" else torch.cuda.device_count()
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count()))

    is_baseline = args.baseline or not os.path.isdir(os.path.join(args.model_path, "llm1"))
    if is_baseline:
        print(f"\n模式: 标准 CausalLM 基线 ({num_gpus} GPUs)")
        evaluator = BaselineEvaluator(args.model_path, num_gpus=num_gpus, batch_size=args.batch_size)
        mode_str = "baseline"
    else:
        print(f"\n模式: C3 压缩评估 ({num_gpus} GPUs)")
        evaluator = C3Evaluator(args.model_path, num_gpus=num_gpus, batch_size=args.batch_size)
        mode_str = "c3_compressed"

    t0 = time.time()
    all_results = {}
    all_meta = {}

    print(f"\n{'='*60}")
    print(f"  Compressed Benchmark v2  [{mode_str}]")
    print(f"{'='*60}")

    for task_name in task_list:
        samples = load_benchmark(task_name, data_dir)
        if not samples:
            print(f"\n  ── {task_name}: 数据不存在，跳过")
            continue

        eval_mode = samples[0].get("eval_mode", "logprob")
        toks = [s["context_tokens"] for s in samples]
        avg_t = sum(toks) / len(toks)
        num_seg = samples[0].get("num_segments", 1)
        print(f"\n  ── {task_name}  ({len(samples)} 样本, avg={avg_t:.0f} tok, "
              f"seg={num_seg}, mode={eval_mode})")

        task_t0 = time.time()
        result = evaluator.evaluate(samples)
        task_elapsed = time.time() - task_t0

        if eval_mode == "generate":
            print(f"    EM={result['em']*100:.2f}%  F1={result['f1']*100:.2f}%  "
                  f"({result['total']} 样本, {task_elapsed:.1f}s)")
            all_results[task_name] = {
                "em": result["em"], "f1": result["f1"],
                "total": result["total"], "primary_metric": "f1",
            }
        else:
            use_norm = samples[0].get("use_acc_norm", False)
            metric = "acc_norm" if use_norm else "acc"
            print(f"    {metric}={result[metric]*100:.2f}%  "
                  f"({result['total']} 样本, {task_elapsed:.1f}s)")
            all_results[task_name] = {
                "acc": result["acc"], "acc_norm": result["acc_norm"],
                "total": result["total"], "correct": result["correct"],
                "correct_norm": result["correct_norm"],
                "primary_metric": metric,
            }

        all_meta[task_name] = {
            "num_samples": len(samples),
            "avg_context_tokens": round(avg_t, 1),
            "num_segments": num_seg,
            "eval_mode": eval_mode,
            "elapsed_s": round(task_elapsed, 1),
        }

    elapsed = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)

    for task_name in all_results:
        task_output = {
            "meta": {
                "model_name": model_name, "model_path": args.model_path,
                "mode": mode_str, "elapsed_s": round(elapsed, 1),
                "task_meta": {task_name: all_meta[task_name]},
            },
            "results": {task_name: all_results[task_name]},
        }
        task_file = os.path.join(out_dir, f"v2_{task_name}.json")
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_output, f, indent=2, ensure_ascii=False)
        print(f"  已保存: {task_file}")

    print(f"\n{'='*60}")
    print(f"  结果汇总  [{mode_str}]")
    print(f"{'='*60}")
    for task_name, r in all_results.items():
        m = r["primary_metric"]
        print(f"  {task_name:20s}  {m}={r[m]*100:.2f}%  ({r['total']} 样本)")
    print(f"\n  总耗时: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
