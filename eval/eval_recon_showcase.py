"""
重建质量展示：用固定长上下文直观对比原文与重建结果。

覆盖多种文本类型：英文叙事、中文新闻、文言文、代码、无意义语料。
每条上下文 200-600 tokens，便于肉眼审阅。

用法：
    CUDA_VISIBLE_DEVICES=0 python eval/eval_recon_showcase.py \
        --model_path ./output/phase1 --model_name c3-phase1 --device cuda
"""

import argparse
import json
import os
import sys
import time
from difflib import SequenceMatcher

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

# ── 固定上下文 ─────────────────────────────────────────────────

CONTEXTS = [
    {
        "id": 1,
        "tag": "English narrative",
        "text": (
            "The old lighthouse keeper had not seen another living soul in seventeen days. "
            "Each morning he climbed the spiral staircase, checked the lamp mechanism, polished "
            "the brass fittings until they gleamed, and recorded the weather in his leather-bound "
            "journal. The sea had been unusually calm — a sheet of grey glass stretching to the "
            "horizon — but he knew from forty years of experience that such stillness often "
            "preceded the worst storms. On the eighteenth morning he noticed a dark line forming "
            "where sky met water. By noon the barometric pressure had dropped sharply, and by "
            "evening the first swells were rolling in from the northwest, each wave a little "
            "taller than the last. He lit the lamp early that night. The beam swept across the "
            "churning water every twelve seconds, a steady heartbeat against the howling wind. "
            "Somewhere out there, he knew, ships would be turning toward the light."
        ),
    },
    {
        "id": 2,
        "tag": "English technical",
        "text": (
            "Transformer architectures rely on self-attention mechanisms to capture long-range "
            "dependencies in sequential data. Unlike recurrent neural networks, which process "
            "tokens one at a time and maintain a hidden state, transformers compute attention "
            "scores between all pairs of positions simultaneously. This parallelism enables "
            "significantly faster training on modern GPU hardware. The core building block is "
            "multi-head attention: the input is projected into query, key, and value matrices "
            "through learned linear transformations, attention weights are computed as scaled "
            "dot products of queries and keys, and the output is a weighted sum of value vectors. "
            "Layer normalization and residual connections stabilize training, while positional "
            "encodings inject information about token order. Recent advances include rotary "
            "position embeddings, grouped-query attention for memory efficiency, and mixture-of-"
            "experts layers that activate only a subset of parameters per token."
        ),
    },
    {
        "id": 3,
        "tag": "中文新闻",
        "text": (
            "据新华社报道，我国科研团队在量子计算领域取得重大突破。该团队成功研制出具有"
            "超过一百个量子比特的超导量子计算原型机，在特定计算任务上的处理速度比目前最快"
            "的经典超级计算机快一亿倍以上。项目负责人介绍，这一成果标志着我国在量子计算"
            "研究方面已进入国际第一梯队。量子计算利用量子叠加和量子纠缠等特性，能够在某些"
            "问题上实现指数级加速。目前该原型机已在密码学、药物分子模拟和材料科学等领域"
            "展开初步应用测试。团队计划在未来三年内将量子比特数提升至一千个以上，并逐步"
            "实现量子纠错功能，向通用量子计算机的目标迈进。这项研究得到了国家重点研发计划"
            "的支持，相关论文已发表在国际顶级学术期刊上。"
        ),
    },
    {
        "id": 4,
        "tag": "文言文",
        "text": (
            "臣闻求木之长者，必固其根本；欲流之远者，必浚其泉源；思国之安者，必积其德义。"
            "源不深而望流之远，根不固而求木之长，德不厚而思国之理，臣虽下愚，知其不可，"
            "而况于明哲乎？人君当神器之重，居域中之大，将崇极天之峻，永保无疆之休。不念"
            "居安思危，戒奢以俭，德不处其厚，情不胜其欲，斯亦伐根以求木茂，塞源而欲流长"
            "者也。凡百元首，承天景命，莫不殷忧而道著，功成而德衰。有善始者实繁，能克终者"
            "盖寡。岂取之易而守之难乎？昔取之而有余，今守之而不足，何也？夫在殷忧，必竭"
            "诚以待下；既得志，则纵情以傲物。竭诚则胡越为一体，傲物则骨肉为行路。虽董之"
            "以严刑，振之以威怒，终苟免而不怀仁，貌恭而不心服。"
        ),
    },
    {
        "id": 5,
        "tag": "Code (Python)",
        "text": (
            "import heapq\n"
            "from collections import defaultdict\n\n"
            "class Graph:\n"
            "    def __init__(self):\n"
            "        self.adj = defaultdict(list)\n\n"
            "    def add_edge(self, u, v, weight):\n"
            "        self.adj[u].append((v, weight))\n"
            "        self.adj[v].append((u, weight))\n\n"
            "    def dijkstra(self, source):\n"
            "        dist = {source: 0}\n"
            "        pq = [(0, source)]\n"
            "        visited = set()\n"
            "        while pq:\n"
            "            d, node = heapq.heappop(pq)\n"
            "            if node in visited:\n"
            "                continue\n"
            "            visited.add(node)\n"
            "            for neighbor, w in self.adj[node]:\n"
            "                new_dist = d + w\n"
            "                if neighbor not in dist or new_dist < dist[neighbor]:\n"
            "                    dist[neighbor] = new_dist\n"
            "                    heapq.heappush(pq, (new_dist, neighbor))\n"
            "        return dist\n\n"
            "    def shortest_path(self, source, target):\n"
            "        dist = self.dijkstra(source)\n"
            "        return dist.get(target, float('inf'))\n"
        ),
    },
    {
        "id": 6,
        "tag": "Nonsense / random",
        "text": (
            "flurbo gax ziptangle 7749 womblex shree dorfkin plaxis UN-3382 quilm barnetho "
            "crendax voop zilquon 42.195 trevnax gobblewort snipfax hexaquad luminotrix "
            "blarvex pendulux fratchet 0xDEADBEEF gromphistle yazzlewick twonkberry subspleen "
            "103.72 nullvex frangiplex hortobagyi spundrel wackovia zimbotic QXJZK-8814 "
            "plimberwick snottifuge glorpmaster 3.14159265 razzledorp frumious bandersnatch "
            "mimsy borogoves outgrabe slithy toves gyre gimble wabe jabberwock frumplestein "
            "24601 zorchblade kringlefax snoodlepop bimtastic glorvenhosen xyloquartz fleep "
            "zangwidget blorftacular trumplesnark MMXXVI quibblezork 88.88 fandangleberry "
            "snorkelwhack gribblesnatch pifflesworth wonkydoodle 0b10110101 crumblefunk"
        ),
    },
    {
        "id": 7,
        "tag": "Mixed zh-en",
        "text": (
            "Machine learning的核心思想是让计算机从数据中自动学习patterns和规律。在supervised "
            "learning中，模型通过大量labeled data进行训练，学习input到output的mapping关系。"
            "常见的算法包括linear regression、decision tree和neural network等。Deep learning"
            "是其中的一个重要分支，它使用多层neural network来提取数据的hierarchical features。"
            "近年来，大语言模型(Large Language Models, LLMs)成为AI领域最热门的研究方向。"
            "这些模型通过在massive text corpus上进行pre-training，学习到了丰富的语言知识和"
            "reasoning能力。GPT系列、LLaMA、Qwen等模型在各种NLP tasks上都展现出了remarkable "
            "的performance。然而，LLMs也面临着hallucination、high computational cost和"
            "alignment等challenges。研究者们正在探索various approaches来解决这些问题。"
        ),
    },
]


# ── Reconstruction ─────────────────────────────────────────────


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


def reconstruct(model, tokenizer, context_text, latent_token_len,
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


def char_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ── Main ───────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="重建质量展示（固定上下文）")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    output = args.output or f"./eval/results/{model_name}/recon_showcase.json"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"模型: {model_name}  |  路径: {args.model_path}")
    print(f"设备: {device}  |  精度: {dtype}")

    print(f"\n加载模型...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=dtype, device_map=device.type,
        low_cpu_mem_usage=True, use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    model.eval()
    model.initialize_special_tokenizer(tokenizer, device=str(device))
    latent_token_len = model.get_model().config.latent_token_len
    stop_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f"加载耗时: {time.time() - t0:.1f}s  |  latent_token_len: {latent_token_len}\n")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    results = []

    for ctx in CONTEXTS:
        text = ctx["text"]
        tok_count = len(tokenizer(text, add_special_tokens=False).input_ids)
        max_new = min(int(tok_count * 1.5), 2048)

        print(f"{'=' * 65}")
        print(f"[{ctx['id']}/{len(CONTEXTS)}] {ctx['tag']}  ({tok_count} tokens)")
        print(f"{'=' * 65}")

        t_start = time.time()
        try:
            generated = reconstruct(
                model, tokenizer, text, latent_token_len,
                device, max_new, stop_token_id,
            )
        except Exception as e:
            print(f"  重建失败: {e}\n")
            results.append({"id": ctx["id"], "tag": ctx["tag"], "error": str(e)})
            continue
        elapsed = round(time.time() - t_start, 2)

        sim = round(char_similarity(text, generated), 4)
        gen_tok = len(tokenizer(generated, add_special_tokens=False).input_ids)
        exact = text.strip() == generated.strip()

        print(f"\n原文 ({tok_count} tok):")
        print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"\n重建 ({gen_tok} tok, {elapsed}s):")
        print(f"  {generated[:300]}{'...' if len(generated) > 300 else ''}")
        print(f"\n  相似度: {sim:.4f}  |  完全匹配: {'✓' if exact else '✗'}  "
              f"|  tokens: {tok_count}→{gen_tok}\n")

        results.append({
            "id": ctx["id"],
            "tag": ctx["tag"],
            "ref_tokens": tok_count,
            "hyp_tokens": gen_tok,
            "char_similarity": sim,
            "exact_match": exact,
            "elapsed_s": elapsed,
            "reference": text,
            "generated": generated,
        })

    output_data = {
        "model_name": model_name,
        "model_path": args.model_path,
        "results": results,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    md_path = output.replace(".json", ".md")
    md_lines = [f"# 重建质量展示 — {model_name}\n"]
    md_lines.append(f"模型路径: `{args.model_path}`\n")
    md_lines.append("| # | 类型 | tokens | 相似度 | 完全匹配 | 耗时 |")
    md_lines.append("|---|------|--------|--------|---------|------|")
    for r in results:
        if "error" in r:
            md_lines.append(f"| {r['id']} | {r['tag']} | - | ERROR | - | - |")
            continue
        em = "✓" if r["exact_match"] else "✗"
        md_lines.append(
            f"| {r['id']} | {r['tag']} | {r['ref_tokens']}→{r['hyp_tokens']} "
            f"| {r['char_similarity']:.4f} | {em} | {r['elapsed_s']}s |"
        )
    md_lines.append("")
    for r in results:
        if "error" in r:
            continue
        md_lines.append(f"---\n\n## [{r['id']}] {r['tag']}\n")
        md_lines.append(f"**原文** ({r['ref_tokens']} tokens):\n")
        md_lines.append(f"```\n{r['reference']}\n```\n")
        md_lines.append(f"**重建** ({r['hyp_tokens']} tokens, 相似度 {r['char_similarity']:.4f}):\n")
        md_lines.append(f"```\n{r['generated']}\n```\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'=' * 65}")
    print(f"  总结  |  {model_name}")
    print(f"{'=' * 65}")
    header = f"  {'#':<3s} {'类型':<20s} {'tok':>5s} {'相似度':>7s} {'匹配':>4s} {'耗时':>6s}"
    print(header)
    print(f"  {'─' * 50}")
    for r in results:
        if "error" in r:
            print(f"  {r['id']:<3d} {r['tag']:<20s} {'ERROR':>5s}")
            continue
        em = "✓" if r["exact_match"] else "✗"
        print(f"  {r['id']:<3d} {r['tag']:<20s} {r['ref_tokens']:>5d} "
              f"{r['char_similarity']:>7.4f} {em:>4s} {r['elapsed_s']:>5.1f}s")
    print(f"\n结果已保存到:")
    print(f"  JSON: {output}")
    print(f"  报告: {md_path}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
