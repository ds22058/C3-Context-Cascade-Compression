"""
对比多个模型的评估结果，生成 Markdown 格式汇总报告。

读取 <results_dir>/<model_name>/ 下的各评估结果文件，
自动识别可用的指标（PPL / lm_eval / 重建 / 生成质量），按模型横向对比。

用法：
    python eval/compare_results.py qwen25-3b c3-decoder
    python eval/compare_results.py qwen25-3b c3-decoder c3-full --output report.md
    python eval/compare_results.py qwen25-3b c3-decoder --results_dir ./eval/results
"""

import argparse
import json
import os
from pathlib import Path


# ── lm-eval 常量 ─────────────────────────────────────────────

TASK_INFO = [
    ("mmlu",          "MMLU",          ["acc,none", "acc"]),
    ("hellaswag",     "HellaSwag",     ["acc_norm,none", "acc_norm", "acc,none"]),
    ("arc_easy",      "ARC-Easy",      ["acc_norm,none", "acc_norm", "acc,none"]),
    ("arc_challenge", "ARC-Challenge", ["acc_norm,none", "acc_norm", "acc,none"]),
    ("winogrande",    "WinoGrande",    ["acc,none", "acc"]),
]


DATASET_TOKEN_INFO = {
    "wikitext-2": "64-512 tok",
    "mixed-1k": "~1024 tok",
    "mixed-2k": "~2048 tok",
    "mixed-4k": "~4096 tok",
    "mixed-8k": "~8192 tok",
}


# ── 工具函数 ─────────────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_lm_eval_results(lm_eval_dir):
    """合并 lm_eval 目录下所有 results_*.json（可能由不同 GPU 分任务产生多个文件）"""
    p = Path(lm_eval_dir)
    if not p.exists():
        return None

    candidates = list(p.rglob("results_*.json")) + list(p.rglob("results.json"))
    if not candidates:
        return None

    merged = {}
    for f in candidates:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        results = data.get("results", {})
        merged.update(results)
    return merged if merged else None


def get_score(task_results, task_key, metric_keys):
    if not task_results or task_key not in task_results:
        return None
    for k in metric_keys:
        if k in task_results[task_key]:
            return task_results[task_key][k]
    return None


def make_md_table(headers, rows):
    all_rows = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = []
    for i, row in enumerate(all_rows):
        line = "| " + " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(headers))) + " |"
        lines.append(line)
        if i == 0:
            lines.append(sep)
    return "\n".join(lines)


# ── 报告各节 ─────────────────────────────────────────────────

def section_perplexity(models, results_dir):
    ppl_data = {}
    for m in models:
        data = load_json(os.path.join(results_dir, m, "perplexity.json"))
        if data and "perplexity" in data:
            ppl_data[m] = data["perplexity"]

    if not ppl_data:
        return ""

    lines = ["## 1. WikiText-2 Perplexity\n"]
    rows = []
    base_name = models[0] if models[0] in ppl_data else None
    base_ppl = ppl_data.get(base_name) if base_name else None

    for m in models:
        ppl = ppl_data.get(m)
        if ppl is None:
            rows.append([m, "N/A", ""])
            continue
        if base_ppl is not None and m != base_name:
            diff = (ppl - base_ppl) / base_ppl * 100
            diff_str = f"{diff:+.2f}%"
        elif m == base_name:
            diff_str = "基准"
        else:
            diff_str = ""
        rows.append([m, f"{ppl:.4f}", diff_str])

    lines.append(make_md_table(["模型", "PPL ↓", "vs 基准"], rows))
    lines.append("")
    return "\n".join(lines)


def section_lm_eval(models, results_dir):
    all_results = {}
    for m in models:
        r = find_lm_eval_results(os.path.join(results_dir, m, "lm_eval"))
        if r:
            all_results[m] = r

    if not all_results:
        return ""

    lines = ["## 2. NLP 基准测试\n"]
    headers = ["任务"] + [m for m in models if m in all_results]
    rows = []
    for task_key, display_name, metric_keys in TASK_INFO:
        row = [display_name]
        for m in models:
            if m not in all_results:
                continue
            score = get_score(all_results[m], task_key, metric_keys)
            row.append(f"{score*100:.2f}%" if score is not None else "N/A")
        rows.append(row)

    lines.append(make_md_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def _build_recon_section(title, models, recon_data):
    """通用重建准确率表格构建。"""
    if not recon_data:
        return ""

    all_ds_names = []
    for m in recon_data:
        for ds in recon_data[m]:
            if ds not in all_ds_names:
                all_ds_names.append(ds)

    lines = [f"{title}\n"]
    metric_labels = [
        ("char_precision", "Char Precision ↑"),
        ("rouge_l", "ROUGE-L ↑"),
        ("token_f1", "Token F1 ↑"),
        ("exact_match", "Exact Match"),
    ]

    for ds_name in all_ds_names:
        token_info = DATASET_TOKEN_INFO.get(ds_name, "")
        ds_label = f"{ds_name} ({token_info})" if token_info else ds_name
        lines.append(f"### {ds_label}\n")
        headers = ["指标"] + [m for m in models if ds_name in recon_data.get(m, {})]
        rows = []
        for key, label in metric_labels:
            row = [label]
            for m in models:
                ds_metrics = recon_data.get(m, {}).get(ds_name)
                if ds_metrics is None:
                    continue
                val = ds_metrics.get(key)
                if val is None:
                    row.append("N/A")
                elif key == "exact_match":
                    row.append(f"{val*100:.1f}%")
                else:
                    row.append(f"{val:.4f}")
            rows.append(row)

        sample_row = ["样本数 / 平均 tokens"]
        for m in models:
            ds_metrics = recon_data.get(m, {}).get(ds_name)
            if ds_metrics is None:
                continue
            n = ds_metrics.get("num_samples", "?")
            avg_tok = ds_metrics.get("avg_ref_tokens", "?")
            sample_row.append(f"{n} / {avg_tok}")
        rows.append(sample_row)

        lines.append(make_md_table(headers, rows))
        lines.append("")

    return "\n".join(lines)


def section_reconstruction(models, results_dir):
    recon_data = {}
    for m in models:
        data = load_json(os.path.join(results_dir, m, "reconstruction.json"))
        if data and "datasets" in data:
            recon_data[m] = data["datasets"]
    return _build_recon_section("## 3. 重建准确率（encoder 压缩→重建）", models, recon_data)


def section_reconstruction_decoder(models, results_dir):
    recon_data = {}
    for m in models:
        data = load_json(os.path.join(results_dir, m, "reconstruction_decoder.json"))
        if data and "datasets" in data:
            recon_data[m] = data["datasets"]
    return _build_recon_section("## 4. 重建准确率（decoder-only 复述）", models, recon_data)


def section_lm_eval_compressed(models, results_dir):
    all_results = {}
    for m in models:
        compressed_dir = os.path.join(results_dir, m, "lm_eval_compressed")
        if not os.path.isdir(compressed_dir):
            continue
        for fname in os.listdir(compressed_dir):
            if fname.startswith("results_compressed") and fname.endswith(".json"):
                data = load_json(os.path.join(compressed_dir, fname))
                if data and "results" in data:
                    threshold = data.get("meta", {}).get("compress_threshold", "?")
                    all_results[m] = {"results": data["results"], "threshold": threshold}
                    break

    if not all_results:
        return ""

    thresholds = set(v["threshold"] for v in all_results.values())
    threshold_str = "/".join(str(t) for t in thresholds)

    lines = [f"## 4. Context-Compressed NLP 基准 (threshold={threshold_str})\n"]
    headers = ["任务"] + [m for m in models if m in all_results]
    rows = []
    for task_key, display_name, metric_keys in TASK_INFO:
        row = [display_name]
        for m in models:
            if m not in all_results:
                continue
            score = get_score(all_results[m]["results"], task_key, metric_keys)
            row.append(f"{score*100:.2f}%" if score is not None else "N/A")
        rows.append(row)

    lines.append(make_md_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def section_generation(models, results_dir):
    gen_data = {}
    for m in models:
        data = load_json(os.path.join(results_dir, m, "generation.json"))
        if data and "results" in data:
            gen_data[m] = data["results"]

    if not gen_data:
        return ""

    lines = ["## 5. 生成质量\n"]

    first_model = next(iter(gen_data))
    for entry in gen_data[first_model]:
        pid = entry.get("prompt_id", "?")
        prompt = entry.get("prompt", "")
        lines.append(f"### Prompt {pid}\n")
        lines.append(f"```\n{prompt}\n```\n")
        for m in gen_data:
            results = gen_data[m]
            match = next((r for r in results if r.get("prompt_id") == pid), None)
            if match:
                text = match.get("generated", "N/A")
                elapsed = match.get("elapsed_s", 0)
                lines.append(f"**{m}**（{elapsed}s）：\n")
                lines.append(f"```\n{text[:500]}{'...' if len(text) > 500 else ''}\n```\n")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对比多个模型的评估结果",
        usage="python eval/compare_results.py model1 model2 [model3 ...] [--results_dir DIR] [--output PATH]",
    )
    parser.add_argument("models", nargs="+", help="要对比的模型名称（对应 results_dir 下的子目录）")
    parser.add_argument("--results_dir", default="./eval/results", help="结果根目录")
    parser.add_argument("--output", default="./eval/results/summary.md", help="报告输出路径")
    args = parser.parse_args()

    # 检查模型目录
    for m in args.models:
        d = os.path.join(args.results_dir, m)
        if not os.path.isdir(d):
            print(f"[警告] 模型结果目录不存在: {d}")

    sections = [
        f"# 模型评估对比报告\n",
        f"> 对比模型: {', '.join(args.models)}\n",
        section_perplexity(args.models, args.results_dir),
        section_lm_eval(args.models, args.results_dir),
        section_reconstruction(args.models, args.results_dir),
        section_lm_eval_compressed(args.models, args.results_dir),
        section_generation(args.models, args.results_dir),
    ]

    report = "\n".join(s for s in sections if s)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n{'='*55}")
    print(f"报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
