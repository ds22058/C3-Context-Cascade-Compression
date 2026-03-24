"""
汇总所有评估结果，生成 Markdown 格式的对比报告。

读取：
  - eval/results/perplexity.json           (eval_perplexity.py 输出)
  - eval/results/lm_eval_qwen25_3b/        (lm-eval-harness 对 Qwen 的输出目录)
  - eval/results/lm_eval_c3_decoder/       (lm-eval-harness 对 C3-Decoder 的输出目录)
  - eval/results/generation.json           (eval_generation.py 输出)

输出：
  - eval/results/summary.md                Markdown 汇总报告
  - 标准输出：终端打印

用法：
    python eval/analyze_results.py
    python eval/analyze_results.py --lm_eval_qwen ./eval/results/lm_eval_qwen25_3b_cpu
"""

import argparse
import json
import os
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# lm-eval-harness 结果解析
# ──────────────────────────────────────────────────────────────

# (task_key, display_name, metric_keys_to_try)
TASK_INFO = [
    ("mmlu",          "MMLU (5-shot)",         ["acc,none", "acc"]),
    ("hellaswag",     "HellaSwag (10-shot)",    ["acc_norm,none", "acc_norm", "acc,none"]),
    ("arc_easy",      "ARC-Easy (25-shot)",     ["acc_norm,none", "acc_norm", "acc,none"]),
    ("arc_challenge", "ARC-Challenge (25-shot)", ["acc_norm,none", "acc_norm", "acc,none"]),
    ("winogrande",    "WinoGrande (5-shot)",    ["acc,none", "acc"]),
]


def find_latest_lm_eval_json(results_dir: str) -> dict | None:
    """在 lm-eval 输出目录中找到最新的 results_*.json 并解析"""
    p = Path(results_dir)
    if not p.exists():
        return None

    # lm-eval v0.4 输出格式：results_dir/**/**/results_<timestamp>.json
    candidates = list(p.rglob("results_*.json")) + list(p.rglob("results.json"))
    if not candidates:
        # 也可能直接是 results_dir/model_name_hash/results_timestamp.json
        candidates = list(p.glob("**/*.json"))

    if not candidates:
        return None

    latest = max(candidates, key=lambda f: f.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        data = json.load(f)

    return data.get("results", {})


def get_score(task_results: dict, task_key: str, metric_keys: list[str]) -> float | None:
    if task_key not in task_results:
        return None
    task_data = task_results[task_key]
    for k in metric_keys:
        if k in task_data:
            return task_data[k]
    return None


# ──────────────────────────────────────────────────────────────
# 表格渲染
# ──────────────────────────────────────────────────────────────

def make_md_table(headers: list[str], rows: list[list]) -> str:
    all_rows = [headers] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = []
    for i, row in enumerate(all_rows):
        line = "| " + " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(headers))) + " |"
        lines.append(line)
        if i == 0:
            lines.append(sep)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 报告各节
# ──────────────────────────────────────────────────────────────

def section_perplexity(ppl_file: str) -> str:
    if not os.path.exists(ppl_file):
        return "## 1. WikiText-2 Perplexity\n\n> 未找到结果文件：`{}`\n".format(ppl_file)

    with open(ppl_file, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["## 1. WikiText-2 Perplexity\n"]
    q_info = data.get("Qwen2.5-3B", {})
    c_info = data.get("C3-Decoder", {})
    q_ppl = q_info.get("perplexity")
    c_ppl = c_info.get("perplexity")

    rows = []
    if q_ppl is not None:
        rows.append(["Qwen2.5-3B (原始)", f"{q_ppl:.4f}", "基准"])
    if c_ppl is not None:
        if q_ppl is not None:
            diff = (c_ppl - q_ppl) / q_ppl * 100
            diff_str = f"{diff:+.2f}% ({'↑ 退化' if diff > 0 else '↓ 提升'})"
        else:
            diff_str = "N/A"
        rows.append(["C3-Decoder", f"{c_ppl:.4f}", diff_str])

    if rows:
        lines.append(make_md_table(["模型", "PPL ↓", "vs 基准"], rows))
        lines.append("")

    if q_ppl is not None and c_ppl is not None:
        diff = (c_ppl - q_ppl) / q_ppl * 100
        if abs(diff) < 5:
            conclusion = f"PPL 差异 **{diff:+.2f}%**，C3 训练对语言建模能力影响**可忽略**（< 5%）。"
        elif abs(diff) < 15:
            conclusion = f"PPL 差异 **{diff:+.2f}%**，C3 训练对语言建模能力有**中等程度**影响（5%~15%）。"
        else:
            conclusion = f"PPL 差异 **{diff:+.2f}%**，C3 训练对语言建模能力有**显著影响**（> 15%）。"
        lines.append(f"> **结论**：{conclusion}")
        lines.append("")

    max_samples = q_info.get("max_samples") or c_info.get("max_samples")
    if max_samples:
        lines.append(f"> ⚠️ 本次使用了 `--max_samples {max_samples}` 快速模式，结果仅供参考。\n")

    return "\n".join(lines)


def section_lm_eval(qwen_dir: str, c3_dir: str) -> str:
    qwen_results = find_latest_lm_eval_json(qwen_dir)
    c3_results = find_latest_lm_eval_json(c3_dir)

    if qwen_results is None and c3_results is None:
        return (
            "## 2. NLP 基准测试 (lm-evaluation-harness)\n\n"
            f"> 未找到结果目录：`{qwen_dir}` 或 `{c3_dir}`\n"
            "> 请先运行 lm-eval 命令（见 README）。\n"
        )

    lines = ["## 2. NLP 基准测试 (lm-evaluation-harness)\n"]
    rows = []

    for task_key, display_name, metric_keys in TASK_INFO:
        q_score = get_score(qwen_results or {}, task_key, metric_keys)
        c_score = get_score(c3_results or {}, task_key, metric_keys)

        q_str = f"{q_score*100:.2f}%" if q_score is not None else "N/A"
        c_str = f"{c_score*100:.2f}%" if c_score is not None else "N/A"

        if q_score is not None and c_score is not None:
            diff = (c_score - q_score) * 100
            diff_str = f"{diff:+.2f}pp ({'↓ 退化' if diff < 0 else '↑ 提升'})"
        else:
            diff_str = "N/A"

        rows.append([display_name, q_str, c_str, diff_str])

    lines.append(make_md_table(["任务", "Qwen2.5-3B ↑", "C3-Decoder ↑", "差异"], rows))
    lines.append("")

    # 总结
    valid_diffs = []
    for task_key, _, metric_keys in TASK_INFO:
        q_s = get_score(qwen_results or {}, task_key, metric_keys)
        c_s = get_score(c3_results or {}, task_key, metric_keys)
        if q_s is not None and c_s is not None:
            valid_diffs.append((c_s - q_s) * 100)

    if valid_diffs:
        avg_diff = sum(valid_diffs) / len(valid_diffs)
        lines.append(
            f"> **平均差异**：{avg_diff:+.2f}pp  "
            + ("（退化）" if avg_diff < 0 else "（提升）")
        )
        lines.append("")

    return "\n".join(lines)


def section_reconstruction(recon_file: str) -> str:
    section_num = "3"
    header = f"## {section_num}. WikiText-2 重建准确率（压缩→重建）\n"

    if not os.path.exists(recon_file):
        return header + f"\n> 未找到结果文件：`{recon_file}`\n"

    with open(recon_file, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    if "error" in summary:
        return header + f"\n> 错误：{summary['error']}\n"

    lines = [header]
    lines.append(
        "> C3 模型将文本压缩为 latent tokens 后重建，"
        "衡量压缩信息的完整性。\n"
    )

    metric_labels = [
        ("char_precision", "Char Precision ↑"),
        ("rouge_l", "ROUGE-L ↑"),
        ("token_f1", "Token F1 ↑"),
        ("exact_match", "完全匹配率"),
    ]
    rows = []
    for key, label in metric_labels:
        info = summary.get(key, {})
        mean_val = info.get("mean")
        if mean_val is None:
            rows.append([label, "N/A", "N/A", "N/A"])
            continue
        if key == "exact_match":
            rows.append([
                label,
                f"{mean_val*100:.1f}%",
                f"{info.get('min', 0)*100:.0f}%",
                f"{info.get('max', 0)*100:.0f}%",
            ])
        else:
            rows.append([
                label,
                f"{mean_val:.4f}",
                f"{info.get('min', 0):.4f}",
                f"{info.get('max', 0):.4f}",
            ])

    lines.append(make_md_table(["指标", "均值", "最小", "最大"], rows))
    lines.append("")

    n_samples = summary.get("num_samples", "?")
    max_tok = summary.get("max_tokens", "?")
    lines.append(f"> 样本数: {n_samples}，上下文上限: {max_tok} tokens\n")

    buckets = summary.get("length_buckets", {})
    if buckets:
        lines.append("**按上下文长度分桶：**\n")
        bucket_rows = []
        for bname, binfo in buckets.items():
            bucket_rows.append([
                bname,
                str(binfo.get("count", 0)),
                f"{binfo.get('rouge_l_mean', 0):.4f}",
            ])
        lines.append(make_md_table(["长度", "样本数", "ROUGE-L"], bucket_rows))
        lines.append("")

    # reconstruction examples
    samples = data.get("samples", [])
    if samples:
        lines.append("**重建示例（前 3 条）：**\n")
        for s in samples[:3]:
            ref = s.get("reference", "")[:300]
            gen = s.get("generated", "")[:300]
            m = s.get("metrics", {})
            lines.append(f"- **原文**：`{ref}{'...' if len(s.get('reference',''))>300 else ''}`")
            lines.append(f"- **重建**：`{gen}{'...' if len(s.get('generated',''))>300 else ''}`")
            lines.append(
                f"- ROUGE-L={m.get('rouge_l',0):.4f}, "
                f"Token-F1={m.get('token_f1',0):.4f}\n"
            )

    return "\n".join(lines)


def section_generation(gen_file: str) -> str:
    if not os.path.exists(gen_file):
        return "## 4. 生成质量对比\n\n> 未找到结果文件：`{}`\n".format(gen_file)

    with open(gen_file, encoding="utf-8") as f:
        data = json.load(f)

    lines = ["## 4. 生成质量对比\n"]
    for entry in data:
        pid = entry.get("prompt_id", "?")
        prompt = entry.get("prompt", "")
        outputs = entry.get("outputs", {})
        lines.append(f"### Prompt {pid}\n")
        lines.append(f"```\n{prompt}\n```\n")
        for name, r in outputs.items():
            text = r.get("generated", "N/A")
            elapsed = r.get("elapsed_s", 0)
            lines.append(f"**{name}**（耗时 {elapsed}s）：\n")
            lines.append(f"```\n{text[:600]}{'...' if len(text) > 600 else ''}\n```\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="汇总 C3 Decoder 评估结果")
    parser.add_argument("--ppl_file",      default="./eval/results/perplexity.json")
    parser.add_argument("--recon_file",    default="./eval/results/reconstruction.json")
    parser.add_argument("--lm_eval_qwen",  default="./eval/results/lm_eval_qwen25_3b")
    parser.add_argument("--lm_eval_c3",    default="./eval/results/lm_eval_c3_decoder")
    parser.add_argument("--gen_file",      default="./eval/results/generation.json")
    parser.add_argument("--output",        default="./eval/results/summary.md")
    args = parser.parse_args()

    sections = [
        "# C3 Decoder 评估报告\n",
        "> **评估目标**：(1) 验证 C3 的上下文压缩→重建质量；"
        "(2) 将 C3 的 decoder 权重提取为独立 Qwen2 模型，"
        "与原始 Qwen2.5-3B 在相同基准上对比，验证 C3 训练是否损害了通用语言能力。\n",
        section_perplexity(args.ppl_file),
        section_lm_eval(args.lm_eval_qwen, args.lm_eval_c3),
        section_reconstruction(args.recon_file),
        section_generation(args.gen_file),
    ]

    report = "\n".join(sections)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n{'='*55}")
    print(f"报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
