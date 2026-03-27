"""
下载 8 个 NLP benchmark 的原始数据到 eval/benchmark_data/raw/。

数据集列表：
  MMLU          cais/mmlu (all)           test + validation(dev)
  HellaSwag     Rowan/hellaswag           validation + train
  ARC-Easy      allenai/ai2_arc (Easy)    test + train
  ARC-Challenge allenai/ai2_arc (Chall.)  test + train
  WinoGrande    allenai/winogrande (xl)   validation + train
  RACE          ehovy/race (high)         test + train
  BoolQ         google/boolq              validation + train
  HotpotQA      hotpot_qa (distractor)    validation + train

用法：
    python eval/benchmark_data/download.py
    HF_ENDPOINT=https://hf-mirror.com python eval/benchmark_data/download.py
"""

import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"    saved {len(rows)} rows → {path}")


def download_one(name, hf_path, config, splits_map):
    """Download one dataset. splits_map: {local_name: hf_split_name}."""
    out_dir = os.path.join(RAW_DIR, name)
    all_exist = all(
        os.path.exists(os.path.join(out_dir, f"{local}.jsonl"))
        for local in splits_map
    )
    if all_exist:
        print(f"  [跳过] {name} (已存在)")
        return

    from datasets import load_dataset

    print(f"  下载 {name} ({hf_path}, config={config}) ...")
    ds = load_dataset(hf_path, config, trust_remote_code=True)
    for local_name, hf_split in splits_map.items():
        rows = [dict(row) for row in ds[hf_split]]
        save_jsonl(rows, os.path.join(out_dir, f"{local_name}.jsonl"))


def main():
    print(f"数据目录: {RAW_DIR}\n")

    tasks = [
        ("mmlu",          "cais/mmlu",            "all",            {"test": "test", "dev": "validation"}),
        ("hellaswag",     "Rowan/hellaswag",      None,             {"test": "validation", "train": "train"}),
        ("arc_easy",      "allenai/ai2_arc",      "ARC-Easy",       {"test": "test", "train": "train"}),
        ("arc_challenge", "allenai/ai2_arc",      "ARC-Challenge",  {"test": "test", "train": "train"}),
        ("winogrande",    "allenai/winogrande",   "winogrande_xl",  {"test": "validation", "train": "train"}),
        ("race",          "ehovy/race",           "high",           {"test": "test", "train": "train"}),
        ("boolq",         "google/boolq",         None,             {"test": "validation", "train": "train"}),
        ("hotpotqa",      "hotpot_qa",            "distractor",     {"test": "validation", "train": "train"}),
    ]

    for name, hf_path, config, splits in tasks:
        try:
            download_one(name, hf_path, config, splits)
        except Exception as e:
            print(f"  [失败] {name}: {e}")
            print(f"         提示: export HF_ENDPOINT=https://hf-mirror.com")

    print("\n完成。")


if __name__ == "__main__":
    main()
